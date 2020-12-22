import argparse
import gc
import math
import os
import random
import sys
import time

import numpy as np
import torch as th
import wandb
from contrastive_learner import ContrastiveLearner, RandomApply
from kornia import augmentation as augs
from scipy.ndimage import gaussian_filter
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm

import validation
from augment import augment
from dataset import MultiResolutionDataset
from distributed import get_rank, reduce_loss_dict, reduce_sum, synchronize
from lookahead_minimax import LookaheadMinimax
from models.stylegan2 import Discriminator, Generator

sys.path.insert(0, "../lookahead_minimax")


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.5 ** (32.0 / 10_000)):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())
    for name, param in model1.named_parameters():
        param.data = decay * par1[name].data + (1 - decay) * par2[name].data


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def make_noise(batch_size, latent_dim, prob):
    if prob > 0 and random.random() < prob:
        return th.randn(2, batch_size, latent_dim, device=device).unbind(0)
    else:
        return [th.randn(batch_size, latent_dim, device=device)]


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()


def d_r1_penalty(real_img, real_pred, args):
    (grad_real,) = th.autograd.grad(real_pred.sum(), real_img, create_graph=True)
    r1_loss = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    r1_loss = r1_loss / 2.0 + 0 * real_pred[0]
    return r1_loss


def g_non_saturating_loss(fake_pred):
    return F.softplus(-fake_pred).mean()


def g_path_length_regularization(generator, mean_path_length, args):
    path_batch_size = max(1, args.batch_size // args.path_batch_shrink)

    noise = make_noise(path_batch_size, args.latent_size, args.mixing_prob)
    fake_img, latents = generator(noise, return_latents=True)

    img_noise = th.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    noisy_img_sum = (fake_img * img_noise).sum()

    (grad,) = th.autograd.grad(noisy_img_sum, latents, create_graph=True)

    path_lengths = th.sqrt(grad.pow(2).sum(2).mean(1))
    path_mean = mean_path_length + 0.01 * (path_lengths.mean() - mean_path_length)
    path_loss = (path_lengths - path_mean).pow(2).mean()
    if not th.isnan(path_mean):
        mean_path_length = path_mean.detach()

    if args.path_batch_shrink:
        path_loss += 0 * fake_img[0, 0, 0, 0]

    return path_loss, mean_path_length


def train(args, loader, generator, discriminator, contrast_learner, g_optim, d_optim, g_ema):
    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
        if contrast_learner is not None:
            cl_module = contrast_learner.module
    else:
        g_module = generator
        d_module = discriminator
        cl_module = contrast_learner

    loader = sample_data(loader)
    sample_z = th.randn(args.n_sample, args.latent_size, device=device)
    mse = th.nn.MSELoss()
    mean_path_length = th.cuda.FloatTensor([0.0])
    ada_aug_signs = th.cuda.FloatTensor([0.0])
    ada_aug_n = th.cuda.FloatTensor([0.0])
    ada_aug_p = th.cuda.FloatTensor([args.augment_p if args.augment_p > 0 else 0.0])
    ada_aug_step = th.cuda.FloatTensor([args.ada_target / args.ada_length])
    r_t_stat = th.cuda.FloatTensor([0.0])
    fids = []

    loss_dict = {
        "Generator": th.cuda.FloatTensor([0.0]),
        "Discriminator": th.cuda.FloatTensor([0.0]),
        "Real Score": th.cuda.FloatTensor([0.0]),
        "Fake Score": th.cuda.FloatTensor([0.0]),
        "Contrastive": th.cuda.FloatTensor([0.0]),
        "Consistency": th.cuda.FloatTensor([0.0]),
        "R1 Penalty": th.cuda.FloatTensor([0.0]),
        "Path Length Regularization": th.cuda.FloatTensor([0.0]),
        "Augment": th.cuda.FloatTensor([0.0]),
        "Rt": th.cuda.FloatTensor([0.0]),
    }

    pbar = range(args.iter)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0)
    for idx in pbar:
        i = idx + args.start_iter
        if i > args.iter:
            print("Done!")
            break
        tick_start = time.time()

        for k, v in loss_dict.items():
            loss_dict[k].mul_(0)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        discriminator.zero_grad()
        for _ in range(args.num_accumulate):
            real_img_og = next(loader).to(device, non_blocking=True)

            noise = make_noise(args.batch_size, args.latent_size, args.mixing_prob)
            fake_img_og, _ = generator(noise)
            if args.augment:
                fake_img, _ = augment(fake_img_og, ada_aug_p)
                real_img, _ = augment(real_img_og, ada_aug_p)
            else:
                fake_img = fake_img_og
                real_img = real_img_og

            fake_pred = discriminator(fake_img)
            real_pred = discriminator(real_img)
            logistic_loss = d_logistic_loss(real_pred, fake_pred)
            loss_dict["Discriminator"] += logistic_loss.detach()
            loss_dict["Real Score"] += real_pred.mean().detach()
            loss_dict["Fake Score"] += fake_pred.mean().detach()
            d_loss = logistic_loss

            if args.contrastive > 0:
                contrast_learner(fake_img_og, fake_img, accumulate=True)
                contrast_learner(real_img_og, real_img, accumulate=True)
                contrast_loss = cl_module.calculate_loss()
                loss_dict["Contrastive"] += contrast_loss.detach()
                d_loss += args.contrastive * contrast_loss

            if args.balanced_consistency > 0:
                consistency_loss = mse(real_pred, discriminator(real_img_og)) + mse(
                    fake_pred, discriminator(fake_img_og)
                )
                loss_dict["Consistency"] += consistency_loss.detach()
                d_loss += args.balanced_consistency * consistency_loss

            d_loss /= args.num_accumulate
            d_loss.backward()
        d_optim.step()

        if args.r1 > 0 and i % args.d_reg_every == 0:
            discriminator.zero_grad()
            for _ in range(args.num_accumulate):
                real_img = next(loader).to(device, non_blocking=True)
                real_img.requires_grad = True
                real_pred = discriminator(real_img)
                r1_loss = d_r1_penalty(real_img, real_pred, args)
                loss_dict["R1 Penalty"] += r1_loss.detach().squeeze()
                r1_loss = args.r1 * args.d_reg_every * r1_loss / args.num_accumulate
                r1_loss.backward()
            d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_signs += th.sign(real_pred).sum().item()
            ada_aug_n += real_pred.shape[0]
            ada_aug_signs, ada_aug_n = reduce_sum(ada_aug_signs), reduce_sum(ada_aug_n)

            if ada_aug_n > 255:
                r_t_stat = ada_aug_signs / ada_aug_n
                loss_dict["Rt"] += r_t_stat
                if r_t_stat > args.ada_target:
                    sign = 1
                else:
                    sign = -1

                ada_aug_p += sign * ada_aug_step * ada_aug_n
                ada_aug_p = th.clamp(ada_aug_p, 0, 1)
                ada_aug_signs.mul_(0)
                ada_aug_n.mul_(0)
                loss_dict["Augment"] += ada_aug_p

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        generator.zero_grad()
        for _ in range(args.num_accumulate):
            noise = make_noise(args.batch_size, args.latent_size, args.mixing_prob)
            fake_img, _ = generator(noise)
            if args.augment:
                fake_img, _ = augment(fake_img, ada_aug_p)
            fake_pred = discriminator(fake_img)
            g_loss = g_non_saturating_loss(fake_pred)
            loss_dict["Generator"] += g_loss.detach()
            g_loss /= args.num_accumulate
            g_loss.backward()
        g_optim.step()

        if args.path_regularize > 0 and i % args.g_reg_every == 0:
            generator.zero_grad()
            for _ in range(args.num_accumulate):
                path_loss, mean_path_length = g_path_length_regularization(generator, mean_path_length, args)
                loss_dict["Path Length Regularization"] += path_loss.detach()
                path_loss = args.path_regularize * args.g_reg_every * path_loss / args.num_accumulate
                path_loss.backward()
            g_optim.step()

        accumulate(g_ema, g_module)

        loss_reduced = reduce_loss_dict(loss_dict)
        log_dict = {k: v.mean().item() / args.num_accumulate for k, v in loss_reduced.items() if v != 0}
        log_dict["Tick Length"] = time.time() - tick_start

        if get_rank() == 0:
            with th.no_grad():
                if args.log_spec_norm:
                    G_norms = []
                    for name, spec_norm in g_module.named_buffers():
                        if "spectral_norm" in name:
                            G_norms.append(spec_norm.cpu().numpy())
                    G_norms = np.array(G_norms)
                    D_norms = []
                    for name, spec_norm in d_module.named_buffers():
                        if "spectral_norm" in name:
                            D_norms.append(spec_norm.cpu().numpy())
                    D_norms = np.array(D_norms)
                    log_dict[f"Spectral Norms/G min spectral norm"] = np.log(G_norms).min()
                    log_dict[f"Spectral Norms/G mean spectral norm"] = np.log(G_norms).mean()
                    log_dict[f"Spectral Norms/G max spectral norm"] = np.log(G_norms).max()
                    log_dict[f"Spectral Norms/D min spectral norm"] = np.log(D_norms).min()
                    log_dict[f"Spectral Norms/D mean spectral norm"] = np.log(D_norms).mean()
                    log_dict[f"Spectral Norms/D max spectral norm"] = np.log(D_norms).max()

                if args.img_every != -1 and i % args.img_every == 0:
                    g_ema.eval()
                    sample = []
                    for sub in range(0, len(sample_z), args.batch_size):
                        subsample, _ = g_ema([sample_z[sub : sub + args.batch_size]])
                        sample.append(subsample.detach().cpu())
                    sample = th.cat(sample).detach()
                    grid = utils.make_grid(sample, nrow=10, normalize=True, range=(-1, 1))
                    log_dict["Generated Images EMA"] = [wandb.Image(grid, caption=f"Step {i}")]

                if args.eval_every != -1 and i % args.eval_every == 0:
                    fid_dict = validation.fid(
                        g_ema, args.val_batch_size, args.fid_n_sample, args.fid_truncation, args.name
                    )

                    fid = fid_dict["FID"]
                    fids.append(fid)
                    density = fid_dict["Density"]
                    coverage = fid_dict["Coverage"]

                    ppl = validation.ppl(
                        g_ema, args.val_batch_size, args.ppl_n_sample, args.ppl_space, args.ppl_crop, args.latent_size,
                    )

                    log_dict["Evaluation/FID"] = fid
                    log_dict["Sweep/FID_smooth"] = gaussian_filter(np.array(fids), [5])[-1]
                    log_dict["Evaluation/Density"] = density
                    log_dict["Evaluation/Coverage"] = coverage
                    log_dict["Evaluation/PPL"] = ppl

                wandb.log(log_dict)

                if args.eval_every != -1:
                    description = (
                        f"FID: {fid:.4f}   PPL: {ppl:.4f}   Dens: {density:.4f}   Cov: {coverage:.4f}   "
                        + f"G: {log_dict['Generator']:.4f}   D: {log_dict['Discriminator']:.4f}"
                    )
                else:
                    description = f"G: {log_dict['Generator']:.4f}   D: {log_dict['Discriminator']:.4f}"
                if "Augment" in log_dict:
                    description += f"   Aug: {log_dict['Augment']:.4f}"  #   Rt: {log_dict['Rt']:.4f}"
                if "R1 Penalty" in log_dict:
                    description += f"   R1: {log_dict['R1 Penalty']:.4f}"
                if "Path Length Regularization" in log_dict:
                    description += f"   Path: {log_dict['Path Length Regularization']:.4f}"
                pbar.set_description(description)

                if i % args.checkpoint_every == 0:
                    check_name = "-".join(
                        [
                            args.name,
                            args.wbname,
                            wandb.run.dir.split("/")[-1].split("-")[-1],
                            # str(int(fid)),
                            str(args.size),
                            str(i).zfill(6),
                        ]
                    )
                    th.save(
                        {
                            "g": g_module.state_dict(),
                            "d": d_module.state_dict(),
                            # "cl": cl_module.state_dict(),
                            "g_ema": g_ema.state_dict(),
                            "g_optim": g_optim.state_dict(),
                            "d_optim": d_optim.state_dict(),
                        },
                        f"/home/hans/modelzoo/maua-sg2/{check_name}.pt",
                    )

        if args.profile_mem:
            gpu_profile(frame=sys._getframe(), event="line", arg=None)


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()

    parser.add_argument("--wbname", type=str, required=True)
    parser.add_argument("--wbproj", type=str, required=True)
    parser.add_argument("--wbgroup", type=str, default=None)

    # data options
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--vflip", type=bool, default=False)
    parser.add_argument("--hflip", type=bool, default=True)

    # training options
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_accumulate", type=int, default=1)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--transfer_mapping_only", type=bool, default=False)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--iter", type=int, default=20_000)

    # model options
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--min_rgb_size", type=int, default=4)
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--n_sample", type=int, default=60)
    parser.add_argument("--constant_input", type=bool, default=False)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--d_skip", type=bool, default=True)

    # optimizer options
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--d_lr_ratio", type=float, default=1.0)
    parser.add_argument("--lookahead", type=bool, default=True)
    parser.add_argument("--la_steps", type=float, default=500)
    parser.add_argument("--la_alpha", type=float, default=0.5)

    # loss options
    parser.add_argument("--r1", type=float, default=1e-5)
    parser.add_argument("--path_regularize", type=float, default=1)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing_prob", type=float, default=0.666)

    # augmentation options
    parser.add_argument("--augment", type=bool, default=True)
    parser.add_argument("--contrastive", type=float, default=0)
    parser.add_argument("--balanced_consistency", type=float, default=0)
    parser.add_argument("--augment_p", type=float, default=0)
    parser.add_argument("--ada_target", type=float, default=0.6)
    parser.add_argument("--ada_length", type=int, default=15_000)

    # validation options
    parser.add_argument("--val_batch_size", type=int, default=6)
    parser.add_argument("--fid_n_sample", type=int, default=2500)
    parser.add_argument("--fid_truncation", type=float, default=None)
    parser.add_argument("--ppl_space", choices=["z", "w"], default="w")
    parser.add_argument("--ppl_n_sample", type=int, default=1250)
    parser.add_argument("--ppl_crop", type=bool, default=False)

    # logging options
    parser.add_argument("--log_spec_norm", type=bool, default=False)
    parser.add_argument("--img_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=-1)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--profile_mem", action="store_true")

    # (multi-)GPU options
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--cudnn_benchmark", type=bool, default=True)

    args = parser.parse_args()
    if args.balanced_consistency > 0 or args.contrastive > 0:
        args.augment = True
    args.name = os.path.splitext(os.path.basename(args.path))[0]
    args.r1 = args.r1 * args.size ** 2

    args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    th.backends.cudnn.benchmark = args.cudnn_benchmark
    args.distributed = args.num_gpus > 1

    # code for updating wandb configs that were incorrect
    # if args.local_rank == 0:
    #     api = wandb.Api()
    #     run = api.run("wav/temperatuur/7kp6g0zt")
    #     run.config = vars(args)
    #     run.update()
    # exit()

    if args.distributed:
        th.cuda.set_device(args.local_rank)
        th.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    generator = Generator(
        args.size,
        args.latent_size,
        args.n_mlp,
        channel_multiplier=args.channel_multiplier,
        constant_input=args.constant_input,
        min_rgb_size=args.min_rgb_size,
    ).to(device, non_blocking=True)
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier, use_skip=args.d_skip).to(
        device
    )

    if args.log_spec_norm:
        for name, parameter in generator.named_parameters():
            if "weight" in name and parameter.squeeze().dim() > 1:
                mod = generator
                for attr in name.replace(".weight", "").split("."):
                    mod = getattr(mod, attr)
                validation.track_spectral_norm(mod)
        for name, parameter in discriminator.named_parameters():
            if "weight" in name and parameter.squeeze().dim() > 1:
                mod = discriminator
                for attr in name.replace(".weight", "").split("."):
                    mod = getattr(mod, attr)
                validation.track_spectral_norm(mod)

    g_ema = Generator(
        args.size,
        args.latent_size,
        args.n_mlp,
        channel_multiplier=args.channel_multiplier,
        constant_input=args.constant_input,
        min_rgb_size=args.min_rgb_size,
    ).to(device, non_blocking=True)
    g_ema.requires_grad_(False)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    if args.contrastive > 0:
        contrast_learner = ContrastiveLearner(
            discriminator,
            args.size,
            augment_fn=nn.Sequential(
                nn.ReflectionPad2d(int((math.sqrt(2) - 1) * args.size / 4)),  # zoom out
                augs.RandomHorizontalFlip(),
                RandomApply(augs.RandomAffine(degrees=0, translate=(0.25, 0.25), shear=(15, 15)), p=0.1),
                RandomApply(augs.RandomRotation(180), p=0.1),
                augs.RandomResizedCrop(size=(args.size, args.size), scale=(1, 1), ratio=(1, 1)),
                RandomApply(augs.RandomResizedCrop(size=(args.size, args.size), scale=(0.5, 0.9)), p=0.1),  # zoom in
                RandomApply(augs.RandomErasing(), p=0.1),
            ),
            hidden_layer=(-1, 0),
        )
    else:
        contrast_learner = None

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = th.optim.Adam(
        generator.parameters(), lr=args.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = th.optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio * args.d_lr_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.lookahead:
        g_optim = LookaheadMinimax(
            g_optim, d_optim, la_steps=args.la_steps, la_alpha=args.la_alpha, accumulate=args.num_accumulate
        )

    if args.checkpoint is not None:
        print("load model:", args.checkpoint)

        checkpoint = th.load(args.checkpoint, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.checkpoint)
            args.start_iter = int(os.path.splitext(ckpt_name)[-1].replace(args.name, ""))
        except ValueError:
            pass

        if args.transfer_mapping_only:
            print("Using generator latent mapping network from checkpoint")
            mapping_state_dict = {}
            for key, val in checkpoint["state_dict"].items():
                if "generator.style" in key:
                    mapping_state_dict[key.replace("generator.", "")] = val
            generator.load_state_dict(mapping_state_dict, strict=False)
        else:
            generator.load_state_dict(checkpoint["g"], strict=False)
            g_ema.load_state_dict(checkpoint["g_ema"], strict=False)

            discriminator.load_state_dict(checkpoint["d"], strict=False)

            if args.lookahead:
                g_optim.load_state_dict(checkpoint["g_optim"], checkpoint["d_optim"])
            else:
                g_optim.load_state_dict(checkpoint["g_optim"])
                d_optim.load_state_dict(checkpoint["d_optim"])

        del checkpoint
        th.cuda.empty_cache()

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

        if contrast_learner is not None:
            contrast_learner = nn.parallel.DistributedDataParallel(
                contrast_learner,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

    transform = transforms.Compose(
        [
            transforms.RandomVerticalFlip(p=0.5 if args.vflip else 0),
            transforms.RandomHorizontalFlip(p=0.5 if args.hflip else 0),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        num_workers=8,
        drop_last=True,
        pin_memory=True,
    )

    if get_rank() == 0:
        validation.get_dataset_inception_features(loader, args.name, args.size)
        if args.wbgroup is None:
            wandb.init(project=args.wbproj, name=args.wbname, config=vars(args))
        else:
            wandb.init(project=args.wbproj, group=args.wbgroup, name=args.wbname, config=vars(args))

    if args.profile_mem:
        os.environ["GPU_DEBUG"] = str(args.local_rank)
        from gpu_profile import gpu_profile

        sys.settrace(gpu_profile)

    train(args, loader, generator, discriminator, contrast_learner, g_optim, d_optim, g_ema)

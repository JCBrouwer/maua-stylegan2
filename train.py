import argparse
import math
import random
import os, gc, sys
import time

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

import wandb
import validation

from models.stylegan2 import Generator, Discriminator
from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

sys.path.insert(0, "../lookahead_minimax")
from lookahead_minimax import LookaheadMinimax

from contrastive_learner import ContrastiveLearner, RandomApply
from kornia import augmentation as augs
from kornia import filters


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
        # print(name)
        param.data = decay * par1[name].data + (1 - decay) * par2[name].data


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def make_noise(batch_size, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return th.randn(2, batch_size, latent_dim, device=device).unbind(0)
    else:
        return [th.randn(batch_size, latent_dim, device=device)]


def train(args, loader, generator, discriminator, contrast_learner, augment, g_optim, d_optim, scaler, g_ema, device):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = th.zeros(size=(1,), device=device)
    g_loss_val = 0
    path_loss = th.zeros(size=(1,), device=device)
    path_lengths = th.zeros(size=(1,), device=device)
    loss_dict = {}
    mse = th.nn.MSELoss()

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
        if contrast_learner is not None:
            cl_module = contrast_learner.module
    else:
        g_module = generator
        d_module = discriminator
        cl_module = contrast_learner

    sample_z = th.randn(args.n_sample, args.latent_size, device=device)

    fids = []

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        discriminator.zero_grad()

        loss_dict["d"], loss_dict["real_score"], loss_dict["fake_score"] = 0, 0, 0
        loss_dict["cl_reg"], loss_dict["bc_reg"] = (
            th.tensor(0, device=device).float(),
            th.tensor(0, device=device).float(),
        )
        for _ in range(args.num_accumulate):
            real_img = next(loader)
            real_img = real_img.to(device)

            with th.cuda.amp.autocast():
                noise = make_noise(args.batch_size, args.latent_size, args.mixing_prob, device)
                fake_img, _ = generator(noise)

                if args.augment_D:
                    fake_pred = discriminator(augment(fake_img))
                    real_pred = discriminator(augment(real_img))
                else:
                    fake_pred = discriminator(fake_img)
                    real_pred = discriminator(real_img)

                # logistic loss
                real_loss = F.softplus(-real_pred)
                fake_loss = F.softplus(fake_pred)
                d_loss = real_loss.mean() + fake_loss.mean()

                loss_dict["d"] += d_loss.detach()
                loss_dict["real_score"] += real_pred.mean().detach()
                loss_dict["fake_score"] += fake_pred.mean().detach()

                if i > 10000 or i == 0:
                    if args.contrastive > 0:
                        contrast_learner(fake_img.clone().detach(), accumulate=True)
                        contrast_learner(real_img, accumulate=True)

                        contrast_loss = cl_module.calculate_loss()
                        loss_dict["cl_reg"] += contrast_loss.detach()

                        d_loss += args.contrastive * contrast_loss

                    if args.balanced_consistency > 0:
                        aug_fake_pred = discriminator(augment(fake_img.clone().detach()))
                        aug_real_pred = discriminator(augment(real_img))

                        consistency_loss = mse(real_pred, aug_real_pred) + mse(fake_pred, aug_fake_pred)
                        loss_dict["bc_reg"] += consistency_loss.detach()

                        d_loss += args.balanced_consistency * consistency_loss

                d_loss /= args.num_accumulate
            scaler.scale(d_loss).backward()
        scaler.step(d_optim)
        scaler.update()

        # R1 regularization
        loss_dict["r1"] = th.tensor(0, device=device).float()
        if args.r1 > 0 and i % args.d_reg_every == 0:

            discriminator.zero_grad()

            for _ in range(args.num_accumulate):
                real_img = next(loader)
                real_img = real_img.to(device)

                real_img.requires_grad = True

                with th.cuda.amp.autocast():
                    if args.augment_D:
                        real_pred = discriminator(
                            augment(real_img)
                        )  # RuntimeError: derivative for grid_sampler_2d_backward is not implemented :(
                    else:
                        real_pred = discriminator(real_img)
                        real_pred_sum = real_pred.sum()

                    (grad_real,) = th.autograd.grad(outputs=real_pred_sum, inputs=real_img, create_graph=True)
                    # grad_real = grad_real * (1.0 / scaler.get_scale())

                    # with th.cuda.amp.autocast():
                    r1_loss = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
                    weighted_r1_loss = args.r1 / 2.0 * r1_loss * args.d_reg_every + 0 * real_pred[0]

                    loss_dict["r1"] += r1_loss.detach()

                    weighted_r1_loss /= args.num_accumulate
                scaler.scale(weighted_r1_loss).backward()
            scaler.step(d_optim)
            scaler.update()

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        generator.zero_grad()
        loss_dict["g"] = 0
        for _ in range(args.num_accumulate):
            with th.cuda.amp.autocast():
                noise = make_noise(args.batch_size, args.latent_size, args.mixing_prob, device)
                fake_img, _ = generator(noise)

                if args.augment_G:
                    fake_img = augment(fake_img)

                fake_pred = discriminator(fake_img)

                # non-saturating loss
                g_loss = F.softplus(-fake_pred).mean()

                loss_dict["g"] += g_loss.detach()

                g_loss /= args.num_accumulate
            scaler.scale(g_loss).backward()
        scaler.step(g_optim)
        scaler.update()

        # path length regularization
        loss_dict["path"], loss_dict["path_length"] = (
            th.tensor(0, device=device).float(),
            th.tensor(0, device=device).float(),
        )
        if args.path_regularize > 0 and i % args.g_reg_every == 0:

            generator.zero_grad()

            for _ in range(args.num_accumulate):
                with th.cuda.amp.autocast():
                    path_batch_size = max(1, args.batch_size // args.path_batch_shrink)

                    noise = make_noise(path_batch_size, args.latent_size, args.mixing_prob, device)
                    fake_img, latents = generator(noise, return_latents=True)

                    img_noise = th.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
                    noisy_img_sum = (fake_img * img_noise).sum()

                    (grad,) = th.autograd.grad(outputs=noisy_img_sum, inputs=latents, create_graph=True)
                    print(grad.min(), grad.mean(), grad.max(), grad.shape)
                print(noisy_img_sum.min(), noisy_img_sum.mean(), noisy_img_sum.max(), noisy_img_sum.shape)
                print(
                    scaler.scale(noisy_img_sum).min(),
                    scaler.scale(noisy_img_sum).mean(),
                    scaler.scale(noisy_img_sum).max(),
                    scaler.scale(noisy_img_sum).shape,
                )
                (grad,) = th.autograd.grad(outputs=scaler.scale(noisy_img_sum), inputs=latents, create_graph=True)
                print(grad.min(), grad.mean(), grad.max(), grad.shape)
                grad = grad * (1.0 / scaler.get_scale())
                print(grad.min(), grad.mean(), grad.max(), grad.shape)

                with th.cuda.amp.autocast():
                    path_lengths = th.sqrt(grad.pow(2).sum(2).mean(1))
                    print(path_lengths)
                    path_mean = mean_path_length + 0.01 * (path_lengths.mean() - mean_path_length)
                    print(path_mean)
                    path_loss = (path_lengths - path_mean).pow(2).mean()
                    print(path_loss)
                    mean_path_length = path_mean.detach()
                    print(mean_path_length)

                    loss_dict["path"] += path_loss.detach()
                    loss_dict["path_length"] += path_lengths.mean().detach()

                    weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss
                    if args.path_batch_shrink:
                        weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

                    weighted_path_loss /= args.num_accumulate
                print(weighted_path_loss)
                scaler.scale(weighted_path_loss).backward()
            scaler.step(g_optim)
            scaler.update()

        accumulate(g_ema, g_module)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item() / args.num_accumulate
        g_loss_val = loss_reduced["g"].mean().item() / args.num_accumulate
        cl_reg_val = loss_reduced["cl_reg"].mean().item() / args.num_accumulate
        bc_reg_val = loss_reduced["bc_reg"].mean().item() / args.num_accumulate
        r1_val = loss_reduced["r1"].mean().item() / args.num_accumulate
        path_loss_val = loss_reduced["path"].mean().item() / args.num_accumulate
        real_score_val = loss_reduced["real_score"].mean().item() / args.num_accumulate
        fake_score_val = loss_reduced["fake_score"].mean().item() / args.num_accumulate
        path_length_val = loss_reduced["path_length"].mean().item() / args.num_accumulate

        if get_rank() == 0:

            log_dict = {
                "Generator": g_loss_val,
                "Discriminator": d_loss_val,
                "Real Score": real_score_val,
                "Fake Score": fake_score_val,
                "Contrastive": cl_reg_val,
                "Consistency": bc_reg_val,
            }

            pbar.set_description(
                (
                    f"G: {g_loss_val}, Fake: {fake_score_val}, Real: {real_score_val}, D: {d_loss_val}, R1: {r1_val}, Path: {path_length_val}, Contrast: {cl_reg_val}, Consist: {bc_reg_val}"
                )
            )

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

            if args.r1 > 0 and i % args.d_reg_every == 0:
                log_dict["R1"] = r1_val

            if args.path_regularize > 0 and i % args.g_reg_every == 0:
                log_dict["Path Length Regularization"] = path_loss_val
                log_dict["Mean Path Length"] = mean_path_length
                log_dict["Path Length"] = path_length_val

            if i % args.img_every == 0:
                gc.collect()
                th.cuda.empty_cache()
                with th.cuda.amp.autocast():
                    with th.no_grad():
                        g_ema.eval()
                        sample = []
                        for sub in range(0, len(sample_z), args.batch_size):
                            subsample, _ = g_ema([sample_z[sub : sub + args.batch_size]])
                            sample.append(subsample.cpu())
                        sample = th.cat(sample)
                        grid = utils.make_grid(sample, nrow=10, normalize=True, range=(-1, 1))
                log_dict["Generated Images EMA"] = [wandb.Image(grid, caption=f"Step {i}")]
            args.eval_every = 25
            if i % args.eval_every == 0:
                with th.cuda.amp.autocast():
                    start_time = time.time()

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

                pbar.set_description(
                    (
                        f"FID: {fid:.4f}; Density: {density:.4f}; Coverage: {coverage:.4f}; PPL: {ppl:.4f} in {time.time() - start_time:.1f}s"
                    )
                )
                log_dict["Evaluation/FID"] = fid
                log_dict["Sweep/FID_smooth"] = gaussian_filter(np.array(fids), [10])[-1]
                log_dict["Evaluation/Density"] = density
                log_dict["Evaluation/Coverage"] = coverage
                log_dict["Evaluation/PPL"] = ppl

                gc.collect()
                th.cuda.empty_cache()

            wandb.log(log_dict)

            if i % args.checkpoint_every == 0:
                th.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        # "cl": cl_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                    },
                    f"/home/hans/modelzoo/maua-sg2/{args.name}-{args.runname}-{wandb.run.dir.split('/')[-1].split('-')[-1]}-{int(fid)}-{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()

    # data options
    parser.add_argument("path", type=str)
    parser.add_argument("--runname", type=str, default=None)
    parser.add_argument("--vflip", type=bool, default=False)
    parser.add_argument("--hflip", type=bool, default=True)

    # training options
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_accumulate", type=int, default=1)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--transfer_mapping_only", type=bool, default=False)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--iter", type=int, default=60_000)

    # model options
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--min_rgb_size", type=int, default=128)
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--n_sample", type=int, default=60)
    parser.add_argument("--constant_input", type=bool, default=False)
    parser.add_argument("--channel_multiplier", type=int, default=2)

    # optimizer options
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--d_lr_ratio", type=float, default=1.0)
    parser.add_argument("--lookahead", type=bool, default=True)
    parser.add_argument("--la_steps", type=float, default=500)
    parser.add_argument("--la_alpha", type=float, default=0.5)

    # loss options
    parser.add_argument("--r1", type=float, default=4)
    parser.add_argument("--path_regularize", type=float, default=1)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing_prob", type=float, default=0.9)
    parser.add_argument("--augment_D", type=bool, default=False)
    parser.add_argument("--augment_G", type=bool, default=False)
    parser.add_argument("--contrastive", type=float, default=0)
    parser.add_argument("--balanced_consistency", type=float, default=0)

    # validation / logging options
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--fid_n_sample", type=int, default=500)
    parser.add_argument("--fid_truncation", type=float, default=None)
    parser.add_argument("--ppl_space", choices=["z", "w"], default="w")
    parser.add_argument("--ppl_n_sample", type=int, default=250)
    parser.add_argument("--ppl_crop", type=bool, default=False)
    parser.add_argument("--log_spec_norm", type=bool, default=False)
    parser.add_argument("--img_every", type=int, default=250)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=1000)

    # (multi-)GPU options
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--cudnn_benchmark", type=bool, default=True)

    args = parser.parse_args()
    if args.balanced_consistency > 0 or args.contrastive > 0:
        args.augment_D = True
        args.augment_G = True
    args.name = os.path.splitext(os.path.basename(args.path))[0]

    args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    th.backends.cudnn.benchmark = args.cudnn_benchmark
    args.distributed = args.num_gpus > 1

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
    ).to(device)
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)

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
    ).to(device)
    g_ema.requires_grad_(False)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    augment_fn = nn.Sequential(
        nn.ReflectionPad2d(int((math.sqrt(2) - 1) * args.size / 4)),  # zoom out
        augs.RandomHorizontalFlip(),
        RandomApply(augs.RandomAffine(degrees=0, translate=(0.25, 0.25), shear=(15, 15)), p=0.1),
        RandomApply(augs.RandomRotation(180), p=0.1),
        augs.RandomResizedCrop(size=(args.size, args.size), scale=(1, 1), ratio=(1, 1)),
        RandomApply(augs.RandomResizedCrop(size=(args.size, args.size), scale=(0.5, 0.9)), p=0.1),  # zoom in
        RandomApply(augs.RandomErasing(), p=0.1),
    )
    contrast_learner = (
        ContrastiveLearner(discriminator, args.size, augment_fn=augment_fn, hidden_layer=(-1, 0))
        if args.contrastive > 0
        else None
    )

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
    )

    if get_rank() == 0:
        validation.get_dataset_inception_features(loader, args.name, args.size)
    if args.runname is not None:
        wandb.init(project=f"maua-stylegan-sweep", name=args.runname, config=vars(args))
    else:
        wandb.init(project=f"maua-stylegan-sweep", config=vars(args))
    scaler = th.cuda.amp.GradScaler()

    train(args, loader, generator, discriminator, contrast_learner, augment_fn, g_optim, d_optim, scaler, g_ema, device)

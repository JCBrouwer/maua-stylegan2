import argparse
import math
import random
import os

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm

import wandb
import validation
from spectral_normalization import track_spectral_norm

from model import Generator, Discriminator
from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


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


def make_noise(batch_size, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return th.randn(2, batch_size, latent_dim, device=device).unbind(0)
    else:
        return [th.randn(batch_size, latent_dim, device=device)]


def train(args, loader, generator, discriminator, g_optim, d_optim, scaler, g_ema, device):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = th.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = th.tensor(0.0, device=device)
    path_lengths = th.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
    else:
        g_module = generator
        d_module = discriminator

    sample_z = th.randn(args.n_sample, args.latent_size, device=device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        # with th.cuda.amp.autocast():
        noise = make_noise(args.batch_size, args.latent_size, args.mixing_prob, device)
        fake_img, _ = generator(noise)
        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img)

        # logistic loss
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)
        d_loss = real_loss.mean() + fake_loss.mean()

        discriminator.zero_grad()
        # scaler.scale(d_loss).backward()
        # scaler.step(d_optim)
        d_loss.backward()
        d_optim.step()

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        # R1 regularization
        if i % args.d_reg_every == 0:
            real_img.requires_grad = True

            # with th.cuda.amp.autocast():
            real_pred = discriminator(real_img)
            real_pred_sum = real_pred.sum()

            (grad_real,) = th.autograd.grad(outputs=real_pred_sum, inputs=real_img, create_graph=True)
            # (grad_real,) = th.autograd.grad(outputs=scaler.scale(real_pred_sum), inputs=real_img, create_graph=True)
            # grad_real = grad_real * (1.0 / scaler.get_scale())

            # with th.cuda.amp.autocast():
            r1_loss = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
            weighted_r1_loss = args.r1 / 2.0 * r1_loss * args.d_reg_every + 0 * real_pred[0]

            discriminator.zero_grad()
            # scaler.scale(weighted_r1_loss).backward()
            # scaler.step(d_optim)
            weighted_r1_loss.backward()
            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        # with th.cuda.amp.autocast():
        noise = make_noise(args.batch_size, args.latent_size, args.mixing_prob, device)
        fake_img, _ = generator(noise)
        fake_pred = discriminator(fake_img)

        # non-saturating loss
        g_loss = F.softplus(-fake_pred).mean()

        generator.zero_grad()
        # scaler.scale(g_loss).backward()
        # scaler.step(g_optim)
        g_loss.backward()
        g_optim.step()

        loss_dict["g"] = g_loss

        # path length regularization
        if i % args.g_reg_every == 0:
            path_batch_size = max(1, args.batch_size // args.path_batch_shrink)

            # with th.cuda.amp.autocast():
            noise = make_noise(path_batch_size, args.latent_size, args.mixing_prob, device)
            fake_img, latents = generator(noise, return_latents=True)

            img_noise = th.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
            noisy_img_sum = (fake_img * img_noise).sum()

            (grad,) = th.autograd.grad(outputs=noisy_img_sum, inputs=latents, create_graph=True)
            # (grad,) = th.autograd.grad(outputs=scaler.scale(noisy_img_sum), inputs=latents, create_graph=True)
            # grad = grad * (1.0 / scaler.get_scale())

            # with th.cuda.amp.autocast():
            path_lengths = th.sqrt(grad.pow(2).sum(2).mean(1))
            path_mean = mean_path_length + 0.01 * (path_lengths.mean() - mean_path_length)
            path_loss = (path_lengths - path_mean).pow(2).mean()
            mean_path_length = path_mean.detach()

            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss
            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            generator.zero_grad()
            # scaler.scale(weighted_path_loss).backward()
            # scaler.step(g_optim)
            weighted_path_loss.backward()
            g_optim.step()

            mean_path_length_avg = reduce_sum(mean_path_length).item() / get_world_size()

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        # scaler.update()

        accumulate(g_ema, g_module)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:

            log_dict = {
                "Generator": g_loss_val,
                "Discriminator": d_loss_val,
                "Real Score": real_score_val,
                "Fake Score": fake_score_val,
            }

            for name, spec_norm in g_module.named_buffers():
                if "spectral_norm" in name:
                    log_dict[f"Spectral Norms/G.{name}"] = spec_norm
            for name, spec_norm in d_module.named_buffers():
                if "spectral_norm" in name:
                    log_dict[f"Spectral Norms/D.{name}"] = spec_norm

            if i % args.d_reg_every == 0:
                log_dict["R1"] = r1_val

            if i % args.g_reg_every == 0:
                log_dict["Path Length Regularization"] = path_loss_val
                log_dict["Mean Path Length"] = mean_path_length
                log_dict["Path Length"] = path_length_val

            if i % 1000 == 0:
                with th.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z])
                    grid = utils.make_grid(sample, nrow=6 * 256 // args.size, normalize=True, range=(-1, 1),)
                log_dict["Generated Images EMA"] = [wandb.Image(grid, caption=f"Step {i}")]
                fid = validation.fid(g_ema, args.val_batch_size, args.fid_n_sample, args.fid_truncation, args.name,)
                ppl = validation.ppl(
                    g_ema, args.val_batch_size, args.ppl_n_sample, args.ppl_space, args.ppl_crop, args.latent_size,
                )
                pbar.set_description((f"FID: {fid:.4f}; PPL: {ppl:.4f}"))
                log_dict["Evaluation/FID"] = fid
                log_dict["Evaluation/PPL"] = ppl

            wandb.log(log_dict)

            if i % 5000 == 0:
                th.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                    },
                    f"checkpoints/{args.name}{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()

    # data options
    parser.add_argument("path", type=str)
    parser.add_argument("--vflip", type=bool, default=False)
    parser.add_argument("--hflip", type=bool, default=True)

    # training options
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--checkpoint", type=str, default=None)

    # model options
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--n_sample", type=int, default=24)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--constant_input", type=bool, default=False)

    # loss options
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing_prob", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)

    # validation / logging options
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--fid_n_sample", type=int, default=5000)
    parser.add_argument("--fid_truncation", type=float, default=0.7)
    parser.add_argument("--ppl_space", choices=["z", "w"], default="w")
    parser.add_argument("--ppl_n_sample", type=int, default=2500)
    parser.add_argument("--ppl_crop", type=bool, default=False)
    parser.add_argument("--log_spec_norm", type=bool, default=True)

    # DevOps options
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--cudnn_benchmark", type=bool, default=True)

    args = parser.parse_args()

    args.name = os.path.splitext(os.path.basename(args.path))[0]

    args.num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    th.backends.cudnn.benchmark = args.cudnn_benchmark
    args.distributed = args.num_gpus > 1

    if args.distributed:
        th.cuda.set_device(args.local_rank)
        th.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent_size = 512
    args.n_mlp = 8

    args.start_iter = 0

    generator = Generator(
        args.size,
        args.latent_size,
        args.n_mlp,
        channel_multiplier=args.channel_multiplier,
        constant_input=args.constant_input,
    ).to(device)
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device)

    if args.log_spec_norm:
        for name, parameter in generator.named_parameters():
            if "weight" in name and parameter.squeeze().dim() > 1:
                mod = generator
                for attr in name.replace(".weight", "").split("."):
                    mod = getattr(mod, attr)
                track_spectral_norm(mod)
        for name, parameter in discriminator.named_parameters():
            if "weight" in name and parameter.squeeze().dim() > 1:
                mod = discriminator
                for attr in name.replace(".weight", "").split("."):
                    mod = getattr(mod, attr)
                track_spectral_norm(mod)

    g_ema = Generator(
        args.size,
        args.latent_size,
        args.n_mlp,
        channel_multiplier=args.channel_multiplier,
        constant_input=args.constant_input,
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = th.optim.Adam(
        generator.parameters(), lr=args.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = th.optim.Adam(
        discriminator.parameters(), lr=args.lr * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.checkpoint is not None:
        print("load model:", args.checkpoint)

        checkpoint = th.load(args.checkpoint, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.checkpoint)
            args.start_iter = int(os.path.splitext(ckpt_name)[0].replace(args.name, ""))

        except ValueError:
            pass

        generator.load_state_dict(checkpoint["g"])
        discriminator.load_state_dict(checkpoint["d"])
        g_ema.load_state_dict(checkpoint["g_ema"])

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

    transform = transforms.Compose(
        [
            transforms.RandomVerticalFlip(p=0.5 if args.vflip else 0),
            transforms.RandomHorizontalFlip(p=0.5 if args.hflip else 0),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
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
        validation.get_dataset_inception_features(loader, args.path, args.name, args.size)
        wandb.init(project=f"maua-stylegan", name="Cyphept 3")
    scaler = th.cuda.amp.GradScaler()

    train(args, loader, generator, discriminator, g_optim, d_optim, scaler, g_ema, device)

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

from autoencoder import LogCoshVAE


def info(x):
    print(x.shape, x.min(), x.mean(), x.max())


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


def train(args, loader, vae, vae_ema, vae_optim, generator, g_ema, g_optim, discriminator, d_optim, scaler, device):
    vae_kld_weight = 1.0 / len(loader)
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    # mean_path_length = 0

    d_loss_val = 0
    # r1_loss = th.tensor(0.0, device=device)
    # g_loss_val = 0
    # path_loss = th.tensor(0.0, device=device)
    # path_lengths = th.tensor(0.0, device=device)
    # mean_path_length_avg = 0
    vae_loss_val = 0
    vae_l1_loss_val = 0
    # vae_adv_loss_val = 0
    loss_dict = {}

    l1_loss = nn.L1Loss()

    if args.distributed:
        # g_module = generator.module
        # d_module = discriminator.module
        vae_module = vae.module
    else:
        # g_module = generator
        # d_module = discriminator
        vae_module = vae

    # sample_z = th.randn(args.n_sample, args.latent_size, device=device)
    sample_imgs = next(loader)[: args.n_sample]
    if get_rank() == 0:
        wandb.log({"Real Images": [wandb.Image(utils.make_grid(sample_imgs, nrow=6, normalize=True, range=(-1, 1)))]})
    sample_z = th.randn(size=(args.n_sample, args.vae_latent_dim))
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        real = next(loader).to(device)

        fake, mu, log_var = vae(real)
        # print(real.item().min(), real.item().mean(), real.item().max(), real.item().shape)
        # print(fake.item().min(), fake.item().mean(), fake.item().max(), fake.item().shape)
        # print()

        # fake_pred = discriminator(fake)
        # real_pred = discriminator(real)

        # real_loss = F.softplus(-real_pred)
        # fake_loss = F.softplus(fake_pred)
        # d_loss = real_loss.mean() + fake_loss.mean()

        diff = fake - real
        vae_reconst_loss = (
            args.vae_alpha * diff + th.log(1.0 + th.exp(-2 * args.vae_alpha * diff)) - th.log(th.tensor(2.0))
        )
        vae_reconst_loss = (1.0 / args.vae_alpha) * vae_reconst_loss.mean()

        vae_kl_loss = th.mean(-0.5 * th.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        # info(fake_pred.detach().cpu())
        # info(th.log(fake_pred).detach().cpu())
        # info(F.softplus(-fake_pred).detach().cpu())
        # vae_adv_loss = F.softplus(-fake_pred).mean()  # th.mean(-th.log(fake_pred))
        # print(vae_adv_loss.detach().item())

        vae_loss = vae_reconst_loss + args.vae_beta * vae_kld_weight * vae_kl_loss  # + args.lambda_adv * vae_adv_loss

        vae.zero_grad()
        # discriminator.zero_grad()

        # requires_grad(vae, True)
        # requires_grad(discriminator, False)
        # vae_loss.backward(retain_graph=True)
        vae_loss.backward()

        # requires_grad(discriminator, True)
        # requires_grad(vae, False)
        # d_loss.backward()

        vae_optim.step()
        # d_optim.step()

        # # R1 regularization
        # if i % args.d_reg_every == 0:
        #     real.requires_grad = True

        #     # with th.cuda.amp.autocast():
        #     real_pred = discriminator(real)
        #     real_pred_sum = real_pred.sum()

        #     (grad_real,) = th.autograd.grad(outputs=real_pred_sum, inputs=real, create_graph=True)
        #     # (grad_real,) = th.autograd.grad(outputs=scaler.scale(real_pred_sum), inputs=real, create_graph=True)
        #     # grad_real = grad_real * (1.0 / scaler.get_scale())

        #     # with th.cuda.amp.autocast():
        #     r1_loss = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        #     weighted_r1_loss = args.r1 / 2.0 * r1_loss * args.d_reg_every + 0 * real_pred[0]

        #     discriminator.zero_grad()
        #     # scaler.scale(weighted_r1_loss).backward()
        #     # scaler.step(d_optim)
        #     weighted_r1_loss.backward()
        #     d_optim.step()

        accumulate(vae_ema, vae_module)

        loss_dict["vae"] = vae_loss
        loss_dict["vae_reconst"] = vae_reconst_loss
        loss_dict["vae_kl"] = vae_kl_loss
        # loss_dict["vae_adv"] = vae_adv_loss

        # loss_dict["d"] = d_loss
        # loss_dict["real_score"] = real_pred.mean()
        # loss_dict["fake_score"] = fake_pred.mean()
        # loss_dict["r1"] = r1_loss

        loss_reduced = reduce_loss_dict(loss_dict)

        # g_loss_val = loss_reduced["g"].mean().item()
        # path_loss_val = loss_reduced["path"].mean().item()
        # path_length_val = loss_reduced["path_length"].mean().item()
        # path_length_val = loss_reduced["path_length"].mean().item()

        # d_loss_val = loss_reduced["d"].mean().item()
        # real_score_val = loss_reduced["real_score"].mean().item()
        # fake_score_val = loss_reduced["fake_score"].mean().item()
        # r1_val = loss_reduced["r1"].mean().item()

        vae_loss_val = loss_reduced["vae"].mean().item()
        vae_reconst_loss_val = loss_reduced["vae_reconst"].mean().item()
        vae_kl_loss_val = loss_reduced["vae_kl"].mean().item()
        # vae_adv_loss_val = loss_reduced["vae_adv"].mean().item()

        if get_rank() == 0:

            log_dict = {
                "Total": vae_loss_val,
                "Reconstruction": vae_reconst_loss_val,
                "Kullback Leibler Divergence": -vae_kl_loss_val,
                # "VAE Adversarial": vae_adv_loss_val,
                # "Generator": g_loss_val,
                # "Discriminator": d_loss_val,
                # "Real Score": real_score_val,
                # "Fake Score": fake_score_val,
            }

            if args.log_spec_norm:
                # G_norms = []
                # for name, spec_norm in g_module.named_buffers():
                #     if "spectral_norm" in name:
                #         G_norms.append(spec_norm.cpu().numpy())
                # G_norms = np.array(G_norms)

                # log_dict[f"Spectral Norms/G min spectral norm"] = np.log(G_norms).min()
                # log_dict[f"Spectral Norms/G median spectral norm"] = np.log(G_norms).mean()
                # log_dict[f"Spectral Norms/G mean spectral norm"] = np.median(np.log(G_norms))
                # log_dict[f"Spectral Norms/G max spectral norm"] = np.log(G_norms).max()

                vae_norms = []
                for name, spec_norm in vae_module.named_buffers():
                    if "spectral_norm" in name:
                        vae_norms.append(spec_norm.cpu().numpy())
                vae_norms = np.array(vae_norms)

                log_dict[f"Spectral Norms/VAE min spectral norm"] = np.log(vae_norms).min()
                log_dict[f"Spectral Norms/VAE mean spectral norm"] = np.log(vae_norms).mean()
                log_dict[f"Spectral Norms/VAE max spectral norm"] = np.log(vae_norms).max()

                # D_norms = []
                # for name, spec_norm in d_module.named_buffers():
                #     if "spectral_norm" in name:
                #         D_norms.append(spec_norm.cpu().numpy())
                # D_norms = np.array(D_norms)

                # log_dict[f"Spectral Norms/D min spectral norm"] = np.log(D_norms).min()
                # log_dict[f"Spectral Norms/D mean spectral norm"] = np.log(D_norms).mean()
                # log_dict[f"Spectral Norms/D max spectral norm"] = np.log(D_norms).max()

            # if i % args.d_reg_every == 0:
            #     log_dict["R1"] = r1_val

            # if i % args.g_reg_every == 0:
            #     log_dict["Path Length Regularization"] = path_loss_val
            #     log_dict["Mean Path Length"] = mean_path_length
            #     log_dict["Path Length"] = path_length_val

            if i % 100 == 0:
                # with th.no_grad():
                #     g_ema.eval()
                #     sample, _ = g_ema([sample_z])
                #     grid = utils.make_grid(sample, nrow=6 * 256 // args.size, normalize=True, range=(-1, 1),)
                # log_dict["Generated Images EMA"] = [wandb.Image(grid, caption=f"Step {i}")]

                with th.no_grad():
                    vae_ema.eval()

                    sample, _, _ = vae_ema(sample_imgs.to(device))
                    grid = utils.make_grid(sample, nrow=6, normalize=True, range=(-1, 1),)
                    del sample

                    log_dict["Reconstructed Images VAE EMA"] = [wandb.Image(grid, caption=f"Step {i}")]

                    sample = vae_ema.decode(sample_z.to(device))
                    grid = utils.make_grid(sample, nrow=6, normalize=True, range=(-1, 1),)
                    del sample

                    log_dict["Generated Images VAE EMA"] = [wandb.Image(grid, caption=f"Step {i}")]

                # pbar.set_description((f"Calculating FID..."))
                # fid_dict = validation.fid(vae_ema, args.val_batch_size, args.fid_n_sample, args.fid_truncation, args.name)
                # fid = fid_dict["FID"]
                # density = fid_dict["Density"]
                # coverage = fid_dict["Coverage"]

                # pbar.set_description((f"Calculating PPL..."))
                # ppl = validation.ppl(
                #     g_ema, args.val_batch_size, args.ppl_n_sample, args.ppl_space, args.ppl_crop, args.latent_size,
                # )

                # pbar.set_description(
                #     (f"FID: {fid:.4f}; Density: {density:.4f}; Coverage: {coverage:.4f}; PPL: {ppl:.4f};")
                # )
                # log_dict["Evaluation/FID"] = fid
                # log_dict["Evaluation/Density"] = density
                # log_dict["Evaluation/Coverage"] = coverage
                # log_dict["Evaluation/PPL"] = ppl

            wandb.log(log_dict)

            if i % 1000 == 0:
                th.save(
                    {
                        # "g": g_module.state_dict(),
                        "vae": vae_module.state_dict(),
                        # "d": d_module.state_dict(),
                        # "g_ema": g_ema.state_dict(),
                        "vae_ema": vae_ema.state_dict(),
                        # "g_optim": g_optim.state_dict(),
                        "vae_optim": vae_optim.state_dict(),
                        # "d_optim": d_optim.state_dict(),
                    },
                    f"checkpoints/vae_{args.name}{str(i).zfill(6)}.pt",
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()

    # data options
    parser.add_argument("path", type=str)
    parser.add_argument("--vflip", type=bool, default=False)
    parser.add_argument("--hflip", type=bool, default=True)
    parser.add_argument("--dataset_size", type=int, default=256)

    # training options
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)

    # VAE options
    parser.add_argument("--vae_resolution", type=int, default=128)
    parser.add_argument("--vae_alpha", type=float, default=10.0)
    parser.add_argument("--vae_beta", type=float, default=1.0)
    # parser.add_argument("--vae_downsample_factor", type=int, default=8)
    # parser.add_argument("--vae_nc", type=int, default=3)
    # parser.add_argument("--vae_ngf", type=int, default=64)
    # parser.add_argument("--vae_ndf", type=int, default=64)
    parser.add_argument("--vae_latent_dim", type=int, default=512)
    parser.add_argument("--vae_lr", type=float, default=5e-3)
    parser.add_argument("--lambda_adv", type=float, default=0.1)

    # # model options
    # parser.add_argument("--latent_size", type=int, default=512)
    # parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--n_sample", type=int, default=24)
    # parser.add_argument("--constant_input", type=bool, default=False)

    # # loss options
    parser.add_argument("--r1", type=float, default=10)
    # parser.add_argument("--path_regularize", type=float, default=2)
    # parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    # parser.add_argument("--g_reg_every", type=int, default=4)
    # parser.add_argument("--mixing_prob", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)

    # # validation / logging options
    # parser.add_argument("--val_batch_size", type=int, default=4)
    # parser.add_argument("--fid_n_sample", type=int, default=5000)
    # parser.add_argument("--fid_truncation", type=float, default=0.7)
    # parser.add_argument("--ppl_space", choices=["z", "w"], default="w")
    # parser.add_argument("--ppl_n_sample", type=int, default=2500)
    # parser.add_argument("--ppl_crop", type=bool, default=False)
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

    # generator = Generator(
    #     args.size,
    #     args.latent_size,
    #     args.n_mlp,
    #     channel_multiplier=args.channel_multiplier,
    #     constant_input=args.constant_input,
    # ).to(device)

    # g_ema = Generator(
    #     args.size,
    #     args.latent_size,
    #     args.n_mlp,
    #     channel_multiplier=args.channel_multiplier,
    #     constant_input=args.constant_input,
    # ).to(device)
    # g_ema.eval()
    # accumulate(g_ema, generator, 0)

    # g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    # g_optim = th.optim.Adam(
    #     generator.parameters(), lr=args.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    # )

    # discriminator = Discriminator(args.vae_resolution, channel_multiplier=args.channel_multiplier).to(device)

    # d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    # d_optim = th.optim.Adam(
    #     discriminator.parameters(), lr=args.lr * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    # )

    vae = LogCoshVAE(
        3, args.vae_latent_dim, hidden_dims=[32, 64, 128, 256, 512, 512], alpha=args.vae_alpha, beta=args.vae_beta
    ).to(device)

    vae_ema = LogCoshVAE(
        3, args.vae_latent_dim, hidden_dims=[32, 64, 128, 256, 512, 512], alpha=args.vae_alpha, beta=args.vae_beta
    ).to(device)
    vae_ema.eval()
    accumulate(vae_ema, vae, 0)

    vae_optim = th.optim.Adam(vae.parameters(), lr=args.vae_lr)

    if args.log_spec_norm:
        # for name, parameter in generator.named_parameters():
        #     if "weight" in name and parameter.squeeze().dim() > 1:
        #         mod = generator
        #         for attr in name.replace(".weight", "").split("."):
        #             mod = getattr(mod, attr)
        #         track_spectral_norm(mod)

        for name, parameter in vae.named_parameters():
            if "weight" in name and parameter.squeeze().dim() > 1:
                mod = vae
                for attr in name.replace(".weight", "").split("."):
                    mod = getattr(mod, attr)
                track_spectral_norm(mod)

        # for name, parameter in discriminator.named_parameters():
        #     if "weight" in name and parameter.squeeze().dim() > 1:
        #         mod = discriminator
        #         for attr in name.replace(".weight", "").split("."):
        #             mod = getattr(mod, attr)
        #         track_spectral_norm(mod)

    if args.checkpoint is not None:
        print("load model:", args.checkpoint)

        checkpoint = th.load(args.checkpoint, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.checkpoint)
            args.start_iter = int(os.path.splitext(ckpt_name)[0].replace(args.name, ""))
        except ValueError:
            pass

        vae.load_state_dict(checkpoint["vae"])
        vae_ema.load_state_dict(checkpoint["vae_ema"])
        vae_optim.load_state_dict(checkpoint["vae_optim"])

        # generator.load_state_dict(checkpoint["g"])
        # g_ema.load_state_dict(checkpoint["g_ema"])
        # g_optim.load_state_dict(checkpoint["g_optim"])

        # discriminator.load_state_dict(checkpoint["d"])
        # d_optim.load_state_dict(checkpoint["d_optim"])

        del checkpoint
        th.cuda.empty_cache()

    if args.distributed:
        #     generator = nn.parallel.DistributedDataParallel(
        #         generator,
        #         device_ids=[args.local_rank],
        #         output_device=args.local_rank,
        #         broadcast_buffers=False,
        #         find_unused_parameters=True,
        #     )

        # discriminator = nn.parallel.DistributedDataParallel(
        #     discriminator,
        #     device_ids=[args.local_rank],
        #     output_device=args.local_rank,
        #     broadcast_buffers=False,
        #     # find_unused_parameters=True,
        # )
        vae = nn.parallel.DistributedDataParallel(
            vae,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            # find_unused_parameters=True,
        )

    transform = transforms.Compose(
        [
            transforms.Resize(args.vae_resolution),
            transforms.RandomVerticalFlip(p=0.5 if args.vflip else 0),
            transforms.RandomHorizontalFlip(p=0.5 if args.hflip else 0),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.dataset_size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        num_workers=12,
        drop_last=True,
    )

    if get_rank() == 0:
        # validation.get_dataset_inception_features(loader, args.path, args.name, args.size)
        wandb.init(project=f"maua-stylegan", name="Cyphept LogCoshVAE")
    # scaler = th.cuda.amp.GradScaler()

    train(
        args=args,
        loader=loader,
        vae=vae,
        vae_ema=vae_ema,
        vae_optim=vae_optim,
        generator=None,
        g_ema=None,
        g_optim=None,
        discriminator=None,  # discriminator,
        d_optim=None,  # d_optim,
        scaler=None,
        device=device,
    )

import gc
import math
import uuid
import numbers
import argparse

import numpy as np
import scipy.ndimage as ndi

import torch as th
import torch.nn.functional as F
import torch.multiprocessing as mp

from render import render
from models.stylegan2 import Generator
from models.stylegan1 import G_style


def gaussian_filter(x, sigma):
    dim = len(x.shape)
    if dim != 3 and dim != 4:
        raise Exception("Only 3- or 4-dimensional tensors are supported.")

    radius = sigma * 4
    channels = x.shape[1]

    kernel = th.arange(-radius, radius + 1, dtype=th.float32, device="cuda")
    kernel = th.exp(-0.5 / sigma ** 2 * kernel ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, len(kernel)).repeat(channels, 1, 1)

    if dim == 4:
        t, c, h, w = x.shape
        x = x.view(t, c, h * w)
    x = x.transpose(0, 2)

    x = F.pad(x, (radius, radius), mode="circular")
    x = F.conv1d(x, weight=kernel, groups=channels)

    x = x.transpose(0, 2)
    if dim == 4:
        x = x.view(t, c, h, w)

    return x


def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def lerp(val, low, high):
    return (1 - val) * low + val * high


def get_spline_loops(base_latent_selection, loop_starting_latents, n_frames, num_loops, smoothing, s=True):
    from scipy import interpolate

    base_latent_selection = np.concatenate([base_latent_selection, base_latent_selection[[0]]])

    x = np.linspace(0, 1, n_frames // max(1, num_loops))
    base_latents = np.zeros((len(x), *base_latent_selection.shape[1:]))
    for lay in range(base_latent_selection.shape[1]):
        for lat in range(base_latent_selection.shape[2]):
            tck = interpolate.splrep(
                np.linspace(0, 1, len(base_latent_selection.shape[0])), base_latent_selection[:, lay, lat]
            )
            base_latents[:, lay, lat] = interpolate.splev(x, tck)

    base_latents = th.cat([base_latents] * int(n_frames / len(base_latents)), axis=0)

    return base_latents


def get_latent_loops(base_latent_selection, loop_starting_latents, n_frames, num_loops, smoothing, s=True):
    base_latents = []

    for n in range(len(base_latent_selection)):
        for val in np.linspace(0.0, 1.0, int(n_frames // max(1, num_loops) // len(base_latent_selection))):
            base_latents.append(
                (slerp if s else lerp)(
                    val,
                    base_latent_selection[(n + loop_starting_latents) % len(base_latent_selection)][0].cpu(),
                    base_latent_selection[(n + loop_starting_latents + 1) % len(base_latent_selection)][0].cpu(),
                )
            )

    base_latents = th.stack(base_latents, axis=0).cuda()
    base_latents = th.cat([base_latents] * int(n_frames / len(base_latents)), axis=0)
    base_latents = th.stack([base_latents] * base_latent_selection.shape[1], axis=1)

    base_latents = gaussian_filter(base_latents, smoothing)

    return base_latents


if "main" in __name__:
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--G_res", type=int, default=1024)
    parser.add_argument("--out_size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--num_frames", type=int, default=5 * 24)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--const", type=bool, default=False)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--truncation", type=int, default=1.5)
    parser.add_argument("--stylegan1", type=bool, default=False)
    parser.add_argument("--slerp", type=bool, default=True)
    parser.add_argument("--latents", type=str, default=None)

    args = parser.parse_args()

    th.set_grad_enabled(False)
    th.backends.cudnn.benchmark = True
    mp.set_start_method("spawn")

    if args.stylegan1:
        generator = G_style(output_size=args.out_size, checkpoint=args.ckpt).cuda()
    else:
        args.latent = 512
        args.n_mlp = 8
        generator = Generator(
            args.G_res,
            args.latent,
            args.n_mlp,
            channel_multiplier=args.channel_multiplier,
            constant_input=args.const,
            checkpoint=args.ckpt,
            output_size=args.out_size,
        ).cuda()
    # generator = th.nn.DataParallel(generator)

    if args.latents is not None:
        styles = th.from_numpy(np.load(args.latents))
    else:
        styles = th.randn((int(args.duration), 512), device="cuda")
        styles = generator(styles, map_latents=True)
    # styles = get_latent_loops(styles, 0, args.num_frames, num_loops=1, smoothing=1, s=args.slerp)
    styles = get_spline_loops(styles, 0, args.num_frames, num_loops=1, smoothing=1, s=args.slerp)

    latents = th.randn((args.num_frames, 512), device="cuda")
    latents = generator(latents, map_latents=True)
    latents = gaussian_filter(latents, 1)
    # print(
    #     th.cat(
    #         [
    #             th.linspace(-1, 3, int(args.num_frames / 2)),
    #             th.linspace(3, -1, args.num_frames - int(args.num_frames / 2)) + 1,
    #         ]

    style_depth = 0
    latents[:, style_depth:] = 1 * styles[:, style_depth:] + 0 * latents[:, style_depth:]

    latents = latents.cpu()
    args.num_frames = len(latents)

    print("latent shape: ")
    print(latents.shape, "\n")

    log_max_res = int(np.log2(args.out_size))
    log_min_res = 2 + (log_max_res - int(np.log2(args.G_res)))

    noise = []
    if args.stylegan1:
        for s in range(log_min_res, log_max_res + 1):
            h = 2 ** s
            w = (2 if args.out_size == 1920 else 1) * 2 ** s
            noise.append(th.randn((args.num_frames, 1, h, w), device="cuda"))
    else:
        for s in range(2 * log_min_res + 1, 2 * (log_max_res + 1), 1):
            h = 2 ** int(s / 2)
            w = (2 if args.out_size == 1920 else 1) * 2 ** int(s / 2)
            noise.append(th.randn((args.num_frames, 1, h, w), device="cuda"))

    print("noise shapes: ")
    for i, n in enumerate(noise):
        if n is None:
            continue
        noise[i] = (gaussian_filter(n, 24)).cpu()
        noise[i] /= noise[i].std()
        print(i, noise[i].shape)
    print()

    # noise = []
    # if args.stylegan1:
    #     for s in range(log_min_res, log_max_res + 1):
    #         h = 2 ** s
    #         w = (2 if args.out_size == 1920 else 1) * 2 ** s
    #         noise.append(np.random.normal(size=(args.num_frames, 1, h, w)))
    # else:
    #     for s in range(2 * log_min_res + 1, 2 * (log_max_res + 1), 1):
    #         h = 2 ** int(s / 2)
    #         w = (2 if args.out_size == 1920 else 1) * 2 ** int(s / 2)
    #         noise.append(np.random.normal(size=(args.num_frames, 1, h, w)))

    # print("noise shapes: ")
    # for i, n in enumerate(noise):
    #     if n is None:
    #         continue
    #     noise[i] = th.from_numpy(ndi.gaussian_filter(n, [15, 0, 0, 0], mode="wrap"))
    #     print(n.shape)
    # print()

    class addNoise(th.nn.Module):
        def __init__(self, noise):
            super(addNoise, self).__init__()
            self.noise = noise

        def forward(self, x):
            return x + self.noise

    manipulations = []
    if log_min_res > 2:
        reflects = []
        for lres in range(2, log_min_res):
            half = 2 ** (lres - 1)
            reflects.append(th.nn.ReplicationPad2d((half, half, half, half)))
        manipulations += [
            {
                "layer": 0,
                "transform": th.nn.Sequential(
                    *reflects, addNoise(2 * th.randn(size=(1, 1, 2 ** log_min_res, 2 ** log_min_res), device="cuda"))
                ),
            }
        ]

    # tl = 4
    # width = lambda s: (2 if args.out_size == 1920 else 1) * 2 ** int(s)
    # translation = (
    #     th.tensor([np.linspace(0, width(tl), args.num_frames + 1), np.zeros((args.num_frames + 1,))]).float().T[:-1]
    # )
    # manipulations += [{"layer": tl, "transform": "translateX", "params": translation}]

    # zl = 6
    # print(
    #     th.cat(
    #         [
    #             th.linspace(-1, 3, int(args.num_frames / 2)),
    #             th.linspace(3, -1, args.num_frames - int(args.num_frames / 2)) + 1,
    #         ]
    #     ).shape
    # )
    # zoom = gaussian_filter(
    #     th.cat(
    #         [
    #             th.linspace(0, 3, int(args.num_frames / 2), dtype=th.float32, device="cuda"),
    #             th.linspace(3, 0, args.num_frames - int(args.num_frames / 2), dtype=th.float32, device="cuda") + 1,
    #         ]
    #     )[:, None, None],
    #     30,
    # ).squeeze()
    # zoom -= zoom.min()
    # zoom /= zoom.max()
    # # zoom *= 1.5
    # zoom += 0.5
    # print(zoom.min().item(), zoom.max().item(), zoom.shape)
    # manipulations += [{"layer": zl, "transform": "zoom", "params": zoom}]

    # rl = 6
    # rotation = th.nn.Sigmoid()(th.tensor(np.linspace(0.0, 1.0, args.num_frames + 1), device="cuda").float())
    # rotation -= rotation.min()
    # rotation /= rotation.max()
    # rotation = rotation[:-1]
    # manipulations += [{"layer": rl, "transform": "rotate", "params": (360.0 * rotation).cpu()}]

    render(
        generator=generator,
        latents=latents,
        noise=noise,
        offset=0,
        duration=args.duration,
        batch_size=args.batch,
        truncation=args.truncation,
        manipulations=manipulations,
        out_size=args.out_size,
        output_file=f"/home/hans/neurout/GAN Bending/{args.ckpt.split('/')[-1].split('.')[0]}-{uuid.uuid4().hex[:8]}.mp4",
    )

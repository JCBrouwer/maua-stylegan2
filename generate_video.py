import argparse
import uuid

import numpy as np
import torch as th
import torch.multiprocessing as mp

import audioreactive as ar
from models.stylegan1 import G_style
from models.stylegan2 import Generator
from render import render

if "main" in __name__:
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--G_res", type=int, default=1024)
    parser.add_argument("--out_size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration", type=int, default=10)
    parser.add_argument("--const", type=bool, default=False)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--truncation", type=int, default=0.7)
    parser.add_argument("--stylegan1", type=bool, default=False)
    parser.add_argument("--latents", type=str, default=None)

    args = parser.parse_args()

    n_frames = args.duration * args.fps

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
        styles = th.randn((args.duration, 512), device="cuda")
        styles = generator(styles, map_latents=True)
    latents = ar.spline_loops(styles[:5], n_frames, 1)
    latents = ar.gaussian_filter(latents, 1).cpu()

    print("latent shape: ")
    print(latents.shape, "\n")

    log_max_res = int(np.log2(args.out_size))
    log_min_res = 2 + (log_max_res - int(np.log2(args.G_res)))

    noise = []
    if args.stylegan1:
        for s in range(log_min_res, log_max_res + 1):
            h = 2 ** s
            w = (2 if args.out_size == 1920 else 1) * 2 ** s
            noise.append(th.randn((n_frames, 1, h, w), device="cuda"))
    else:
        for s in range(2 * log_min_res + 1, 2 * (log_max_res + 1), 1):
            h = 2 ** int(s / 2)
            w = (2 if args.out_size == 1920 else 1) * 2 ** int(s / 2)
            noise.append(th.randn((n_frames, 1, h, w), device="cuda"))

    print("noise shapes: ")
    for i, n in enumerate(noise):
        if n is None:
            continue
        if i > 14:
            noise[i] = None
            continue

        # xs = 8 * np.pi * th.linspace(0, 1, n.shape[-1])
        # ys = th.linspace(0, 2 * np.pi, n.shape[-2])
        # ts = 8 * np.pi * th.linspace(0, 1, n.shape[0])
        # horiz = xs[None, None, None, :] + ys[None, None, :, None] + ts[:, None, None, None]
        # vert = (
        #     xs[None, None, None, :] / (4 * np.pi)
        #     + 4 * np.pi * ys[None, None, :, None]
        #     + 2 * ts[:, None, None, None]
        # )
        # moving_noise = th.sin(horiz.cuda() * vert.cuda() + n / 4)
        # moving_noise = gaussian_filter(moving_noise, 6).cpu()
        # moving_noise /= moving_noise.std() / 2
        # moving_noise = ar.perlin_noise((n.shape[0], n.shape[-2], n.shape[-1]), (10, 4, 4))[:, None, ...]
        moving_noise = ar.gaussian_filter(n, 12)
        moving_noise /= moving_noise.std() * 1.2
        noise[i] = moving_noise.cpu()

        print(i, noise[i].shape, noise[i].std())
    print()

    class addNoise(th.nn.Module):
        def __init__(self, noise):
            super(addNoise, self).__init__()
            self.noise = noise

        def forward(self, x):
            return x + self.noise

    # manipulations = []
    # if log_min_res > 2:
    #     reflects = []
    #     for lres in range(2, log_min_res):
    #         half = 2 ** (lres - 1)
    #         reflects.append(th.nn.ReplicationPad2d((half, half, half, half)))
    #     manipulations += [
    #         {
    #             "layer": 0,
    #             "transform": th.nn.Sequential(
    #                 *reflects, addNoise(2 * th.randn(size=(1, 1, 2 ** log_min_res, 2 ** log_min_res), device="cuda"))
    #             ),
    #         }
    #     ]

    # tl = 4
    # width = lambda s: (2 if args.out_size == 1920 else 1) * 2 ** int(s)
    # translation = (
    #     th.tensor([np.linspace(0, width(tl), n_frames + 1), np.zeros((n_frames + 1,))]).float().T[:-1]
    # )
    # manipulations += [{"layer": tl, "transform": "translateX", "params": translation}]

    # zl = 6
    # print(
    #     th.cat(
    #         [
    #             th.linspace(-1, 3, int(n_frames / 2)),
    #             th.linspace(3, -1, n_frames - int(n_frames / 2)) + 1,
    #         ]
    #     ).shape
    # )
    # zoom = gaussian_filter(
    #     th.cat(
    #         [
    #             th.linspace(0, 3, int(n_frames / 2), dtype=th.float32, device="cuda"),
    #             th.linspace(3, 0, n_frames - int(n_frames / 2), dtype=th.float32, device="cuda") + 1,
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
    # rotation = th.nn.Sigmoid()(th.tensor(np.linspace(0.0, 1.0, n_frames + 1), device="cuda").float())
    # rotation -= rotation.min()
    # rotation /= rotation.max()
    # rotation = rotation[:-1]
    # manipulations += [{"layer": rl, "transform": "rotate", "params": (360.0 * rotation).cpu()}]

    output_name = f"/home/hans/modelzoo/nada/{args.ckpt.split('/')[-1].split('.')[0]}_{uuid.uuid4().hex[:8]}"
    render(
        generator=generator,
        latents=latents,
        noise=noise,
        offset=0,
        duration=args.duration,
        batch_size=args.batch,
        truncation=args.truncation,
        # manipulations=manipulations,
        out_size=args.out_size,
        output_file=f"{output_name}.mp4",
    )

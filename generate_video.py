import argparse
import numpy as np
import torch as th
from render import render
import scipy.ndimage as ndi
from models.stylegan2 import Generator


def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def lerp(val, low, high):
    return (1 - val) * low + val * high


def get_latent_loops(base_latent_selection, loop_starting_latents, n_frames, num_loops, smoothing, s=True):
    interp = slerp if s else lerp
    base_latents = []
    for n in range(len(base_latent_selection)):
        for val in np.linspace(0.0, 1.0, int(n_frames // max(1, num_loops) // len(base_latent_selection))):
            base_latents.append(
                interp(
                    val,
                    base_latent_selection[(n + loop_starting_latents) % len(base_latent_selection)][0],
                    base_latent_selection[(n + loop_starting_latents + 1) % len(base_latent_selection)][0],
                ).numpy()
            )
    base_latents = ndi.gaussian_filter(np.array(base_latents), [smoothing, 0], mode="wrap")
    base_latents = np.concatenate([base_latents] * int(n_frames / len(base_latents)), axis=0)
    base_latents = np.concatenate([base_latents[:, None, :]] * 18, axis=1)
    base_latents = th.from_numpy(base_latents)
    return base_latents


if __name__ == "__main__":
    th.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--G_res", type=int, default=1024)
    parser.add_argument("--out_size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--num_frames", type=int, default=8 * 30)
    parser.add_argument("--duration", type=int, default=8)
    parser.add_argument("--const", type=bool, default=False)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--truncation", type=int, default=1.5)

    args = parser.parse_args()
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

    with th.no_grad():
        # latents = th.randn((args.num_frames, 512)).cuda()
        # latents = generator(latents, map_latents=True).cpu().numpy()
        # latents = ndi.gaussian_filter(latents, [15, 0, 0], mode="wrap")
        # latents = th.from_numpy(latents).cuda()
        latent_selection = th.from_numpy(np.load("/home/hans/neurout/stylesheets/scrollin_patrolin_latents.npy"))[
            [0, 5, 0] + [5, 4, 0, 5, 6, 5]
        ]
        latents = get_latent_loops(latent_selection, 0, args.num_frames, num_loops=1, smoothing=15, s=False)
        latents[:, 5:] = 0.8 * latent_selection[[4], 5:] + 0.2 * latents[:, 5:]
        latents = latents.cuda()
        args.num_frames = len(latents)

        print("latent shape: ")
        print(latents.shape)

        log_min_res = 2
        log_max_res = 9
        noise = [
            np.random.normal(
                size=(args.num_frames, 1, 2 ** int(s / 2), (2 if args.out_size == 1920 else 1) * 2 ** int(s / 2),)
            )
            for s in range(2 * log_min_res + 1, 2 * log_max_res, 1)
        ]

        noise = [
            th.from_numpy(ndi.gaussian_filter(n, [5, 0, 0, 0], mode="wrap"))
            * (4 - 3 * th.sin(th.linspace(0, 2 * np.pi, args.num_frames)[:, None, None, None]) ** 4)
            for n in noise
        ]
        print("noise shapes: ")
        [print(n.shape if n is not None else None) for n in noise]
        print()
        noise += [None] * 40

        # class addNoise(th.nn.Module):
        #     def forward(self, x):
        #         return x + th.normal(th.zeros_like(x), std=x.std())

        width = lambda s: (2 if args.out_size == 1920 else 1) * 2 ** int(s)

        translation = th.tensor([np.linspace(0, width(4), args.num_frames), np.zeros((args.num_frames,))]).float().T

        rotation = th.nn.Sigmoid()(th.tensor(np.linspace(0.0, 1.0, args.num_frames)).float()).numpy()
        rotation = ndi.gaussian_filter(rotation, [5])
        rotation -= rotation.min()
        rotation /= rotation.max()
        rotation = th.from_numpy(rotation)

        manipulations = [
            # {
            #     "layer": 0,
            #     "transform": th.nn.Sequential(
            #         th.nn.ReflectionPad2d((2, 2, 0, 0)), th.nn.ReflectionPad2d((4, 4, 2, 2))  # , addNoise()
            #     ),
            # },
            {"layer": 4, "transform": "translate", "params": translation,},
            {"layer": 6, "transform": "rotate", "params": 360.0 * rotation,},
        ]

        render(
            generator=generator,
            latents=latents,
            noise=noise,
            duration=args.duration,
            batch_size=args.batch,
            truncation=args.truncation,
            manipulations=manipulations,
        )

import argparse
import numpy as np
import torch as th
from render import render
import scipy.ndimage as ndi
from models.stylegan2 import Generator


if __name__ == "__main__":
    th.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--G_res", type=int, default=1024)
    parser.add_argument("--out_size", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--num_frames", type=int, default=150)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--const", type=bool, default=False)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--truncation", type=int, default=1.5)

    args = parser.parse_args()
    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.G_res,
        args.latent,
        args.n_mlp,
        channel_multiplier=args.channel_multiplier,
        constant_input=args.const,
        checkpoint=args.ckpt,
        output_size=args.out_size,
    )
    g_ema = th.nn.DataParallel(g_ema.cuda())

    with th.no_grad():
        latents = th.randn((args.num_frames, 512)).cuda()
        latents = g_ema(latents, map_latents=True).cpu().numpy()
        latents = ndi.gaussian_filter(latents, [5, 0, 0])
        latents = th.from_numpy(latents).cuda()
        print("input latents: ", latents.shape)

        noise = [
            np.random.normal(
                size=(args.num_frames, 1, 2 ** int(s / 2), (2 if args.out_size == 1920 else 1) * 2 ** int(s / 2),)
            )
            for s in range(5, 2 * 8, 1)
        ]
        noise = [th.from_numpy(ndi.gaussian_filter(n, [5, 0, 0, 0])).cuda() for n in noise]
        [print(n.shape) for n in noise]

        manipulations = [
            {"layer": 0, "transform": "double-width", "params": None, "indicies": None},
            {"layer": 4, "transform": "translate", "params": [0.25, 0.0], "indicies": "all"},
        ]

        render(
            generator=g_ema,
            latents=latents,
            noise=noise,
            duration=args.duration,
            batch_size=args.batch,
            truncation=args.truncation,
            manipulations=manipulations,
        )

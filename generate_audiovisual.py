import os, gc
import time, uuid, json, math
import argparse, random

import numpy as np
import scipy
import scipy.signal as signal
import sklearn.cluster

import matplotlib.pyplot as plt

import madmom as mm
import librosa as rosa
import librosa.display

import torch as th
import torch.nn.functional as F

from functools import partial
import kornia.augmentation as kA
import kornia.geometry.transform as kT

import render
import generate
from models.stylegan1 import G_style
from models.stylegan2 import Generator


def get_noise_range(size, generator_resolution, is_stylegan1):
    log_max_res = int(np.log2(size))
    log_min_res = 2 + (log_max_res - int(np.log2(generator_resolution)))
    if is_stylegan1:
        range_min = log_min_res
        range_max = log_max_res + 1
        side_fn = lambda x: x
        max_noise_scale = 8
    else:
        range_min = 2 * log_min_res + 1
        range_max = 2 * (log_max_res + 1)
        side_fn = lambda x: int(x / 2)
        max_noise_scale = 2 * (8 + 1)
    return log_min_res, range_min, min(range_max, max_noise_scale), side_fn


def get_latent_selection(latent_file):
    try:
        latent_selection = th.from_numpy(np.load(latent_file))
    except:
        print("generating random latents...")
        generator = Generator(
            args.G_res,
            512,
            8,
            channel_multiplier=args.channel_multiplier,
            constant_input=args.const,
            checkpoint=args.ckpt,
            output_size=args.size,
        ).cuda()
        styles = th.randn((12, 512), device="cuda")
        latent_selection = generator(styles, map_latents=True).cpu()
        del generator, styles
        gc.collect()
        th.cuda.empty_cache()
    return latent_selection


def load_generator(ckpt, is_stylegan1, G_res, size, const, latent_dim, n_mlp, channel_multiplier, dataparallel):
    if is_stylegan1:
        generator = G_style(output_size=size, checkpoint=ckpt).cuda()
    else:
        generator = Generator(
            G_res,
            latent_dim,
            n_mlp,
            channel_multiplier=channel_multiplier,
            constant_input=const,
            checkpoint=ckpt,
            output_size=size,
        ).cuda()
    if dataparallel:
        generator = th.nn.DataParallel(generator)
    return generator


if __name__ == "__main__":
    time_taken = time.time()
    th.set_grad_enabled(False)

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--audio_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--offset", type=float, default=0)
    parser.add_argument("--duration", default=None)
    parser.add_argument("--latent_file", type=str, default=None)
    parser.add_argument("--shuffle_latents", action="store_true")
    parser.add_argument("--G_res", type=int, default=1024)
    parser.add_argument("--size", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dataparallel", action="store_true")
    parser.add_argument("--truncation", type=int, default=1)
    parser.add_argument("--stylegan1", action="store_true")
    parser.add_argument("--const", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--randomize_noise", action="store_true")

    args = parser.parse_args()

    if args.duration is None:
        args.duration = rosa.get_duration(filename=args.audio_file)
    num_frames = int(round(args.duration * args.fps))

    main_audio, sr = rosa.load(args.audio_file, offset=args.offset, duration=args.duration)

    smf = args.fps / 43.066666  # smoothing factor, makes sure visual smoothness is independent of frame rate

    # generate audiovisual latents
    latent_selection = get_latent_selection(args.latent_file)
    if args.shuffle_latents:
        random_indices = random.sample(range(len(latent_selection)), len(latent_selection))
        latent_selection = latent_selection[random_indices]
    np.save("workspace/last-latents.npy", latent_selection.numpy())

    latents = get_latents(main_audio, latent_selection)
    print("latents:")
    for level in range(latents.shape[1]):
        print(list(latents[:, level].shape), f"amplitude={latents[:, level].std()}")
    print()

    # generate audiovisual noise
    print("noise:")
    noise = []
    log_min_res, range_min, range_max, exponent = get_noise_range(args.size, args.G_res, args.stylegan1)
    for scale in range(range_min, range_max):
        h = 2 ** exponent(scale)
        w = (2 if args.size == 1920 else 1) * 2 ** exponent(scale)

        noise.append(get_noise(main_audio, h, w, scale))

        if noise[-1] is not None:
            print(list(noise[-1].shape), f"amplitude={noise[-1].std()}")
    print()

    # generate audiovisual network bending manipulations
    bends = get_manipulations(main_audio)

    # generate audiovisual model rewriting manipulations
    rewrites = get_rewrites(main_audio)

    # render the given (latent, noise, bends, rewrites, truncation) interpolation
    checkpoint_title = args.ckpt.split("/")[-1].split(".")[0].lower()
    track_title = args.audio_file.split("/")[-1].split(".")[0].lower()
    title = f"{output_dir}}/{track_title}_{checkpoint_title}_{uuid.uuid4().hex[:8]}.mp4"

    print(f"rendering {num_frames} frames...")
    render.render(
        generator=load_generator(
            ckpt=args.ckpt,
            is_stylegan1=args.stylegan1,
            G_res=args.G_res,
            size=args.size,
            const=args.const,
            latent_dim=args.latent_dim,
            n_mlp=args.n_mlp,
            channel_multiplier=args.channel_multiplier,
            dataparallel=args.dataparallel,
        ),
        latents=latents,
        noise=noise,
        audio_file=args.audio_file,
        offset=args.offset,
        duration=args.duration,
        batch_size=args.batch,
        truncation=args.truncation,
        bends=bends,
        rewrites=rewrites,
        out_size=args.size,
        output_file=title,
        randomize_noise=args.randomize_noise,
    )

    print(f"Took {(time.time() - time_taken)/60:.2f} minutes")

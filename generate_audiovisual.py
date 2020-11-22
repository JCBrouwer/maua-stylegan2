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
import audioreactive as ar


def get_noise_range(size, generator_resolution, is_stylegan1):
    log_max_res = int(np.log2(size))
    log_min_res = 2 + (log_max_res - int(np.log2(generator_resolution)))
    if is_stylegan1:
        range_min = log_min_res
        range_max = log_max_res + 1
        side_fn = lambda x: x
    else:
        range_min = 2 * log_min_res + 1
        range_max = 2 * (log_max_res + 1)
        side_fn = lambda x: int(x / 2)
    return range_min, range_max, side_fn


def load_or_generate_latents(latent_file):
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
    parser.add_argument("--audioreactive_file", type=str, default="audioreactive_example.py")
    parser.add_argument("--offset", type=float, default=0)
    parser.add_argument("--duration", default=None)
    parser.add_argument("--latent_file", type=str, default=None)
    parser.add_argument("--shuffle_latents", action="store_true")
    parser.add_argument("--G_res", type=int, default=1024)
    parser.add_argument("--size", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dataparallel", action="store_true")
    parser.add_argument("--truncation", default=1.0)
    parser.add_argument("--stylegan1", action="store_true")
    parser.add_argument("--const", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--randomize_noise", action="store_true")

    args = parser.parse_args()

    args.audioreactive_file = args.audioreactive_file.replace(".py", "")  # strip extension from file name

    if args.duration is None:
        args.duration = rosa.get_duration(filename=args.audio_file)
    num_frames = int(round(args.duration * args.fps))

    main_audio, sr = rosa.load(args.audio_file, offset=args.offset, duration=args.duration)

    # ====================================================================================
    # =========================== generate audiovisual latents ===========================
    # ====================================================================================
    print("generating latents...")
    try:
        get_latents = __import__(args.audioreactive_file).get_latents
    except:
        print("No 'get_latents()' function found in --audioreactive_file, using default...")
        from audioreactive_example import get_latents

    latent_selection = load_or_generate_latents(args.latent_file)
    if args.shuffle_latents:
        random_indices = random.sample(range(len(latent_selection)), len(latent_selection))
        latent_selection = latent_selection[random_indices]
    np.save("workspace/last-latents.npy", latent_selection.numpy())

    latents = get_latents(audio=main_audio, sr=sr, num_frames=num_frames, selection=latent_selection).cpu()

    print(f"{list(latents.shape)} amplitude={latents.std()}\n")

    # ====================================================================================
    # ============================ generate audiovisual noise ============================
    # ====================================================================================
    print("generating noise...")
    try:
        get_noise = __import__(args.audioreactive_file).get_noise
    except:
        print("No 'get_noise()' function found in --audioreactive_file, using default...")
        from audioreactive_example import get_noise

    noise = []
    range_min, range_max, exponent = get_noise_range(args.size, args.G_res, args.stylegan1)
    for scale in range(range_min, range_max):
        h = 2 ** exponent(scale)
        w = (2 if args.size == 1920 else 1) * 2 ** exponent(scale)

        noise.append(
            get_noise(
                audio=main_audio,
                sr=sr,
                num_frames=num_frames,
                num_scales=range_max - range_min,
                height=h,
                width=w,
                scale=scale - range_min,
            )
        )

        if noise[-1] is not None:
            print(list(noise[-1].shape), f"amplitude={noise[-1].std()}")
    print()

    # ====================================================================================
    # ================ generate audiovisual network bending manipulations ================
    # ====================================================================================
    print("generating network bends...")
    try:
        get_bends = __import__(args.audioreactive_file).get_bends
    except:
        print("No 'get_bends()' function found in --audioreactive_file, using default...")
        from audioreactive_example import get_bends

    bends = get_bends(audio=main_audio)

    # ====================================================================================
    # ================ generate audiovisual model rewriting manipulations ================
    # ====================================================================================
    print("generating model rewrites...")
    try:
        get_rewrites = __import__(args.audioreactive_file).get_rewrites
    except:
        print("No 'get_rewrites()' function found in --audioreactive_file, using default...")
        from audioreactive_example import get_rewrites

    rewrites = get_rewrites(audio=main_audio)

    # ====================================================================================
    # ========================== generate audiovisual truncation =========================
    # ====================================================================================
    if args.truncation == "reactive":
        print("generating truncation...")
        try:
            get_truncation = __import__(args.audioreactive_file).get_truncation
        except:
            print("No 'get_truncation()' function found in --audioreactive_file, using default...")
            from audioreactive_example import get_truncation

        truncation = get_truncation(audio=main_audio)
    else:
        truncation = float(args.truncation)

    # ====================================================================================
    # ==== render the given (latent, noise, bends, rewrites, truncation) interpolation ===
    # ====================================================================================
    print(f"\npreprocessing took {time.time() - time_taken:.2f}s\n")
    time_taken = time.time()

    checkpoint_title = args.ckpt.split("/")[-1].split(".")[0].lower()
    track_title = args.audio_file.split("/")[-1].split(".")[0].lower()
    title = f"{args.output_dir}/{track_title}_{checkpoint_title}_{uuid.uuid4().hex[:8]}.mp4"

    ar.CACHE.clear()
    gc.collect()
    th.cuda.empty_cache()

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
        truncation=truncation,
        bends=bends,
        rewrites=rewrites,
        out_size=args.size,
        output_file=title,
        randomize_noise=args.randomize_noise,
    )

    print(f"\nrendering took {(time.time() - time_taken)/60:.2f} minutes")

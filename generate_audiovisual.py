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


def generate(
    get_latents,
    get_noise,
    get_bends,
    get_rewrites,
    get_truncation,
    ckpt,
    audio_file,
    output_dir,
    audioreactive_file,
    offset,
    duration,
    latent_file,
    shuffle_latents,
    G_res,
    size,
    fps,
    batch,
    dataparallel,
    truncation,
    stylegan1,
    const,
    latent_dim,
    n_mlp,
    channel_multiplier,
    randomize_noise,
):
    time_taken = time.time()
    th.set_grad_enabled(False)

    if duration is None:
        duration = rosa.get_duration(filename=audio_file)
    num_frames = int(round(duration * fps))

    main_audio, sr = rosa.load(audio_file, offset=offset, duration=duration)

    # ====================================================================================
    # =========================== generate audiovisual latents ===========================
    # ====================================================================================
    print("generating latents...")

    latent_selection = load_or_generate_latents(latent_file)
    if shuffle_latents:
        random_indices = random.sample(range(len(latent_selection)), len(latent_selection))
        latent_selection = latent_selection[random_indices]
    np.save("workspace/last-latents.npy", latent_selection.numpy())

    latents = get_latents(audio=main_audio, sr=sr, num_frames=num_frames, selection=latent_selection).cpu()

    print(f"{list(latents.shape)} amplitude={latents.std()}\n")

    # ====================================================================================
    # ============================ generate audiovisual noise ============================
    # ====================================================================================
    print("generating noise...")

    noise = []
    range_min, range_max, exponent = get_noise_range(size, G_res, stylegan1)
    for scale in range(range_min, range_max):
        h = 2 ** exponent(scale)
        w = (2 if size == 1920 else 1) * 2 ** exponent(scale)

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
    bends = get_bends(audio=main_audio, num_frames=num_frames, duration=duration, fps=fps)

    # ====================================================================================
    # ================ generate audiovisual model rewriting manipulations ================
    # ====================================================================================
    print("generating model rewrites...")
    rewrites = get_rewrites(audio=main_audio, num_frames=num_frames)

    # ====================================================================================
    # ========================== generate audiovisual truncation =========================
    # ====================================================================================
    if truncation == "reactive":
        print("generating truncation...")
        truncation = get_truncation(audio=main_audio)
    else:
        truncation = float(truncation)

    # ====================================================================================
    # ==== render the given (latent, noise, bends, rewrites, truncation) interpolation ===
    # ====================================================================================
    print(f"\npreprocessing took {time.time() - time_taken:.2f}s\n")
    time_taken = time.time()

    checkpoint_title = ckpt.split("/")[-1].split(".")[0].lower()
    track_title = audio_file.split("/")[-1].split(".")[0].lower()
    title = f"{output_dir}/{track_title}_{checkpoint_title}_{uuid.uuid4().hex[:8]}.mp4"

    ar.CACHE.clear()
    gc.collect()
    th.cuda.empty_cache()

    print(f"rendering {num_frames} frames...")
    render.render(
        generator=load_generator(
            ckpt=ckpt,
            is_stylegan1=stylegan1,
            G_res=G_res,
            size=size,
            const=const,
            latent_dim=latent_dim,
            n_mlp=n_mlp,
            channel_multiplier=channel_multiplier,
            dataparallel=dataparallel,
        ),
        latents=latents,
        noise=noise,
        audio_file=audio_file,
        offset=offset,
        duration=duration,
        batch_size=batch,
        truncation=truncation,
        bends=bends,
        rewrites=rewrites,
        out_size=size,
        output_file=title,
        randomize_noise=randomize_noise,
    )

    print(f"\nrendering took {(time.time() - time_taken)/60:.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--audio_file", type=str)
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--audioreactive_file", type=str, default="audioreactive/default.py")
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

    modnames = args.audioreactive_file.replace(".py", "").replace("/", ".").split(".")

    func_names = ["get_latents", "get_noise", "get_bends", "get_rewrites"]
    if args.truncation == "reactive":
        func_names += "get_truncation"
    else:
        funcs = {"get_truncation": None}
    for func in func_names:
        try:
            file = __import__(".".join(modnames[:-1]), fromlist=[modnames[-1]]).__dict__[modnames[-1]]
            funcs[func] = getattr(file, func)
        except:
            print(f"No '{func}' function found in --audioreactive_file, using default...")
            file = __import__("audioreactive", fromlist=["default"]).__dict__["default"]
            funcs[func] = getattr(file, func)

    generate(**funcs, **vars(args))

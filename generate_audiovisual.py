import argparse
import gc
import random
import time
import traceback
import uuid
import warnings

import librosa as rosa
import librosa.display
import numpy as np
import torch as th

import audioreactive as ar
import generate
import render
from models.stylegan1 import G_style
from models.stylegan2 import Generator


def get_noise_range(out_size, generator_resolution, is_stylegan1):
    """Gets the correct number of noise resolutions for a given resolution of StyleGAN 1 or 2"""
    log_max_res = int(np.log2(out_size))
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


def load_generator(
    ckpt, is_stylegan1, G_res, out_size, noconst, latent_dim, n_mlp, channel_multiplier, dataparallel, base_res_factor
):
    """Loads a StyleGAN 1 or 2 generator"""
    if is_stylegan1:
        generator = G_style(output_size=out_size, checkpoint=ckpt).cuda()
    else:
        generator = Generator(
            G_res,
            latent_dim,
            n_mlp,
            channel_multiplier=channel_multiplier,
            constant_input=not noconst,
            checkpoint=ckpt,
            output_size=out_size,
            base_res_factor=base_res_factor,
        ).cuda()
    if dataparallel:
        generator = th.nn.DataParallel(generator)
    return generator


def generate(
    ckpt,
    audio_file,
    initialize=None,
    get_latents=None,
    get_noise=None,
    get_bends=None,
    get_rewrites=None,
    get_truncation=None,
    output_dir="./output",
    audioreactive_file="audioreactive/examples/default.py",
    offset=0,
    duration=None,
    latent_file=None,
    shuffle_latents=False,
    G_res=1024,
    out_size=1024,
    fps=30,
    batch=8,
    dataparallel=False,
    truncation=1.0,
    stylegan1=False,
    noconst=False,
    latent_dim=512,
    n_mlp=8,
    channel_multiplier=2,
    randomize_noise=False,
    ffmpeg_preset="slow",
    base_res_factor=1,
    args=None,
):
    # if args is empty (i.e. generate() called directly instead of through __main__)
    # create args Namespace with all locally available variables
    if args is None:
        kwargs = locals()
        args = argparse.Namespace()
        for k, v in kwargs.items():
            setattr(args, k, v)

    # ensures smoothing is independent of frame rate
    ar.set_SMF(args.fps / 30)

    time_taken = time.time()
    th.set_grad_enabled(False)

    audio_dur = rosa.get_duration(filename=audio_file)
    if duration is None or audio_dur < duration:
        duration = audio_dur

    n_frames = int(round(duration * fps))
    args.duration = duration
    args.n_frames = n_frames

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
        audio, sr = rosa.load(audio_file, offset=offset, duration=duration)
    args.audio = audio
    args.sr = sr

    if initialize is not None:
        args = initialize(args)

    # ====================================================================================
    # =========================== generate audiovisual latents ===========================
    # ====================================================================================
    print("\ngenerating latents...")
    if get_latents is None:
        from audioreactive.default import get_latents

    if latent_file is not None:
        latent_selection = ar.load_latents(latent_file)
    else:
        latent_selection = ar.generate_latents(12, ckpt, G_res, noconst, latent_dim, n_mlp, channel_multiplier)
    if shuffle_latents:
        random_indices = random.sample(range(len(latent_selection)), len(latent_selection))
        latent_selection = latent_selection[random_indices]
    np.save("workspace/last-latents.npy", latent_selection.numpy())

    latents = get_latents(selection=latent_selection, args=args).cpu()

    print(f"{list(latents.shape)} amplitude={latents.std()}\n")

    # ====================================================================================
    # ============================ generate audiovisual noise ============================
    # ====================================================================================
    print("generating noise...")
    if get_noise is None:
        from audioreactive.default import get_noise

    noise = []
    range_min, range_max, exponent = get_noise_range(out_size, G_res, stylegan1)
    for scale in range(range_min, range_max):
        h = (2 if out_size == 1080 else 1) * 2 ** exponent(scale)
        w = (2 if out_size == 1920 else 1) * 2 ** exponent(scale)

        noise.append(get_noise(height=h, width=w, scale=scale - range_min, num_scales=range_max - range_min, args=args))

        if noise[-1] is not None:
            print(list(noise[-1].shape), f"amplitude={noise[-1].std()}")
    print()

    # ====================================================================================
    # ================ generate audiovisual network bending manipulations ================
    # ====================================================================================
    if get_bends is not None:
        print("generating network bends...")
        bends = get_bends(args=args)
    else:
        bends = []

    # ====================================================================================
    # ================ generate audiovisual model rewriting manipulations ================
    # ====================================================================================
    if get_rewrites is not None:
        print("generating model rewrites...")
        rewrites = get_rewrites(args=args)
    else:
        rewrites = {}

    # ====================================================================================
    # ========================== generate audiovisual truncation =========================
    # ====================================================================================
    if get_truncation is not None:
        print("generating truncation...")
        truncation = get_truncation(args=args)
    else:
        truncation = float(truncation)

    # ====================================================================================
    # ==== render the given (latent, noise, bends, rewrites, truncation) interpolation ===
    # ====================================================================================
    gc.collect()
    th.cuda.empty_cache()

    generator = load_generator(
        ckpt=ckpt,
        is_stylegan1=stylegan1,
        G_res=G_res,
        out_size=out_size,
        noconst=noconst,
        latent_dim=latent_dim,
        n_mlp=n_mlp,
        channel_multiplier=channel_multiplier,
        dataparallel=dataparallel,
        base_res_factor=base_res_factor,
    )

    print(f"\npreprocessing took {time.time() - time_taken:.2f}s\n")

    print(f"rendering {n_frames} frames...")
    checkpoint_title = ckpt.split("/")[-1].split(".")[0].lower()
    track_title = audio_file.split("/")[-1].split(".")[0].lower()
    title = f"{output_dir}/{track_title}_{checkpoint_title}_{uuid.uuid4().hex[:8]}.mp4"
    render.render(
        generator=generator,
        latents=latents,
        noise=noise,
        audio_file=audio_file,
        offset=offset,
        duration=duration,
        batch_size=batch,
        truncation=truncation,
        bends=bends,
        rewrites=rewrites,
        out_size=out_size,
        output_file=title,
        randomize_noise=randomize_noise,
        ffmpeg_preset=ffmpeg_preset,
    )

    print(f"\ntotal time taken: {(time.time() - time_taken)/60:.2f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--audio_file", type=str)
    parser.add_argument("--audioreactive_file", type=str, default="audioreactive/examples/default.py")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--offset", type=float, default=0)
    parser.add_argument("--duration", default=None)
    parser.add_argument("--latent_file", type=str, default=None)
    parser.add_argument("--shuffle_latents", action="store_true")
    parser.add_argument("--G_res", type=int, default=1024)
    parser.add_argument("--out_size", type=int, default=1024)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--dataparallel", action="store_true")
    parser.add_argument("--truncation", type=float, default=1.0)
    parser.add_argument("--stylegan1", action="store_true")
    parser.add_argument("--noconst", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--randomize_noise", action="store_true")
    parser.add_argument("--base_res_factor", type=float, default=1)
    parser.add_argument("--ffmpeg_preset", type=str, default="slow")
    args = parser.parse_args()

    # transform file path to python module string
    modnames = args.audioreactive_file.replace(".py", "").replace("/", ".").split(".")

    # try to load each of the standard functions from the specified file
    func_names = ["initialize", "get_latents", "get_noise", "get_bends", "get_rewrites", "get_truncation"]
    funcs = {}
    for func in func_names:
        try:
            file = __import__(".".join(modnames[:-1]), fromlist=[modnames[-1]]).__dict__[modnames[-1]]
            funcs[func] = getattr(file, func)
        except AttributeError as error:
            print(f"No '{func}' function found in --audioreactive_file, using default...")
            funcs[func] = None
        except:
            if funcs.get(func, "error") == "error":
                print("Error while loading --audioreactive_file...")
                traceback.print_exc()
                exit(1)

    # override with args from the OVERRIDE dict in the specified file
    arg_dict = vars(args).copy()
    try:
        file = __import__(".".join(modnames[:-1]), fromlist=[modnames[-1]]).__dict__[modnames[-1]]
        for arg, val in getattr(file, "OVERRIDE").items():
            arg_dict[arg] = val
            setattr(args, arg, val)
    except:
        pass  # no overrides, just continue

    ckpt = arg_dict.pop("ckpt", None)
    audio_file = arg_dict.pop("audio_file", None)

    # splat kwargs to function call
    # (generate() has all kwarg defaults specified again to make it amenable to ipynb usage)
    generate(ckpt=ckpt, audio_file=audio_file, **funcs, **arg_dict, args=args)

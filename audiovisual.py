import os, gc
import time, uuid, json
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch as th
import torch.nn.functional as F

import madmom as mm
import librosa as rosa
import librosa.display
import scipy.signal as signal

from functools import partial
import kornia.augmentation as kA
import kornia.geometry.transform as kT

import render
import generate
from models.stylegan1 import G_style
from models.stylegan2 import Generator

time_taken = time.time()
th.set_grad_enabled(False)
plt.rcParams["axes.facecolor"] = "black"
plt.rcParams["figure.facecolor"] = "black"
VERBOSE = False
# VERBOSE = True

parser = argparse.ArgumentParser()

parser.add_argument("--ckpt", type=str)
parser.add_argument("--G_res", type=int, default=1024)
parser.add_argument("--size", type=int, default=1920)
parser.add_argument("--batch", type=int, default=12)
parser.add_argument("--duration", type=int, default=None)
parser.add_argument("--const", type=bool, default=False)
parser.add_argument("--channel_multiplier", type=int, default=2)
parser.add_argument("--truncation", type=int, default=1.5)
parser.add_argument("--stylegan1", type=bool, default=False)
parser.add_argument("--slerp", type=bool, default=True)
parser.add_argument("--latents", type=str, default=None)
parser.add_argument("--random_latents", action="store_true")
parser.add_argument("--color_latents", type=str, default=None)
parser.add_argument("--color_layer", type=int, default=6)

args = parser.parse_args()


def gaussian_filter(x, sigma, causal=False):
    dim = len(x.shape)
    while len(x.shape) < 3:
        x = x[:, None]

    radius = int(sigma * 4)
    channels = x.shape[1]

    kernel = th.arange(-radius, radius + 1, dtype=th.float32, device=x.device)
    kernel = th.exp(-0.5 / sigma ** 2 * kernel ** 2)
    if causal:
        kernel[:radius] *= 0.1
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

    if len(x.shape) > dim:
        x = x.squeeze()

    return x


def info(arr):
    if isinstance(arr, list):
        print([(list(a.shape), f"{a.min():.2f}", f"{a.float().mean():.2f}", f"{a.max():.2f}") for a in arr])
    else:
        print(list(arr.shape), f"{arr.min():.2f}", f"{arr.float().mean():.2f}", f"{arr.max():.2f}")


def percentile(y, p):
    k = 1 + round(0.01 * float(p) * (y.numel() - 1))
    return y.view(-1).kthvalue(k).values.item()


def percentile_clip(y, p):
    locs = th.arange(0, y.shape[0])
    peaks = th.ones(y.shape, dtype=bool)
    main = y.take(locs)

    plus = y.take((locs + 1).clamp(0, y.shape[0] - 1))
    minus = y.take((locs - 1).clamp(0, y.shape[0] - 1))
    peaks &= th.gt(main, plus)
    peaks &= th.gt(main, minus)

    y = y.clamp(0, percentile(y[peaks], p))
    y /= y.max()
    return y


def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def normalize(y):
    y -= y.min()
    y /= y.max()
    return y


def wrapping_slice(tensor, start, length, return_indices=False):
    if start + length <= tensor.shape[0]:
        indices = th.arange(start, start + length)
    else:
        indices = th.cat((th.arange(start, tensor.shape[0]), th.arange(0, (start + length) % tensor.shape[0])))
    if tensor.shape[0] == 1:
        indices = th.zeros(1, dtype=th.int64)
    if return_indices:
        return indices
    return tensor[indices]


def plot_signals(signals, vlines=None):
    if not VERBOSE:
        return
    info(signals)
    plt.figure(figsize=(30, 6 * len(signals)))
    for sbplt, y in enumerate(signals):
        plt.subplot(len(signals), 1, sbplt + 1)
        if vlines is not None:
            plt.vlines(vlines, 0.0, 1.0)
        plt.plot(y.squeeze())
        plt.tight_layout()
    plt.show()


def plot_spectra(spectra, chroma=False):
    if not VERBOSE:
        return
    plt.figure(figsize=(30, 8 * len(spectra)))
    for sbplt, spectrum in enumerate(spectra):
        rosa.display.specshow(spectrum, y_axis="chroma" if chroma else None, x_axis="time")
        plt.show()


if __name__ == "__main__":
    # args.ckpt = "/home/hans/modelzoo/lakspe_stylegan1.pt"
    args.ckpt = "/home/hans/modelzoo/maua-sg2/cyphept-CYPHEPT-2q5b2lk6-33-1024-145000.pt"
    args.audio_file = f"../../datasets/RKU_LHH_Keys_Loop_Cool_70_Amin.wav"
    args.bpm = 70
    args.fps = 30
    args.offset = 0
    args.duration = None

    file_root = args.audio_file.split("/")[-1].split(".")[0]
    metadata_file = f"workspace/{file_root}_metadata.json"
    intro_file = f"workspace/naamloos_intro_latents.npy"
    drop_file = f"workspace/naamloos_drop_latents.npy"
    latent_file = f"workspace/{file_root}_latents.npy"
    noise_file = f"workspace/{file_root}_noise.npy"

    if args.duration is None:
        args.duration = rosa.get_duration(filename=args.audio_file)

    if os.path.exists(metadata_file):
        with open(metadata_file) as json_file:
            data = json.load(json_file)
        total_frames = data["total_frames"]
    else:
        audio, sr = rosa.load(args.audio_file)
        onset = rosa.onset.onset_strength(audio, sr=sr)
        total_frames = len(onset)
        with open(metadata_file, "w") as outfile:
            json.dump({"total_frames": total_frames}, outfile)

    smf = args.fps / 43.066666
    num_frames = int(round(args.duration * args.fps))

    print(f"processing audio files...")
    main_audio, sr = rosa.load(args.audio_file, offset=args.offset, duration=args.duration)

    y_harm = rosa.effects.harmonic(y=main_audio, margin=8)
    chroma_harm = rosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma = np.minimum(chroma_harm, rosa.decompose.nn_filter(chroma_harm, aggregate=np.median, metric="cosine")).T
    chroma = signal.resample(chroma, num_frames)
    chroma = th.from_numpy(chroma / chroma.sum(1)[:, None])

    def get_chroma_loops(base_latent_selection, n_frames, chroma, loop=True):
        # chromhalf = th.stack([chroma[2 * i : 2 * i + 2].sum(0) for i in range(int(len(chroma) / 2))])
        base_latents = (chroma[..., None, None] * base_latent_selection[None, ...]).sum(1)
        return base_latents

    latent_files_exist = os.path.exists(drop_file)
    if latent_files_exist:
        latent_selection = th.from_numpy(np.load(drop_file))
    if args.random_latents or not latent_files_exist:
        print("generating random latents")
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

    info(chroma)
    info(latent_selection)
    latents = get_chroma_loops(
        base_latent_selection=wrapping_slice(latent_selection, 0, 12), n_frames=num_frames, chroma=chroma
    )
    info(latents)
    latents = gaussian_filter(latents.float().cuda(), int(round(1 * smf))).cpu()
    info(latents)

    log_max_res = int(np.log2(args.size))
    log_min_res = 2 + (log_max_res - int(np.log2(args.G_res)))

    noise = []
    if args.stylegan1:
        range_min = log_min_res
        range_max = log_max_res + 1
        side_fn = lambda x: x
    else:
        range_min = 2 * log_min_res + 1
        range_max = 2 * (log_max_res + 1)
        side_fn = lambda x: int(x / 2)

    max_noise_scale = 2 * (7 + 1)
    # max_noise_scale = 8
    for s in range(range_min, min(max_noise_scale, range_max)):
        h = 2 ** side_fn(s)
        w = (2 if args.size == 1920 else 1) * 2 ** side_fn(s)
        print(num_frames, 1, h, w)

        noise.append(
            gaussian_filter(th.randn((num_frames, 1, h, w), device="cuda"), max(1, int(round(20 * smf)))).cpu()
        )
        noise[-1] /= noise[-1].std() * 2

        gc.collect()
        th.cuda.empty_cache()
    noise += [None] * (17 - len(noise))

    if args.stylegan1:
        generator = G_style(output_size=args.size, checkpoint=args.ckpt).cuda()
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
            output_size=args.size,
        ).cuda()
    # generator = th.nn.DataParallel(generator)

    manipulations = []

    print(f"rendering {num_frames} frames...")
    checkpoint_title = args.ckpt.split("/")[-1].split(".")[0].lower()
    track_title = args.audio_file.split("/")[-1].split(".")[0].lower()
    title = f"/home/hans/neurout/{track_title}_{checkpoint_title}_{uuid.uuid4().hex[:8]}.mp4"
    render.render(
        generator=generator,
        latents=latents,
        noise=noise,
        audio_file=args.audio_file,
        offset=args.offset,
        duration=args.duration,
        batch_size=args.batch,
        truncation=args.truncation,
        manipulations=manipulations,
        out_size=args.size,
        output_file=title,
    )


print(f"Took {(time.time() - time_taken)/60:.2f} minutes")


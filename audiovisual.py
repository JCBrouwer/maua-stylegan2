import os, gc
import time, uuid, json
import argparse

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


def gaussian_filter(x, sigma, causal=None):
    dim = len(x.shape)
    while len(x.shape) < 3:
        x = x[:, None]

    radius = int(sigma * 4)
    channels = x.shape[1]

    kernel = th.arange(-radius, radius + 1, dtype=th.float32, device=x.device)
    kernel = th.exp(-0.5 / sigma ** 2 * kernel ** 2)
    if causal is not None:
        kernel[radius + 1 :] *= 0.1 if not isinstance(causal, float) else causal
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
        print([(list(a.shape), f"{a.min():.2f}", f"{a.mean():.2f}", f"{a.max():.2f}") for a in arr])
    else:
        print(list(arr.shape), f"{arr.min():.2f}", f"{arr.mean():.2f}", f"{arr.max():.2f}")


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
    plt.figure(figsize=(8, 2 * len(signals)))
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
    plt.figure(figsize=(8, 3 * len(spectra)))
    for sbplt, spectrum in enumerate(spectra):
        rosa.display.specshow(spectrum, y_axis="chroma" if chroma else None, x_axis="time")
    plt.show()


def get_noise_params(size, generator_resolution, is_stylegan1):
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
        max_noise_scale = 2 * (9 + 1)
    return range_min, range_max, side_fn, max_noise_scale


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
    plt.rcParams["axes.facecolor"] = "black"
    plt.rcParams["figure.facecolor"] = "black"
    VERBOSE = False
    VERBOSE = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--audio_file", type=str)
    parser.add_argument("--bpm", type=float)
    parser.add_argument("--offset", type=float, default=0)
    parser.add_argument("--duration", default=None)
    parser.add_argument("--latent_file", type=str, default=None)
    parser.add_argument("--G_res", type=int, default=1024)
    parser.add_argument("--size", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--dataparallel", action="store_true")
    parser.add_argument("--truncation", type=int, default=1)
    parser.add_argument("--stylegan1", action="store_true")
    parser.add_argument("--const", action="store_true")
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--channel_multiplier", type=int, default=2)

    args = parser.parse_args()

    if args.duration is None:
        args.duration = rosa.get_duration(filename=args.audio_file)
    num_frames = int(round(args.duration * args.fps))

    main_audio, sr = rosa.load(args.audio_file, offset=args.offset, duration=args.duration)

    smf = args.fps / 43.066666  # smoothing factor, makes sure visual smoothness is independent of frame rate

    # =========================================================================================
    # ========================== generate audiovisual latents =================================
    # =========================================================================================

    latent_selection = get_latent_selection(args.latent_file)
    np.save("workspace/last-latents.npy", latent_selection.numpy())

    # get drum onsets
    sig = mm.audio.signal.Signal(args.audio_file, num_channels=1)
    sig_frames = mm.audio.signal.FramedSignal(sig, frame_size=2048, hop_size=441)
    stft = mm.audio.stft.ShortTimeFourierTransform(sig_frames, ciruclar_shift=True)
    spec = mm.audio.spectrogram.Spectrogram(stft, ciruclar_shift=True)

    def get_onsets(spec, fmin, fmax, smooth, clip, power):
        log_filt_spec = mm.audio.spectrogram.FilteredSpectrogram(spec, num_bands=24, fmin=fmin, fmax=fmax)
        onset = np.sum(
            [
                mm.features.onsets.high_frequency_content(log_filt_spec),
                mm.features.onsets.spectral_diff(log_filt_spec),
                mm.features.onsets.spectral_flux(log_filt_spec),
                mm.features.onsets.superflux(log_filt_spec),
                mm.features.onsets.complex_flux(log_filt_spec),
                mm.features.onsets.modified_kullback_leibler(log_filt_spec),
            ],
            axis=0,
        )
        onset = np.clip(signal.resample(onset, num_frames), onset.min(), onset.max())
        onset = th.from_numpy(onset).float()
        onset = gaussian_filter(onset, smooth * smf, causal=True)
        onset = percentile_clip(onset, clip)
        onset = onset ** power
        return onset

    kick_onset = get_onsets(spec, fmin=30, fmax=200, smooth=9, clip=99, power=2)
    snare_onset = get_onsets(spec, fmin=250, fmax=350, smooth=5, clip=95, power=1)
    hats_onset = get_onsets(spec, fmin=1000, fmax=18000, smooth=3, clip=90, power=2)
    # hats_onset -= snare_onset
    # hats_onset = gaussian_filter(hats_onset, 2 * smf, causal=0.1)
    # hats_onset = percentile_clip(hats_onset, 90)
    # plot_signals([kick_onset, snare_onset, hats_onset])

    # def get_chroma(audio, num_frames):
    #     y_harm = rosa.effects.harmonic(y=audio, margin=16)
    #     chroma = rosa.feature.chroma_cqt(y=y_harm, sr=sr)
    #     chroma = np.minimum(chroma, rosa.decompose.nn_filter(chroma, aggregate=np.median, metric="cosine")).T
    #     chroma = signal.resample(chroma, num_frames)
    #     chroma = th.from_numpy(chroma / chroma.sum(1)[:, None])
    #     return chroma

    # def get_chroma_latents(chroma, base_latent_selection):
    #     base_latents = (chroma[..., None, None] * base_latent_selection[None, ...]).sum(1)
    #     return base_latents

    # # separate bass and main harmonic frequencies
    # mid_chroma = get_chroma(signal.sosfilt(signal.butter(24, 220, "hp", fs=sr, output="sos"), main_audio), num_frames)
    # mid_latents = get_chroma_latents(chroma=mid_chroma, base_latent_selection=wrapping_slice(latent_selection, 0, 12))
    # bass_chroma = get_chroma(signal.sosfilt(signal.butter(24, 80, "lp", fs=sr, output="sos"), main_audio), num_frames)
    # latents = get_chroma_latents(chroma=bass_chroma, base_latent_selection=wrapping_slice(latent_selection, 0, 12))
    # mid_layer = 9
    # latents[:, mid_layer:] = mid_latents[:, mid_layer:]
    latents = th.ones((num_frames, 1, 1)) * latent_selection[[1]]
    latents = (
        0.75 * snare_onset[:, None, None] * th.ones((num_frames, 1, 1)) * latent_selection[[0]]
        + (1 - 0.75 * snare_onset[:, None, None]) * latents
    )

    # plt.figure(figsize=(12, 4))
    # plt.subplot(2, 1, 1)
    # librosa.display.specshow(mid_chroma.T.numpy(), y_axis="chroma")
    # plt.colorbar()
    # plt.ylabel("Mid")
    # plt.subplot(2, 1, 2)
    # librosa.display.specshow(bass_chroma.T.numpy(), y_axis="chroma", x_axis="time")
    # plt.colorbar()
    # plt.ylabel("Bass")
    # plt.tight_layout()
    # plt.show()
    # plt.close()

    # smooth the final latents just a bit to prevent any jitter or jerks
    latents = gaussian_filter(latents.float().cuda(), max(1, int(round(2 * smf))), causal=0.2).cpu()
    # color_layr = 5
    # latents[:, color_layr:] = latent_selection[[1], color_layr:]

    # =========================================================================================
    # ============================== generate audiovisual noise ===============================
    # =========================================================================================

    noise = []
    range_min, range_max, side_fn, max_noise_scale = get_noise_params(args.size, args.G_res, args.stylegan1)
    for s in range(range_min, min(max_noise_scale, range_max)):
        h = 2 ** side_fn(s)
        w = (2 if args.size == 1920 else 1) * 2 ** side_fn(s)
        # print(num_frames, 1, h, w)

        noise.append(
            gaussian_filter(th.randn((num_frames, 1, h, w), device="cuda"), max(1, int(round(20 * smf)))).cpu()
        )
        if s < int((2 * (9 + 1)) / 3) + 2:
            # print("kick", s)
            noise[-1] *= 1 - kick_onset[:, None, None, None]
            noise[-1] += (
                kick_onset[:, None, None, None]
                * gaussian_filter(th.randn((num_frames, 1, h, w), device="cuda"), max(1, int(round(5 * smf)))).cpu()
            )
        else:
            # print("hats", s)
            noise[-1] *= 1 - hats_onset[:, None, None, None]
            noise[-1] += (
                hats_onset[:, None, None, None]
                * gaussian_filter(th.randn((num_frames, 1, h, w), device="cuda"), max(1, int(round(3 * smf)))).cpu()
            )
        noise[-1] /= noise[-1].std()

        gc.collect()
        th.cuda.empty_cache()
    noise += [None] * (17 - len(noise))

    # =========================================================================================
    # ================== generate audiovisual network bending manipulations ===================
    # =========================================================================================

    manipulations = []

    # =========================================================================================
    # ============== render the given latent + noise + manipulation interpolation =============
    # =========================================================================================

    checkpoint_title = args.ckpt.split("/")[-1].split(".")[0].lower()
    track_title = args.audio_file.split("/")[-1].split(".")[0].lower()
    title = f"/home/hans/neurout/{track_title}_{checkpoint_title}_{uuid.uuid4().hex[:8]}.mp4"

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
        manipulations=manipulations,
        out_size=args.size,
        output_file=title,
    )


print(f"Took {(time.time() - time_taken)/60:.2f} minutes")

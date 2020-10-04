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
    fig, ax = plt.subplots(len(spectra), 1, figsize=(8, 3 * len(spectra)))
    for sbplt, spectrum in enumerate(spectra):
        rosa.display.specshow(spectrum, y_axis="chroma" if chroma else None, x_axis="time", ax=ax[sbplt])
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
        max_noise_scale = 2 * (8 + 1)
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
    # plt.rcParams["axes.facecolor"] = "black"
    # plt.rcParams["figure.facecolor"] = "black"
    # VERBOSE = False
    VERBOSE = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--audio_file", type=str)
    parser.add_argument("--bpm", type=float)
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
    if args.shuffle_latents:
        random_indices = random.sample(range(len(latent_selection)), len(latent_selection))
        latent_selection = latent_selection[random_indices]
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

    kick_onset = get_onsets(spec, fmin=125, fmax=200, smooth=5, clip=99, power=1)
    # snare_onset = get_onsets(spec, fmin=250, fmax=350, smooth=5, clip=95, power=1)
    snare_onset = get_onsets(spec, fmin=1000, fmax=18000, smooth=7, clip=97, power=2)
    # plot_signals([kick_onset, snare_onset])  # , hats_onset])

    def get_chroma(audio, num_frames):
        y_harm = rosa.effects.harmonic(y=audio, margin=16)
        chroma = rosa.feature.chroma_cqt(y=y_harm, sr=sr)
        chroma = np.minimum(chroma, rosa.decompose.nn_filter(chroma, aggregate=np.median, metric="cosine")).T
        chroma = signal.resample(chroma, num_frames)
        chroma = th.from_numpy(chroma / chroma.sum(1)[:, None])
        return chroma

    def get_chroma_latents(chroma, base_latent_selection):
        base_latents = (chroma[..., None, None] * base_latent_selection[None, ...]).sum(1)
        return base_latents

    # separate bass and main harmonic frequencies
    mid_chroma = get_chroma(
        signal.sosfilt(signal.butter(24, [100, 1000], "bp", fs=sr, output="sos"), main_audio), num_frames
    )
    latents = get_chroma_latents(chroma=mid_chroma, base_latent_selection=wrapping_slice(latent_selection, 0, 12))

    bass_chroma = get_chroma(
        signal.sosfilt(signal.butter(24 * 4, 72, "lp", fs=sr, output="sos"), main_audio), num_frames
    )
    bass_latents = get_chroma_latents(chroma=bass_chroma, base_latent_selection=wrapping_slice(latent_selection, 0, 12))
    crossover = 3
    latents[:, :crossover] = bass_latents[:, :crossover]

    high_mel = rosa.feature.melspectrogram(rosa.effects.harmonic(y=main_audio, margin=1), sr=sr, fmin=1000)
    high_onset = rosa.onset.onset_strength(S=high_mel, sr=sr)
    high_onset = th.from_numpy(signal.resample(high_onset, num_frames))
    high_onset = gaussian_filter(high_onset, 3 * smf, causal=0.3)
    # high_onset = high_onset ** 2
    high_onset = percentile_clip(high_onset, 96)
    high_onset = gaussian_filter(high_onset, 2 * smf, causal=0.1)

    # # fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    # # ax[0].plot(high_onset.squeeze())
    # # rosa.display.specshow(high_mel[:32], ax=ax[1])
    # # plt.tight_layout()
    # # plt.show()
    # # exit()

    # # latents = th.ones((num_frames, 1, 1)) * latent_selection[[1]]

    latents = (
        high_onset[:, None, None] * th.ones((num_frames, 1, 1)) * latent_selection[[-1]]
        + (1 - high_onset[:, None, None]) * latents
    )

    latents = (
        0.5 * snare_onset[:, None, None] * th.ones((num_frames, 1, 1)) * latent_selection[[-4]]
        + (1 - 0.5 * snare_onset[:, None, None]) * latents
    )

    moar_latents = get_latent_selection("workspace/cyphept-flat.npy")
    if args.shuffle_latents:
        random_indices = random.sample(range(len(moar_latents)), len(moar_latents))
        moar_latents = moar_latents[random_indices]
    intro_latents = get_chroma_latents(chroma=mid_chroma, base_latent_selection=wrapping_slice(moar_latents, 0, 12))
    intro_latents = gaussian_filter(intro_latents.float().cuda(), max(1, int(round(10 * smf))), causal=0.1).cpu()

    bass_audio = signal.sosfilt(signal.butter(24 * 4, 50, "lp", fs=sr, output="sos"), main_audio)
    bass_spec = np.abs(rosa.stft(bass_audio))
    bass_sum = bass_spec.sum(0)
    bass_sum = np.clip(signal.resample(bass_sum, num_frames), bass_sum.min(), bass_sum.max())
    bass_sum = percentile_clip(th.from_numpy(bass_sum).float(), 75) ** 2
    bass_sum = gaussian_filter(bass_sum, 100 * smf, causal=True)

    rms = rosa.feature.rms(S=np.abs(rosa.stft(y=main_audio, hop_length=512)))[0]
    rms = np.clip(signal.resample(rms, num_frames), rms.min(), rms.max())
    rms = percentile_clip(th.from_numpy(rms).float(), 75) ** 2
    rms = gaussian_filter(rms, 100 * smf, causal=True)

    drop_weight = percentile_clip(rms + bass_sum, 55) ** 2
    drop_weight[:700] = 0.1 * drop_weight[:700]
    drop_weight = gaussian_filter(drop_weight, 20 * smf, causal=0)
    # plot_signals([rms, bass_sum, drop_weight])
    # plot_signals([high_onset, snare_onset, kick_onset * drop_weight, drop_weight])

    latents = drop_weight[:, None, None] * latents + (1 - drop_weight[:, None, None]) * intro_latents

    # smooth the final latents just a bit to prevent any jitter or jerks
    latents = gaussian_filter(latents.float().cuda(), max(1, int(round(4 * smf))), causal=0.1).cpu()

    # =========================================================================================
    # ============================== generate audiovisual noise ===============================
    # =========================================================================================

    noise = []
    range_min, range_max, side_fn, max_noise_scale = get_noise_params(args.size, args.G_res, args.stylegan1)
    for s in range(range_min, min(max_noise_scale, range_max)):
        h = 2 ** side_fn(s)
        w = (2 if args.size == 1920 else 1) * 2 ** side_fn(s)

        noise.append(
            6 * gaussian_filter(th.randn((num_frames, 1, h, w), device="cuda"), max(1, int(round(100 * smf)))).cpu()
        )
        # # if s < int((2 * (9 + 1)) / 3) + 2:
        # # print("kick", s)
        # # noise[-1] *= 1 - snare_onset[:, None, None, None]
        noise[-1] += (
            high_onset[:, None, None, None]
            * drop_weight[:, None, None, None]
            * gaussian_filter(th.randn((num_frames, 1, h, w), device="cuda"), max(1, int(round(10 * smf)))).cpu()
        )
        # # noise[-1] *= 1 - kick_onset[:, None, None, None]
        noise[-1] += (
            0.5
            * kick_onset[:, None, None, None]
            # * drop_weight[:, None, None, None]
            * gaussian_filter(th.randn((num_frames, 1, h, w), device="cuda"), max(1, int(round(5 * smf)))).cpu()
        )
        # # else:
        # # print("hats", s)
        # # noise[-1] *= 1 - hats_onset[:, None, None, None]
        noise[-1] += (
            0.5
            * snare_onset[:, None, None, None]
            * gaussian_filter(th.randn((num_frames, 1, h, w), device="cuda"), max(1, int(round(3 * smf)))).cpu()
        )
        noise[-1] /= noise[-1].std()
        noise[-1] = gaussian_filter(noise[-1].cuda(), 2, causal=0).cpu()

        print(num_frames, 1, h, w, f"std.dev.={noise[-1].std()}")
        gc.collect()
        th.cuda.empty_cache()
    noise += [None] * (17 - len(noise))

    # =========================================================================================
    # ================== generate audiovisual network bending manipulations ===================
    # =========================================================================================

    class Manipulation(th.nn.Module):
        def __init__(self, sequential_fn, batch):
            super(Manipulation, self).__init__()
            self.sequential = sequential_fn(batch)

        def forward(self, x):
            return self.sequential(x)

    class addNoise(th.nn.Module):
        def __init__(self, noise):
            super(addNoise, self).__init__()
            self.noise = noise

        def forward(self, x):
            return x + self.noise.to(x.device)

    class Print(th.nn.Module):
        def forward(self, x):
            print(x.shape, [x.min().item(), x.mean().item(), x.max().item()], th.std(x).item())
            return x

    manipulations = [
        {
            "layer": 0,
            "transform": th.nn.Sequential(
                th.nn.ReplicationPad2d((2, 2, 0, 0)), addNoise(0.05 * th.randn(size=(1, 1, 4, 8), device="cuda")),
            ),
        }
    ]

    tl = 6
    width = (2 if args.size == 1920 else 1) * 2 ** int(2 + math.ceil(tl / 2))
    print(width)

    tl8_dist = int(width / 32)
    print(tl8_dist)
    smooth_noise = 0.3 * th.randn(size=(1, 1, 2 ** int(2 + math.ceil(tl / 2)), 2 * tl8_dist + width), device="cuda")

    class Translate(Manipulation):
        def __init__(self, layer, batch):
            layer_h = 2 ** int(2 + math.ceil(layer / 2))
            layer_w = (2 if args.size == 1920 else 1) * 2 ** int(2 + math.ceil(layer / 2))
            sequential_fn = lambda b: th.nn.Sequential(
                th.nn.ReflectionPad2d((tl8_dist, tl8_dist, 0, 0)),
                addNoise(smooth_noise),
                kT.Translate(b),
                kA.CenterCrop((layer_h, layer_w)),
            )
            super(Translate, self).__init__(sequential_fn, batch)

    translation = th.tensor([snare_onset.numpy() * tl8_dist, np.zeros(num_frames)]).float().T
    manipulations += [
        {"layer": tl, "transform": lambda batch, layer=tl: Translate(layer, batch), "params": translation}
    ]

    class Zoom(Manipulation):
        def __init__(self, layer, batch):
            layer_h = 2 ** int(2 + math.ceil(layer / 2))
            layer_w = (2 if args.size == 1920 else 1) * 2 ** int(2 + math.ceil(layer / 2))
            sequential_fn = lambda b: th.nn.Sequential(kT.Scale(b), kA.CenterCrop((layer_h, layer_w)))
            super(Zoom, self).__init__(sequential_fn, batch)

    zl = 6
    # zoom = 0.25 * kick_onset + 1
    zoom = 0.25 * (kick_onset * drop_weight) + 1
    manipulations += [{"layer": zl, "transform": lambda batch, layer=zl: Zoom(layer, batch), "params": zoom}]

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

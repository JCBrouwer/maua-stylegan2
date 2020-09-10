import os
import gc
import time
import uuid
import json
import render
import generate
import argparse
import numpy as np
import torch as th
import madmom as mm
import librosa as rosa
import scipy.signal as signal
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.stylegan2 import Generator
from models.stylegan1 import G_style

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
parser.add_argument("--color_layer", type=int, default=8)

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
        kernel[:radius] = 0
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
        # plt.subplot(len(spectrum), 1, sbplt + 1)
        rosa.display.specshow(spectrum, y_axis="chroma" if chroma else None, x_axis="time")
        # plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    args.ckpt = f"/home/hans/modelzoo/maua-sg2/cyphept-CYPHEPT-2q5b2lk6-33-1024-145000.pt"
    args.audio_file = f"../../datasets/naamloos.wav"
    args.bpm = 170
    args.fps = 30
    args.duration = None
    args.offset = 0

    file_root = args.audio_file.split("/")[-1].split(".")[0]
    metadata_file = f"workspace/{file_root}_metadata.json"
    intro_file = f"workspace/{file_root}_drop_latents.npy"
    drop_file = f"workspace/{file_root}_intro_latents.npy"
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

    prms = {
        "intro_num_beats": 64,
        "intro_loop_smoothing": 30,
        "intro_loop_factor": 0.4,
        "intro_loop_len": 12,
        "drop_num_beats": 32,
        "drop_loop_smoothing": 15,
        "drop_loop_factor": 1,
        "drop_loop_len": 6,
        "onset_smooth": 2,
        "onset_clip": 95,
        "freq_mod": 10,
        "freq_mod_offset": 0,
        "freq_smooth": 5,
        "freq_latent_smooth": 4,
        "freq_latent_layer": 1,
        "freq_latent_weight": 2,
        "high_freq_mod": 10,
        "high_freq_mod_offset": 0,
        "high_freq_smooth": 4,
        "high_freq_latent_smooth": 5,
        "high_freq_latent_layer": 2,
        "high_freq_latent_weight": 1.5,
        "rms_smooth": 5,
        "bass_smooth": 5,
        "bass_clip": 65,
        "drop_clip": 75,
        "drop_smooth": 5,
        "drop_weight": 1,
        "high_noise_clip": 100,
        "high_noise_weight": 1.5,
        "low_noise_weight": 1,
    }

    with open(metadata_file.replace("metadata", "params"), "w") as outfile:
        json.dump(prms, outfile)

    if not os.path.exists(f"workspace/{file_root}_onsets.npy"):
        print(f"processing audio files...")
        main_audio, sr = rosa.load(args.audio_file)  # , offset=args.offset, duration=args.duration)

        sig = mm.audio.signal.Signal(
            args.audio_file, num_channels=1
        )  # , start=args.offset, stop=args.offset + args.duration)
        sig_frames = mm.audio.signal.FramedSignal(sig, frame_size=2048, hop_size=441)
        stft = mm.audio.stft.ShortTimeFourierTransform(sig_frames, ciruclar_shift=True)
        spec = mm.audio.spectrogram.Spectrogram(stft, ciruclar_shift=True)
        log_filt_spec = mm.audio.spectrogram.FilteredSpectrogram(spec, num_bands=24)

        high_frequency_content = mm.features.onsets.high_frequency_content(log_filt_spec)
        spectral_diff = mm.features.onsets.spectral_diff(log_filt_spec)
        spectral_flux = mm.features.onsets.spectral_flux(log_filt_spec)
        superflux = mm.features.onsets.superflux(log_filt_spec)
        complex_flux = mm.features.onsets.complex_flux(log_filt_spec)
        modified_kullback_leibler = mm.features.onsets.modified_kullback_leibler(log_filt_spec)
        phase_deviation = mm.features.onsets.phase_deviation(log_filt_spec)
        weighted_phase_deviation = mm.features.onsets.weighted_phase_deviation(spec)
        normalized_weighted_phase_deviation = mm.features.onsets.normalized_weighted_phase_deviation(spec)
        complex_domain = mm.features.onsets.complex_domain(spec)
        rectified_complex_domain = mm.features.onsets.rectified_complex_domain(spec)
        onsets = np.sum(
            [
                high_frequency_content,
                spectral_diff,  # ***
                spectral_flux,
                superflux,
                complex_flux,
                modified_kullback_leibler,
                phase_deviation,
                weighted_phase_deviation,
                normalized_weighted_phase_deviation,
                complex_domain,
                rectified_complex_domain,
            ],
            axis=0,
        )

        onsets = np.clip(signal.resample(onsets, num_frames), onsets.min(), onsets.max())

        pitches, magnitudes = rosa.core.piptrack(y=main_audio, sr=sr, hop_length=512, fmin=40, fmax=4000)
        pitches_mean = pitches.mean(0)
        pitches_mean = np.clip(signal.resample(pitches_mean, num_frames), pitches_mean.min(), pitches_mean.max())
        average_pitch = np.average(pitches, axis=0, weights=magnitudes + 1e-8)
        average_pitch = np.clip(signal.resample(average_pitch, num_frames), average_pitch.min(), average_pitch.max())

        high_pitch_cutoff = 40
        high_pitches, high_magnitudes = pitches[high_pitch_cutoff:], magnitudes[high_pitch_cutoff:]
        high_pitches_mean = high_pitches.mean(0)
        high_pitches_mean = np.clip(
            signal.resample(high_pitches_mean, num_frames), high_pitches_mean.min(), high_pitches_mean.max(),
        )
        high_average_pitch = np.average(high_pitches, axis=0, weights=high_magnitudes + 1e-8)
        high_average_pitch = np.clip(
            signal.resample(high_average_pitch, num_frames), high_average_pitch.min(), high_average_pitch.max()
        )

        rms = rosa.feature.rms(S=np.abs(rosa.stft(y=main_audio, hop_length=512)))[0]
        rms = np.clip(signal.resample(rms, num_frames), rms.min(), rms.max())
        rms = normalize(rms)

        bass_audio = signal.sosfilt(signal.butter(12, 100, "lp", fs=sr, output="sos"), main_audio)
        bass_spec = np.abs(rosa.stft(bass_audio))
        bass_sum = bass_spec.sum(0)
        bass_sum = np.clip(signal.resample(bass_sum, num_frames), bass_sum.min(), bass_sum.max())
        bass_sum = normalize(bass_sum)

        np.save(f"workspace/{file_root}_onsets.npy", onsets.astype(np.float32))
        np.save(f"workspace/{file_root}_pitches_mean.npy", pitches_mean.astype(np.float32))
        np.save(f"workspace/{file_root}_average_pitch.npy", average_pitch.astype(np.float32))
        np.save(f"workspace/{file_root}_high_pitches_mean.npy", high_pitches_mean.astype(np.float32))
        np.save(f"workspace/{file_root}_high_average_pitch.npy", high_average_pitch.astype(np.float32))
        np.save(f"workspace/{file_root}_rms.npy", rms.astype(np.float32))
        np.save(f"workspace/{file_root}_bass_sum.npy", bass_sum.astype(np.float32))

    onsets = th.from_numpy(np.load(f"workspace/{file_root}_onsets.npy"))
    pitches_mean = th.from_numpy(np.load(f"workspace/{file_root}_pitches_mean.npy"))
    average_pitch = th.from_numpy(np.load(f"workspace/{file_root}_average_pitch.npy"))
    high_pitches_mean = th.from_numpy(np.load(f"workspace/{file_root}_high_pitches_mean.npy"))
    high_average_pitch = th.from_numpy(np.load(f"workspace/{file_root}_high_average_pitch.npy"))
    rms = th.from_numpy(np.load(f"workspace/{file_root}_rms.npy"))
    bass_sum = th.from_numpy(np.load(f"workspace/{file_root}_bass_sum.npy"))

    def get_latent_loops(base_latent_selection, loop_starting_latents, n_frames, num_loops, smoothing):
        base_latents = []
        for n in range(len(base_latent_selection)):
            for val in np.linspace(0.0, 1.0, int(n_frames // max(1, num_loops) // len(base_latent_selection))):
                base_latents.append(
                    slerp(
                        val,
                        base_latent_selection[(n + loop_starting_latents) % len(base_latent_selection)][0],
                        base_latent_selection[(n + loop_starting_latents + 1) % len(base_latent_selection)][0],
                    )
                )
        base_latents = th.stack(base_latents)
        base_latents = gaussian_filter(base_latents, smoothing * smf)
        base_latents = th.cat([base_latents] * int(n_frames / len(base_latents)), axis=0)
        base_latents = th.cat([base_latents[:, None, :]] * 18, axis=1)
        return base_latents

    intro_selection = th.from_numpy(np.load(intro_file))
    drop_selection = th.from_numpy(np.load(drop_file))

    if args.random_latents:
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
        styles = th.randn((18, 512), device="cuda")
        intro_selection = generator(styles, map_latents=True).cpu()
        styles = th.randn((18, 512), device="cuda")
        drop_selection = generator(styles, map_latents=True).cpu()
        del generator, styles
        gc.collect()
        th.cuda.empty_cache()

    intro_loops = get_latent_loops(
        base_latent_selection=wrapping_slice(intro_selection, 0, prms["intro_loop_len"]) * 2,
        loop_starting_latents=0,
        n_frames=num_frames,
        num_loops=args.bpm / 60.0 * args.duration / prms["intro_num_beats"],
        smoothing=prms["intro_loop_smoothing"] * smf,
    )
    intro_latents = th.cat(
        [
            prms["intro_loop_factor"] * intro_loops + (1 - prms["intro_loop_factor"]) * intro_selection[[0], :],
            prms["intro_loop_factor"] * intro_loops[0 : num_frames - len(intro_loops)]
            + (1 - prms["intro_loop_factor"]) * intro_selection[[0], :],
        ]
    )

    drop_loops = get_latent_loops(
        base_latent_selection=wrapping_slice(drop_selection, 0, prms["drop_loop_len"]),
        loop_starting_latents=0,
        n_frames=num_frames,
        num_loops=args.bpm / 60.0 * args.duration / prms["drop_num_beats"],
        smoothing=prms["drop_loop_smoothing"] * smf,
    )
    drop_latents = (
        prms["drop_loop_factor"] * th.cat([drop_loops, drop_loops[0 : num_frames - len(drop_loops)]])
        + (1 - prms["drop_loop_factor"]) * drop_selection[[0], :]
    )

    rms = gaussian_filter(rms, prms["rms_smooth"] * smf, causal=True)
    main_weight = gaussian_filter(
        normalize(rms) * normalize(onsets), prms["onset_smooth"] * smf * 86 / args.bpm, causal=True
    )
    main_weight = percentile_clip(main_weight, prms["onset_clip"])
    main_weight = main_weight[:, None, None]
    plot_signals([main_weight])

    freqs = (average_pitch + pitches_mean) / prms["freq_mod"]
    freqs = gaussian_filter(freqs, prms["freq_smooth"] * smf, causal=True)

    plot_signals([average_pitch, pitches_mean, freqs])
    freqs = (freqs + 0 + prms["freq_mod_offset"]) % (drop_selection.shape[0] - 1)
    freqs = freqs.int()

    plot_signals([freqs])
    reactive_latents = th.from_numpy(drop_selection.numpy()[freqs, :, :])  # torch indexing doesn't work the same :(
    reactive_latents = gaussian_filter(reactive_latents, prms["freq_latent_smooth"] * smf)

    layr = prms["freq_latent_layer"]
    drop_latents[:, layr:] = (1 - main_weight) * drop_latents[:, layr:] + reactive_latents[:, layr:] * prms[
        "freq_latent_weight"
    ] * main_weight

    high_freqs = (high_average_pitch + high_pitches_mean) / prms["high_freq_mod"]
    high_freqs = gaussian_filter(high_freqs, prms["high_freq_smooth"] * smf, causal=True)
    plot_signals([high_average_pitch, high_pitches_mean, high_freqs])
    high_freqs = (high_freqs + 0 + prms["high_freq_mod_offset"]) % (intro_selection.shape[0] - 1)
    high_freqs = high_freqs.int()

    plot_signals([high_freqs])
    reactive_latents = th.from_numpy(intro_selection.numpy()[high_freqs, :, :])  # torch indexing doesn't work
    reactive_latents = gaussian_filter(reactive_latents, prms["high_freq_latent_smooth"] * smf)

    layr = prms["high_freq_latent_layer"]
    intro_latents[:, layr:] = (1 - main_weight) * intro_latents[:, layr:] + reactive_latents[:, layr:] * prms[
        "high_freq_latent_weight"
    ] * main_weight

    bass_weight = gaussian_filter(bass_sum, prms["bass_smooth"] * smf, causal=True)
    bass_weight = percentile_clip(bass_weight, prms["bass_clip"])

    drop_weight = rms ** 2 + bass_weight
    drop_weight = percentile_clip(drop_weight, prms["drop_clip"])
    drop_weight = gaussian_filter(drop_weight, prms["drop_smooth"] * smf, causal=True)
    drop_weight = normalize(drop_weight) * prms["drop_weight"]
    drop_weight[: int(3473 * (68 / args.duration))] *= 0.4 * drop_weight[: int(3473 * (68 / args.duration))]
    drop_weight = drop_weight[:, None, None]
    plot_signals([drop_weight])

    latents = drop_weight * drop_latents + (1 - drop_weight) * intro_latents

    if args.color_latents is not None:
        color_latents = th.from_numpy(np.load(args.color_latents))
        latents[:, args.color_layer :, :] = (
            latents[:, args.color_layer :, :] * 3 / 5 + color_latents[[0], args.color_layer :, :] * 2 / 5
        )
        latents[:, args.color_layer :, :] = (
            latents[:, args.color_layer :, :] * 3 / 5 + color_latents[[0], args.color_layer :, :] * 2 / 5
        )

    latents = gaussian_filter(latents.float().cuda(), int(round(3 * smf))).cpu()

    high_noise_mod = percentile_clip(main_weight.squeeze() ** 2, prms["high_noise_clip"])
    high_noise_mod *= prms["high_noise_weight"]
    high_noise_mod = high_noise_mod[:, None, None, None].float()
    plot_signals([high_noise_mod])

    low_noise_mod = (1 - drop_weight) * main_weight
    low_noise_mod = normalize(low_noise_mod.squeeze())
    low_noise_mod *= prms["low_noise_weight"]
    low_noise_mod = low_noise_mod[:, None, None, None].float()
    plot_signals([low_noise_mod])

    high_noise_mod = gaussian_filter(high_noise_mod, 3.5 * smf)
    low_noise_mod = gaussian_filter(low_noise_mod, 2.5 * smf)

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
    for s in range(range_min, min(max_noise_scale, range_max)):
        h = 2 ** side_fn(s)
        w = (2 if args.size == 1920 else 1) * 2 ** side_fn(s)
        print(num_frames, 1, h, w)

        noise_noisy = gaussian_filter(th.randn((num_frames, 1, h, w), device="cuda"), int(round(smf))).cpu()
        noise_vox = gaussian_filter(th.randn((num_frames, 1, h, w), device="cuda"), int(round(10 * smf))).cpu()
        noise_smooth = gaussian_filter(th.randn((num_frames, 1, h, w), device="cuda"), int(round(20 * smf))).cpu()

        noise.append(
            high_noise_mod * noise_noisy
            + (1 - high_noise_mod) * noise_smooth
            + low_noise_mod * noise_vox
            + (1 - low_noise_mod) * noise_smooth
        )

        del noise_noisy, noise_vox, noise_smooth
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

    class addNoise(th.nn.Module):
        def __init__(self, noise):
            super(addNoise, self).__init__()
            self.noise = noise

        def forward(self, x):
            return x + self.noise.to(x.device)

    class Print(th.nn.Module):
        def forward(self, x):
            print(x.shape)
            return x

    manipulations = [
        {
            "layer": 0,
            "transform": th.nn.Sequential(
                # Print(),
                th.nn.ReplicationPad2d((2, 2, 0, 0)),
                addNoise(0.25 * th.randn(size=(1, 1, 2 ** log_min_res, 2 * 2 ** log_min_res), device="cuda")),
                # Print(),
            ),
        }
    ]

    # if log_min_res > 2:
    #     reflects = []
    #     for lres in range(2, log_min_res):
    #         half = 2 ** (lres - 1)
    #         reflects.append(th.nn.ReplicationPad2d((half, half, half, half)))
    #     manipulations += [
    #         {
    #             "layer": 0,
    #             "transform": th.nn.Sequential(
    #                 *reflects, addNoise(2 * th.randn(size=(1, 1, 2 ** log_min_res, 2 ** log_min_res), device="cuda")),
    #             ),
    #         }
    #     ]

    tl = 4
    width = lambda s: (2 if args.size == 1920 else 1) * 2 ** int(s)
    intro_start = int(3473 * (22.5 / args.duration))
    intro_end = int(3473 * (45 / args.duration))
    intro_scroll_loop_length = int(12 * args.fps)
    intro_scroll_loop_num = int((intro_end - intro_start) / intro_scroll_loop_length)
    intro_scroll_truncated = (intro_end - intro_start) - intro_scroll_loop_num * intro_scroll_loop_length

    drop_start = int(3473 * (67.5 / args.duration))
    drop_end = int(3473 * (113 / args.duration))
    drop_scroll_loop_length = int(6 * args.fps)
    drop_scroll_loop_num = int((drop_end - drop_start) / drop_scroll_loop_length)
    drop_scroll_truncated = (drop_end - drop_start) - drop_scroll_loop_num * drop_scroll_loop_length
    translation = (
        th.tensor(
            [
                np.concatenate(
                    [
                        np.zeros(intro_start),
                        # intro drums scroll
                        np.concatenate([np.linspace(0, width(tl), intro_scroll_loop_length)] * intro_scroll_loop_num),
                        np.linspace(0, width(tl), intro_scroll_loop_length)[:intro_scroll_truncated],
                        np.linspace(
                            np.linspace(0, width(tl), intro_scroll_loop_length)[intro_scroll_truncated + 1],
                            0,
                            drop_start - intro_end,
                        ),
                        # drop scroll
                        np.concatenate([np.linspace(0, width(tl), drop_scroll_loop_length)] * drop_scroll_loop_num),
                        np.linspace(0, width(tl), drop_scroll_loop_length)[:drop_scroll_truncated],
                        np.ones(num_frames - drop_end)
                        * np.linspace(0, width(tl), drop_scroll_loop_length)[drop_scroll_truncated + 1],
                    ]
                ),
                np.zeros(num_frames),
            ]
        )
        .float()
        .T
    )
    assert latents.shape[0] == translation.shape[0]
    manipulations += [{"layer": tl, "transform": "translateX", "params": translation}]

    # zl = 6
    # print(
    #     th.cat(
    #         [
    #             th.linspace(-1, 3, int(num_frames / 2)),
    #             th.linspace(3, -1, num_frames - int(num_frames / 2)) + 1,
    #         ]
    #     ).shape
    # )
    # zoom = gaussian_filter(
    #     th.cat(
    #         [
    #             th.linspace(0, 3, int(num_frames / 2), dtype=th.float32, device="cuda"),
    #             th.linspace(3, 0, num_frames - int(num_frames / 2), dtype=th.float32, device="cuda") + 1,
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

    # # rl = 6
    # # rotation = th.nn.Sigmoid()(th.tensor(np.linspace(0.0, 1.0, num_frames + 1), device="cuda").float())
    # # rotation -= rotation.min()
    # # rotation /= rotation.max()
    # # rotation = rotation[:-1]
    # # manipulations += [{"layer": rl, "transform": "rotate", "params": (360.0 * rotation).cpu()}]

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


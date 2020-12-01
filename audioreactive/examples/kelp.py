import math
from functools import partial

import numpy as np
import torch as th
import librosa as rosa
import kornia.augmentation as kA
import kornia.geometry.transform as kT

import audioreactive as ar

OVERRIDE = dict(audio_file="audioreactive/examples/Wavefunk - Dwelling in the Kelp.mp3", out_size=1920, duration=120)

bpm = 130


def get_latents(selection, args):
    low_onsets = ar.onsets(
        args.audio, args.sr, args.n_frames, fmax=100, smooth=7, clip=95, power=8, type="rosa"
    ).clone()
    low_onsets = low_onsets[:, None, None]

    high_onsets = ar.onsets(
        args.audio, args.sr, args.n_frames, fmin=200, smooth=7, clip=95, power=8, type="rosa"
    ).clone()
    high_onsets = high_onsets[:, None, None]

    timestamps, labels = ar.laplacian_segmentation(args.audio, args.sr, k=7)

    drop_selection = th.from_numpy(np.load("workspace/cyphept_kelp_drop_latents.npy"))

    rms = ar.rms(args.audio, args.sr, args.n_frames, fmin=20, fmax=200, smooth=180, clip=55, power=4)
    rms = rms[:, None, None]

    color_layer = 9

    latents = []
    for (start, stop), l in zip(zip(timestamps, timestamps[1:]), labels):
        start_frame = int(round(start / args.duration * args.n_frames))
        stop_frame = int(round(stop / args.duration * args.n_frames))
        section_frames = stop_frame - start_frame
        section_bars = (stop - start) * (bpm / 60) / 4

        latent_selection_slice = ar.wrapping_slice(selection, l, 4)
        latent_section = ar.get_spline_loops(latent_selection_slice, section_frames, section_bars / 4)
        latent_section[:, color_layer:] = th.cat([selection[[l], color_layer:]] * section_frames)

        drop_selection_slice = ar.wrapping_slice(drop_selection, l, 4)
        drop_section = ar.get_spline_loops(drop_selection_slice, section_frames, section_bars / 2)
        drop_section[:, color_layer:] = th.cat([drop_selection[[l], color_layer:]] * section_frames)

        latents.append((1 - rms[start_frame:stop_frame]) * latent_section + rms[start_frame:stop_frame] * drop_section)
    len_latents = sum([len(l) for l in latents])
    if len_latents != args.n_frames:
        latents.append(th.cat([latents[-1][[-1]]] * (args.n_frames - len_latents)))
    latents = th.cat(latents).float()
    latents = ar.gaussian_filter(latents, 5)

    latents = high_onsets * selection[[1]] + (1 - high_onsets) * latents
    latents = low_onsets * selection[[2]] + (1 - low_onsets) * latents

    latents = ar.gaussian_filter(latents, 1, causal=0.2)
    return latents


def get_noise(height, width, scale, num_scales, args):
    if width > 256:
        return None
    low_onsets = ar.onsets(
        args.audio, args.sr, args.n_frames, fmax=100, smooth=7, clip=95, power=8, type="rosa"
    ).clone()
    low_onsets = low_onsets[:, None, None, None].cuda()

    high_onsets = ar.onsets(
        args.audio, args.sr, args.n_frames, fmin=200, smooth=7, clip=95, power=8, type="rosa"
    ).clone()
    high_onsets = high_onsets[:, None, None, None].cuda()

    noise_noisy = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 5)

    noise = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 128)
    if width > 8:
        noise = low_onsets * noise_noisy + (1 - low_onsets) * noise
        noise = high_onsets * noise_noisy + (1 - high_onsets) * noise

    noise /= noise.std() * 2.5

    return noise.cpu()


def get_bends(args):
    transform = th.nn.Sequential(
        th.nn.ReplicationPad2d((2, 2, 0, 0)), ar.AddNoise(0.025 * th.randn(size=(1, 1, 4, 8), device="cuda")),
    )
    bends = [{"layer": 0, "transform": transform}]

    return bends

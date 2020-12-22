import librosa as rosa
import torch as th
import numpy as np

import audioreactive as ar


def initialize(args):
    drums, drsr = rosa.load(args.audio_file.replace(".wav", "/drums.wav"))

    args.lo_onsets = ar.onsets(drums, drsr, args.n_frames, fmax=150, clip=96, smooth=3)
    args.hi_onsets = ar.onsets(drums, drsr, args.n_frames, fmin=150, clip=96, smooth=3)

    bass, basr = rosa.load(args.audio_file.replace(".wav", "/bass.wav"))
    args.bass_onsets = ar.rms(bass, basr, args.n_frames, smooth=4, clip=99, power=1)

    # ar.plot_signals([args.bass_onsets, args.hi_onsets])

    return args


def get_latents(selection, args):
    chroma = ar.chroma(args.audio, args.sr, args.n_frames)
    chroma_latents = ar.chroma_weight_latents(chroma, selection[:12])
    latents = ar.gaussian_filter(chroma_latents, 4)

    lo_onsets = args.lo_onsets[:, None, None]
    hi_onsets = args.hi_onsets[:, None, None]
    bass_onsets = args.bass_onsets[:, None, None]

    latents = hi_onsets * selection[[-4]] + (1 - hi_onsets) * latents
    latents = lo_onsets * selection[[-7]] + (1 - lo_onsets) * latents
    latents = bass_onsets * selection[[-5]] + (1 - bass_onsets) * latents

    latents = ar.gaussian_filter(latents, 4, causal=0.2)

    return latents


def get_noise(height, width, scale, num_scales, args):
    bass_onsets = args.bass_onsets[:, None, None, None]
    hi_onsets = args.hi_onsets[:, None, None, None]

    noise_noisy = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 7).cpu()
    noise_noiser = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 2).cpu()
    noise = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 64).cpu()

    noise = bass_onsets * noise_noisy + (1 - bass_onsets) * noise
    if width > 8:
        noise = hi_onsets * noise_noiser + (1 - hi_onsets) * noise

    noise /= noise.std() * (0.85 + np.random.rand())

    return noise

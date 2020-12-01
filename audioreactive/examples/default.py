import torch as th

import audioreactive as ar


def initialize(args):
    args.lo_onsets = ar.onsets(args.audio, args.sr, args.n_frames, fmax=150, smooth=5, clip=97, power=2)
    args.hi_onsets = ar.onsets(args.audio, args.sr, args.n_frames, fmin=500, smooth=5, clip=99, power=2)
    return args


def get_latents(selection, args):
    chroma = ar.chroma(args.audio, args.sr, args.n_frames)
    chroma_latents = ar.chroma_weight_latents(chroma, selection)
    latents = ar.gaussian_filter(chroma_latents, 4)

    lo_onsets = args.lo_onsets[:, None, None]
    hi_onsets = args.hi_onsets[:, None, None]

    latents = hi_onsets * selection[[-4]] + (1 - hi_onsets) * latents
    latents = lo_onsets * selection[[-7]] + (1 - lo_onsets) * latents

    latents = ar.gaussian_filter(latents, 2, causal=0.2)

    return latents


def get_noise(height, width, scale, num_scales, args):
    if width > 256:
        return None

    lo_onsets = args.lo_onsets[:, None, None, None].cuda()
    hi_onsets = args.hi_onsets[:, None, None, None].cuda()

    noise_noisy = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 5)
    noise = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 128)

    if width < 128:
        noise = lo_onsets * noise_noisy + (1 - lo_onsets) * noise
    if width > 32:
        noise = hi_onsets * noise_noisy + (1 - hi_onsets) * noise

    noise /= noise.std() * 2.5

    return noise.cpu()

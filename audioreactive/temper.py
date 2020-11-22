import torch as th

import audioreactive as ar


def get_latents(audio, sr, num_frames, selection):
    chroma = ar.get_chroma(audio, sr, num_frames)
    chroma_latents = ar.get_chroma_latents(chroma, selection)
    latents = ar.gaussian_filter(chroma_latents, 4)

    lo_onsets = ar.get_onsets(audio, sr, num_frames, fmax=150, smooth=5, clip=97, power=2)
    hi_onsets = ar.get_onsets(audio, sr, num_frames, fmin=500, smooth=5, clip=99, power=2)

    lo_onsets = lo_onsets[:, None, None]
    hi_onsets = hi_onsets[:, None, None]

    latents = hi_onsets * selection[[-4]] + (1 - hi_onsets) * latents
    latents = lo_onsets * selection[[-7]] + (1 - lo_onsets) * latents

    latents = ar.gaussian_filter(latents, 2, causal=0.2)

    return latents


def get_noise(audio, sr, num_frames, num_scales, height, width, scale):
    if height > 256:
        return None

    lo_onsets = ar.get_onsets(audio, sr, num_frames, fmax=150, smooth=5, clip=97, power=2)
    hi_onsets = ar.get_onsets(audio, sr, num_frames, fmin=500, smooth=5, clip=99, power=2)

    lo_onsets = lo_onsets[:, None, None, None].cuda()
    hi_onsets = hi_onsets[:, None, None, None].cuda()

    mask = ar.create_circular_mask(height, width, radius=int(width / 2), soft=2)[None, None, ...].float().cuda()

    noise_noisy = ar.gaussian_filter(th.randn((num_frames, 1, height, width), device="cuda"), 5)

    noise = ar.gaussian_filter(th.randn((num_frames, 1, height, width), device="cuda"), 128)

    if height < 128:
        noise = 2 * mask * lo_onsets * noise_noisy + (1 - mask) * (1 - lo_onsets) * noise
    if height > 32:
        noise = 0.75 * (1 - mask) * hi_onsets * noise_noisy + mask * (1 - 0.75 * hi_onsets) * noise

    noise /= noise.std() * 2

    return noise.cpu()


def get_bends(audio, num_frames):
    return []


def get_rewrites(audio, num_frames):
    return {}

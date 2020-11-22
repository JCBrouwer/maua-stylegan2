from functools import partial

import numpy as np
import torch as th
import kornia.augmentation as kA
import kornia.geometry.transform as kT

import audioreactive as ar


def get_latents(audio, sr, num_frames, selection):
    chroma = ar.get_chroma(audio, sr, num_frames)
    chroma_latents = ar.get_chroma_latents(chroma, selection)
    latents = ar.gaussian_filter(chroma_latents, 5)

    lo_onsets = ar.get_onsets(audio, sr, num_frames, fmax=150, smooth=5, clip=97, power=2)
    hi_onsets = ar.get_onsets(audio, sr, num_frames, fmin=500, smooth=5, clip=99, power=2)

    lo_onsets = lo_onsets[:, None, None]
    hi_onsets = hi_onsets[:, None, None]

    latents = hi_onsets * selection[[-4]] + (1 - hi_onsets) * latents
    latents = lo_onsets * selection[[-7]] + (1 - lo_onsets) * latents

    latents = ar.gaussian_filter(latents, 3, causal=0.3)

    return latents


def get_noise(audio, sr, num_frames, num_scales, height, width, scale):
    if width > 256:
        return None

    lo_onsets = ar.get_onsets(audio, sr, num_frames, fmax=150, smooth=5, clip=97, power=2)
    hi_onsets = ar.get_onsets(audio, sr, num_frames, fmin=500, smooth=5, clip=99, power=2)

    lo_onsets = lo_onsets[:, None, None, None].cuda()
    hi_onsets = hi_onsets[:, None, None, None].cuda()

    noise_noisy = ar.gaussian_filter(th.randn((num_frames, 1, height, width), device="cuda"), 5)

    noise = ar.gaussian_filter(th.randn((num_frames, 1, height, width), device="cuda"), 128)
    if width > 8:
        noise = lo_onsets * noise_noisy + (1 - lo_onsets) * noise
        noise = hi_onsets * noise_noisy + (1 - hi_onsets) * noise

    noise /= noise.std() * 3

    return noise.cpu()


def get_bends(audio, num_frames, duration, fps):
    transform = th.nn.Sequential(
        th.nn.ReplicationPad2d((2, 2, 0, 0)), ar.AddNoise(0.025 * th.randn(size=(1, 1, 4, 8), device="cuda")),
    )
    bends = [{"layer": 0, "transform": transform}]

    tl = 4
    width = lambda s: 2 * 2 ** int(s)
    smooth_noise = 0.1 * th.randn(size=(1, 1, 2 ** int(tl), 5 * width(tl)), device="cuda")

    class Translate(ar.NetworkBend):
        def __init__(self, layer, batch):
            layer_h = 2 ** int(layer)
            layer_w = 2 * 2 ** int(layer)
            sequential_fn = lambda b: th.nn.Sequential(
                # ar.Print(),
                th.nn.ReflectionPad2d((int(layer_w / 2), int(layer_w / 2), 0, 0)),
                th.nn.ReflectionPad2d((layer_w, layer_w, 0, 0)),
                th.nn.ReflectionPad2d((layer_w, 0, 0, 0)),
                ar.AddNoise(smooth_noise),
                # ar.Print(),
                kT.Translate(b),
                kA.CenterCrop((layer_h, layer_w)),
                # ar.Print(),
            )
            super(Translate, self).__init__(sequential_fn, batch)

    drop_start = int(5591 * (45 / duration))
    drop_end = int(5591 * (135 / duration))
    scroll_loop_length = int(6 * fps)
    scroll_loop_num = int((drop_end - drop_start) / scroll_loop_length)
    scroll_trunc = (drop_end - drop_start) - scroll_loop_num * scroll_loop_length
    translation = (
        th.tensor(
            [
                np.concatenate(
                    [
                        np.zeros(drop_start),
                        np.concatenate([np.linspace(0, width(tl), scroll_loop_length)] * scroll_loop_num),
                        np.linspace(0, width(tl), scroll_loop_length)[:scroll_trunc],
                        np.ones(num_frames - drop_end) * np.linspace(0, width(tl), scroll_loop_length)[scroll_trunc + 1]
                        if num_frames - drop_end > 0
                        else [1],
                    ]
                ),
                np.zeros(num_frames),
            ]
        )
        .float()
        .T
    )[:num_frames]
    translation.T[0, drop_start - fps : drop_start + fps] = ar.gaussian_filter(
        translation.T[0, drop_start - 5 * fps : drop_start + 5 * fps], 5
    )[4 * fps : -4 * fps]

    transform = lambda batch: partial(Translate, tl)(batch)
    bends += [{"layer": tl, "transform": transform, "modulation": translation}]

    return bends


def get_rewrites(audio, num_frames):
    return {}

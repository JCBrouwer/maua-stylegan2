from functools import partial

import kornia.augmentation as kA
import kornia.geometry.transform as kT
import numpy as np
import torch as th

import audioreactive as ar

OVERRIDE = dict(audio_file="audioreactive/examples/Wavefunk - Tau Ceti Alpha.mp3", out_size=1920, dataparallel=False)


def initialize(args):
    args.low_onsets = ar.onsets(args.audio, args.sr, args.n_frames, fmax=150, smooth=5, clip=97, power=2)
    args.high_onsets = ar.onsets(args.audio, args.sr, args.n_frames, fmin=500, smooth=5, clip=99, power=2)
    return args


def get_latents(selection, args):
    chroma = ar.chroma(args.audio, args.sr, args.n_frames)
    chroma_latents = ar.get_chroma_latents(chroma, selection[:12])  # shape [n_frames, 18, 512]
    latents = ar.gaussian_filter(chroma_latents, 5)

    lo_onsets = args.low_onsets[:, None, None]  # expand to same shape as latents [n_frames, 1, 1]
    hi_onsets = args.high_onsets[:, None, None]

    latents = hi_onsets * selection[[-4]] + (1 - hi_onsets) * latents
    latents = lo_onsets * selection[[-7]] + (1 - lo_onsets) * latents

    latents = ar.gaussian_filter(latents, 5, causal=0)

    drop_start = int(5591 * (45 / args.duration))
    drop_end = int(5591 * (135 / args.duration))

    color_layer = 9
    color_latent_selection = th.from_numpy(np.load("workspace/cyphept-multicolor-latents.npy"))

    color_latents = [latents[:drop_start, color_layer:]]

    drop_length = drop_end - drop_start
    section_length = int(drop_length / 4)
    for i, section_start in enumerate(range(0, drop_length, section_length)):
        if i > 3:
            break
        color_latents.append(th.cat([color_latent_selection[[i], color_layer:]] * section_length))

    if drop_length - 4 * section_length != 0:
        color_latents.append(th.cat([color_latent_selection[[i], color_layer:]] * (drop_length - 4 * section_length)))
    color_latents.append(latents[drop_end:, color_layer:])
    color_latents = th.cat(color_latents, axis=0)

    color_latents = ar.gaussian_filter(color_latents, 5)

    latents[:, color_layer:] = color_latents

    return latents


def get_noise(height, width, scale, num_scales, args):
    if width > 256:
        return None

    lo_onsets = 1.25 * args.low_onsets[:, None, None, None].cuda()
    hi_onsets = 1.25 * args.high_onsets[:, None, None, None].cuda()

    noise_noisy = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 5)

    noise = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 128)
    if width > 8:
        noise = lo_onsets * noise_noisy + (1 - lo_onsets) * noise
        noise = hi_onsets * noise_noisy + (1 - hi_onsets) * noise

    noise /= noise.std() * 2.5

    return noise.cpu()


def get_bends(args):
    transform = th.nn.Sequential(
        th.nn.ReplicationPad2d((2, 2, 0, 0)), ar.AddNoise(0.025 * th.randn(size=(1, 1, 4, 8), device="cuda")),
    )
    bends = [{"layer": 0, "transform": transform}]

    drop_start = int(5591 * (45 / args.duration))
    drop_end = int(5591 * (135 / args.duration))

    scroll_loop_length = int(6 * args.fps)
    scroll_loop_num = int((drop_end - drop_start) / scroll_loop_length)
    scroll_trunc = (drop_end - drop_start) - scroll_loop_num * scroll_loop_length

    intro_tl8 = np.zeros(drop_start)
    loops_tl8 = np.concatenate([np.linspace(0, width, scroll_loop_length)] * scroll_loop_num)
    last_loop_tl8 = np.linspace(0, width, scroll_loop_length)[:scroll_trunc]
    outro_tl8 = np.ones(args.n_frames - drop_end) * np.linspace(0, width, scroll_loop_length)[scroll_trunc + 1]
    x_tl8 = np.concatenate([intro_tl8, loops_tl8, last_loop_tl8, outro_tl8])
    y_tl8 = np.zeros(args.n_frames)
    translation = (th.tensor([x_tl8, y_tl8]).float().T)[: args.n_frames]

    translation.T[0, drop_start - args.fps : drop_start + args.fps] = ar.gaussian_filter(
        translation.T[0, drop_start - 5 * args.fps : drop_start + 5 * args.fps], 5
    )[4 * args.fps : -4 * args.fps]

    tl = 4
    transform = lambda batch: partial(ar.Translate, h=2 ** tl, w=2 * 2 ** tl)(batch)
    bends += [{"layer": tl, "transform": transform, "modulation": translation}]

    return bends

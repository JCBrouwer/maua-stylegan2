"""
This file shows an example of network bending
The latents and noise are similar to temper.py (although without spatial noise controls)
The latents cycle through different colors for different sections of the drop
During the drop, a translation is applied which makes the video seem to scroll endlessly
"""

from functools import partial

import numpy as np
import torch as th

import audioreactive as ar

OVERRIDE = dict(
    audio_file="audioreactive/examples/Wavefunk - Tau Ceti Alpha.mp3",
    out_size=1920,  # get bends assumes 1920x1080 output size
    dataparallel=False,  # makes use of a kornia transform during network bending => not compatible with dataparallel
    fps=30,  # 5591 magic number below is based on number of frames in output video with fps of 30
)


def initialize(args):
    args.low_onsets = ar.onsets(args.audio, args.sr, args.n_frames, fmax=150, smooth=5, clip=97, power=2)
    args.high_onsets = ar.onsets(args.audio, args.sr, args.n_frames, fmin=500, smooth=5, clip=99, power=2)
    return args


def get_latents(selection, args):
    chroma = ar.chroma(args.audio, args.sr, args.n_frames)
    chroma_latents = ar.chroma_weight_latents(chroma, selection[:12])  # shape [n_frames, 18, 512]
    latents = ar.gaussian_filter(chroma_latents, 5)

    lo_onsets = args.low_onsets[:, None, None]  # expand to same shape as latents [n_frames, 1, 1]
    hi_onsets = args.high_onsets[:, None, None]

    latents = hi_onsets * selection[[-4]] + (1 - hi_onsets) * latents
    latents = lo_onsets * selection[[-7]] + (1 - lo_onsets) * latents

    latents = ar.gaussian_filter(latents, 5, causal=0)

    # cheating a little, you could probably do this with laplacian segmentation, but is it worth the effort?
    drop_start = int(5591 * (45 / args.duration))
    drop_end = int(5591 * (135 / args.duration))

    # selection of latents with different colors (chosen with select_latents.py)
    color_latent_selection = th.from_numpy(np.load("workspace/cyphept-multicolor-latents.npy"))

    # build sequence of latents for just the upper layers
    color_layer = 9
    color_latents = [latents[:drop_start, color_layer:]]

    # for 4 different sections in the drop, use a different color latent
    drop_length = drop_end - drop_start
    section_length = int(drop_length / 4)
    for i, section_start in enumerate(range(0, drop_length, section_length)):
        if i > 3:
            break
        color_latents.append(th.cat([color_latent_selection[[i], color_layer:]] * section_length))

    # ensure color sequence is correct length and concatenate
    if drop_length - 4 * section_length != 0:
        color_latents.append(th.cat([color_latent_selection[[i], color_layer:]] * (drop_length - 4 * section_length)))
    color_latents.append(latents[drop_end:, color_layer:])
    color_latents = th.cat(color_latents, axis=0)

    color_latents = ar.gaussian_filter(color_latents, 5)

    # set upper layers of latent sequence to the colored sequence
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
    # repeat the intermediate features outwards on both sides (2:1 aspect ratio)
    # + add some noise to give the whole thing a little variation (disguises the repetition)
    transform = th.nn.Sequential(
        th.nn.ReplicationPad2d((2, 2, 0, 0)), ar.AddNoise(0.025 * th.randn(size=(1, 1, 4, 8), device="cuda")),
    )
    bends = [{"layer": 0, "transform": transform}]

    # during the drop, create scrolling effect
    drop_start = int(5591 * (45 / args.duration))
    drop_end = int(5591 * (135 / args.duration))

    # calculate length of loops, number of loops, and remainder at end of drop
    scroll_loop_length = int(6 * args.fps)
    scroll_loop_num = int((drop_end - drop_start) / scroll_loop_length)
    scroll_trunc = (drop_end - drop_start) - scroll_loop_num * scroll_loop_length

    # apply network bending to 4th layer in StyleGAN
    # lower layer network bends have more fluid outcomes
    tl = 4
    h = 2 ** tl
    w = 2 * h

    # create values between 0 and 1 corresponding to fraction of scroll from left to right completed
    # all 0s during intro
    intro_tl8 = np.zeros(drop_start)
    # repeating linear interpolation from 0 to 1 during drop
    loops_tl8 = np.concatenate([np.linspace(0, w, scroll_loop_length)] * scroll_loop_num)
    # truncated interp
    last_loop_tl8 = np.linspace(0, w, scroll_loop_length)[:scroll_trunc]
    # static at final truncated value during outro
    outro_tl8 = np.ones(args.n_frames - drop_end) * np.linspace(0, w, scroll_loop_length)[scroll_trunc + 1]

    # create 2D array of translations in x and y directions
    x_tl8 = np.concatenate([intro_tl8, loops_tl8, last_loop_tl8, outro_tl8])
    y_tl8 = np.zeros(args.n_frames)
    translation = (th.tensor([x_tl8, y_tl8]).float().T)[: args.n_frames]

    # smooth the transition from intro to drop to prevent jerk
    translation.T[0, drop_start - args.fps : drop_start + args.fps] = ar.gaussian_filter(
        translation.T[0, drop_start - 5 * args.fps : drop_start + 5 * args.fps], 5
    )[4 * args.fps : -4 * args.fps]

    class Translate(NetworkBend):
        """From audioreactive/examples/bend.py"""

        def __init__(self, modulation, h, w, noise):
            sequential_fn = lambda b: th.nn.Sequential(
                th.nn.ReflectionPad2d((int(w / 2), int(w / 2), 0, 0)),  #  < Reflect out to 5x width (so that after
                th.nn.ReflectionPad2d((w, w, 0, 0)),  #                    < translating w pixels, center crop gives
                th.nn.ReflectionPad2d((w, 0, 0, 0)),  #                    < same features as translating 0 pixels)
                AddNoise(noise),  # add some noise to disguise reflections
                kT.Translate(b),
                kA.CenterCrop((h, w)),
            )
            super(Translate, self).__init__(sequential_fn, modulation)

    # create static noise for translate bend
    noise = 0.2 * th.randn((1, 1, h, 5 * w), device="cuda")
    # create function which returns an initialized Translate object when fed a batch of modulation
    # this is so that creation of the object is delayed until the specific batch is sent into the generator
    # (there's probably an easier way to do this without the kornia transforms, e.g. using Broad et al.'s transform implementations)
    transform = lambda batch: partial(Translate, h=h, w=w, noise=noise)(batch)
    bends += [{"layer": tl, "transform": transform, "modulation": translation}]  # add network bend to list dict

    return bends

"""
This file shows an example of a loop based interpolation
Here sections are identified with laplacian segmentation and looping latents are generated for each section
The noise is looping perlin noise
Long term section analysis is done with the RMS to interpolate between latent sequences for the intro/outro and drop
"""


import librosa as rosa
import torch as th

import audioreactive as ar

OVERRIDE = dict(audio_file="audioreactive/examples/Wavefunk - Dwelling in the Kelp.mp3", out_size=1920)
BPM = 130


def initialize(args):
    # RMS can be used to distinguish between the drop sections and intro/outros
    rms = ar.rms(args.audio, args.sr, args.n_frames, smooth=10, clip=60, power=1)
    rms = ar.expand(rms, threshold=0.8, ratio=10)
    rms = ar.gaussian_filter(rms, 4)
    rms = ar.normalize(rms)
    args.rms = rms

    # cheating a little here, this my song so I have the multitracks
    # this is much easier than fiddling with onsets until you have envelopes that dance nicely to the drums
    audio, sr = rosa.load("workspace/kelpkick.wav", offset=args.offset, duration=args.duration)
    args.kick_onsets = ar.onsets(audio, sr, args.n_frames, margin=1, smooth=4)
    audio, sr = rosa.load("workspace/kelpsnare.wav", offset=args.offset, duration=args.duration)
    args.snare_onsets = ar.onsets(audio, sr, args.n_frames, margin=1, smooth=4)

    ar.plot_signals([args.rms, args.kick_onsets, args.snare_onsets])

    return args


def get_latents(selection, args):
    # expand envelopes to latent shape
    rms = args.rms[:, None, None]
    low_onsets = args.kick_onsets[:, None, None]
    high_onsets = args.snare_onsets[:, None, None]

    # get timestamps and labels with laplacian segmentation
    # k is the number of labels the algorithm may use
    # try multiple values with plot=True to see which value correlates best with the sections of the song
    timestamps, labels = ar.laplacian_segmentation(args.audio, args.sr, k=7)

    # a second set of latents for the drop section, the 'selection' variable is the other set for the intro
    drop_selection = ar.load_latents("workspace/cyphept_kelp_drop_latents.npy")
    color_layer = 9

    latents = []
    for (start, stop), l in zip(zip(timestamps, timestamps[1:]), labels):
        start_frame = int(round(start / args.duration * args.n_frames))
        stop_frame = int(round(stop / args.duration * args.n_frames))
        section_frames = stop_frame - start_frame
        section_bars = (stop - start) * (BPM / 60) / 4

        # get portion of latent selection (wrapping around to start)
        latent_selection_slice = ar.wrapping_slice(selection, l, 4)
        # spline interpolation loops through selection slice
        latent_section = ar.spline_loops(latent_selection_slice, n_frames=section_frames, n_loops=section_bars / 4)
        # set the color with laplacian segmentation label, (1 latent repeated for entire section in upper layers)
        latent_section[:, color_layer:] = th.cat([selection[[l], color_layer:]] * section_frames)

        # same as above but for the drop latents (with faster loops)
        drop_selection_slice = ar.wrapping_slice(drop_selection, l, 4)
        drop_section = ar.spline_loops(drop_selection_slice, n_frames=section_frames, n_loops=section_bars / 2)
        drop_section[:, color_layer:] = th.cat([drop_selection[[l], color_layer:]] * section_frames)

        # merged based on RMS (drop section or not)
        latents.append((1 - rms[start_frame:stop_frame]) * latent_section + rms[start_frame:stop_frame] * drop_section)

    # concatenate latents to correct length & smooth over the junctions
    len_latents = sum([len(l) for l in latents])
    if len_latents != args.n_frames:
        latents.append(th.cat([latents[-1][[-1]]] * (args.n_frames - len_latents)))
    latents = th.cat(latents).float()
    latents = ar.gaussian_filter(latents, 3)

    # use onsets to modulate towards latents
    latents = 0.666 * low_onsets * selection[[2]] + (1 - 0.666 * low_onsets) * latents
    latents = 0.666 * high_onsets * selection[[1]] + (1 - 0.666 * high_onsets) * latents

    latents = ar.gaussian_filter(latents, 1, causal=0.2)
    return latents


def get_noise(height, width, scale, num_scales, args):
    if width > 512:  # larger sizes don't fit in VRAM, just use default or randomize
        return

    num_bars = int(round(args.duration * (BPM / 60) / 4))
    frames_per_loop = int(args.n_frames / num_bars * 2)  # loop every 2 bars

    def perlin_pls(resolution):
        perlin = ar.perlin_noise(shape=(frames_per_loop, height, width), res=resolution)[:, None, ...].cpu()
        perlin = th.cat([perlin] * int(num_bars / 2))  # concatenate multiple copies for looping
        if args.n_frames - len(perlin) > 0:
            perlin = th.cat([perlin, th.cat([perlin[[-1]]] * (args.n_frames - len(perlin)))])  # fix up rounding errors
        return perlin

    smooth = perlin_pls(resolution=(1, 1, 1))  # (time res, x res, y res)
    noise = perlin_pls(resolution=(8, 4, 4))  # higher resolution => higher frequency noise => more movement in video

    rms = args.rms[:, None, None, None]
    noise = rms * noise + (1 - rms) * smooth  # blend between noises based on drop (high rms) or not

    return noise


def get_bends(args):
    # repeat the intermediate features outwards on both sides (2:1 aspect ratio)
    # + add some noise to give the whole thing a little variation (disguises the repetition)
    transform = th.nn.Sequential(
        th.nn.ReplicationPad2d((2, 2, 0, 0)), ar.AddNoise(0.025 * th.randn(size=(1, 1, 4, 8), device="cuda")),
    )
    bends = [{"layer": 0, "transform": transform}]

    return bends

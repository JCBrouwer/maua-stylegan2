from functools import partial

import numpy as np
import torch as th
import librosa as rosa
import kornia.augmentation as kA
import kornia.geometry.transform as kT

import audioreactive as ar

OVERRIDE = dict(audio_file="audioreactive/wavefunk - tau ceti alpha.flac", size=1920, dataparallel=False)

duration = rosa.get_duration(filename=OVERRIDE["audio_file"])
drop_start = int(5591 * (45 / duration))
drop_end = int(5591 * (135 / duration))


def get_latents(audio, sr, num_frames, selection):
    chroma = ar.get_chroma(audio, sr, num_frames)
    chroma_latents = ar.get_chroma_latents(chroma, selection[:12])
    latents = ar.gaussian_filter(chroma_latents, 5)

    low_onsets = ar.get_onsets(audio, sr, num_frames, fmax=150, smooth=5, clip=97, power=2)
    high_onsets = ar.get_onsets(audio, sr, num_frames, fmin=500, smooth=5, clip=99, power=2)
    lo_onsets = low_onsets[:, None, None]
    hi_onsets = high_onsets[:, None, None]

    latents = hi_onsets * selection[[-4]] + (1 - hi_onsets) * latents
    latents = lo_onsets * selection[[-7]] + (1 - lo_onsets) * latents

    latents = ar.gaussian_filter(latents, 3, causal=0.3)

    color_layer = 8
    color_latent_selection = th.from_numpy(np.load("workspace/cyphept-multicolor-latents.npy"))

    color_latents = [latents[:drop_start, color_layer:]]

    drop_length = drop_end - drop_start
    section_length = int(drop_length / 4)
    for i, section_start in enumerate(range(0, drop_length, section_length)):
        color_latents.append(th.cat([color_latent_selection[[i], color_layer:]] * section_length))

    if drop_length - 4 * section_length != 0:
        color_latents.append(th.cat([color_latent_selection[[i], color_layer:]] * (drop_length - 4 * section_length)))
    color_latents.append(latents[drop_end:, color_layer:])
    color_latents = th.cat(color_latents, axis=0)

    color_latents = ar.gaussian_filter(color_latents, 5)

    latents[:, color_layer:] = color_latents

    return latents


def get_noise(audio, sr, num_frames, num_scales, height, width, scale):
    if width > 256:
        return None

    low_onsets = ar.get_onsets(audio, sr, num_frames, fmax=150, smooth=5, clip=97, power=2)
    high_onsets = ar.get_onsets(audio, sr, num_frames, fmin=500, smooth=5, clip=99, power=2)
    lo_onsets = low_onsets[:, None, None, None].cuda()
    hi_onsets = high_onsets[:, None, None, None].cuda()

    noise_noisy = ar.gaussian_filter(th.randn((num_frames, 1, height, width), device="cuda"), 5)

    noise = ar.gaussian_filter(th.randn((num_frames, 1, height, width), device="cuda"), 128)
    if width > 8:
        noise = lo_onsets * noise_noisy + (1 - lo_onsets) * noise
        noise = hi_onsets * noise_noisy + (1 - hi_onsets) * noise

    noise /= noise.std() * 2.5

    # drop_length = drop_end - drop_start
    # section_length = int(drop_length / 4)
    # for i, section_start in enumerate(range(0, drop_length, section_length)):
    #     section_frame = drop_start + section_start

    #     x = th.linspace(0, 20, width, device="cuda")[None, None, None, :]
    #     y = th.linspace(0, 100, int(height / 2), device="cuda")[None, None, :, None]
    #     t = th.linspace(0, 300, section_length, device="cuda")[:, None, None, None]
    #     upward_streaks = th.cos(np.pi * x * y / 800) ** 2 * th.sin(3 * np.pi * x / 20 + (y + t) / 64) ** 40
    #     upward_streaks[: int(section_length / 2)] *= 0
    #     upward_streaks = ar.gaussian_filter(upward_streaks, 10)
    #     upward_streaks = 5 * ar.normalize(upward_streaks)

    #     noise[section_frame : section_frame + section_length, :, : int(height / 2)] += upward_streaks

    # if width == 256:
    #     import render

    #     output = noise.permute(0, 2, 3, 1)
    #     output = ar.normalize(output) * 255
    #     output = output.cpu().numpy()
    #     output = np.concatenate([output, output, output], axis=3)
    #     render.write_video(output, "/home/hans/neurout/tauceti-noise.mp4", 30)

    return noise.cpu()


def get_bends(audio, num_frames, duration, fps):
    transform = th.nn.Sequential(
        th.nn.ReplicationPad2d((2, 2, 0, 0)), ar.AddNoise(0.025 * th.randn(size=(1, 1, 4, 8), device="cuda")),
    )
    bends = [{"layer": 0, "transform": transform}]

    tl = 4
    width = 2 * 2 ** tl
    smooth_noise = 0.2 * th.randn(size=(1, 1, 2 ** int(tl), 5 * width), device="cuda")

    class Translate(ar.NetworkBend):
        def __init__(self, layer, batch):
            layer_h = 2 ** int(layer)
            layer_w = 2 * 2 ** int(layer)
            sequential_fn = lambda b: th.nn.Sequential(
                th.nn.ReflectionPad2d((int(layer_w / 2), int(layer_w / 2), 0, 0)),
                th.nn.ReflectionPad2d((layer_w, layer_w, 0, 0)),
                th.nn.ReflectionPad2d((layer_w, 0, 0, 0)),
                ar.AddNoise(smooth_noise),
                kT.Translate(b),
                kA.CenterCrop((layer_h, layer_w)),
            )
            super(Translate, self).__init__(sequential_fn, batch)

    scroll_loop_length = int(6 * fps)
    scroll_loop_num = int((drop_end - drop_start) / scroll_loop_length)
    scroll_trunc = (drop_end - drop_start) - scroll_loop_num * scroll_loop_length

    intro_tl8 = np.zeros(drop_start)
    loops_tl8 = np.concatenate([np.linspace(0, width, scroll_loop_length)] * scroll_loop_num)
    last_loop_tl8 = np.linspace(0, width, scroll_loop_length)[:scroll_trunc]
    outro_tl8 = np.ones(num_frames - drop_end) * np.linspace(0, width, scroll_loop_length)[scroll_trunc + 1]
    x_tl8 = np.concatenate([intro_tl8, loops_tl8, last_loop_tl8, outro_tl8])
    y_tl8 = np.zeros(num_frames)
    translation = (th.tensor([x_tl8, y_tl8]).float().T)[:num_frames]

    translation.T[0, drop_start - fps : drop_start + fps] = ar.gaussian_filter(
        translation.T[0, drop_start - 5 * fps : drop_start + 5 * fps], 5
    )[4 * fps : -4 * fps]

    transform = lambda batch: partial(Translate, tl)(batch)
    bends += [{"layer": tl, "transform": transform, "modulation": translation}]

    return bends

"""
This file shows an example of spatial control of the noise using a simple circular mask
The latents are a chromagram weighted sequence, modulated by drum onsets
"""

import scipy.ndimage.filters as ndi
import torch as th

import audioreactive as ar

OVERRIDE = dict(audio_file="audioreactive/examples/Wavefunk - Temper.mp3", out_size=1024)


def initialize(args):
    # these onsets can definitely use some tweaking, the drum reactivity isn't great for this one
    # the main bass makes it hard to identify both the kick and the snare because it is so loud and covers the whole spectrum
    args.lo_onsets = ar.onsets(args.audio, args.sr, args.n_frames, fmax=150, smooth=5, clip=97, power=2)
    args.hi_onsets = ar.onsets(args.audio, args.sr, args.n_frames, fmin=500, smooth=5, clip=99, power=2)
    return args


def get_latents(selection, args):
    # create chromagram weighted sequence
    chroma = ar.chroma(args.audio, args.sr, args.n_frames)
    chroma_latents = ar.chroma_weight_latents(chroma, selection)
    latents = ar.gaussian_filter(chroma_latents, 4)

    # expand onsets to latent shape
    lo_onsets = args.lo_onsets[:, None, None]
    hi_onsets = args.hi_onsets[:, None, None]

    # modulate latents to specific latent vectors
    latents = hi_onsets * selection[[-4]] + (1 - hi_onsets) * latents
    latents = lo_onsets * selection[[-7]] + (1 - lo_onsets) * latents

    latents = ar.gaussian_filter(latents, 2, causal=0.2)

    return latents


def circular_mask(h, w, center=None, radius=None, soft=0):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius

    if soft > 0:
        mask = ndi.gaussian_filter(mask, sigma=int(round(soft)))  # blur mask for smoother transition

    return th.from_numpy(mask)


def get_noise(height, width, scale, num_scales, args):
    if width > 256:  # larger sizes don't fit in VRAM, just use default or randomize
        return None

    # expand onsets to noise shape
    # send to GPU as gaussian_filter on large noise tensors with high standard deviation is slow
    lo_onsets = args.lo_onsets[:, None, None, None].cuda()
    hi_onsets = args.hi_onsets[:, None, None, None].cuda()

    # 1s inside circle of radius, 0s outside
    mask = circular_mask(height, width, radius=int(width / 2), soft=2)[None, None, ...].float().cuda()

    # create noise which changes quickly (small standard deviation smoothing)
    noise_noisy = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 5)

    # create noise which changes slowly (large standard deviation smoothing)
    noise = ar.gaussian_filter(th.randn((args.n_frames, 1, height, width), device="cuda"), 128)

    # for lower layers, noise inside circle are affected by low onsets
    if width < 128:
        noise = 2 * mask * lo_onsets * noise_noisy + (1 - mask) * (1 - lo_onsets) * noise
    # for upper layers, noise outside circle are affected by high onsets
    if width > 32:
        noise = 0.75 * (1 - mask) * hi_onsets * noise_noisy + mask * (1 - 0.75 * hi_onsets) * noise

    # ensure amplitude of noise is close to standard normal distribution (dividing by std. dev. gets it exactly there)
    noise /= noise.std() * 2

    return noise.cpu()

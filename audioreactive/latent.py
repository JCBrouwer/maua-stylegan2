import argparse
import gc
import json
import math
import os
import random
import time
import uuid
import warnings

import cachetools
import kornia.augmentation as kA
import kornia.geometry.transform as kT
import librosa as rosa
import librosa.display
import madmom as mm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal as signal
import sklearn.cluster
import torch as th
import torch.nn.functional as F
from scipy import interpolate

import generate
import render
from models.stylegan1 import G_style
from models.stylegan2 import Generator

# ====================================================================================
# ================================= latent/noise ops =================================
# ====================================================================================


def get_chroma_latents(chroma, base_latent_selection):
    base_latents = (chroma[..., None, None] * base_latent_selection[None, ...]).sum(1)
    return base_latents


def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def get_gaussian_loops(base_latent_selection, loop_starting_latents, n_frames, n_loops, smoothing):
    base_latents = []
    for n in range(len(base_latent_selection)):
        for val in np.linspace(0.0, 1.0, int(n_frames // max(1, n_loops) // len(base_latent_selection))):
            base_latents.append(
                slerp(
                    val,
                    base_latent_selection[(n + loop_starting_latents) % len(base_latent_selection)][0],
                    base_latent_selection[(n + loop_starting_latents + 1) % len(base_latent_selection)][0],
                )
            )
    base_latents = th.stack(base_latents)
    base_latents = gaussian_filter(base_latents, smoothing * smf)
    base_latents = th.cat([base_latents] * int(n_frames / len(base_latents)), axis=0)
    base_latents = th.cat([base_latents[:, None, :]] * 18, axis=1)
    if n_frames - len(base_latents) != 0:
        base_latents = th.cat([base_latents, base_latents[0 : n_frames - len(base_latents)]])
    return base_latents


def get_spline_loops(base_latent_selection, n_frames, n_loops, loop=True):
    if loop:
        base_latent_selection = np.concatenate([base_latent_selection, base_latent_selection[[0]]])

    x = np.linspace(0, 1, int(n_frames // max(1, n_loops)))
    base_latents = np.zeros((len(x), *base_latent_selection.shape[1:]))
    for lay in range(base_latent_selection.shape[1]):
        for lat in range(base_latent_selection.shape[2]):
            tck = interpolate.splrep(
                np.linspace(0, 1, base_latent_selection.shape[0]), base_latent_selection[:, lay, lat]
            )
            base_latents[:, lay, lat] = interpolate.splev(x, tck)

    base_latents = th.cat([th.from_numpy(base_latents)] * int(n_frames / len(base_latents)), axis=0)
    if n_frames - len(base_latents) > 0:
        base_latents = th.cat([base_latents, base_latents[0 : n_frames - len(base_latents)]])
    return base_latents[:n_frames]


def wrapping_slice(tensor, start, length, return_indices=False):
    if start + length <= tensor.shape[0]:
        indices = th.arange(start, start + length)
    else:
        indices = th.cat((th.arange(start, tensor.shape[0]), th.arange(0, (start + length) % tensor.shape[0])))
    if tensor.shape[0] == 1:
        indices = th.zeros(1, dtype=th.int64)
    if return_indices:
        return indices
    return tensor[indices]


def generate_latents(n_latents, ckpt, G_res, out_size, noconst=False, latent_dim=512, n_mlp=8, channel_multiplier=2):
    generator = Generator(
        G_res,
        latent_dim,
        n_mlp,
        channel_multiplier=channel_multiplier,
        constant_input=not noconst,
        checkpoint=ckpt,
        output_size=out_size,
    ).cuda()
    zs = th.randn((n_latents, latent_dim), device="cuda")
    latent_selection = generator(zs, map_latents=True).cpu()
    del generator, zs
    gc.collect()
    th.cuda.empty_cache()
    return latent_selection


def load_latents(latent_file):
    return th.from_numpy(np.load(latent_file))


def create_circular_mask(h, w, center=None, radius=None, soft=0):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    if soft > 0:
        import scipy.ndimage.filters as ndi

        mask = ndi.gaussian_filter(mask, sigma=int(round(soft)))
    return th.from_numpy(mask)


def perlinterpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin_noise(shape, res, tileable=(True, False, False), interpolant=perlinterpolant):
    """Generate a 3D tensor of perlin noise.
    Args:
        shape: The shape of the generated tensor (tuple of three ints).
            This must be a multiple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of three ints). Note shape must be a multiple
            of res.
        tileable: If the noise should be tileable along each axis
            (tuple of three bools). Defaults to (False, False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).
    Returns:
        A tensor of shape shape with the generated noise.
    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1], 0 : res[2] : delta[2]]
    grid = grid.transpose(1, 2, 3, 0) % 1
    grid = th.from_numpy(grid).cuda()
    # Gradients
    theta = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1)
    phi = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1, res[2] + 1)
    gradients = np.stack((np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)), axis=3)
    if tileable[0]:
        gradients[-1, :, :] = gradients[0, :, :]
    if tileable[1]:
        gradients[:, -1, :] = gradients[:, 0, :]
    if tileable[2]:
        gradients[:, :, -1] = gradients[:, :, 0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    gradients = th.from_numpy(gradients).cuda()
    g000 = gradients[: -d[0], : -d[1], : -d[2]]
    g100 = gradients[d[0] :, : -d[1], : -d[2]]
    g010 = gradients[: -d[0], d[1] :, : -d[2]]
    g110 = gradients[d[0] :, d[1] :, : -d[2]]
    g001 = gradients[: -d[0], : -d[1], d[2] :]
    g101 = gradients[d[0] :, : -d[1], d[2] :]
    g011 = gradients[: -d[0], d[1] :, d[2] :]
    g111 = gradients[d[0] :, d[1] :, d[2] :]
    # Ramps
    n000 = th.sum(th.stack((grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2]), axis=3) * g000, 3)
    n100 = th.sum(th.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1], grid[:, :, :, 2]), axis=3) * g100, 3)
    n010 = th.sum(th.stack((grid[:, :, :, 0], grid[:, :, :, 1] - 1, grid[:, :, :, 2]), axis=3) * g010, 3)
    n110 = th.sum(th.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1] - 1, grid[:, :, :, 2]), axis=3) * g110, 3)
    n001 = th.sum(th.stack((grid[:, :, :, 0], grid[:, :, :, 1], grid[:, :, :, 2] - 1), axis=3) * g001, 3)
    n101 = th.sum(th.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1], grid[:, :, :, 2] - 1), axis=3) * g101, 3)
    n011 = th.sum(th.stack((grid[:, :, :, 0], grid[:, :, :, 1] - 1, grid[:, :, :, 2] - 1), axis=3) * g011, 3)
    n111 = th.sum(th.stack((grid[:, :, :, 0] - 1, grid[:, :, :, 1] - 1, grid[:, :, :, 2] - 1), axis=3) * g111, 3)
    # Interpolation
    t = interpolant(grid)
    n00 = n000 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n100
    n10 = n010 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n110
    n01 = n001 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n101
    n11 = n011 * (1 - t[:, :, :, 0]) + t[:, :, :, 0] * n111
    n0 = (1 - t[:, :, :, 1]) * n00 + t[:, :, :, 1] * n10
    n1 = (1 - t[:, :, :, 1]) * n01 + t[:, :, :, 1] * n11
    perlin = (1 - t[:, :, :, 2]) * n0 + t[:, :, :, 2] * n1
    return perlin * 2 - 1  # stretch from -1 to 1

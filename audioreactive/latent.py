import gc

import numpy as np
import torch as th
from scipy import interpolate

from models.stylegan2 import Generator
from .signal import gaussian_filter

# ====================================================================================
# ================================= latent/noise ops =================================
# ====================================================================================


def chroma_weight_latents(chroma, latents):
    """Creates chromagram weighted latent sequence

    Args:
        chroma (th.tensor): Chromagram
        latents (th.tensor): Latents (must have same number as number of notes in chromagram)

    Returns:
        th.tensor: Chromagram weighted latent sequence
    """
    base_latents = (chroma[..., None, None] * latents[None, ...]).sum(1)
    return base_latents


def slerp(val, low, high):
    """Interpolation along geodesic of n-dimensional unit sphere
    from https://github.com/soumith/dcgan.torch/issues/14#issuecomment-200025792

    Args:
        val (float): Value between 0 and 1 representing fraction of interpolation completed
        low (float): Starting value
        high (float): Ending value

    Returns:
        float: Interpolated value
    """
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def slerp_loops(latent_selection, n_frames, n_loops, smoothing=1, loop=True):
    """Get looping latents using geodesic interpolation. Total length of n_frames with n_loops repeats.

    Args:
        latent_selection (th.tensor): Set of latents to loop between (in order)
        n_frames (int): Total length of output looping sequence
        n_loops (int): Number of times to loop
        smoothing (int, optional): Standard deviation of gaussian smoothing kernel. Defaults to 1.
        loop (bool, optional): Whether to return to first latent. Defaults to True.

    Returns:
        th.tensor: Sequence of smoothly looping latents
    """
    if loop:
        latent_selection = np.concatenate([latent_selection, latent_selection[[0]]])

    base_latents = []
    for n in range(len(latent_selection)):
        for val in np.linspace(0.0, 1.0, int(n_frames // max(1, n_loops) // len(latent_selection))):
            base_latents.append(
                th.from_numpy(
                    slerp(
                        val,
                        latent_selection[n % len(latent_selection)][0],
                        latent_selection[(n + 1) % len(latent_selection)][0],
                    )
                )
            )
    base_latents = th.stack(base_latents)
    base_latents = gaussian_filter(base_latents, smoothing)
    base_latents = th.cat([base_latents] * int(n_frames / len(base_latents)), axis=0)
    base_latents = th.cat([base_latents[:, None, :]] * 18, axis=1)
    if n_frames - len(base_latents) != 0:
        base_latents = th.cat([base_latents, base_latents[0 : n_frames - len(base_latents)]])
    return base_latents


def spline_loops(latent_selection, n_frames, n_loops, loop=True):
    """Get looping latents using spline interpolation. Total length of n_frames with n_loops repeats.

    Args:
        latent_selection (th.tensor): Set of latents to loop between (in order)
        n_frames (int): Total length of output looping sequence
        n_loops (int): Number of times to loop
        loop (bool, optional): Whether to return to first latent. Defaults to True.

    Returns:
        th.tensor: Sequence of smoothly looping latents
    """
    if loop:
        latent_selection = np.concatenate([latent_selection, latent_selection[[0]]])

    x = np.linspace(0, 1, int(n_frames // max(1, n_loops)))
    base_latents = np.zeros((len(x), *latent_selection.shape[1:]))
    for lay in range(latent_selection.shape[1]):
        for lat in range(latent_selection.shape[2]):
            tck = interpolate.splrep(np.linspace(0, 1, latent_selection.shape[0]), latent_selection[:, lay, lat])
            base_latents[:, lay, lat] = interpolate.splev(x, tck)

    base_latents = th.cat([th.from_numpy(base_latents)] * int(n_frames / len(base_latents)), axis=0)
    if n_frames - len(base_latents) > 0:
        base_latents = th.cat([base_latents, base_latents[0 : n_frames - len(base_latents)]])
    return base_latents[:n_frames]


def wrapping_slice(tensor, start, length, return_indices=False):
    """Gets slice of tensor of a given length that wraps around to beginning

    Args:
        tensor (th.tensor): Tensor to slice
        start (int): Starting index
        length (int): Size of slice
        return_indices (bool, optional): Whether to return indices rather than values. Defaults to False.

    Returns:
        th.tensor: Values or indices of slice
    """
    if start + length <= tensor.shape[0]:
        indices = th.arange(start, start + length)
    else:
        indices = th.cat((th.arange(start, tensor.shape[0]), th.arange(0, (start + length) % tensor.shape[0])))
    if tensor.shape[0] == 1:
        indices = th.zeros(1, dtype=th.int64)
    if return_indices:
        return indices
    return tensor[indices]


def generate_latents(n_latents, ckpt, G_res, noconst=False, latent_dim=512, n_mlp=8, channel_multiplier=2):
    """Generates random, mapped latents

    Args:
        n_latents (int): Number of mapped latents to generate 
        ckpt (str): Generator checkpoint to use
        G_res (int): Generator's training resolution
        noconst (bool, optional): Whether the generator was trained without constant starting layer. Defaults to False.
        latent_dim (int, optional): Size of generator's latent vectors. Defaults to 512.
        n_mlp (int, optional): Number of layers in the generator's mapping network. Defaults to 8.
        channel_multiplier (int, optional): Scaling multiplier for generator's channel depth. Defaults to 2.

    Returns:
        th.tensor: Set of mapped latents
    """
    generator = Generator(
        G_res, latent_dim, n_mlp, channel_multiplier=channel_multiplier, constant_input=not noconst, checkpoint=ckpt,
    ).cuda()
    zs = th.randn((n_latents, latent_dim), device="cuda")
    latent_selection = generator(zs, map_latents=True).cpu()
    del generator, zs
    gc.collect()
    th.cuda.empty_cache()
    return latent_selection


def save_latents(latents, filename):
    """Saves latent vectors to file

    Args:
        latents (th.tensor): Latent vector(s) to save
        filename (str): Filename to save to
    """
    np.save(filename, latents)


def load_latents(filename):
    """Load latents from numpy file

    Args:
        filename (str): Filename to load from

    Returns:
        th.tensor: Latent vectors
    """
    return th.from_numpy(np.load(filename))


def _perlinterpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin_noise(shape, res, tileable=(True, False, False), interpolant=_perlinterpolant):
    """Generate a 3D tensor of perlin noise.

    Args:
        shape: The shape of the generated tensor (tuple of three ints). This must be a multiple of res.
        res: The number of periods of noise to generate along each axis (tuple of three ints). Note shape must be a multiple of res.
        tileable: If the noise should be tileable along each axis (tuple of three bools). Defaults to (False, False, False).
        interpolant: The interpolation function, defaults to t*t*t*(t*(t*6 - 15) + 10).

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

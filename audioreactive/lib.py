import os, gc
import time, uuid, json, math
import argparse, random
import cachetools, warnings

import numpy as np
import scipy
from scipy import interpolate
import scipy.signal as signal
import sklearn.cluster

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import madmom as mm
import librosa as rosa
import librosa.display

import torch as th
import torch.nn.functional as F

import kornia.augmentation as kA
import kornia.geometry.transform as kT

import render
import generate
from models.stylegan1 import G_style
from models.stylegan2 import Generator

# ====================================================================================
# ==================================== utilities =====================================
# ====================================================================================

CACHE = cachetools.LRUCache(maxsize=128)
SMF = 30 / 43.066666  # ensures smoothing is independent of frame rate


def subsample_hash(a):
    rng = np.random.RandomState(42)
    inds = rng.randint(low=0, high=a.size, size=int(a.size / 64))
    b = a.flat[inds]
    b.flags.writeable = False
    return hash(b.data.tobytes())


def lru_cache(func):
    """
    numpy friendly caching
    https://github.com/alekk/lru_cache_numpy/blob/master/numpy_caching.py
    """

    def hashing_first_numpy_arg(*args, **kwargs):
        """ sum up the hash of all the arguments """
        hash_total = 0
        for x in [*args, *kwargs.values()]:
            if isinstance(x, np.ndarray):
                hash_total += subsample_hash(x)
            else:
                hash_total += hash(x)
        return hash_total

    return cachetools.cached(CACHE, hashing_first_numpy_arg)(func)


def info(arr):
    if isinstance(arr, list):
        print([(list(a.shape), f"{a.min():.2f}", f"{a.mean():.2f}", f"{a.max():.2f}") for a in arr])
    else:
        print(list(arr.shape), f"{arr.min():.2f}", f"{arr.mean():.2f}", f"{arr.max():.2f}")


def plot_signals(signals):
    info(signals)
    plt.figure(figsize=(8, 2 * len(signals)))
    for sbplt, y in enumerate(signals):
        try:
            signal = signal.cpu().numpy()
        except:
            pass
        plt.subplot(len(signals), 1, sbplt + 1)
        plt.plot(y.squeeze())
    plt.tight_layout()
    plt.show()


def plot_spectra(spectra, chroma=False):
    fig, axes = plt.subplots(len(spectra), 1, figsize=(8, 3 * len(spectra)))
    for ax, spectrum in zip(axes if len(spectra) > 1 else [axes], spectra):
        try:
            spectrum = spectrum.cpu().numpy()
        except:
            pass
        if spectrum.shape[1] == 12:
            spectrum = spectrum.T
        rosa.display.specshow(spectrum, y_axis="chroma" if chroma else None, x_axis="time", ax=ax)
    plt.tight_layout()
    plt.show()


def plot_audio(audio, sr):
    rosa.display.specshow(
        rosa.power_to_db(rosa.feature.melspectrogram(y=audio, sr=sr), ref=np.max), y_axis="mel", x_axis="time"
    )
    plt.colorbar(format="%+2.f dB")
    plt.tight_layout()
    plt.show()


def plot_chroma_comparison(audio, sr):
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 9))
    for col, types in enumerate([["cens", "cqt"], ["deep", "clp"], ["stft"]]):
        for row, type in enumerate(types):
            ch = raw_chroma(audio, sr, type=type)
            if ch.shape[1] == 12:
                ch = ch.T
            librosa.display.specshow(ch, y_axis="chroma", x_axis="time", ax=ax[row, col])
            ax[row, col].set(title=type)
            ax[row, col].label_outer()
    plt.tight_layout()
    plt.show()


# ====================================================================================
# ==================================== signal ops ====================================
# ====================================================================================


@lru_cache
def onsets(audio, sr, n_frames, fmin=20, fmax=16000, smooth=1, clip=100, power=1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_perc = rosa.effects.percussive(y=audio)
        sig = mm.audio.signal.Signal(y_perc, num_channels=1, sample_rate=sr)
        sig_frames = mm.audio.signal.FramedSignal(sig, frame_size=2048, hop_size=441)
        stft = mm.audio.stft.ShortTimeFourierTransform(sig_frames, ciruclar_shift=True)
        spec = mm.audio.spectrogram.Spectrogram(stft, ciruclar_shift=True)
        filt_spec = mm.audio.spectrogram.FilteredSpectrogram(spec, num_bands=24, fmin=fmin, fmax=fmax)
        onset = np.sum(
            [
                mm.features.onsets.spectral_diff(filt_spec),
                mm.features.onsets.spectral_flux(filt_spec),
                mm.features.onsets.superflux(filt_spec),
                mm.features.onsets.complex_flux(filt_spec),
                mm.features.onsets.modified_kullback_leibler(filt_spec),
            ],
            axis=0,
        )
    onset = np.clip(signal.resample(onset, n_frames), onset.min(), onset.max())
    onset = th.from_numpy(onset).float()
    onset = gaussian_filter(onset, smooth, causal=0.2)
    onset = percentile_clip(onset, clip)
    onset = onset ** power
    return onset


@lru_cache
def rms(y, sr, n_frames, fmin, fmax, smooth, clip, power):
    y_filt = signal.sosfilt(signal.butter(12, [fmin, fmax], "bp", fs=sr, output="sos"), y)
    rms = rosa.feature.rms(S=np.abs(rosa.stft(y=y_filt, hop_length=512)))[0]
    rms = np.clip(signal.resample(rms, n_frames), rms.min(), rms.max())
    rms = th.from_numpy(rms).float()
    rms = gaussian_filter(rms, smooth, causal=0.2)
    rms = percentile_clip(rms, clip)
    rms = rms ** power
    return rms


def raw_chroma(audio, sr, type="cens", nearest_neighbor=True):
    if type == "cens":
        ch = rosa.feature.chroma_cens(y=audio, sr=sr)
    elif type == "cqt":
        ch = rosa.feature.chroma_cqt(y=audio, sr=sr)
    elif type == "stft":
        ch = rosa.feature.chroma_stft(y=audio, sr=sr)
    elif type == "deep":
        sig = mm.audio.signal.Signal(audio, num_channels=1, sample_rate=sr)
        ch = mm.audio.chroma.DeepChromaProcessor().process(sig)
    elif type == "clp":
        sig = mm.audio.signal.Signal(audio, num_channels=1, sample_rate=sr)
        ch = mm.audio.chroma.CLPChromaProcessor().process(sig)
    else:
        print("chroma type not recognized, options are: [cens, cqt, deep, clp, or stft]. defaulting to cens...")
        ch = rosa.feature.chroma_cens(y=audio, sr=sr)

    if nearest_neighbor:
        ch = np.minimum(ch, rosa.decompose.nn_filter(ch, aggregate=np.median, metric="cosine"))

    return ch


@lru_cache
def chroma(audio, sr, n_frames, margin=16, type="cens", top_k=12):
    y_harm = rosa.effects.harmonic(y=audio, margin=margin)
    chroma = raw_chroma(y_harm, sr, type=type).T
    chroma = signal.resample(chroma, n_frames)
    top_k_indices = np.argsort(np.median(chroma, axis=0))[:top_k]
    chroma = chroma[top_k_indices]
    chroma = th.from_numpy(chroma / chroma.sum(1)[:, None]).float()
    return chroma


@lru_cache
def laplacian_segmentation(y, sr, k=5, plot=False):
    """
    Based on https://librosa.org/doc/latest/auto_examples/plot_segmentation.html#sphx-glr-auto-examples-plot-segmentation-py%22
    """
    BINS_PER_OCTAVE = 12 * 3
    N_OCTAVES = 7
    C = librosa.amplitude_to_db(
        np.abs(librosa.cqt(y=y, sr=sr, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_OCTAVES * BINS_PER_OCTAVE)), ref=np.max
    )

    # make CQT beat-synchronous to reduce dimensionality
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    Csync = librosa.util.sync(C, beats, aggregate=np.median)

    # build a weighted recurrence matrix using beat-synchronous CQT
    R = librosa.segment.recurrence_matrix(Csync, width=3, mode="affinity", sym=True)
    # enhance diagonals with a median filter
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))

    # build the sequence matrix using mfcc-similarity
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    Msync = librosa.util.sync(mfcc, beats)
    path_distance = np.sum(np.diff(Msync, axis=1) ** 2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    # compute the balanced combination
    deg_path = np.sum(R_path, axis=1)
    deg_rec = np.sum(Rf, axis=1)
    mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec) ** 2)

    A = mu * Rf + (1 - mu) * R_path
    # compute the normalized laplacian and its spectral decomposition
    L = scipy.sparse.csgraph.laplacian(A, normed=True)
    evals, evecs = scipy.linalg.eigh(L)
    # median filter to smooth over small discontinuities
    evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))
    # cumulative normalization for symmetric normalized laplacian eigenvectors
    Cnorm = np.cumsum(evecs ** 2, axis=1) ** 0.5

    X = evecs[:, :k] / Cnorm[:, k - 1 : k]

    # use first k components to cluster beats into segments
    seg_ids = sklearn.cluster.KMeans(n_clusters=k).fit_predict(X)

    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])  # locate segment boundaries from the label sequence
    bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)  # count beat 0 as a boundary
    bound_segs = list(seg_ids[bound_beats])  # compute the segment label for each boundary
    bound_frames = beats[bound_beats]  # convert beat indices to frames
    bound_frames = librosa.util.fix_frames(bound_frames, x_min=None, x_max=C.shape[1] - 1)
    bound_times = librosa.frames_to_time(bound_frames)
    if bound_times[0] != 0:
        bound_times[0] = 0

    if plot:
        freqs = librosa.cqt_frequencies(
            n_bins=C.shape[0], fmin=librosa.note_to_hz("C1"), bins_per_octave=BINS_PER_OCTAVE
        )
        fig, ax = plt.subplots()
        colors = plt.get_cmap("Paired", k)
        librosa.display.specshow(C, y_axis="cqt_hz", sr=sr, bins_per_octave=BINS_PER_OCTAVE, x_axis="time", ax=ax)
        for interval, label in zip(zip(bound_times, bound_times[1:]), bound_segs):
            ax.add_patch(
                patches.Rectangle(
                    (interval[0], freqs[0]), interval[1] - interval[0], freqs[-1], facecolor=colors(label), alpha=0.50
                )
            )
        plt.show()

    return list(bound_times), list(bound_segs)


def normalize(y):
    y -= y.min()
    y /= y.max()
    return y


def percentile(y, p):
    k = 1 + round(0.01 * float(p) * (y.numel() - 1))
    return y.view(-1).kthvalue(k).values.item()


def percentile_clip(y, p):
    locs = th.arange(0, y.shape[0])
    peaks = th.ones(y.shape, dtype=bool)
    main = y.take(locs)

    plus = y.take((locs + 1).clamp(0, y.shape[0] - 1))
    minus = y.take((locs - 1).clamp(0, y.shape[0] - 1))
    peaks &= th.gt(main, plus)
    peaks &= th.gt(main, minus)

    y = y.clamp(0, percentile(y[peaks], p))
    y /= y.max()
    return y


def compress(audio, threshold, ratio, invert=False):
    if invert:
        audio[audio < threshold] *= ratio
    else:
        audio[audio > threshold] *= ratio
    return normalize(audio)


def gaussian_filter(x, sigma, causal=None):
    dim = len(x.shape)
    while len(x.shape) < 3:
        x = x[:, None]

    radius = int(sigma * 4 * SMF)
    channels = x.shape[1]

    kernel = th.arange(-radius, radius + 1, dtype=th.float32, device=x.device)
    kernel = th.exp(-0.5 / sigma ** 2 * kernel ** 2)
    if causal is not None:
        kernel[: radius + 1] *= 0 if not isinstance(causal, float) else causal
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, len(kernel)).repeat(channels, 1, 1)

    if dim == 4:
        t, c, h, w = x.shape
        x = x.view(t, c, h * w)
    x = x.transpose(0, 2)

    x = F.pad(x, (radius, radius), mode="circular")
    x = F.conv1d(x, weight=kernel, groups=channels)

    x = x.transpose(0, 2)
    if dim == 4:
        x = x.view(t, c, h, w)

    if len(x.shape) > dim:
        x = x.squeeze()

    return x


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


def get_gaussian_loops(base_latent_selection, loop_starting_latents, n_frames, num_loops, smoothing):
    base_latents = []
    for n in range(len(base_latent_selection)):
        for val in np.linspace(0.0, 1.0, int(n_frames // max(1, num_loops) // len(base_latent_selection))):
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


def get_spline_loops(base_latent_selection, n_frames, num_loops, loop=True):
    if loop:
        base_latent_selection = np.concatenate([base_latent_selection, base_latent_selection[[0]]])

    x = np.linspace(0, 1, int(n_frames // max(1, num_loops)))
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
    return (1 - t[:, :, :, 2]) * n0 + t[:, :, :, 2] * n1


# ====================================================================================
# ================================= network bending ==================================
# ====================================================================================


class NetworkBend(th.nn.Module):
    def __init__(self, sequential_fn, modulation):
        super(NetworkBend, self).__init__()
        self.sequential = sequential_fn(modulation)

    def forward(self, x):
        return self.sequential(x)


class AddNoise(th.nn.Module):
    def __init__(self, noise):
        super(AddNoise, self).__init__()
        self.noise = noise

    def forward(self, x):
        return x + self.noise.to(x.device)


class Print(th.nn.Module):
    def forward(self, x):
        print(x.shape, [x.min().item(), x.mean().item(), x.max().item()], th.std(x).item())
        return x


class Translate(NetworkBend):
    def __init__(self, h, w, modulation):
        sequential_fn = lambda b: th.nn.Sequential(
            th.nn.ReflectionPad2d((int(w / 2), int(w / 2), 0, 0)),
            th.nn.ReflectionPad2d((w, w, 0, 0)),
            th.nn.ReflectionPad2d((w, 0, 0, 0)),
            AddNoise(th.randn((1, 1, h, w), device="cuda")),
            kT.Translate(b),
            kA.CenterCrop((h, w)),
        )
        super(Translate, self).__init__(sequential_fn, modulation)


class Zoom(NetworkBend):
    def __init__(self, h, w, modulation):
        padding = int(max(h, w)) - 1
        sequential_fn = lambda b: th.nn.Sequential(th.nn.ReflectionPad2d(padding), kT.Scale(b), kA.CenterCrop((h, w)))
        super(Zoom, self).__init__(sequential_fn, modulation)


class Rotate(NetworkBend):
    def __init__(self, h, w, modulation):
        # worst case rotation brings sqrt(2) * max_side_length out-of-frame pixels into frame
        # padding should cover that exactly
        padding = int(max(h, w) * (1 - math.sqrt(2) / 2))
        sequential_fn = lambda b: th.nn.Sequential(th.nn.ReflectionPad2d(padding), kT.Rotate(b), kA.CenterCrop((h, w)))
        super(Rotate, self).__init__(sequential_fn, modulation)

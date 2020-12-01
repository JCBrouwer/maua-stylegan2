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

from .util import *

SMF = 30 / 43.066666  # ensures smoothing is independent of frame rate
CACHE = cachetools.LRUCache(maxsize=128)
USE_CACHE = False


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
        print("caching")
        return hash_total

    return cachetools.cached(CACHE, hashing_first_numpy_arg)(func)


# ====================================================================================
# ==================================== signal ops ====================================
# ====================================================================================


@lru_cache
def onsets(audio, sr, n_frames, margin=8, fmin=20, fmax=8000, smooth=1, clip=100, power=1, type="rosa"):
    y_perc = rosa.effects.percussive(y=audio, margin=margin)
    if type == "rosa":
        onset = rosa.onset.onset_strength(y=y_perc, sr=sr, fmin=fmin, fmax=fmax)
    elif type == "mm":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
    onset = gaussian_filter(onset, smooth, causal=True)
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


@lru_cache
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
        kernel[radius + 1 :] *= 0 if not isinstance(causal, float) else causal
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

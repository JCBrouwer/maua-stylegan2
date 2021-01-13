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

SMF = None  # this is set by generate_audiovisual.py based on rendering fps


def set_SMF(smf):
    global SMF
    SMF = smf


# ====================================================================================
# ==================================== signal ops ====================================
# ====================================================================================


def onsets(audio, sr, n_frames, margin=8, fmin=20, fmax=8000, smooth=1, clip=100, power=1, type="mm"):
    """Creates onset envelope from audio

    Args:
        audio (np.array): Audio signal
        sr (int): Sampling rate of the audio
        n_frames (int): Total number of frames to resample envelope to
        margin (int, optional): For percussive source separation, higher values create more extreme separations. Defaults to 8.
        fmin (int, optional): Minimum frequency for onset analysis. Defaults to 20.
        fmax (int, optional): Maximum frequency for onset analysis. Defaults to 8000.
        smooth (int, optional): Standard deviation of gaussian kernel to smooth with. Defaults to 1.
        clip (int, optional): Percentile to clip onset signal to. Defaults to 100.
        power (int, optional): Exponent to raise onset signal to. Defaults to 1.
        type (str, optional): ["rosa", "mm"]. Whether to use librosa or madmom for onset analysis. Madmom is slower but often more accurate. Defaults to "mm".

    Returns:
        th.tensor, shape=(n_frames,): Onset envelope
    """
    y_perc = rosa.effects.percussive(y=audio, margin=margin)
    if type == "rosa":
        onset = rosa.onset.onset_strength(y=y_perc, sr=sr, fmin=fmin, fmax=fmax)
    elif type == "mm":
        sig = mm.audio.signal.Signal(y_perc, num_channels=1, sample_rate=sr)
        sig_frames = mm.audio.signal.FramedSignal(sig, frame_size=2048, hop_size=441)
        stft = mm.audio.stft.ShortTimeFourierTransform(sig_frames, circular_shift=True)
        spec = mm.audio.spectrogram.Spectrogram(stft, circular_shift=True)
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
    onset = gaussian_filter(onset, smooth, causal=0)
    onset = percentile_clip(onset, clip)
    onset = onset ** power
    return onset


def rms(y, sr, n_frames, fmin=20, fmax=8000, smooth=180, clip=50, power=6):
    """Creates RMS envelope from audio

    Args:
        audio (np.array): Audio signal
        sr (int): Sampling rate of the audio
        n_frames (int): Total number of frames to resample envelope to
        fmin (int, optional): Minimum frequency for onset analysis. Defaults to 20.
        fmax (int, optional): Maximum frequency for onset analysis. Defaults to 8000.
        smooth (int, optional): Standard deviation of gaussian kernel to smooth with. Defaults to 180.
        clip (int, optional): Percentile to clip onset signal to. Defaults to 50.
        power (int, optional): Exponent to raise onset signal to. Defaults to 6.

    Returns:
        th.tensor, shape=(n_frames,): RMS envelope
    """
    y_filt = signal.sosfilt(signal.butter(12, [fmin, fmax], "bp", fs=sr, output="sos"), y)
    rms = rosa.feature.rms(S=np.abs(rosa.stft(y=y_filt, hop_length=512)))[0]
    rms = np.clip(signal.resample(rms, n_frames), rms.min(), rms.max())
    rms = th.from_numpy(rms).float()
    rms = gaussian_filter(rms, smooth, causal=0.2)
    rms = percentile_clip(rms, clip)
    rms = rms ** power
    return rms


def raw_chroma(audio, sr, type="cens", nearest_neighbor=True):
    """Creates chromagram

    Args:
        audio (np.array): Audio signal
        sr (int): Sampling rate of the audio
        type (str, optional): ["cens", "cqt", "stft", "deep", "clp"]. Which strategy to use to calculate the chromagram. Defaults to "cens".
        nearest_neighbor (bool, optional): Whether to post process using nearest neighbor smoothing. Defaults to True.

    Returns:
        np.array, shape=(12, n_frames): Chromagram
    """
    if type == "cens":
        ch = rosa.feature.chroma_cens(y=audio, sr=sr)
    elif type == "cqt":
        ch = rosa.feature.chroma_cqt(y=audio, sr=sr)
    elif type == "stft":
        ch = rosa.feature.chroma_stft(y=audio, sr=sr)
    elif type == "deep":
        sig = mm.audio.signal.Signal(audio, num_channels=1, sample_rate=sr)
        ch = mm.audio.chroma.DeepChromaProcessor().process(sig).T
    elif type == "clp":
        sig = mm.audio.signal.Signal(audio, num_channels=1, sample_rate=sr)
        ch = mm.audio.chroma.CLPChromaProcessor().process(sig).T
    else:
        print("chroma type not recognized, options are: [cens, cqt, deep, clp, or stft]. defaulting to cens...")
        ch = rosa.feature.chroma_cens(y=audio, sr=sr)

    if nearest_neighbor:
        ch = np.minimum(ch, rosa.decompose.nn_filter(ch, aggregate=np.median, metric="cosine"))

    return ch


def chroma(audio, sr, n_frames, margin=16, type="cens", notes=12):
    """Creates chromagram for the harmonic component of the audio

    Args:
        audio (np.array): Audio signal
        sr (int): Sampling rate of the audio
        n_frames (int): Total number of frames to resample envelope to
        margin (int, optional): For harmonic source separation, higher values create more extreme separations. Defaults to 16.
        type (str, optional): ["cens", "cqt", "stft", "deep", "clp"]. Which strategy to use to calculate the chromagram. Defaults to "cens".
        notes (int, optional): Number of notes to use in output chromagram (e.g. 5 for pentatonic scale, 7 for standard western scales). Defaults to 12.

    Returns:
        th.tensor, shape=(n_frames, 12): Chromagram
    """
    y_harm = rosa.effects.harmonic(y=audio, margin=margin)
    chroma = raw_chroma(y_harm, sr, type=type).T
    chroma = signal.resample(chroma, n_frames)
    notes_indices = np.argsort(np.median(chroma, axis=0))[:notes]
    chroma = chroma[:, notes_indices]
    chroma = th.from_numpy(chroma / chroma.sum(1)[:, None]).float()
    return chroma


def laplacian_segmentation(signal, sr, k=5, plot=False):
    """Segments the audio with pattern recurrence analysis
    From https://librosa.org/doc/latest/auto_examples/plot_segmentation.html#sphx-glr-auto-examples-plot-segmentation-py%22

    Args:
        signal (np.array): Audio signal
        sr (int): Sampling rate of the audio
        k (int, optional): Number of labels to use during segmentation. Defaults to 5.
        plot (bool, optional): Whether to show plot of found segmentation. Defaults to False.

    Returns:
        tuple(list, list): List of starting timestamps and labels of found segments
    """
    BINS_PER_OCTAVE = 12 * 3
    N_OCTAVES = 7
    C = librosa.amplitude_to_db(
        np.abs(librosa.cqt(y=signal, sr=sr, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_OCTAVES * BINS_PER_OCTAVE)),
        ref=np.max,
    )

    # make CQT beat-synchronous to reduce dimensionality
    tempo, beats = librosa.beat.beat_track(y=signal, sr=sr, trim=False)
    Csync = librosa.util.sync(C, beats, aggregate=np.median)

    # build a weighted recurrence matrix using beat-synchronous CQT
    R = librosa.segment.recurrence_matrix(Csync, width=3, mode="affinity", sym=True)
    # enhance diagonals with a median filter
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, 7))

    # build the sequence matrix using mfcc-similarity
    mfcc = librosa.feature.mfcc(y=signal, sr=sr)
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


def normalize(signal):
    """Normalize signal between 0 and 1

    Args:
        signal (np.array/th.tensor): Signal to normalize

    Returns:
        np.array/th.tensor: Normalized signal
    """
    signal -= signal.min()
    signal /= signal.max()
    return signal


def percentile(signal, p):
    """Calculate percentile of signal

    Args:
        signal (np.array/th.tensor): Signal to normalize
        p (int): [0-100]. Percentile to find

    Returns:
        int: Percentile signal value
    """
    k = 1 + round(0.01 * float(p) * (signal.numel() - 1))
    return signal.view(-1).kthvalue(k).values.item()


def percentile_clip(signal, p):
    """Normalize signal between 0 and 1, clipping peak values above given percentile

    Args:
        signal (th.tensor): Signal to normalize
        p (int): [0-100]. Percentile to clip to

    Returns:
        th.tensor: Clipped signal
    """
    locs = th.arange(0, signal.shape[0])
    peaks = th.ones(signal.shape, dtype=bool)
    main = signal.take(locs)

    plus = signal.take((locs + 1).clamp(0, signal.shape[0] - 1))
    minus = signal.take((locs - 1).clamp(0, signal.shape[0] - 1))
    peaks &= th.gt(main, plus)
    peaks &= th.gt(main, minus)

    signal = signal.clamp(0, percentile(signal[peaks], p))
    signal /= signal.max()
    return signal


def compress(signal, threshold, ratio, invert=False):
    """Expand or compress signal. Values above/below (depending on invert) threshold are multiplied by ratio.

    Args:
        signal (th.tensor): Signal to normalize
        threshold (int): Signal value above/below which to change signal
        ratio (float): Value to multiply signal with
        invert (bool, optional): Specifies if values above or below threshold are affected. Defaults to False.

    Returns:
        th.tensor: Compressed/expanded signal
    """
    if invert:
        signal[signal < threshold] *= ratio
    else:
        signal[signal > threshold] *= ratio
    return normalize(signal)


def expand(signal, threshold, ratio, invert=False):
    """Alias for compress. Whether compression or expansion occurs is determined by values of threshold and ratio"""
    return compress(signal, threshold, ratio, invert)


def gaussian_filter(x, sigma, causal=None):
    """Smooth 3 or 4 dimensional tensors along time (first) axis with gaussian kernel.

    Args:
        x (th.tensor): Tensor to be smoothed
        sigma (float): Standard deviation for gaussian kernel (higher value gives smoother result)
        causal (float, optional): Factor to multiply right side of gaussian kernel with. Lower value decreases effect of "future" values. Defaults to None.

    Returns:
        th.tensor: Smoothed tensor
    """
    dim = len(x.shape)
    while len(x.shape) < 3:
        x = x[:, None]

    # radius =  min(int(sigma * 4 * SMF), int(len(x) / 2) - 1)  # prevent padding errors on short sequences
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

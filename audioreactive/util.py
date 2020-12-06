import librosa as rosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# ====================================================================================
# ==================================== utilities =====================================
# ====================================================================================


def info(arr):
    """Shows statistics and shape information of (lists of) np.arrays/th.tensors

    Args:
        arr (np.array/th.tensor/list): List of or single np.array or th.tensor
    """
    if isinstance(arr, list):
        print([(list(a.shape), f"{a.min():.2f}", f"{a.mean():.2f}", f"{a.max():.2f}") for a in arr])
    else:
        print(list(arr.shape), f"{arr.min():.2f}", f"{arr.mean():.2f}", f"{arr.max():.2f}")


def plot_signals(signals):
    """Shows plot of (multiple) 1D signals

    Args:
        signals (np.array/th.tensor): List of signals (1 non-unit dimension)
    """
    plt.figure(figsize=(16, 4 * len(signals)))
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
    """Shows plot of (multiple) spectrograms

    Args:
        spectra (np.array/th.tensor): List of spectrograms
        chroma (bool, optional): Whether to plot with chromagram y-axis label. Defaults to False.
    """
    fig, axes = plt.subplots(len(spectra), 1, figsize=(16, 4 * len(spectra)))
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
    """Shows spectrogram of audio signal

    Args:
        audio (np.array): Audio signal to be plotted
        sr (int): Sampling rate of the audio
    """
    plt.figure(figsize=(16, 9))
    rosa.display.specshow(
        rosa.power_to_db(rosa.feature.melspectrogram(y=audio, sr=sr), ref=np.max), y_axis="mel", x_axis="time"
    )
    plt.colorbar(format="%+2.f dB")
    plt.tight_layout()
    plt.show()


def plot_chroma_comparison(audio, sr):
    """Shows plot comparing different chromagram strategies.

    Args:
        audio (np.array): Audio signal to be plotted
        sr (int): Sampling rate of the audio
    """
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

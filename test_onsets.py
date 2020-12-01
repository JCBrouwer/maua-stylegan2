# %% (vscode python interactive cell, alternatively, paste these into ipynb cells)
import librosa as rosa
import audioreactive as ar

audio_file = "audioreactive/examples/Wavefunk - Dwelling in the Kelp.mp3"
fps = 24
# audio, sr = rosa.load(audio_file, offset=0, duration=120)
# n_frames = 120 * fps
audio, sr = rosa.load(audio_file, offset=0, duration=None)
duration = rosa.get_duration(filename=audio_file)
n_frames = int(round(duration * fps))

# %%
import importlib

importlib.reload(ar)

low_onsets = ar.onsets(audio, sr, n_frames, margin=1, fmax=100, smooth=5, clip=96, power=2)
high_onsets = ar.compress(low_onsets, 0.4, 2)

high_onsets = ar.onsets(audio, sr, n_frames, margin=4, fmin=200, smooth=5, clip=97, power=2)
high_onsets = ar.compress(high_onsets, 0.6, 2)

ar.plot_signals([low_onsets, high_onsets])

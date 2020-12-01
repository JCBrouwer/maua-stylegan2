# %%
import librosa as rosa
import audioreactive as ar

audio, sr = rosa.load("audioreactive/examples/Wavefunk - Dwelling in the Kelp.mp3", offset=0, duration=120)
fps = 24
n_frames = 120 * fps

# %%
import importlib

importlib.reload(ar)

low_onsets = ar.onsets(audio, sr, n_frames, fmax=100, smooth=7, clip=95, power=8, type="rosa").clone()

high_onsets = ar.onsets(audio, sr, n_frames, fmin=200, smooth=7, clip=95, power=8, type="rosa").clone()

ar.plot_signals([low_onsets[1500:2000], high_onsets[1500:2000]])

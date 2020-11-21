import audioreactive as ar


def get_latents(aud, ls):
    chroma = ar.get_chroma(aud, num_frames)
    chroma_latents = ar.get_chroma_latents(chroma, ls)
    return chroma_latents


def get_noise(audio, height, width, scale, max_scale):
    spec = mm.audio.spectrogram.Spectrogram(
        mm.audio.stft.ShortTimeFourierTransform(
            mm.audio.signal.FramedSignal(mm.audio.signal.Signal(aud, num_channels=1), frame_size=2048, hop_size=441),
            ciruclar_shift=True,
        ),
        ciruclar_shift=True,
    )
    low_onsets = ar.get_onsets(spec, fmin=30, fmax=500, smooth=5 * smf, clip=95, power=2)
    hi_onsets = ar.get_onsets(spec, fmin=500, fmax=18000, smooth=5 * smf, clip=95, power=2)

    mask = create_circular_mask(height, width, radius=int(width * 5 / 6))

    noise_noisy = gaussian_filter(th.randn((num_frames, 1, h, w), device="cuda"), max(1, int(round(smf)))).cpu()
    noise_smooth = gaussian_filter(th.randn((num_frames, 1, h, w), device="cuda"), max(1, int(round(20 * smf)))).cpu()

    if scale < max_scale / 3:
        noise_scale = mask * low_onsets * noise_noisy + (1 - mask) * (1 - low_onsets) * noise_smooth
    else:
        noise_scale = (1 - mask) * hi_onsets * noise_noisy + mask * (1 - hi_onsets) * noise_smooth

    return noise_scale


def old():
    latent_selection = get_latent_selection(args.latent_file)
    if args.shuffle_latents:
        random_indices = random.sample(range(len(latent_selection)), len(latent_selection))
        latent_selection = latent_selection[random_indices]
    np.save("workspace/last-latents.npy", latent_selection.numpy())

    # get drum onsets
    sig = mm.audio.signal.Signal(args.audio_file, num_channels=1)
    sig_frames = mm.audio.signal.FramedSignal(sig, frame_size=2048, hop_size=441)
    stft = mm.audio.stft.ShortTimeFourierTransform(sig_frames, ciruclar_shift=True)
    spec = mm.audio.spectrogram.Spectrogram(stft, ciruclar_shift=True)

    kick_onset = get_onsets(spec, fmin=125, fmax=200, smooth=5, clip=99, power=1)
    # snare_onset = get_onsets(spec, fmin=250, fmax=350, smooth=5, clip=95, power=1)
    snare_onset = get_onsets(spec, fmin=1000, fmax=18000, smooth=7, clip=97, power=2)
    # plot_signals([kick_onset, snare_onset])  # , hats_onset])

    # plot_signals([sig])
    # sig, fs = rosa.load(args.audio_file)
    # S = rosa.feature.melspectrogram(y=sig, sr=fs)
    # rosa.display.specshow(rosa.power_to_db(S, ref=np.max))
    # plt.show()
    # rosa.display.specshow(rosa.feature.chroma_cqt(y=main_audio, sr=sr), y_axis="chroma", x_axis="time")
    # plt.show()
    # plt.close()
    # exit()

    # separate bass and main harmonic frequencies
    mid_chroma = get_chroma(
        signal.sosfilt(signal.butter(24, [100, 1000], "bp", fs=sr, output="sos"), main_audio), num_frames
    )
    # chromhalf = th.stack([mid_chroma[2 * i : 2 * i + 2].sum(0) for i in range(int(len(mid_chroma) / 2))])
    # chromhalf[chromhalf > 0.333] *= 2
    # chromhalf = gaussian_filter(chromhalf.T, 3, causal=0.1).T
    # # chromhalf = chromhalf ** 2
    # chromhalf /= chromhalf.sum(0)
    # fig, ax = plt.subplots(6, 1, figsize=(16, 9), sharey=True)
    # for i, ch in enumerate(chromhalf):
    #     ax[i].plot(ch.squeeze())
    #     ax[i].set_xlim(0, len(ch))
    #     ax[i].axis("off")
    # ax[i].axis("on")
    # ax[i].spines["top"].set_visible(False)
    # ax[i].spines["right"].set_visible(False)
    # ax[i].spines["left"].set_visible(False)
    # ax[i].axes.get_yaxis().set_visible(False)
    # ax[i].axes.xaxis.set_ticklabels([])
    # ax[i].set_xlabel("Time")
    # fig.text(0.04, 0.5, "Note Presence", va="center", rotation="vertical")
    # # plt.tight_layout()
    # plt.show()
    # exit()

    latents = get_chroma_latents(chroma=mid_chroma, base_latent_selection=latent_selection)
    latents = gaussian_filter(latents.float().cuda(), max(1, int(round(8 * smf))), causal=0.4).cpu()
    # latents = th.cat([latent_selection[[0]]]*num_frames, axis=0)

    # bass_chroma = get_chroma(
    #     signal.sosfilt(signal.butter(24 * 4, 72, "lp", fs=sr, output="sos"), main_audio), num_frames
    # )
    # bass_latents = get_chroma_latents(chroma=bass_chroma, base_latent_selection=wrapping_slice(latent_selection, 0, 12))
    # crossover = 3
    # latents[:, :crossover] = bass_latents[:, :crossover]

    high_mel = rosa.feature.melspectrogram(rosa.effects.harmonic(y=main_audio, margin=1), sr=sr, fmin=1000)
    high_onset = rosa.onset.onset_strength(S=high_mel, sr=sr)
    high_onset = th.from_numpy(signal.resample(high_onset, num_frames))
    high_onset = gaussian_filter(high_onset, 3 * smf, causal=0.3)
    high_onset = high_onset ** 2
    high_onset = percentile_clip(high_onset, 93)
    high_onset = gaussian_filter(high_onset, 2 * smf, causal=0.1)

    # # fig, ax = plt.subplots(2, 1, figsize=(8, 6))
    # # ax[0].plot(high_onset.squeeze())
    # # rosa.display.specshow(high_mel[:32], ax=ax[1])
    # # plt.tight_layout()
    # # plt.show()
    # # exit()

    # # latents = th.ones((num_frames, 1, 1)) * latent_selection[[1]]

    high_latents = (
        high_onset[:, None, None] * th.ones((num_frames, 1, 1)) * latent_selection[[-1]]
        + (1 - high_onset[:, None, None]) * latents
    )
    crossover = 11
    latents[:, :crossover] = high_latents[:, :crossover]

    latents = (
        0.5 * snare_onset[:, None, None] * th.ones((num_frames, 1, 1)) * latent_selection[[-4]]
        + (1 - 0.5 * snare_onset[:, None, None]) * latents
    )

    # moar_latents = get_latent_selection("workspace/cyphept-flat.npy")
    # if args.shuffle_latents:
    #     random_indices = random.sample(range(len(moar_latents)), len(moar_latents))
    #     moar_latents = moar_latents[random_indices]
    # intro_latents = get_chroma_latents(chroma=mid_chroma, base_latent_selection=wrapping_slice(moar_latents, 0, 12))
    # intro_latents = gaussian_filter(intro_latents.float().cuda(), max(1, int(round(10 * smf))), causal=0.1).cpu()

    # bass_audio = signal.sosfilt(signal.butter(24 * 4, 50, "lp", fs=sr, output="sos"), main_audio)
    # bass_spec = np.abs(rosa.stft(bass_audio))
    # bass_sum = bass_spec.sum(0)
    # bass_sum = np.clip(signal.resample(bass_sum, num_frames), bass_sum.min(), bass_sum.max())
    # bass_sum = percentile_clip(th.from_numpy(bass_sum).float(), 75) ** 2
    # bass_sum = gaussian_filter(bass_sum, 100 * smf, causal=True)

    # rms = rosa.feature.rms(S=np.abs(rosa.stft(y=main_audio, hop_length=512)))[0]
    # rms = np.clip(signal.resample(rms, num_frames), rms.min(), rms.max())
    # rms = percentile_clip(th.from_numpy(rms).float(), 75) ** 2
    # rms = gaussian_filter(rms, 100 * smf, causal=True)
    # # plot_signals([rms])

    # drop_weight = percentile_clip(rms + bass_sum, 55) ** 2
    # drop_weight[:700] = 0.1 * drop_weight[:700]
    # drop_weight = gaussian_filter(drop_weight, 20 * smf, causal=0)
    # plot_signals([drop_weight])

    # plot_signals([rms, bass_sum, drop_weight])
    # plot_signals([high_onset, snare_onset, kick_onset * drop_weight, drop_weight])

    # latents = drop_weight[:, None, None] * latents + (1 - drop_weight[:, None, None]) * intro_latents

    # smooth the final latents just a bit to prevent any jitter or jerks
    latents = gaussian_filter(latents.float().cuda(), max(1, int(round(4 * smf))), causal=0.1).cpu()

    class Translate(NetworkBend):
        def __init__(self, layer, batch):
            layer_h = 2 ** int(2 + math.ceil(layer / 2))
            layer_w = (2 if args.size == 1920 else 1) * 2 ** int(2 + math.ceil(layer / 2))
            sequential_fn = lambda b: th.nn.Sequential(
                th.nn.ReflectionPad2d((tl8_dist, tl8_dist, 0, 0)),
                # addNoise(smooth_noise),
                kT.Translate(b),
                kA.CenterCrop((layer_h, layer_w)),
            )
            super(Translate, self).__init__(sequential_fn, batch)

    class Zoom(NetworkBend):
        def __init__(self, layer, batch):
            layer_h = 2 ** int(2 + math.ceil(layer / 2))
            layer_w = (2 if args.size == 1920 else 1) * 2 ** int(2 + math.ceil(layer / 2))
            sequential_fn = lambda b: th.nn.Sequential(kT.Scale(b), kA.CenterCrop((layer_h, layer_w)))
            super(Zoom, self).__init__(sequential_fn, batch)

    tl = 4
    width = (2 if args.size == 1920 else 1) * 2 ** int(2 + math.ceil(tl / 2))
    print(width)

    tl8_dist = int(width / 8)
    print(tl8_dist)
    # smooth_noise = 0.3 * th.randn(size=(1, 1, 2 ** int(2 + math.ceil(tl / 2)), 2 * tl8_dist + width), device="cuda")

    translation = th.tensor([snare_onset.numpy() * tl8_dist, np.zeros(num_frames)]).float().T
    bends += [{"layer": tl, "transform": lambda batch, layer=tl: Translate(layer, batch), "modulation": translation}]

    zl = 6
    zoom = 0.5 * kick_onset + 1
    # zoom = 0.25 * (kick_onset * drop_weight) + 1
    bends += [{"layer": zl, "transform": lambda batch, layer=zl: Zoom(layer, batch), "modulation": zoom}]

    if log_min_res > 2:
        reflects = []
        for lres in range(2, log_min_res):
            half = 2 ** (lres - 1)
            reflects.append(th.nn.ReplicationPad2d((half, half, half, half)))
        transform = th.nn.Sequential(
            *reflects, addNoise(0.2 * th.randn(size=(1, 1, 2 ** log_min_res, 2 ** log_min_res), device="cuda"))
        )
        bends += [{"layer": 0, "transform": transform,}]
    if args.size == 1920:
        transform = th.nn.Sequential(
            th.nn.ReplicationPad2d((2, 2, 0, 0)),
            addNoise(0.05 * th.randn(size=(1, 1, 2 ** log_min_res, 2 * 2 ** log_min_res), device="cuda")),
        )
        bends += [{"layer": 0, "transform": transform}]

    rewrite_env = (
        th.cat(
            [
                th.zeros((int(len(latents) / 12))),
                th.linspace(0, 1, int(len(latents) / 12)) ** 1.75,
                th.linspace(1, 0, int(len(latents) / 12)) ** 3,
                th.linspace(0, 0.3, int(len(latents) / 24)),
                th.linspace(0.3, 1, int(len(latents) / 48)),
                th.linspace(1, 0, int(len(latents) / 48)),
                th.zeros((int(3 * len(latents) / 24))),
                th.linspace(0, 1, int(len(latents) / 48)),
                th.linspace(1, 0, int(len(latents) / 48)),
                th.zeros((int(len(latents) / 12))),
                1 - th.linspace(1, 0, int(len(latents) / 12)) ** 2,
                th.linspace(1, 0, int(len(latents) / 3)),
            ],
            axis=0,
        )
        .float()
        .contiguous()
        .pin_memory()
    )
    rewrite_env = th.cat([rewrite_env, th.zeros((len(latents) - len(rewrite_env)))]) ** 1.5

    orig_weights = [getattr(generator.convs, f"{i}").conv.weight.clone() for i in range(len(generator.convs)) if i <= 7]
    [print(ogw.shape) for ogw in orig_weights]
    _, filin, filout, kh, kw = orig_weights[0].shape

    rewrite_noise = gaussian_filter(th.randn((len(latents), filin * filout, kh, kw)) - 1, 3)
    print(rewrite_noise.min(), rewrite_noise.mean(), rewrite_noise.max())
    rewrite_noise = rewrite_noise.reshape((len(latents), filin, filout, kh, kw)).float().contiguous().pin_memory()

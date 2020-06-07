import gc
import uuid
import queue
import ffmpeg
import PIL.Image
import numpy as np
import torch as th
from tqdm import tqdm
from skimage import io
import librosa as rosa
import librosa.display
from threading import Thread
from wrappers import G_style, G_style2
from skimage.transform import resize

th.set_grad_enabled(False)


def info(arr):
    if isinstance(arr, list):
        print([(list(a.shape), f"{a.min():.2f}", f"{a.mean():.2f}", f"{a.max():.2f}") for a in arr])
    else:
        print(list(arr.shape), f"{arr.min():.2f}", f"{arr.mean():.2f}", f"{arr.max():.2f}")


def load_stylegan(checkpoint, size):
    generator = (G_style2 if not "stylegan1" in checkpoint else G_style)(checkpoint=checkpoint, output_size=size).eval()
    generator = th.nn.DataParallel(generator.cuda())
    return generator


def render(audio_file, generator, latents, noise, offset, duration, size, batch_size, output_file):
    output_file = (
        output_file
        if output_file is not None
        else f"output/{audio_file.split('/')[-1].split('.')[0]}-{uuid.uuid4().hex[:8]}.mp4"
    )

    split_queue = queue.Queue()
    render_queue = queue.Queue()

    def split_batches(jobs_in, jobs_out):
        while True:
            try:
                imgs = jobs_in.get(timeout=3)
            except queue.Empty:
                return
            imgs = (imgs.clamp_(-1, 1) + 1) * 127.5
            imgs = imgs.permute(0, 2, 3, 1)
            for img in imgs:
                jobs_out.put(img.cpu().numpy().astype(np.uint8))
            jobs_in.task_done()

    audio = ffmpeg.input(audio_file, ss=offset, to=offset + duration, guess_layout_max=0)
    video = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", framerate=len(latents) / duration, s="1920x1080")
        .output(
            audio,
            output_file,
            framerate=len(latents) / duration,
            vcodec="libx264",
            preset="slow",
            audio_bitrate="320K",
            ac=2,
            v="warning",
        )
        .global_args("-benchmark", "-stats", "-hide_banner")
        .overwrite_output()
        .run_async(pipe_stdin=True)  # , pipe_stdout=True, pipe_stderr=True)
    )

    # PBAR = tqdm(total=len(latents), smoothing=0.1)

    def make_video(jobs_in):
        for _ in range(len(latents)):
            img = jobs_in.get(timeout=3)
            if size == 1920:
                img = img[:, 112:-112, :]
                im = PIL.Image.fromarray(img)
                img = np.array(im.resize((1920, 1080), PIL.Image.BILINEAR))
            video.stdin.write(img.tobytes())
            # PBAR.update(1)
            jobs_in.task_done()
        video.stdin.close()
        video.wait()

    splitter = Thread(target=split_batches, args=(split_queue, render_queue))
    splitter.daemon = True
    renderer = Thread(target=make_video, args=(render_queue,))
    renderer.daemon = True

    latents = latents.float()
    noise = [(noise_scale.float() if noise_scale is not None else None) for noise_scale in noise]
    for n in range(0, len(latents), batch_size):
        latent_slice = latents[n : n + batch_size]
        noise_slice = [(noise_scale[n : n + batch_size] if noise_scale is not None else None) for noise_scale in noise]

        outputs = generator(latent=latent_slice, noise=noise_slice, truncation=1.5)

        split_queue.put(outputs)

        if n == 0:
            splitter.start()
            renderer.start()

    splitter.join()
    renderer.join()


def generate_video(
    audio_file, checkpoint, latents, noise, audio_offset, audio_duration, bpm, size, batch_size, output_file=None,
):
    generator = load_stylegan(checkpoint, size)
    render(audio_file, generator, latents, noise, audio_offset, audio_duration, size, batch_size, output_file)

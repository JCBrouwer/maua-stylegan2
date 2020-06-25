import uuid
import queue
import ffmpeg
import PIL.Image
import numpy as np
import torch as th
from threading import Thread

th.set_grad_enabled(False)


def render(generator, latents, noise, batch_size, duration, truncation=1, manipulations=[], output_file=None):
    output_file = output_file if output_file is not None else f"output/{uuid.uuid4().hex[:8]}.mp4"

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

    video = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", framerate=len(latents) / duration, s="1920x1080")
        .output(output_file, framerate=len(latents) / duration, vcodec="libx264", preset="slow", v="warning",)
        .global_args("-benchmark", "-stats", "-hide_banner")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    def make_video(jobs_in):
        for _ in range(len(latents)):
            img = jobs_in.get(timeout=3)
            if img.shape[1] == 2048:
                img = img[:, 112:-112, :]
                im = PIL.Image.fromarray(img)
                img = np.array(im.resize((1920, 1080), PIL.Image.BILINEAR))
            video.stdin.write(img.tobytes())
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

        outputs, _ = generator(
            styles=latent_slice,
            noise=noise_slice,
            truncation=truncation,
            transform_dict_list=manipulations,
            randomize_noise=True,
            input_is_latent=True,
        )
        print(outputs.min(), outputs.mean(), outputs.max(), outputs.shape)

        split_queue.put(outputs)

        if n == 0:
            splitter.start()
            renderer.start()

    splitter.join()
    renderer.join()

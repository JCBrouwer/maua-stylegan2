import gc, uuid, math
import queue
import ffmpeg
import PIL.Image
import numpy as np
import torch as th
import torch.utils.data as data
from threading import Thread

th.set_grad_enabled(False)
th.backends.cudnn.benchmark = True


class Print(th.nn.Module):
    def forward(self, x, *args, **kwargs):
        print(x.shape)
        return x


def render(
    generator,
    latents,
    noise,
    offset,
    duration,
    batch_size,
    out_size,
    audio_file=None,
    truncation=1,
    manipulations=[],
    output_file=None,
):
    output_file = output_file if output_file is not None else f"/home/hans/neurout/{uuid.uuid4().hex[:8]}.mp4"

    split_queue = queue.Queue()
    render_queue = queue.Queue()

    def split_batches(jobs_in, jobs_out):
        while True:
            try:
                imgs = jobs_in.get(timeout=10)
            except queue.Empty:
                return
            imgs = (imgs.clamp_(-1, 1) + 1) * 127.5
            imgs = imgs.permute(0, 2, 3, 1)
            for img in imgs:
                jobs_out.put(img.cpu().numpy().astype(np.uint8))
            jobs_in.task_done()

    res = "1024x1024" if out_size == 1024 else ("512x512" if out_size == 512 else "1920x1080")
    if audio_file is not None:
        audio = ffmpeg.input(audio_file, ss=offset, to=offset + duration, guess_layout_max=0)
        video = (
            ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", framerate=len(latents) / duration, s=res)
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
            .run_async(pipe_stdin=True)
        )
    else:
        video = (
            ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", framerate=len(latents) / duration, s=res)
            .output(output_file, framerate=len(latents) / duration, vcodec="libx264", preset="slow", v="warning",)
            .global_args("-benchmark", "-stats", "-hide_banner")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def make_video(jobs_in):
        for _ in range(len(latents)):
            img = jobs_in.get(timeout=10)
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

    latents = latents.float().pin_memory()

    for ni, noise_scale in enumerate(noise):
        noise[ni] = noise_scale.float().pin_memory() if noise_scale is not None else None

    for n in range(0, len(latents), batch_size):
        latent_batch = latents[n : n + batch_size].cuda()
        noise_batch = [
            (noise_scale[n : n + batch_size].cuda() if noise_scale is not None else None) for noise_scale in noise
        ]

        manipulation_batch = []
        if manipulations is not None:
            for manip in manipulations:
                if "params" in manip:
                    manipulation_batch.append(
                        {
                            "layer": manip["layer"],
                            "transform": manip["transform"](manip["params"][n : n + batch_size].cuda()),
                        }
                    )
                else:
                    manipulation_batch.append({"layer": manip["layer"], "transform": manip["transform"]})

        outputs, _ = generator(
            styles=latent_batch,
            noise=noise_batch,
            truncation=truncation,
            transform_dict_list=manipulation_batch,
            randomize_noise=False,
            input_is_latent=True,
        )

        split_queue.put(outputs)

        if n == 0:
            splitter.start()
            renderer.start()

    splitter.join()
    renderer.join()

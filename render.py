import uuid, math
import queue
import ffmpeg
import PIL.Image
import numpy as np
import torch as th
from threading import Thread
from functools import partial
import kornia.geometry.transform as kT
import kornia.augmentation as kA

th.set_grad_enabled(False)


def render(generator, latents, noise, batch_size, duration, truncation=1, manipulations=[], output_file=None):
    output_file = (
        output_file if output_file is not None else f"/home/hans/neurout/GAN Bending/{uuid.uuid4().hex[:8]}.mp4"
    )

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

    res = "1920x1080"
    video = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", framerate=24, s=res)
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

    latents = latents.float()
    noise = [(noise_scale.float() if noise_scale is not None else None) for noise_scale in noise]
    for n in range(0, len(latents), batch_size):
        latent_slice = latents[n : n + batch_size]
        noise_slice = [(noise_scale[n : n + batch_size] if noise_scale is not None else None) for noise_scale in noise]

        manipulation_slice = []
        for manip in manipulations:
            manipulation_slice.append({"layer": manip["layer"]})
            _, _, layer_h, layer_w = noise[manip["layer"]].shape
            if manip["transform"] == "translate":
                manipulation_slice[-1]["transform"] = th.nn.Sequential(
                    th.nn.ReflectionPad2d((int(layer_w / 2), int(layer_w / 2), 0, 0)),
                    th.nn.ReflectionPad2d((layer_w, layer_w, 0, 0)),
                    th.nn.ReflectionPad2d((layer_w, 0, 0, 0)),
                    kT.Translate(manip["params"][n : n + batch_size].cuda()),
                    kA.CenterCrop((layer_h, layer_w)),
                )
            elif manip["transform"] == "rotate":
                manipulation_slice[-1]["transform"] = th.nn.Sequential(
                    th.nn.ReflectionPad2d(int(max(layer_h, layer_w) * (1 - math.sqrt(2) / 2))),
                    kT.Rotate(manip["params"][n : n + batch_size].cuda()),
                    kA.CenterCrop((layer_h, layer_w)),
                )
            else:
                manipulation_slice[-1]["transform"] = manip["transform"]

        outputs, _ = generator(
            styles=latent_slice,
            noise=noise_slice,
            truncation=truncation,
            transform_dict_list=manipulation_slice,
            randomize_noise=False,
            input_is_latent=True,
        )
        # print(outputs.min(), outputs.mean(), outputs.max(), outputs.shape)

        split_queue.put(outputs)

        if n == 0:
            splitter.start()
            renderer.start()

    splitter.join()
    renderer.join()

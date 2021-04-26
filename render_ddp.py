import queue
import sys
import uuid
from math import ceil
from threading import Thread

import ffmpeg
import numpy as np
import PIL
import torch as th
import torch.nn.functional as F
from tqdm import tqdm

from audioreactive import gaussian_filter, spline_loops

sys.path.append("nvsg2a")
from nvsg2a import dnnlib, legacy

th.set_grad_enabled(False)
th.backends.cudnn.benchmark = True


@th.no_grad()
def render_ddp(synthesis, latents, noise, batch_size, duration, output_size, output_file):
    video = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", framerate=len(latents) / duration, s=output_size)
        .output(
            output_file,
            framerate=len(latents) / duration,
            vcodec="libx264",
            pix_fmt="yuv420p",
            preset="slow",
            v="warning",
        )
        .global_args("-hide_banner")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    def split_batches(jobs_in, jobs_out):
        while True:
            try:
                imgs = jobs_in.get(timeout=10)
            except queue.Empty:
                return
            imgs = (imgs.clamp_(-1, 1) + 1) * 127.5
            if imgs.shape[-1] == 2048:
                imgs = imgs[..., 112:-112]
            imgs = imgs.permute(0, 2, 3, 1)
            for img in imgs:
                jobs_out.put(img.cpu().numpy().astype(np.uint8))
            jobs_in.task_done()

    def make_video(jobs_in):
        w, h = [int(dim) for dim in output_size.split("x")]
        for _ in tqdm(range(len(latents)), position=0, leave=True, ncols=80):
            img = jobs_in.get(timeout=10)
            im = PIL.Image.fromarray(img)
            img = np.array(im.resize((1920, 1080), PIL.Image.BILINEAR))
            video.stdin.write(img.tobytes())
            jobs_in.task_done()
        video.stdin.close()
        video.wait()

    split_queue = queue.Queue()
    render_queue = queue.Queue()
    splitter = Thread(target=split_batches, args=(split_queue, render_queue))
    splitter.daemon = True
    renderer = Thread(target=make_video, args=(render_queue,))
    renderer.daemon = True

    latents = latents.float().contiguous().pin_memory()
    for ni, noise_scale in enumerate(noise):
        noise[ni] = noise_scale.float().contiguous().pin_memory() if noise_scale is not None else None

    for batch_idx in range(0, len(latents), batch_size):

        latent_slice = latents.narrow(0, batch_idx, min(batch_size, len(latents) - batch_idx)).cuda(non_blocking=True)

        i = 0
        for name, param in synthesis.named_parameters():
            if i == len(noise) and batch_idx != 0:
                break
            if "noise_strength" in name:
                net = synthesis
                for mod in name.split(".")[:-1]:
                    net = getattr(net, mod)
                setattr(
                    net,
                    "noise_const",
                    noise[i].narrow(0, batch_idx, min(batch_size, len(latents) - batch_idx)).cuda(non_blocking=True)
                    if i < len(noise)
                    else th.nn.ReflectionPad2d(
                        (int(net.noise_const.shape[-1] / 2), int(net.noise_const.shape[-1] / 2), 0, 0)
                    )(net.noise_const[None, None, ...]),
                )
                i += 1

        outputs = synthesis(latent_slice, noise_mode="const")

        if batch_idx == 0:
            splitter.start()
            renderer.start()

        split_queue.put(outputs)

    splitter.join()
    renderer.join()


if __name__ == "__main__":
    num_bars = int(np.random.choice([2, 4, 8]))
    duration = 4 / (172 / 60) * num_bars
    num_frames = int(24 * duration)
    output_size = "1920x1080"
    network_pkl = "/home/hans/modelzoo/00031-naomo-mirror-wav-gamma10.4858-resumeffhq1024/network-snapshot-000320.pkl"
    net = "naomo"

    print('Loading networks from "%s"...' % network_pkl)
    device = th.device("cuda")
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(device)  # type: ignore

    lats = th.randn(size=(ceil(num_bars / 4), 512), device="cuda")
    lats = G.mapping(lats, c=None).cpu().numpy()

    num_loops = int(np.random.choice([1, 2, 4]))
    print(num_bars, "bars\t", duration, "sec\t", num_loops, "loops")
    lats = spline_loops(lats, num_frames, num_loops)

    nois = []
    for scale in range(5, 7 * 2 + 2):
        nois.append(
            gaussian_filter(
                th.randn(size=(num_frames, 1, 2 ** int(scale / 2), 2 ** int(scale / 2)), device="cuda"), 5
            ).cpu()
        )

    del G.mapping, G.z_dim, G.c_dim, G.w_dim, G.img_resolution, G.img_channels, G.num_ws
    render_ddp(
        G.synthesis,
        lats,
        nois,
        24,
        duration,
        output_size,
        f"/home/hans/neurout/tvgf/nvsg2a-{net}-{str(uuid.uuid4())[:8]}.mp4",
    )

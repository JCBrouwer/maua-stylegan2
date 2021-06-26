import queue
import sys
import uuid
from threading import Thread

import ffmpeg
import numpy as np
import torch as th
from madmom.features import RNNOnsetProcessor
from madmom.features.tempo import TempoEstimationProcessor
from madmom.processors import IOProcessor, ParallelProcessor, SequentialProcessor, process_online
from PIL import Image
from tqdm import tqdm

np.set_printoptions(precision=3)


def zero_lag_ema(
    close, ec_var, ema_var, length=20.0, length_multiplier=1.0, gain_limit=500.0, gain_precision=100.0, dtype=th.float32
):
    alpha = 2.0 / ((length * length_multiplier) + 1.0)
    ec_1 = close if ec_var == 0 else ec_var
    ema_1 = close if ema_var == 0 else ema_var
    ema = alpha * close + (1.0 - alpha) * ema_1
    grid = th.arange(-gain_limit, gain_limit + 1.0, delta=1.0, dtype=dtype)
    gains = grid / gain_precision

    def fn(gain):
        ec = alpha * (ema + gain * (close - ec_1)) + (1.0 - alpha) * ec_1
        error = th.norm(close - ec)
        return error

    errors = th.vmap(fn, gains)
    least_error_idx = th.argmin(errors)
    best_gain = gains[least_error_idx]
    least_error = errors[least_error_idx]
    ec = alpha * (ema + best_gain * (close - ec_1)) + (1.0 - alpha) * ec_1
    return ec, ema


sys.path.append("nvsg2a")

# import nvsg2a.torch_utils.persistence
# def remove_shape_asserts(meta):
#     print("removing shape asserts")
#     meta.module_src = meta.module_src.replace("misc.assert_shape", "# misc.assert_shape")
#     return meta
# nvsg2a.torch_utils.persistence.import_hook(remove_shape_asserts)

import nvsg2a.dnnlib
import nvsg2a.legacy

th.set_grad_enabled(False)
th.backends.cudnn.benchmark = True

if __name__ == "__main__":
    device = th.device("cuda")
    num_bars = 24
    duration = num_bars * 4 / (172 / 60)
    num_frames = int(24 * duration)
    output_size = "1920x1080"
    fps = 24
    audio_file = "/home/hans/datasets/DONDERSLAG3/09 Mitekiss - Some People.wav"

    network_pkl = np.random.choice(
        [
            # "/home/hans/modelzoo/00005-stylegan2-neurout-2gpu-config-f/network-snapshot-000314.pkl",
            "/home/hans/modelzoo/00031-naomo-mirror-wav-gamma10.4858-resumeffhq1024/network-snapshot-000320.pkl",
            "/home/hans/modelzoo/00038-naomo-mirror-wav-resumeffhq1024/network-snapshot-000140.pkl",
            "/home/hans/modelzoo/00033-naomo-mirror-wav-gamma0.1-resumeffhq1024/network-snapshot-000120.pkl",
            "/home/hans/modelzoo/00045-naomo-mirror-wav-gamma500-resumeffhq1024/network-snapshot-000240.pkl",
        ]
    )
    net = network_pkl.split("/")[-2].split("-")[1]
    if net == "stylegan2":
        net = network_pkl.split("/")[-2].split("-")[2]
    output_file = f"/home/hans/neurout/{net}-{str(uuid.uuid4())[:8]}.mp4"

    with nvsg2a.dnnlib.util.open_url(network_pkl) as fp:
        G = nvsg2a.legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(device)

    double_width = th.nn.ReflectionPad2d((2, 2, 0, 0))
    a_lil_noise = 0.5 * th.randn(size=(1, 1, 4, 8), device=device)

    def fullHDpls(_, __, output):
        return double_width(output) + a_lil_noise

    G.synthesis.b4.conv1.register_forward_hook(fullHDpls)

    max_noise_power = 8
    powers = th.arange(3, max_noise_power)
    noise_numel = 4 * 8 + th.sum(2 ** (2 * powers + 2))

    noise_parents = []
    for name, param in G.synthesis.named_parameters():
        if "noise_strength" in name:
            net = G.synthesis
            for mod in name.split(".")[:-1]:
                net = getattr(net, mod)
            h, w = net.noise_const.shape
            if h > 4:
                setattr(net, "noise_const", th.randn(size=(1, 1, h, 2 * w), device=device))
                if w < 2 ** max_noise_power:
                    noise_parents.append(net)

    def process_audio(queue_out):
        def enqueue(events, f):
            onset, tempos = events
            # f.write(bytes(f"{onset[0]:.3f} {tempos[0, 0]:.1f}\n".encode("utf8")))
            # f.flush()
            queue_out.put((onset[0], tempos[0, 0]))

        onset_processor = RNNOnsetProcessor(fps=fps, origin="stream", num_frames=1, num_threads=1)
        tempo_processor = TempoEstimationProcessor(method="dbn", fps=fps, online=True)
        processor = ParallelProcessor([onset_processor, SequentialProcessor([onset_processor, tempo_processor])])
        process_online(
            processor=IOProcessor(processor, enqueue), infile=open(audio_file, "rb"), outfile=sys.stdout.buffer, fps=fps
        )

    audio_queue = queue.Queue()
    audio_thread = Thread(target=process_audio, args=(audio_queue,))
    audio_thread.daemon = True

    def gaussian_kernel(sigma, causal=None):
        radius = int(sigma * 4)
        kernel = th.arange(-radius, radius + 1, dtype=th.float32, device=device)
        kernel = th.exp(-0.5 / sigma ** 2 * kernel ** 2)
        if causal is not None:
            kernel[radius + 1 :] *= 0 if not isinstance(causal, float) else causal
        return kernel / kernel.sum()

    def slerp(val, low, high):
        omega = th.arccos(th.clamp(th.dot(low / th.norm(low, p=2), high / th.norm(high, p=2)), -1, 1))
        so = th.sin(omega)
        if so == 0:
            return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
        return th.sin((1.0 - val) * omega) / so * low + th.sin(val * omega) / so * high

    def generate_input(queue_in, queue_out):
        window = gaussian_kernel(5)[:, None]
        z = th.randn(size=(512,), device=device)
        onset_latent = G.mapping(th.randn(size=(1, 512), device=device), c=None)
        noise_buffer = th.randn(size=(len(window), noise_numel), device=device)
        for _ in range(num_frames):
            onset, tempo = queue_in.get()

            z = slerp(0.05, z, th.randn(size=(512,), device=device))
            latent = G.mapping(z[None, :], c=None)
            latent = onset * onset_latent + (1 - onset) * latent

            noise_buffer = th.roll(noise_buffer, shifts=1, dims=0)
            noise_buffer[-1] = th.randn(noise_numel, device=device)
            noise = th.sum(window * noise_buffer, axis=0)

            queue_out.put((latent, noise))

    inference_queue = queue.Queue()
    generate_thread = Thread(target=generate_input, args=(audio_queue, inference_queue))
    generate_thread.daemon = True

    def inference(queue_in, queue_out):
        for _ in range(num_frames):
            latent, noise = queue_in.get()

            idx = 0
            for parent in noise_parents:
                b, c, h, w = parent.noise_const.shape
                numel = b * c * h * w
                setattr(parent, "noise_const", noise[idx : idx + numel].reshape(b, c, h, w))
                idx += numel
                if idx == noise_numel:
                    break

            outputs = G.synthesis(latent, noise_mode="const")
            queue_out.put(outputs)

    split_queue = queue.Queue()
    inference_thread = Thread(target=inference, args=(inference_queue, split_queue))
    inference_thread.daemon = True

    def offload(queue_in, queue_out):
        for _ in range(num_frames):
            imgs = queue_in.get()
            imgs = (imgs.clamp_(-1, 1) + 1) * 127.5
            if imgs.shape[-1] == 2048:
                imgs = imgs[..., 112:-112]
            imgs = imgs.permute(0, 2, 3, 1)
            for img in imgs:
                queue_out.put(img.cpu().numpy().astype(np.uint8))
            queue_in.task_done()

    render_queue = queue.Queue()
    offload_thread = Thread(target=offload, args=(split_queue, render_queue))
    offload_thread.daemon = True

    video = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", framerate=num_frames / duration, s=output_size)
        .output(
            ffmpeg.input(audio_file, ss=0, t=duration, guess_layout_max=2),
            output_file,
            framerate=num_frames / duration,
            vcodec="libx264",
            pix_fmt="yuv420p",
            preset="slow",
            v="warning",
        )
        .global_args("-hide_banner")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    def render(queue_in):
        w, h = [int(dim) for dim in output_size.split("x")]
        for _ in tqdm(range(num_frames), position=0, leave=True, ncols=80):
            img = queue_in.get()
            im = Image.fromarray(img)
            img = np.array(im.resize((w, h), Image.BILINEAR))
            video.stdin.write(img.tobytes())
            queue_in.task_done()
        video.stdin.close()
        video.wait()

    render_thread = Thread(target=render, args=(render_queue,))
    render_thread.daemon = True

    audio_thread.start()
    generate_thread.start()
    inference_thread.start()
    offload_thread.start()
    render_thread.start()

    audio_thread.join()
    generate_thread.join()
    inference_thread.join()
    offload_thread.join()
    render_thread.join()

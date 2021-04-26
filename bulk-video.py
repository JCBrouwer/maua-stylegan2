import sys
import uuid

import numpy as np
import torch as th

from audioreactive import gaussian_filter, spline_loops
from render_ddp import render_ddp

sys.path.append("nvsg2a")
from nvsg2a import dnnlib, legacy

# import nvsg2a.torch_utils.persistence
# def remove_shape_asserts(meta):
#     meta.module_src = meta.module_src.replace("misc.assert_shape", "# misc.assert_shape")
#     return meta
# nvsg2a.torch_utils.persistence.import_hook(remove_shape_asserts)

th.set_grad_enabled(False)
th.backends.cudnn.benchmark = True

if __name__ == "__main__":
    device = th.device("cuda")
    num_bars = int(np.random.choice([4, 8, 16]))
    duration = 4 / (172 / 60) * num_bars
    num_frames = int(24 * duration)
    output_size = "1920x1080"

    network_pkl = np.random.choice(
        [
            # "/home/hans/modelzoo/-visionary/VisionaryArt.pkl",
            # "/home/hans/modelzoo/-visionary/VisionaryArt.pkl",
            # "/home/hans/modelzoo/-visionary/VisionaryArt.pkl",
            # "/home/hans/modelzoo/-visionary/VisionaryArt.pkl",
            # "/home/hans/modelzoo/00005-stylegan2-neurout-2gpu-config-f/network-snapshot-000314.pkl",
            # "/home/hans/modelzoo/00005-stylegan2-neurout-2gpu-config-f/network-snapshot-000221.pkl",
            "/home/hans/modelzoo/00005-stylegan2-neurout-2gpu-config-f/network-snapshot-000105.pkl",
            "/home/hans/modelzoo/00005-stylegan2-neurout-2gpu-config-f/network-snapshot-000424.pkl",
            # "/home/hans/modelzoo/00004-stylegan2-animacht-2gpu-config-f/network-snapshot-000393.pkl",
            # "/home/hans/modelzoo/00004-stylegan2-animacht-2gpu-config-f/network-snapshot-000240.pkl",
            # "/home/hans/modelzoo/00004-stylegan2-animacht-2gpu-config-f/network-snapshot-000135.pkl",
            # "/home/hans/modelzoo/00004-stylegan2-animacht-2gpu-config-f/network-snapshot-000086.pkl",
            # "/home/hans/modelzoo/00026-XXL-mirror-mirrory-aydao-resumecustom/network-snapshot-000564.pkl",
            # "/home/hans/modelzoo/00021-XXL-mirror-mirrory-aydao-resumecustom/network-snapshot-000241.pkl",
            # "/home/hans/modelzoo/00019-lyreca-mirror-mirrory-auto2-resumeffhq1024/network-snapshot-000288.pkl",
            "/home/hans/modelzoo/00006-stylegan2-lyreca-2gpu-config-f/network-snapshot-000019.pkl",
            # "/home/hans/modelzoo/00006-stylegan2-lyreca-2gpu-config-f/network-snapshot-000086.pkl",
            "/home/hans/modelzoo/00007-stylegan2-lyreca-2gpu-config-f/network-snapshot-000049.pkl",
            # "/home/hans/modelzoo/00008-stylegan2-dohr-2gpu-config-f/network-snapshot-000314.pkl",
            # "/home/hans/modelzoo/00008-stylegan2-dohr-2gpu-config-f/network-snapshot-000013.pkl",
            # "/home/hans/modelzoo/00008-stylegan2-dohr-2gpu-config-f/network-snapshot-000019.pkl",
            # "/home/hans/modelzoo/00008-stylegan2-dohr-2gpu-config-f/network-snapshot-000172.pkl",
            # "/home/hans/modelzoo/00031-naomo-mirror-wav-gamma10.4858-resumeffhq1024/network-snapshot-000320.pkl",
            # "/home/hans/modelzoo/00031-naomo-mirror-wav-gamma10.4858-resumeffhq1024/network-snapshot-000320.pkl",
            # "/home/hans/modelzoo/00038-naomo-mirror-wav-resumeffhq1024/network-snapshot-000140.pkl",
            # "/home/hans/modelzoo/00033-naomo-mirror-wav-gamma0.1-resumeffhq1024/network-snapshot-000120.pkl",
            # "/home/hans/modelzoo/00045-naomo-mirror-wav-gamma500-resumeffhq1024/network-snapshot-000240.pkl",
        ]
    )
    net = network_pkl.split("/")[-2].split("-")[1]
    if net == "stylegan2":
        net = network_pkl.split("/")[-2].split("-")[2]

    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(device)

    double_width = th.nn.ReflectionPad2d((2, 2, 0, 0))
    a_lil_noise = th.randn(size=(1, 1, 4, 8), device="cuda")

    def fullHDpls(self, input, output):
        return double_width(output) + a_lil_noise

    G.synthesis.b4.conv1.register_forward_hook(fullHDpls)

    lats = th.randn(size=(np.random.choice([3, 4, 5]), 512), device="cuda")
    lats = G.mapping(lats, c=None).cpu().numpy()
    lats = spline_loops(lats, num_frames, 1).float()

    style = th.randn(size=(1, 512), device="cuda")
    style = G.mapping(style, c=None)
    layer = 8
    lats[:, layer:] = style[:, layer:]
    lats = gaussian_filter(lats, 2)

    nois = []
    for scale in range(5, 8 * 2 + 2):
        nois.append(
            gaussian_filter(
                th.randn(
                    size=(num_frames, 1, 2 ** int(scale / 2), (2 if not scale == 5 else 1) * 2 ** int(scale / 2)),
                    device="cuda",
                ),
                5,
            ).cpu()
        )

    del G.mapping, G.z_dim, G.c_dim, G.w_dim, G.img_resolution, G.img_channels, G.num_ws
    render_ddp(
        synthesis=G.synthesis,
        latents=lats,
        noise=nois,
        batch_size=6,
        duration=duration,
        output_size=output_size,
        output_file=f"/home/hans/neurout/tvgf/{net}-{str(uuid.uuid4())[:8]}.mp4",
    )

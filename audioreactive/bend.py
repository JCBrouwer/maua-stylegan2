import math

import kornia.augmentation as kA
import kornia.geometry.transform as kT
import torch as th

# ====================================================================================
# ================================= network bending ==================================
# ====================================================================================


class NetworkBend(th.nn.Module):
    """Base network bending class

    Args:
        sequential_fn (function): Function that takes a batch of modulation and creates th.nn.Sequential
        modulation (th.tensor): Modulation batch
    """

    def __init__(self, sequential_fn, modulation):
        super(NetworkBend, self).__init__()
        self.sequential = sequential_fn(modulation)

    def forward(self, x):
        return self.sequential(x)


class AddNoise(th.nn.Module):
    """Adds static noise to output

    Args:
        noise (th.tensor): Noise to be added
    """

    def __init__(self, noise):
        super(AddNoise, self).__init__()
        self.noise = noise

    def forward(self, x):
        return x + self.noise.to(x.device)


class Print(th.nn.Module):
    """Prints intermediate feature statistics (useful for debugging complicated network bends)."""

    def forward(self, x):
        print(x.shape, [x.min().item(), x.mean().item(), x.max().item()], th.std(x).item())
        return x


class Translate(NetworkBend):
    """Creates horizontal translating effect where repeated linear interpolations from 0 to 1 (saw tooth wave) creates seamless scrolling effect.

    Args:
        modulation (th.tensor): [0.0-1.0]. Batch of modulation
        h (int): Height of intermediate features that the network bend is applied to
        w (int): Width of intermediate features that the network bend is applied to
        noise (int): Noise to be added (must be 5 * width wide)
    """

    def __init__(self, modulation, h, w, noise):
        sequential_fn = lambda b: th.nn.Sequential(
            th.nn.ReflectionPad2d((int(w / 2), int(w / 2), 0, 0)),
            th.nn.ReflectionPad2d((w, w, 0, 0)),
            th.nn.ReflectionPad2d((w, 0, 0, 0)),
            AddNoise(noise),
            kT.Translate(b),
            kA.CenterCrop((h, w)),
        )
        super(Translate, self).__init__(sequential_fn, modulation)


class Zoom(NetworkBend):
    """Creates zooming effect.

    Args:
        modulation (th.tensor): [0.0-1.0]. Batch of modulation
        h (int): height of intermediate features that the network bend is applied to
        w (int): width of intermediate features that the network bend is applied to
    """

    def __init__(self, modulation, h, w):
        padding = int(max(h, w)) - 1
        sequential_fn = lambda b: th.nn.Sequential(th.nn.ReflectionPad2d(padding), kT.Scale(b), kA.CenterCrop((h, w)))
        super(Zoom, self).__init__(sequential_fn, modulation)


class Rotate(NetworkBend):
    """Creates rotation effect.

    Args:
        modulation (th.tensor): [0.0-1.0]. Batch of modulation
        h (int): height of intermediate features that the network bend is applied to
        w (int): width of intermediate features that the network bend is applied to
    """

    def __init__(self, modulation, h, w):
        # worst case rotation brings sqrt(2) * max_side_length out-of-frame pixels into frame
        # padding should cover that exactly
        padding = int(max(h, w) * (1 - math.sqrt(2) / 2))
        sequential_fn = lambda b: th.nn.Sequential(th.nn.ReflectionPad2d(padding), kT.Rotate(b), kA.CenterCrop((h, w)))
        super(Rotate, self).__init__(sequential_fn, modulation)

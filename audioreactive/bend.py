import os, gc
import time, uuid, json, math
import argparse, random
import cachetools, warnings

import numpy as np
import scipy
from scipy import interpolate
import scipy.signal as signal
import sklearn.cluster

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import madmom as mm
import librosa as rosa
import librosa.display

import torch as th
import torch.nn.functional as F

import kornia.augmentation as kA
import kornia.geometry.transform as kT

import render
import generate
from models.stylegan1 import G_style
from models.stylegan2 import Generator

# ====================================================================================
# ================================= network bending ==================================
# ====================================================================================


class NetworkBend(th.nn.Module):
    def __init__(self, sequential_fn, modulation):
        super(NetworkBend, self).__init__()
        self.sequential = sequential_fn(modulation)

    def forward(self, x):
        return self.sequential(x)


class AddNoise(th.nn.Module):
    def __init__(self, noise):
        super(AddNoise, self).__init__()
        self.noise = noise

    def forward(self, x):
        return x + self.noise.to(x.device)


class Print(th.nn.Module):
    def forward(self, x):
        print(x.shape, [x.min().item(), x.mean().item(), x.max().item()], th.std(x).item())
        return x


class Translate(NetworkBend):
    def __init__(self, h, w, modulation):
        sequential_fn = lambda b: th.nn.Sequential(
            th.nn.ReflectionPad2d((int(w / 2), int(w / 2), 0, 0)),
            th.nn.ReflectionPad2d((w, w, 0, 0)),
            th.nn.ReflectionPad2d((w, 0, 0, 0)),
            AddNoise(th.randn((1, 1, h, 5 * w), device="cuda")),
            kT.Translate(b),
            kA.CenterCrop((h, w)),
        )
        super(Translate, self).__init__(sequential_fn, modulation)


class Zoom(NetworkBend):
    def __init__(self, h, w, modulation):
        padding = int(max(h, w)) - 1
        sequential_fn = lambda b: th.nn.Sequential(th.nn.ReflectionPad2d(padding), kT.Scale(b), kA.CenterCrop((h, w)))
        super(Zoom, self).__init__(sequential_fn, modulation)


class Rotate(NetworkBend):
    def __init__(self, h, w, modulation):
        # worst case rotation brings sqrt(2) * max_side_length out-of-frame pixels into frame
        # padding should cover that exactly
        padding = int(max(h, w) * (1 - math.sqrt(2) / 2))
        sequential_fn = lambda b: th.nn.Sequential(th.nn.ReflectionPad2d(padding), kT.Rotate(b), kA.CenterCrop((h, w)))
        super(Rotate, self).__init__(sequential_fn, modulation)

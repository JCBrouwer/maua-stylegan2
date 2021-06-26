"""
Describe your source and target class. These describe the direction of change you're trying to apply (e.g. "photo" to "sketch", "dog" to "the joker" or "dog" to "avocado dog").

For changes that do not require drastic shape modifications, we reccomend lambda_direction = 1.0, lambda_patch = 0.0, lambda_global = 0.0.

More drastic changes may require turning on the global loss (and / or modifying the number of iterations).


As a rule of thumb:
- Style and minor domain changes ('photo' -> 'sketch') require ~200-400 iterations.
- Identity changes ('person' -> 'taylor swift') require ~150-200 iterations.
- Simple in-domain changes ('face' -> 'smiling face') may require as few as 50.
"""

import os
import gc
from argparse import Namespace
from pathlib import Path

import torch
from tqdm import tqdm

from ZSSGAN.model.ZSSGAN import ZSSGAN
from ZSSGAN.utils.file_utils import save_images

output_dir = f"/home/hans/modelzoo/nada/"

torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

source_class = "abstract art"
target_class = "black and white vector line art"

lambda_direction = 1.0
lambda_patch = 0
lambda_global = 0

training_iterations = 151
output_interval = 50
truncation = 0.7

checkpoint_path = "/home/hans/modelzoo/maua-sg2/cyphept-CYPHEPT-2q5b2lk6-33-1024-145000.pt"
name = Path(checkpoint_path).stem + "-" + target_class.replace(" ", "_")

training_args = {
    "size": 1024,
    "batch": 2,
    "n_sample": 6,
    "output_dir": output_dir,
    "lr": 0.002,
    "frozen_gen_ckpt": checkpoint_path,
    "train_gen_ckpt": checkpoint_path,
    "iter": training_iterations,
    "source_class": source_class,
    "target_class": target_class,
    "lambda_direction": lambda_direction,
    "lambda_patch": lambda_patch,
    "lambda_global": lambda_global,
    "phase": None,
    "sample_truncation": truncation,
}
args = Namespace(**training_args)

os.makedirs(output_dir, exist_ok=True)

net = ZSSGAN(args)

g_reg_ratio = 4 / 5
g_optim = torch.optim.Adam(
    net.generator_trainable.parameters(), lr=args.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
)

fixed_z = torch.randn(args.n_sample, 512, device=device)

for i in tqdm(range(args.iter)):

    sample_z = torch.randn(args.batch, 512, device=device)
    [sampled_src, sampled_dst], [cycle_dst, cycle_src], clip_loss, cycle_loss = net([sample_z])

    net.zero_grad()
    clip_loss.backward()
    g_optim.step()

    if i % output_interval == 0:
        with torch.no_grad():
            [sampled_src, sampled_dst], [cycle_dst, cycle_src], clip_loss, cycle_loss = net(
                [fixed_z], truncation=args.sample_truncation
            )
            save_images(sampled_dst, args.output_dir, f"{name}_{str(i).zfill(4)}", 3)
            state_dict = net.generator_trainable.state_dict()
            state_dict = {k.replace("generator.", ""): v for k, v in state_dict.items()}
            torch.save(state_dict, f"{output_dir}/{name}_{str(i).zfill(4)}.pt")
            del sampled_src, sampled_dst, cycle_dst, cycle_src, clip_loss, cycle_loss
            gc.collect()
            torch.cuda.empty_cache()

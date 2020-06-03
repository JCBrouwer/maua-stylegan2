import argparse
import numpy as np
import multiprocessing
from functools import partial

import lmdb
from tqdm import tqdm

import torch as th
from autoencoder import ConvSegNet
from torchvision import datasets, utils
import torchvision.transforms as transforms


def lmdmb_write_worker(i_code, env, size):
    i, code = i_code.cpu().numpy()
    key = f"{size}-{str(i).zfill(5)}".encode("utf-8")
    with env.begin(write=True) as txn:
        txn.put(key, code)


def prepare(env, vae, loader, total, batch_size, n_worker, size=1024):
    write_fn = partial(lmdmb_write_worker, env=env, size=size)

    b = 0
    with multiprocessing.Pool(n_worker) as pool:
        for batch in tqdm(loader):
            code_nums = np.arange(b * batch_size, (b + 1) * batch_size)

            with th.no_grad():
                codes = vae.module.encode(batch[0].cuda())

            pool.imap_unordered(write_fn, zip(code_nums, codes))

            b += 1

    with env.begin(write=True) as txn:
        txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--n_worker", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--resample", type=str, default="bilinear")
    parser.add_argument("data_path", type=str)
    parser.add_argument("vae_checkpoint", type=str)

    args = parser.parse_args()

    print(f"Make dataset of image size:", args.size)

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    imgset = datasets.ImageFolder(args.data_path, transform=transform)
    loader = th.utils.data.DataLoader(imgset, batch_size=args.batch_size, num_workers=int(args.n_worker / 2))
    print(args.batch_size)
    print(loader)

    vae = ConvSegNet()
    vae.load_state_dict(th.load(args.vae_checkpoint)["vae"])
    vae = th.nn.DataParallel(vae).eval().cuda()

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare(
            env,
            vae,
            loader,
            total=len(imgset),
            batch_size=args.batch_size,
            n_worker=int(args.n_worker / 2),
            size=args.size,
        )

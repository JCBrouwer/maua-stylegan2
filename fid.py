import argparse
import pickle

import os
import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm

from model import Generator
from inception import InceptionV3


@torch.no_grad()
def extract_feature_from_samples(generator, inception, truncation, truncation_latent, batch_size, n_sample, device):
    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in tqdm(batch_sizes):
        latent = torch.randn(batch, 512, device=device)
        img, _ = g([latent], truncation=truncation, truncation_latent=truncation_latent)
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to("cpu"))

    features = torch.cat(features, 0)

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print("product of cov matrices is singular")
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f"Imaginary component {m}")

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


def get_dataset_inception_features(loader, path, inception_name, size):
    if not os.path.exists(f"inception_{inception_name}.pkl"):
        print("calculating inception features for FID....")
        inception = InceptionV3([3], normalize_input=False, init_weights=False)
        inception = nn.DataParallel(inception).eval().cuda()

        feature_list = []
        for img in tqdm(loader):
            img = img.cuda()
            feature = inception(img)[0].view(img.shape[0], -1)
            feature_list.append(feature.to("cpu"))
        features = torch.cat(feature_list, 0).numpy()

        mean = np.mean(features, 0)
        cov = np.cov(features, rowvar=False)

        with open(f"inception_{inception_name}.pkl", "wb") as f:
            pickle.dump({"mean": mean, "cov": cov, "size": size, "path": path}, f)
    else:
        print(f"Found inception features: inception_{inception_name}.pkl")


def validation_fid(generator, batch_size, n_sample, truncation, inception_name):
    generator.eval()

    with torch.no_grad():
        mean_latent = generator.mean_latent(2 ** 14)

        inception = InceptionV3([3], normalize_input=False, init_weights=False)
        inception = inception.eval().to(next(generator.parameters()).device)

        n_batch = n_sample // batch_size
        resid = n_sample - (n_batch * batch_size)
        if resid == 0:
            batch_sizes = [batch_size] * n_batch
        else:
            batch_sizes = [batch_size] * n_batch + [resid]
        features = []

        for batch in batch_sizes:
            latent = torch.randn(batch, 512).cuda()
            img, _ = generator([latent], truncation=truncation, truncation_latent=mean_latent)
            feat = inception(img)[0].view(img.shape[0], -1)
            features.append(feat.to("cpu"))
        features = torch.cat(features, 0).numpy()

        sample_mean = np.mean(features, 0)
        sample_cov = np.cov(features, rowvar=False)

        with open(f"inception_{inception_name}.pkl", "rb") as f:
            embeds = pickle.load(f)
            real_mean = embeds["mean"]
            real_cov = embeds["cov"]

        cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

        if not np.isfinite(cov_sqrt).all():
            print("product of cov matrices is singular")
            offset = np.eye(sample_cov.shape[0]) * 1e-6
            cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

        if np.iscomplexobj(cov_sqrt):
            if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_sqrt.imag))

                raise ValueError(f"Imaginary component {m}")

            cov_sqrt = cov_sqrt.real

        mean_diff = sample_mean - real_mean
        mean_norm = mean_diff @ mean_diff

        trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

        fid = mean_norm + trace

        del inception

    return torch.tensor(fid)


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()

    parser.add_argument("--truncation", type=float, default=1)
    parser.add_argument("--truncation_mean", type=int, default=4096 * 8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--n_sample", type=int, default=50000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--inception", type=str, default=None, required=True)
    parser.add_argument("ckpt", metavar="CHECKPOINT")

    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)

    g = Generator(args.size, 512, 8).to(device)
    g.load_state_dict(ckpt["g_ema"])
    g = nn.DataParallel(g)
    g.eval()

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g.mean_latent(args.truncation_mean)

    else:
        mean_latent = None

    inception = InceptionV3([3], normalize_input=False, init_weights=False)
    inception = nn.DataParallel(inception).eval().cuda()

    features = extract_feature_from_samples(
        g, inception, args.truncation, mean_latent, args.batch, args.n_sample, device
    ).numpy()
    print(f"extracted {features.shape[0]} features")

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    with open(args.inception, "rb") as f:
        embeds = pickle.load(f)
        real_mean = embeds["mean"]
        real_cov = embeds["cov"]

    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

    print("fid:", fid)

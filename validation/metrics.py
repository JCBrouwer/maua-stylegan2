import os
import pickle

from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import torch
from torch.nn import functional as F
import numpy as np
from scipy import linalg

from .inception import InceptionV3
from . import lpips


@torch.no_grad()
def vae_fid(vae, batch_size, latent_dim, n_sample, inception_name, calculate_prdc=True):
    vae.eval()

    inception = InceptionV3([3], normalize_input=False, init_weights=False)
    inception = inception.eval().to(next(vae.parameters()).device)

    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    if resid == 0:
        batch_sizes = [batch_size] * n_batch
    else:
        batch_sizes = [batch_size] * n_batch + [resid]
    features = []

    for batch in batch_sizes:
        latent = torch.randn(batch, *latent_dim).cuda()
        img = vae.decode(latent)
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to("cpu"))
    features = torch.cat(features, 0).numpy()

    del inception

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    with open(f"inception_{inception_name}_stats.pkl", "rb") as f:
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

    ret_dict = {"FID": fid}

    if calculate_prdc:
        with open(f"inception_{inception_name}_features.pkl", "rb") as f:
            embeds = pickle.load(f)
            real_feats = embeds["features"]
        _, _, density, coverage = prdc(real_feats[:80000], features[:80000])
        ret_dict["Density"] = density
        ret_dict["Coverage"] = coverage

    return ret_dict


@torch.no_grad()
def fid(generator, batch_size, n_sample, truncation, inception_name, calculate_prdc=True):
    generator.eval()
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

    del inception

    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)

    with open(f"inception_{inception_name}_stats.pkl", "rb") as f:
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

    ret_dict = {"FID": fid}

    if calculate_prdc:
        with open(f"inception_{inception_name}_features.pkl", "rb") as f:
            embeds = pickle.load(f)
            real_feats = embeds["features"]
        _, _, density, coverage = prdc(real_feats[:80000], features[:80000])
        ret_dict["Density"] = density
        ret_dict["Coverage"] = coverage

    return ret_dict


def get_dataset_inception_features(loader, inception_name, size):
    if not os.path.exists(f"inception_{inception_name}_stats.pkl"):
        print("calculating inception features for FID....")
        inception = InceptionV3([3], normalize_input=False, init_weights=False)
        inception = torch.nn.DataParallel(inception).eval().cuda()

        feature_list = []
        for img in tqdm(loader):
            img = img.cuda()
            feature = inception(img)[0].view(img.shape[0], -1)
            feature_list.append(feature.to("cpu"))
        features = torch.cat(feature_list, 0).numpy()

        mean = np.mean(features, 0)
        cov = np.cov(features, rowvar=False)

        with open(f"inception_{inception_name}_stats.pkl", "wb") as f:
            pickle.dump({"mean": mean, "cov": cov, "size": size, "feat": features}, f)
        with open(f"inception_{inception_name}_features.pkl", "wb") as f:
            pickle.dump({"features": features}, f)
    else:
        print(f"Found inception features: inception_{inception_name}_stats.pkl")


def compute_pairwise_distance(data_x, data_y=None, metric="l2"):
    if data_y is None:
        data_y = data_x
    dists = pairwise_distances(
        data_x.reshape((len(data_x), -1)), data_y.reshape((len(data_y), -1)), metric=metric, n_jobs=24
    )
    return dists


def get_kth_value(unsorted, k, axis=-1):
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k, metric):
    distances = compute_pairwise_distance(input_features, metric=metric)
    radii = get_kth_value(distances, k=nearest_k + 1, axis=-1)
    return radii


def prdc(real_features, fake_features, nearest_k=10, metric="l2"):
    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(real_features, nearest_k, metric=metric)
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(fake_features, nearest_k, metric=metric)
    distance_real_fake = compute_pairwise_distance(real_features, fake_features, metric=metric)

    precision = (distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)).any(axis=0).mean()
    recall = (distance_real_fake < np.expand_dims(fake_nearest_neighbour_distances, axis=0)).any(axis=1).mean()

    density = (1.0 / float(nearest_k)) * (
        distance_real_fake < np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()
    coverage = (distance_real_fake.min(axis=1) < real_nearest_neighbour_distances).mean()

    return precision, recall, density, coverage


def lerp(a, b, t):
    return a + (b - a) * t


@torch.no_grad()
def ppl(generator, batch_size, n_sample, space, crop, latent_dim, eps=1e-4):
    generator.eval()

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=True, gpu_ids=[next(generator.parameters()).device.index]
    )

    distances = []

    n_batch = n_sample // batch_size
    resid = n_sample - (n_batch * batch_size)
    if resid == 0:
        batch_sizes = [batch_size] * n_batch
    else:
        batch_sizes = [batch_size] * n_batch + [resid]

    for batch_size in batch_sizes:
        noise = generator.make_noise()

        inputs = torch.randn([batch_size * 2, latent_dim]).cuda()
        lerp_t = torch.rand(batch_size).cuda()

        if space == "w":
            latent = generator.get_latent(inputs)
            latent_t0, latent_t1 = latent[::2], latent[1::2]
            latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None])
            latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None] + eps)
            latent_e = torch.stack([latent_e0, latent_e1], 1).view(*latent.shape)

        image, _ = generator(latent_e, input_is_latent=True, noise=noise)

        if crop:
            c = image.shape[2] // 8
            image = image[:, :, c * 3 : c * 7, c * 2 : c * 6]

        factor = image.shape[2] // 256

        if factor > 1:
            image = F.interpolate(image, size=(256, 256), mode="bilinear", align_corners=False)

        dist = percept(image[::2], image[1::2]).view(image.shape[0] // 2) / (eps ** 2)
        distances.append(dist.to("cpu").numpy())

    distances = np.concatenate(distances, 0)

    lo = np.percentile(distances, 1, interpolation="lower")
    hi = np.percentile(distances, 99, interpolation="higher")
    filtered_dist = np.extract(np.logical_and(lo <= distances, distances <= hi), distances)
    path_length = filtered_dist.mean()

    del percept, inputs, lerp_t, image, dist

    return path_length

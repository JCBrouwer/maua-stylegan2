import os
import gc
import wandb
import argparse
import validation
import torch as th
from tqdm import tqdm
from torch.utils import data
from autoencoder import LogCoshVAE
from dataset import MultiResolutionDataset
from torchvision import transforms, utils, models


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class VGG19(th.nn.Module):
    """
    Adapted from https://github.com/NVIDIA/pix2pixHD
    See LICENSE-VGG
    """

    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = th.nn.Sequential()
        self.slice2 = th.nn.Sequential()
        self.slice3 = th.nn.Sequential()
        self.slice4 = th.nn.Sequential()
        self.slice5 = th.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(th.nn.Module):
    """
    Adapted from https://github.com/NVIDIA/pix2pixHD
    See LICENSE-VGG
    """

    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = th.nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


device = "cuda"
th.backends.cudnn.benchmark = True

wandb.init(project=f"maua-stylegan")


def train(latent_dim, learning_rate, number_filters, vae_alpha, vae_beta, kl_divergence_weight):
    print(
        f"latent_dim={latent_dim}",
        f"learning_rate={learning_rate}",
        f"number_filters={number_filters}",
        f"vae_alpha={vae_alpha}",
        f"vae_beta={vae_beta}",
        f"kl_divergence_weight={kl_divergence_weight}",
    )

    batch_size = 64
    i = None
    while batch_size >= 1:
        try:
            transform = transforms.Compose(
                [
                    transforms.Resize(128),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
                ]
            )
            data_path = "/home/hans/trainsets/cyphis"
            name = os.path.splitext(os.path.basename(data_path))[0]
            dataset = MultiResolutionDataset(data_path, transform, 256)
            dataloader = data.DataLoader(
                dataset,
                batch_size=int(batch_size),
                sampler=data_sampler(dataset, shuffle=True, distributed=False),
                num_workers=12,
                drop_last=True,
            )
            loader = sample_data(dataloader)
            sample_imgs = next(loader)[:24]
            wandb.log(
                {"Real Images": [wandb.Image(utils.make_grid(sample_imgs, nrow=6, normalize=True, range=(-1, 1)))]}
            )

            hidden_dims = [min(int(number_filters) * 2 ** i, latent_dim) for i in range(5)] + [latent_dim]
            vae, vae_optim = None, None
            vae = LogCoshVAE(
                3, latent_dim, hidden_dims=hidden_dims, alpha=vae_alpha, beta=vae_beta, kld_weight=kl_divergence_weight,
            ).to(device)
            vae.train()
            vae_optim = th.optim.Adam(vae.parameters(), lr=learning_rate)

            mse_loss = th.nn.MSELoss()
            vgg = VGGLoss()

            sample_z = th.randn(size=(24, latent_dim))

            scores = []
            num_iters = 100_000
            pbar = range(num_iters)
            pbar = tqdm(pbar, smoothing=0.1)
            for i in pbar:
                vae.train()

                real = next(loader).to(device)
                fake, mu, log_var = vae(real)

                loss_dict = vae.loss(real, fake, mu, log_var)
                vgg_loss = vgg(fake, real)
                loss = loss_dict["Total"] + vgg_loss

                vae.zero_grad()
                loss.backward()
                vae_optim.step()

                wandb.log(
                    {
                        "Total": loss,
                        "VGG": vgg_loss,
                        "Reconstruction": loss_dict["Reconstruction"],
                        "Kullback Leibler Divergence": loss_dict["Kullback Leibler Divergence"],
                    }
                )

                if i % int(num_iters / 1000) == 0 or i + 1 == num_iters:
                    with th.no_grad():
                        vae.eval()

                        sample, _, _ = vae(sample_imgs.to(device))
                        grid = utils.make_grid(sample, nrow=6, normalize=True, range=(-1, 1),)
                        del sample
                        wandb.log({"Reconstructed Images VAE": [wandb.Image(grid, caption=f"Step {i}")]})

                        sample = vae.decode(sample_z.to(device))
                        grid = utils.make_grid(sample, nrow=6, normalize=True, range=(-1, 1),)
                        del sample
                        wandb.log({"Generated Images VAE": [wandb.Image(grid, caption=f"Step {i}")]})

                if i % int(num_iters / 40) == 0 or i + 1 == num_iters:
                    with th.no_grad():
                        fid_dict = validation.vae_fid(vae, int(batch_size), (latent_dim,), 5000, name)
                        wandb.log(fid_dict)
                        mse = mse_loss(fake, real) * 5000
                        score = fid_dict["FID"] + mse + 1000 * vgg_loss
                        wandb.log({"Score": score})
                        pbar.set_description(f"FID: {fid_dict['FID']:.2f} MSE: {mse:.2f} VGG: {1000 * vgg_loss:.2f}")

                    if i >= num_iters / 2:
                        scores.append(score)

                if th.isnan(loss).any() or th.isinf(loss).any():
                    print("NaN losses, exiting...")
                    print(
                        {
                            "Total": loss.detach().cpu().item(),
                            "\nVGG": vgg_loss.detach().cpu().item(),
                            "\nReconstruction": loss_dict["Reconstruction"].detach().cpu().item(),
                            "\nKullback Leibler Divergence": loss_dict["Kullback Leibler Divergence"]
                            .detach()
                            .cpu()
                            .item(),
                        }
                    )
                    wandb.log({"Score": 27000})
                    return

            return

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                batch_size = batch_size / 2

                if batch_size < 1:
                    print("This configuration does not fit into memory, exiting...")
                    wandb.log({"Score": 27000})
                    return

                print(f"Out of memory, halving batch size... {batch_size}")
                if vae is not None:
                    del vae
                if vae_optim is not None:
                    del vae_optim
                gc.collect()
                th.cuda.empty_cache()

            else:
                print(e)
                return


parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=1024)
parser.add_argument("--learning_rate", type=float, default=0.005)
parser.add_argument("--number_filters", type=int, default=64)
parser.add_argument("--vae_alpha", type=float, default=10.0)
parser.add_argument("--vae_beta", type=float, default=1.0)
parser.add_argument("--kl_divergence_weight", type=float, default=1.0)
args = parser.parse_args()

train(
    args.latent_dim, args.learning_rate, args.number_filters, args.vae_alpha, args.vae_beta, args.kl_divergence_weight,
)


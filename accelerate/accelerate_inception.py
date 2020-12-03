import os
import gc
import wandb
import argparse
import torch as th
from tqdm import tqdm
from torch.utils import data
import torch.nn.functional as F
from inception_vae import InceptionVAE
from dataset import MultiResolutionDataset
from torchvision import transforms, utils, models


def info(x):
    print(x.shape, x.detach().cpu().min(), x.detach().cpu().mean(), x.detach().cpu().max())


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


def train(latent_dim, num_repeats, learning_rate, lambda_vgg, lambda_mse):
    print(
        f"latent_dim={latent_dim:.4f}",
        f"num_repeats={num_repeats:.4f}",
        f"learning_rate={learning_rate:.4f}",
        f"lambda_vgg={lambda_vgg:.4f}",
        f"lambda_mse={lambda_mse:.4f}",
    )

    transform = transforms.Compose(
        [
            transforms.Resize(128),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    batch_size = 72
    data_path = "/home/hans/trainsets/cyphis"
    name = os.path.splitext(os.path.basename(data_path))[0]
    dataset = MultiResolutionDataset(data_path, transform, 256)
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size, sampler=data.RandomSampler(dataset), num_workers=12, drop_last=True,
    )
    loader = sample_data(dataloader)
    sample_imgs = next(loader)[:24]
    wandb.log({"Real Images": [wandb.Image(utils.make_grid(sample_imgs, nrow=6, normalize=True, range=(0, 1)))]})

    vae, vae_optim = None, None
    vae = InceptionVAE(latent_dim=latent_dim, repeat_per_block=num_repeats).to(device)
    vae_optim = th.optim.Adam(vae.parameters(), lr=learning_rate)

    vgg = VGGLoss()

    # sample_z = th.randn(size=(24, 512))

    scores = []
    num_iters = 100_000
    pbar = tqdm(range(num_iters), smoothing=0.1)
    for i in pbar:
        vae.train()

        real = next(loader).to(device)

        fake, mu, log_var = vae(real)

        bce = F.binary_cross_entropy(fake, real, size_average=False)
        kld = -0.5 * th.sum(1 + log_var - mu.pow(2) - log_var.exp())
        vgg_loss = vgg(fake, real)
        mse_loss = th.sqrt((fake - real).pow(2).mean())

        loss = bce + kld + lambda_vgg * vgg_loss + lambda_mse * mse_loss

        loss_dict = {
            "Total": loss,
            "BCE": bce,
            "Kullback Leibler Divergence": kld,
            "MSE": mse_loss,
            "VGG": vgg_loss,
        }

        vae.zero_grad()
        loss.backward()
        vae_optim.step()

        wandb.log(loss_dict)

        with th.no_grad():
            if i % int(num_iters / 100) == 0 or i + 1 == num_iters:
                vae.eval()

                sample, _, _ = vae(sample_imgs.to(device))
                grid = utils.make_grid(sample, nrow=6, normalize=True, range=(0, 1))
                del sample
                wandb.log({"Reconstructed Images VAE": [wandb.Image(grid, caption=f"Step {i}")]})

                sample = vae.sampling()
                grid = utils.make_grid(sample, nrow=6, normalize=True, range=(0, 1))
                del sample
                wandb.log({"Generated Images VAE": [wandb.Image(grid, caption=f"Step {i}")]})

                gc.collect()
                th.cuda.empty_cache()

                th.save(
                    {"vae": vae.state_dict(), "vae_optim": vae_optim.state_dict()},
                    f"/home/hans/modelzoo/maua-sg2/vae-{name}-{wandb.run.dir.split('/')[-1].split('-')[-1]}.pt",
                )

        if th.isnan(loss).any() or th.isinf(loss).any():
            print("NaN losses, exiting...")
            print(
                {
                    "Total": loss,
                    "\nBCE": bce,
                    "\nKullback Leibler Divergence": kld,
                    "\nMSE": mse_loss,
                    "\nVGG": vgg_loss,
                }
            )
            wandb.log({"Total": 27000})
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=float, default=512)
    parser.add_argument("--num_repeats", type=float, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--lambda_vgg", type=float, default=1.0)
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    args = parser.parse_args()

    device = "cuda"
    th.backends.cudnn.benchmark = True

    wandb.init(project=f"maua-stylegan")

    train(
        args.latent_dim, args.num_repeats, args.learning_rate, args.lambda_vgg, args.lambda_mse,
    )


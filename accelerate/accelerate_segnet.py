import os
import gc
import wandb
import argparse
import torch as th
from tqdm import tqdm
from torch.utils import data
from autoencoder import ConvSegNet
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


def align(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform(x, t=2):
    return (th.pdist(x.view(x.size(0), -1), p=2).pow(2).mul(-t).exp().mean() + 1e-27).log()


def train(learning_rate, lambda_mse):
    print(
        f"learning_rate={learning_rate:.4f}", f"lambda_mse={lambda_mse:.4f}",
    )

    transform = transforms.Compose(
        [
            transforms.Resize(128),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
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
    wandb.log({"Real Images": [wandb.Image(utils.make_grid(sample_imgs, nrow=6, normalize=True, range=(-1, 1)))]})

    vae, vae_optim = None, None
    vae = ConvSegNet().to(device)
    vae_optim = th.optim.Adam(vae.parameters(), lr=learning_rate)

    vgg = VGGLoss()

    sample_z = th.randn(size=(24, 512, 16, 16))
    sample_z /= sample_z.abs().max()

    scores = []
    num_iters = 100_000
    pbar = tqdm(range(num_iters), smoothing=0.1)
    for i in pbar:
        vae.train()

        real = next(loader).to(device)

        z = vae.encode(real)
        fake = vae.decode(z)

        vgg_loss = vgg(fake, real)

        mse_loss = th.sqrt((fake - real).pow(2).mean())

        # diff = fake - real
        # recons_loss = recons_alpha * diff + th.log(1.0 + th.exp(-2 * recons_alpha * diff)) - th.log(th.tensor(2.0))
        # recons_loss = (1.0 / recons_alpha) * recons_loss.mean()
        # recons_loss = recons_loss if not th.isinf(recons_loss).any() else 0

        # x, y = z.chunk(2)
        # align_loss = align(x, y, alpha=align_alpha)
        # unif_loss = -(uniform(x, t=unif_t) + uniform(y, t=unif_t)) / 2.0

        loss = (
            vgg_loss
            + lambda_mse * mse_loss
            # + lambda_recons * recons_loss
            # + lambda_align * align_loss
            # + lambda_unif * unif_loss
        )
        # print(vgg_loss.detach().cpu().item())
        # print(lambda_mse * mse_loss.detach().cpu().item())
        # # print(lambda_recons * recons_loss.detach().cpu().item())
        # print(lambda_align * align_loss.detach().cpu().item())
        # print(lambda_unif * unif_loss.detach().cpu().item())

        loss_dict = {
            "Total": loss,
            "MSE": mse_loss,
            "VGG": vgg_loss,
            # "Reconstruction": recons_loss,
            # "Alignment": align_loss,
            # "Uniformity": unif_loss,
        }

        vae.zero_grad()
        loss.backward()
        vae_optim.step()

        wandb.log(loss_dict)
        # pbar.set_description(" ".join())

        with th.no_grad():
            if i % int(num_iters / 100) == 0 or i + 1 == num_iters:
                vae.eval()

                sample = vae(sample_imgs.to(device))
                grid = utils.make_grid(sample, nrow=6, normalize=True, range=(-1, 1))
                del sample
                wandb.log({"Reconstructed Images VAE": [wandb.Image(grid, caption=f"Step {i}")]})

                sample = vae.decode(sample_z.to(device))
                grid = utils.make_grid(sample, nrow=6, normalize=True, range=(-1, 1))
                del sample
                wandb.log({"Generated Images VAE": [wandb.Image(grid, caption=f"Step {i}")]})

                gc.collect()
                th.cuda.empty_cache()

                th.save(
                    {"vae": vae.state_dict(), "vae_optim": vae_optim.state_dict()},
                    f"/home/hans/modelzoo/maua-sg2/vae-{name}-{wandb.run.dir.split('/')[-1].split('-')[-1]}.pt",
                )

        if th.isnan(loss).any():
            print("NaN losses, exiting...")
            wandb.log({"Total": 27000})
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    # parser.add_argument("--lambda_recons", type=float, default=0.0)
    # parser.add_argument("--recons_alpha", type=float, default=5.0)
    # parser.add_argument("--lambda_align", type=float, default=1.0)
    # parser.add_argument("--align_alpha", type=float, default=2.0)
    # parser.add_argument("--lambda_unif", type=float, default=1.0)
    # parser.add_argument("--unif_t", type=float, default=0.001)
    args = parser.parse_args()

    device = "cuda"
    th.backends.cudnn.benchmark = True

    wandb.init(project=f"maua-stylegan")

    train(
        args.learning_rate,
        args.lambda_mse,
        # args.lambda_recons,
        # args.recons_alpha,
        # args.lambda_align,
        # args.align_alpha,
        # args.lambda_unif,
        # args.unif_t,
    )


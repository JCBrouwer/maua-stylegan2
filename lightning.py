import os
import math
import wandb
import random
import argparse
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
from collections import OrderedDict
from torch.utils import data
from dataset import MultiResolutionDataset
from model import Generator, Discriminator
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def get_spectral_norms(model):
    spectral_norms = {}
    for name, param in model.named_parameters():
        if param.numel() > 0:
            spectral_norms[name] = nn.utils.spectral_norm(param)
    return spectral_norms


class StyleGAN2(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        for key in kwargs:
            setattr(self, key, kwargs[key])

        self.generator = Generator(
            self.size, self.latent_size, self.n_mlp, channel_multiplier=self.channel_multiplier
        )  # .to(device)
        self.g_ema = Generator(
            self.size, self.latent_size, self.n_mlp, channel_multiplier=self.channel_multiplier
        )  # .to(device)
        self.g_ema.eval()
        self.accumulate_g(0)

        self.discriminator = Discriminator(self.size, channel_multiplier=self.channel_multiplier)  # .to(device)

        if self.checkpoint is not None:
            print("load model:", self.checkpoint)
            checkpoint = th.load(self.checkpoint)
            try:
                checkpoint_name = os.path.basename(self.checkpoint)
                self.start_iter = int(os.path.splitext(checkpoint_name)[0])
            except ValueError:
                pass
            self.generator.load_state_dict(checkpoint["g"])
            self.discriminator.load_state_dict(checkpoint["d"])
            self.g_ema.load_state_dict(checkpoint["g_ema"])

        self.sample_z = th.randn(self.n_sample, self.latent_size)

        self.mean_path_length = 0
        self.d_loss = 0
        self.r1_loss = th.tensor(0.0)
        self.g_loss = 0
        self.path_loss = th.tensor(0.0)
        self.path_lengths = th.tensor(0.0)

    def forward(self, z):
        return self.generator(z)

    def configure_optimizers(self):
        g_reg_ratio = self.g_reg_every / (self.g_reg_every + 1)
        d_reg_ratio = self.d_reg_every / (self.d_reg_every + 1)

        self.g_optim = th.optim.Adam(
            self.generator.parameters(), lr=self.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
        self.d_optim = th.optim.Adam(
            self.discriminator.parameters(), lr=self.lr * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

        if self.checkpoint is not None:
            checkpoint = th.load(self.checkpoint)
            try:
                checkpoint_name = os.path.basename(self.checkpoint)
                self.start_iter = int(os.path.splitext(checkpoint_name)[0])
            except ValueError:
                pass
            self.g_optim.load_state_dict(checkpoint["g_optim"])
            self.d_optim.load_state_dict(checkpoint["d_optim"])

        return [self.g_optim, self.d_optim], []

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomCrop(256),
                # transforms.RandomApply(
                #     transforms.RandomChoice(
                #         [
                #             transforms.RandomRotation((-90, -90)),
                #             transforms.RandomRotation((0, 0)),
                #             transforms.RandomRotation((90, 90)),
                #             transforms.RandomRotation((180, 180)),
                #         ]
                #     ),
                #     p=0.75,
                # ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        dataset = MultiResolutionDataset(self.path, transform, self.size)
        # loader = data.RandomSampler(dataset)

        # def sample_data(loader):
        #     while True:
        #         for batch in loader:
        #             yield batch

        return data.DataLoader(dataset)

    def accumulate_g(self, decay=0.5 ** (32 / (10 * 1000))):
        par1 = dict(self.g_ema.named_parameters())
        par2 = dict(self.generator.named_parameters())
        for k in par1.keys():
            par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

    def make_noise(self, batch_size, device):
        if self.mixing_prob > 0 and random.random() < self.mixing_prob:
            return th.randn(2, batch_size, self.latent_size, device=device).unbind(0)
        else:
            return [th.randn(batch_size, self.latent_size, device=device)]

    def d_logistic_loss(self, real_pred, fake_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)
        return real_loss.mean() + fake_loss.mean()

    def d_r1_loss(self, real_pred, real_img):
        (grad_real,) = th.autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
        grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty

    def g_nonsaturating_loss(self, fake_pred):
        loss = F.softplus(-fake_pred).mean()
        return loss

    def g_path_regularize(self, fake_img, latents, mean_path_length, decay=0.01):
        noise = th.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
        (grad,) = th.autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)
        path_lengths = th.sqrt(grad.pow(2).sum(2).mean(1))
        path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
        path_penalty = (path_lengths - path_mean).pow(2).mean()
        return path_penalty, path_mean.detach(), path_lengths

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_img = batch
        log_dict = {}
        # print("batch is: ", batch)

        # train generator
        if optimizer_idx == 0:
            requires_grad(self.generator, True)
            requires_grad(self.discriminator, False)

            noise = self.make_noise(self.batch_size, device="cuda")  # real_img.device.index)
            fake_img, _ = self.generator(noise)
            fake_pred = self.discriminator(fake_img)
            g_loss = self.g_nonsaturating_loss(fake_pred)

            log_dict["Generator"] = g_loss

            if batch_idx % self.g_reg_every == 0:
                path_batch_size = max(1, self.batch_size // self.path_batch_shrink)
                noise = self.make_noise(path_batch_size, device="cuda")  # real_img.device.index)
                fake_img, latents = self.generator(noise, return_latents=True)

                self.path_loss, self.mean_path_length, self.path_lengths = self.g_path_regularize(
                    fake_img, latents, self.mean_path_length
                )
                self.weighted_path_loss = self.path_regularize * self.g_reg_every * self.path_loss
                if self.path_batch_shrink:
                    self.weighted_path_loss += 0 * fake_img[0, 0, 0, 0]
                # log_dict["weighted_path_loss"] = self.weighted_path_loss
                g_loss += self.weighted_path_loss

            log_dict["Path Length Regularization"] = self.path_loss
            log_dict["Mean Path Length"] = self.path_lengths.mean()
            # log_dict["Spectral Norms/Generator"] = get_spectral_norms(self.generator)

            # log sampled images
            # sample_imgs = self.generated_imgs[:6]
            # grid = thvision.utils.make_grid(sample_imgs)
            # self.logger.experiment.add_image("generated_images", grid, 0)

            return OrderedDict({"loss": g_loss, "progress_bar": log_dict, "log": log_dict})

        # train discriminator
        if optimizer_idx == 1:
            requires_grad(self.generator, False)
            requires_grad(self.discriminator, True)

            noise = self.make_noise(self.batch_size, device="cuda")  # real_img.device.index)
            fake_img, _ = self.generator(noise)
            fake_pred = self.discriminator(fake_img)

            real_pred = self.discriminator(real_img)
            d_loss = self.d_logistic_loss(real_pred, fake_pred)

            log_dict["Discriminator"] = d_loss
            log_dict["Real Score"] = real_pred.mean()
            log_dict["Fake Score"] = fake_pred.mean()

            if batch_idx % self.d_reg_every == 0:
                real_img.requires_grad = True
                real_pred = self.discriminator(real_img)
                self.r1_loss = self.d_r1_loss(real_pred, real_img)
                self.weighted_r1_loss = self.r1 / 2 * self.r1_loss * self.d_reg_every + 0 * real_pred[0]
                log_dict["R1"] = self.r1_loss
                d_loss += self.weighted_r1_loss

            # log_dict["Spectral Norms/Discriminator"] = get_spectral_norms(self.discriminator)

            return OrderedDict({"loss": d_loss, "progress_bar": log_dict, "log": log_dict})

    def optimizer_step(self, current_epoch, batch_idx, optimizer, optimizer_i, second_order_closure=None):
        if optimizer_i == 0:
            optimizer.step()
            optimizer.zero_grad()

            # path regularization
            # if batch_idx % self.g_reg_every == 0:
            #     self.weighted_path_loss.backward()
            #     optimizer.step()
            #     optimizer.zero_grad()

            self.accumulate_g()

        if optimizer_i == 1:
            optimizer.step()
            optimizer.zero_grad()

            # r1 regularization
            # if batch_idx % self.d_reg_every == 0:
            #     self.weighted_r1_loss.backward()
            #     optimizer.step()
            #     optimizer.zero_grad()

    def on_epoch_end(self):
        with th.no_grad():
            self.g_ema.eval()
            sample, _ = self.g_ema([self.sample_z.cuda()])
            grid = tv.utils.make_grid(sample, nrow=int(args.n_sample ** 0.5), normalize=True, range=(-1, 1),)
            self.logger.experiment.log({"generated_images": [wandb.Image(grid, caption=f"Epoch {self.current_epoch}")]})

        if self.current_epoch % 5 == 0:
            th.save(
                {
                    "g": self.generator.state_dict(),
                    "d": self.discriminator.state_dict(),
                    "g_ema": self.g_ema.state_dict(),
                    "g_optim": self.g_optim.state_dict(),
                    "d_optim": self.d_optim.state_dict(),
                },
                f"checkpoint/{str(self.current_epoch).zfill(6)}.pt",
            )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--start_iter", type=int, default=0)
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_sample", type=int, default=64)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing_prob", type=float, default=0.9)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    # th.autograd.set_detect_anomaly(True)

    stylegan2 = StyleGAN2(**vars(args))
    stylegan2.prepare_data()
    stylegan2.train_dataloader()

    wandb_logger = WandbLogger(name="Hello World", project="maua-stylegan")
    trainer = pl.Trainer(gpus=1, max_epochs=10, logger=wandb_logger)  # , amp_level="O1", precision=16)
    trainer.fit(stylegan2)

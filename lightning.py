import os
import gc
import fid
import ppl
import math
import apex
import wandb
import pickle
import random
import argparse
import numpy as np
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
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams  # for automatic param saving with lightning
        [setattr(self, k, v) for k, v in vars(hparams).items()]  # for easy access within module

        self.generator = Generator(self.size, self.latent_size, self.n_mlp, channel_multiplier=self.channel_multiplier)
        self.g_ema = Generator(self.size, self.latent_size, self.n_mlp, channel_multiplier=self.channel_multiplier)
        self.g_ema.eval()
        self.accumulate_g(0)

        self.discriminator = Discriminator(self.size, channel_multiplier=self.channel_multiplier)

        self.sample_z = th.randn(self.n_sample, self.latent_size)

        self.mean_path_length = th.tensor(0.0)

    def forward(self, z):
        return self.generator(z)

    def accumulate_g(self, decay=0.5 ** (32.0 / (10 * 1000))):
        par1 = dict(self.g_ema.named_parameters())
        par2 = dict(self.generator.named_parameters())
        for k in par1.keys():
            par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

    def configure_optimizers(self):
        g_reg_ratio = self.g_reg_every / (self.g_reg_every + 1)
        d_reg_ratio = self.d_reg_every / (self.d_reg_every + 1)

        g_optim = th.optim.Adam(
            self.generator.parameters(), lr=self.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
        d_optim = th.optim.Adam(
            self.discriminator.parameters(), lr=self.lr * d_reg_ratio, betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

        return [g_optim, g_optim, d_optim, d_optim], []

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomVerticalFlip(p=0.5 if self.vflip else 0),
                transforms.RandomHorizontalFlip(p=0.5 if self.hflip else 0),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            ]
        )
        dataset = MultiResolutionDataset(self.path, transform, self.size)
        loader = data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        return loader

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

    def make_noise(self, batch, batch_size=None):
        if batch_size is None:
            batch_size = batch.size(0)
        if self.mixing_prob > 0 and random.random() < self.mixing_prob:
            return th.randn(2, batch_size, self.latent_size).type_as(batch).unbind(0)
        else:
            return [th.randn(batch_size, self.latent_size).type_as(batch)]

    def training_step(self, real_img, batch_idx, optimizer_idx):
        log_dict = {}

        # train generator
        if optimizer_idx == 0:
            requires_grad(self.generator, True)
            requires_grad(self.discriminator, False)

            noise = self.make_noise(real_img)
            fake_img, _ = self.generator(noise)
            fake_pred = self.discriminator(fake_img)
            g_loss = self.g_nonsaturating_loss(fake_pred)

            log_dict["Generator"] = g_loss
            # log_dict["Spectral Norms/Generator"] = get_spectral_norms(self.generator)

            return OrderedDict({"loss": g_loss, "log": log_dict})

        # maybe regularize generator
        if optimizer_idx == 1:
            if batch_idx % self.g_reg_every == 0:
                path_batch_size = max(1, self.batch_size // self.path_batch_shrink)
                noise = self.make_noise(real_img, path_batch_size)
                fake_img, latents = self.generator(noise, return_latents=True)
                path_loss, self.mean_path_length, path_lengths = self.g_path_regularize(
                    fake_img, latents, self.mean_path_length.type_as(real_img)
                )
                weighted_path_loss = self.path_regularize * self.g_reg_every * path_loss
                if self.path_batch_shrink:
                    weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

                log_dict["Path Length Regularization"] = path_loss
                log_dict["Mean Path Length"] = path_lengths.mean()

                return OrderedDict({"loss": weighted_path_loss, "log": log_dict})
            return OrderedDict({"loss": th.tensor(-69).type_as(real_img)})

        # train discriminator
        if optimizer_idx == 2:
            requires_grad(self.generator, False)
            requires_grad(self.discriminator, True)

            noise = self.make_noise(real_img)
            fake_img, _ = self.generator(noise)
            fake_pred = self.discriminator(fake_img)

            real_pred = self.discriminator(real_img)
            d_loss = self.d_logistic_loss(real_pred, fake_pred)

            log_dict["Discriminator"] = d_loss
            log_dict["Real Score"] = real_pred.mean()
            log_dict["Fake Score"] = fake_pred.mean()
            # log_dict["Spectral Norms/Discriminator"] = get_spectral_norms(self.discriminator)

            return OrderedDict({"loss": d_loss, "log": log_dict})

        # maybe regularize discriminator
        if optimizer_idx == 3:
            if batch_idx % self.d_reg_every == 0:
                real_img.requires_grad = True
                real_pred = self.discriminator(real_img)
                r1_loss = self.d_r1_loss(real_pred, real_img)
                weighted_r1_loss = self.r1 / 2 * r1_loss * self.d_reg_every + 0 * real_pred[0]

                log_dict["R1"] = r1_loss

                return OrderedDict({"loss": weighted_r1_loss, "log": log_dict})
            return OrderedDict({"loss": th.tensor(-69).type_as(real_img)})

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        if optimizer_idx == 0:
            super(StyleGAN2, self).backward(trainer, loss, optimizer, optimizer_idx)
        if optimizer_idx == 1 and loss != -69:
            super(StyleGAN2, self).backward(trainer, loss, optimizer, optimizer_idx)
        if optimizer_idx == 2:
            super(StyleGAN2, self).backward(trainer, loss, optimizer, optimizer_idx)
        if optimizer_idx == 3 and loss != -69:
            super(StyleGAN2, self).backward(trainer, loss, optimizer, optimizer_idx)

    def optimizer_step(self, cur_epoch, batch_idx, optimizer, optimizer_idx, clos=None):
        if optimizer_idx == 0:
            super(StyleGAN2, self).optimizer_step(cur_epoch, batch_idx, optimizer, optimizer_idx, clos)
        if optimizer_idx == 1 and batch_idx % self.g_reg_every == 0:
            super(StyleGAN2, self).optimizer_step(cur_epoch, batch_idx, optimizer, optimizer_idx, clos)
        if optimizer_idx == 2:
            super(StyleGAN2, self).optimizer_step(cur_epoch, batch_idx, optimizer, optimizer_idx, clos)
        if optimizer_idx == 3 and batch_idx % self.d_reg_every == 0:
            super(StyleGAN2, self).optimizer_step(cur_epoch, batch_idx, optimizer, optimizer_idx, clos)

    def prepare_data(self):
        fid.get_dataset_inception_features(self.train_dataloader(), self.path, self.name, self.size)

    def val_dataloader(self):
        return [[th.zeros(size=(1,))]]

    def validation_step(self, batch, batch_idx):
        return OrderedDict({"batch": batch})

    def validation_epoch_end(self, outputs):
        gc.collect()
        th.cuda.empty_cache()
        val_fid = fid.validation_fid(
            self.g_ema.to(outputs[0]["batch"].device),
            self.val_batch_size,
            self.fid_n_sample,
            self.fid_truncation,
            self.name,
        )
        val_ppl = ppl.validation_ppl(
            self.g_ema.to(outputs[0]["batch"].device),
            self.val_batch_size,
            self.ppl_n_sample,
            self.ppl_space,
            self.ppl_crop,
            self.latent_size,
        )
        with th.no_grad():
            self.g_ema.eval()
            sample, _ = self.g_ema([self.sample_z.to(outputs[0]["batch"].device)])
            grid = tv.utils.make_grid(sample, nrow=int(16.0 / 9 * self.n_sample ** 0.5), normalize=True, range=(-1, 1))
            self.logger.experiment.log({"generated_images": [wandb.Image(grid, caption=f"Step {self.global_step}")]})
            del sample, grid
        gc.collect()
        th.cuda.empty_cache()
        return {"val_loss": val_fid, "log": {"Validation/FID": val_fid, "Validation/PPL": val_ppl}}


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()

    # data options
    parser.add_argument("path", type=str)
    parser.add_argument("--vflip", type=bool, default=False)
    parser.add_argument("--hflip", type=bool, default=True)

    # training options
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--checkpoint", type=str, default=None)

    # model options
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--n_sample", type=int, default=32)
    parser.add_argument("--size", type=int, default=256)

    # loss hyperparams
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing_prob", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)

    # validation / logging options
    parser.add_argument("--wandb", type=bool, default=True)
    parser.add_argument("--validation_interval", type=float, default=0.25)
    parser.add_argument("--val_batch_size", type=int, default=24)
    parser.add_argument("--fid_n_sample", type=int, default=10000)
    parser.add_argument("--fid_truncation", type=float, default=0.7)
    parser.add_argument("--ppl_space", choices=["z", "w"], default="w")
    parser.add_argument("--ppl_n_sample", type=int, default=5000)
    parser.add_argument("--ppl_crop", type=bool, default=False)

    # DevOps options
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--cudnn_benchmark", type=bool, default=True)
    parser.add_argument("--distributed_backend", type=str, default="dp")

    args = parser.parse_args()

    args.name = os.path.splitext(os.path.basename(args.path))[0]

    # th.autograd.set_detect_anomaly(True)

    stylegan2 = StyleGAN2(args)
    stylegan2.prepare_data()
    stylegan2.train_dataloader()

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath="checkpoints/" + args.name + "-{epoch}-FID{val_loss:.2f}", save_top_k=10
    )

    wandb_logger = pl.loggers.WandbLogger(name="Hello World", project="maua-stylegan")

    trainer = pl.Trainer(
        gpus=args.num_gpus,
        max_epochs=args.epochs,
        logger=wandb_logger,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=None,
        distributed_backend=args.distributed_backend,
        benchmark=args.cudnn_benchmark,
        val_check_interval=args.validation_interval,
        num_sanity_val_steps=0,
        terminate_on_nan=False,
        resume_from_checkpoint=args.checkpoint,
        # amp_level="O1",
        # precision=16,
    )
    trainer.fit(stylegan2)

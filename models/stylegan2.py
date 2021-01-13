import math
import os
import random
import sys

import torch as th
from torch import nn
from torch.nn import functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs * th.rsqrt(th.mean(inputs ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = th.tensor(k, dtype=th.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, inputs):
        out = upfirdn2d(inputs, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, inputs):
        out = upfirdn2d(inputs, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, inputs):
        out = upfirdn2d(inputs, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(th.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(th.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, inputs):
        out = F.conv2d(inputs, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding,)

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()

        self.weight = nn.Parameter(th.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(th.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, inputs):
        if self.activation:
            out = F.linear(inputs, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(inputs, self.weight * self.scale, bias=self.bias * self.lr_mul)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, inputs):
        out = F.leaky_relu(inputs, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(th.randn(1, out_channel, in_channel, kernel_size, kernel_size))

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, inputs, style):
        batch, in_channel, height, width = inputs.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = th.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        if self.upsample:
            inputs = inputs.view(1, batch * in_channel, height, width)
            weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(inputs, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            inputs = self.blur(inputs)
            _, _, height, width = inputs.shape
            inputs = inputs.view(1, batch * in_channel, height, width)
            out = F.conv2d(inputs, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            inputs = inputs.view(1, batch * in_channel, height, width)
            out = F.conv2d(inputs, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(th.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise.to(image.device)


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(th.randn(1, channel, size, size))

    def forward(self, inputs):
        batch = inputs.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out


class LatentInput(nn.Module):
    def __init__(self, latent_dim, channel, size=4):
        super().__init__()
        self.channel = channel
        self.size = size
        self.linear = EqualLinear(latent_dim, channel * size * size, activation="fused_lrelu")
        self.activate = FusedLeakyReLU(channel * size * size)
        self.input = nn.Parameter(th.randn(1))

    def forward(self, inputs):
        batch = inputs.shape[0]
        out = self.linear(inputs[:, 0])
        out = self.activate(out)
        return out.reshape((batch, self.channel, self.size, self.size))


class ManipulationLayer(th.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, input, tranforms_dict_list):
        out = input
        for transform_dict in tranforms_dict_list:
            if transform_dict["layer"] == self.layer:
                out = transform_dict["transform"].to(out.device)(out)
        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        layerID=-1,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)
        self.manipulation = ManipulationLayer(layerID)

    def forward(self, inputs, style, noise=None, transform_dict_list=[]):
        out = self.conv(inputs, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)
        out = self.manipulation(out, transform_dict_list)
        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(th.zeros(1, 3, 1, 1))

    def forward(self, inputs, style, skip=None):
        out = self.conv(inputs, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        constant_input=False,
        checkpoint=None,
        output_size=None,
        min_rgb_size=4,
        base_res_factor=1,
    ):
        super().__init__()

        self.size = size
        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"))

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1
        self.n_latent = self.log_size * 2 - 2
        self.min_rgb_size = min_rgb_size

        if constant_input:
            self.input = ConstantInput(self.channels[4])
        else:
            self.input = LatentInput(style_dim, self.channels[4])

        self.const_manipulation = ManipulationLayer(0)

        layerID = 1
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel, layerID=layerID
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", th.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            layerID += 1
            self.convs.append(
                StyledConv(
                    in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel, layerID=layerID
                )
            )

            layerID += 1
            self.convs.append(
                StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel, layerID=layerID)
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.truncation_latent = None

        if checkpoint is not None:
            self.load_state_dict(th.load(checkpoint)["g_ema"])

        if size != output_size or base_res_factor != 1:
            for layer_idx in range(self.num_layers):
                res = (layer_idx + 5) // 2
                shape = [
                    1,
                    1,
                    int(base_res_factor * 2 ** res * (2 if output_size == 1080 else 1)),
                    int(base_res_factor * 2 ** res * (2 if output_size == 1920 else 1)),
                ]
                print(shape)
                setattr(self.noises, f"noise_{layer_idx}", th.randn(*shape))

    def make_noise(self):
        device = self.input.input.device

        noises = [th.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(th.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = th.randn(n_latent, self.style_dim, device=self.input.input.device)
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, inputs):
        return self.style(inputs)

    def forward(
        self,
        styles,
        return_latents=False,
        return_activation_maps=False,
        inject_index=None,
        truncation=th.cuda.FloatTensor([1]),
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        transform_dict_list=[],
        map_latents=False,
    ):
        if map_latents:
            latent = th.cat([self.style(s[None, None, :]) for s in styles], axis=0)
            latent = latent.repeat(1, self.n_latent, 1)
            return latent

        if not input_is_latent:
            styles = [self.style(s) for s in styles]

            if len(styles) < 2:
                inject_index = self.n_latent
                if styles[0].ndim < 3:
                    latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                else:
                    latent = styles[0]
            else:
                if inject_index is None:
                    inject_index = random.randint(1, self.n_latent - 1)
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
                latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)
                latent = th.cat([latent, latent2], 1)
        else:
            latent = styles
            if latent.dim() == 2:
                latent = latent[:, None, :].repeat(1, self.n_latent, 1)

        if noise is None:
            noise = [None] * self.num_layers
        for ns, noise_scale in enumerate(noise):
            if not randomize_noise and noise_scale is None:
                noise[ns] = getattr(self.noises, f"noise_{ns}")

        if self.truncation_latent is None:
            self.truncation_latent = truncation_latent if truncation_latent is not None else self.mean_latent(2 ** 14)
        latent = self.truncation_latent[None, ...] + truncation.to(latent.device)[:, None, None] * (
            latent - self.truncation_latent[None, ...]
        )

        activation_map_list = []

        out = self.input(latent)
        out = self.const_manipulation(out, transform_dict_list)
        out = self.conv1(out, latent[:, 0], noise=noise[0], transform_dict_list=transform_dict_list)
        activation_map_list.append(out)

        current_size = 4
        if self.min_rgb_size <= current_size:
            image = self.to_rgb1(out, latent[:, 1])
        else:
            image = None

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1, transform_dict_list=transform_dict_list)
            current_size *= 2
            activation_map_list.append(out)
            out = conv2(out, latent[:, i + 1], noise=noise2, transform_dict_list=transform_dict_list)
            activation_map_list.append(out)
            if self.min_rgb_size <= current_size:
                image = to_rgb(out, latent[:, i + 2], image)
            i += 2

        if return_activation_maps:
            return image, activation_map_list
        elif return_latents:
            return image, latent
        else:
            return image, None


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel, out_channel, kernel_size, padding=self.padding, stride=stride, bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], use_skip=True):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        if use_skip:
            self.skip = ConvLayer(in_channel, out_channel, 1, downsample=True, activate=False, bias=False)
        else:
            self.skip = None

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)

        if self.skip is not None:
            skip = self.skip(inputs)
            out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], use_skip=True):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel, use_skip=use_skip))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"), EqualLinear(channels[4], 1),
        )

    def forward(self, inputs):
        out = self.convs(inputs)

        batch, channel, height, width = out.shape

        try:
            group = min(batch, self.stddev_group)
            stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
            stddev = th.sqrt(stddev.var(0, unbiased=False) + 1e-8)
            stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
            stddev = stddev.repeat(group, 1, height, width)
            out = th.cat([out, stddev], 1)
        except RuntimeError:
            group = batch
            stddev = out.view(group, -1, self.stddev_feat, channel // self.stddev_feat, height, width)
            stddev = th.sqrt(stddev.var(0, unbiased=False) + 1e-8)
            stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
            stddev = stddev.repeat(group, 1, height, width)
            out = th.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

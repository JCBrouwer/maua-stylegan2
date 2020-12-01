# from https://github.com/lernapparat/lernapparat/blob/master/style_gan/pyth_style_gan.ipynb

import gc
from collections import OrderedDict

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class MyLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(
        self, input_size, output_size, gain=2 ** (0.5), use_wscale=False, lrmul=1, bias=True,
    ):
        super().__init__()
        he_std = gain * input_size ** (-0.5)  # He init
        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = th.nn.Parameter(th.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = th.nn.Parameter(th.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)


class MyConv2d(nn.Module):
    """Conv layer with equalized learning rate and custom learning rate multiplier."""

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        gain=2 ** (0.5),
        use_wscale=False,
        lrmul=1,
        bias=True,
        intermediate=None,
        upscale=False,
    ):
        super().__init__()
        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = th.nn.Parameter(th.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = th.nn.Parameter(th.zeros(output_channels))
            self.b_mul = lrmul
        else:
            self.bias = None
        self.intermediate = intermediate

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul

        have_convolution = False
        if self.upscale is not None and min(x.shape[2:]) * 2 >= 128:
            # this is the fused upscale + conv from StyleGAN, sadly this seems incompatible with the non-fused way
            # this really needs to be cleaned up and go into the conv...
            w = self.weight * self.w_mul
            w = w.permute(1, 0, 2, 3)
            # probably applying a conv on w would be more efficient. also this quadruples the weight (average)?!
            w = F.pad(w, (1, 1, 1, 1))
            w = w[:, :, 1:, 1:] + w[:, :, :-1, 1:] + w[:, :, 1:, :-1] + w[:, :, :-1, :-1]
            x = F.conv_transpose2d(x, w, stride=2, padding=(w.size(-1) - 1) // 2)
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)

        if not have_convolution and self.intermediate is None:
            return F.conv2d(x, self.weight * self.w_mul, bias, padding=self.kernel_size // 2)
        elif not have_convolution:
            x = F.conv2d(x, self.weight * self.w_mul, None, padding=self.kernel_size // 2)

        if self.intermediate is not None:
            x = self.intermediate(x)
        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        return x


class NoiseLayer(nn.Module):
    """adds noise. noise is per pixel (constant over channels) with per-channel weight"""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(th.zeros(channels))
        self.noise = None

    def forward(self, x):
        if self.noise is None:
            noise = th.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        else:
            noise = self.noise.to(x.device)
        # print(noise.shape, noise.min(), noise.mean(), noise.max())
        x = x + self.weight.view(1, -1, 1, 1) * noise
        return x


class StyleMod(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(StyleMod, self).__init__()
        self.lin = MyLinear(latent_size, channels * 2, gain=1.0, use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.lin(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.0) + style[:, 1]
        return x


class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * th.rsqrt(th.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class BlurLayer(nn.Module):
    def __init__(self, kernel=[1, 2, 1], normalize=True, flip=False, stride=1):
        super(BlurLayer, self).__init__()
        kernel = [1, 2, 1]
        kernel = th.tensor(kernel, dtype=th.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer("kernel", kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(x, kernel, stride=self.stride, padding=int((self.kernel.size(2) - 1) / 2), groups=x.size(1),)
        return x


def upscale2d(x, factor=2, gain=1):
    assert x.dim() == 4
    if gain != 1:
        x = x * gain
    if factor != 1:
        shape = x.shape
        x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, factor, -1, factor)
        x = x.contiguous().view(shape[0], shape[1], factor * shape[2], factor * shape[3])
    return x


class Upscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        return upscale2d(x, factor=self.factor, gain=self.gain)


class G_mapping(nn.Sequential):
    def __init__(self, nonlinearity="lrelu", use_wscale=True):
        act, gain = {"relu": (th.relu, np.sqrt(2)), "lrelu": (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[
            nonlinearity
        ]
        layers = [
            ("pixel_norm", PixelNormLayer()),
            ("dense0", MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale),),
            ("dense0_act", act),
            ("dense1", MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale),),
            ("dense1_act", act),
            ("dense2", MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale),),
            ("dense2_act", act),
            ("dense3", MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale),),
            ("dense3_act", act),
            ("dense4", MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale),),
            ("dense4_act", act),
            ("dense5", MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale),),
            ("dense5_act", act),
            ("dense6", MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale),),
            ("dense6_act", act),
            ("dense7", MyLinear(512, 512, gain=gain, lrmul=0.01, use_wscale=use_wscale),),
            ("dense7_act", act),
        ]
        super().__init__(OrderedDict(layers))

    def forward(self, x):
        x = super().forward(x)
        # Broadcast
        x = x.unsqueeze(1).expand(-1, 18, -1)
        return x


class Truncation(nn.Module):
    def __init__(self, avg_latent, max_layer=8, threshold=0.7):
        super().__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.register_buffer("avg_latent", avg_latent)

    def forward(self, x):
        assert x.dim() == 3
        interp = th.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (th.arange(x.size(1)) < self.max_layer).view(1, -1, 1)
        return th.where(do_trunc, interp, x)


class LayerEpilogue(nn.Module):
    """Things to do at the end of each layer."""

    def __init__(
        self,
        channels,
        dlatent_size,
        use_wscale,
        use_noise,
        use_pixel_norm,
        use_instance_norm,
        use_styles,
        activation_layer,
    ):
        super().__init__()
        layers = []
        if use_noise:
            layers.append(("noise", NoiseLayer(channels)))
        layers.append(("activation", activation_layer))
        if use_pixel_norm:
            layers.append(("pixel_norm", PixelNormLayer()))
        if use_instance_norm:
            layers.append(("instance_norm", nn.InstanceNorm2d(channels)))
        self.top_epi = nn.Sequential(OrderedDict(layers))
        if use_styles:
            self.style_mod = StyleMod(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

        # if use_noise:
        #     # layers.append(("noise", NoiseLayer(channels)))
        #     self.noise = NoiseLayer(channels)
        # else:
        #     self.noise = None

        # self.activation = activation_layer

        # if use_pixel_norm:
        #     self.pixel_norm = PixelNormLayer()
        # else:
        #     self.pixel_norm = None

        # if use_instance_norm:
        #     self.instance_norm = nn.InstanceNorm2d(channels)
        # else:
        #     self.instance_norm = None

        # if use_styles:
        #     self.style_mod = StyleMod(dlatent_size, channels, use_wscale=use_wscale)
        # else:
        #     self.style_mod = None

    def forward(self, x, dlatents_in_slice, noise):
        # if self.noise is not None:
        #     x = self.noise(x, noise)

        # x = self.activation(x)

        # if self.pixel_norm is not None:
        #     x = self.pixel_norm(x)

        # if self.instance_norm is not None:
        #     x = self.instance_norm(x)

        if noise is not None:
            self.top_epi.noise.noise = noise

        x = self.top_epi(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        else:
            assert dlatents_in_slice is None

        if noise is not None:
            del self.top_epi.noise.noise
            gc.collect()

        return x


class InputBlock(nn.Module):
    def __init__(
        self,
        nf,
        dlatent_size,
        const_input_layer,
        gain,
        use_wscale,
        use_noise,
        use_pixel_norm,
        use_instance_norm,
        use_styles,
        activation_layer,
    ):
        super().__init__()
        self.const_input_layer = const_input_layer
        self.nf = nf
        if self.const_input_layer:
            # called 'const' in tf
            self.const = nn.Parameter(th.ones(1, nf, 4, 4))
            self.bias = nn.Parameter(th.ones(nf))
        else:
            self.dense = MyLinear(
                dlatent_size, nf * 16, gain=gain / 4, use_wscale=use_wscale
            )  # tweak gain to match the official implementation of Progressing GAN
        self.epi1 = LayerEpilogue(
            nf, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer,
        )
        self.conv = MyConv2d(nf, nf, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(
            nf, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer,
        )

    def forward(self, dlatents_in_range, noise):
        batch_size = dlatents_in_range.size(0)
        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_in_range[:, 0]).view(batch_size, self.nf, 4, 4)
        x = self.epi1(x, dlatents_in_range[:, 0], noise=noise)
        x = self.conv(x)
        x = self.epi2(x, dlatents_in_range[:, 1], noise=noise)
        return x


class GSynthesisBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        blur_filter,
        dlatent_size,
        gain,
        use_wscale,
        use_noise,
        use_pixel_norm,
        use_instance_norm,
        use_styles,
        activation_layer,
    ):
        # 2**res x 2**res # res = 3..resolution_log2
        super().__init__()
        if blur_filter:
            blur = BlurLayer(blur_filter)
        else:
            blur = None
        self.conv0_up = MyConv2d(
            in_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale, intermediate=blur, upscale=True,
        )
        self.epi1 = LayerEpilogue(
            out_channels,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer,
        )
        self.conv1 = MyConv2d(out_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(
            out_channels,
            dlatent_size,
            use_wscale,
            use_noise,
            use_pixel_norm,
            use_instance_norm,
            use_styles,
            activation_layer,
        )

    def forward(self, x, dlatents_in_range, noise):
        x = self.conv0_up(x)
        x = self.epi1(x, dlatents_in_range[:, 0], noise=noise)
        x = self.conv1(x)
        x = self.epi2(x, dlatents_in_range[:, 1], noise=noise)
        return x


class G_synthesis(nn.Module):
    def __init__(
        self,
        dlatent_size=512,  # Disentangled latent (W) dimensionality.
        num_channels=3,  # Number of output color channels.
        resolution=1024,  # Output resolution.
        fmap_base=8192,  # Overall multiplier for the number of feature maps.
        fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
        fmap_max=512,  # Maximum number of feature maps in any layer.
        use_styles=True,  # Enable style inputs?
        const_input_layer=True,  # First layer is a learned constant?
        use_noise=True,  # Enable noise inputs?
        randomize_noise=False,  # True = randomize noise inputs every time (non-deterministic) or from variables passed,
        nonlinearity="lrelu",  # Activation function: 'relu', 'lrelu'
        use_wscale=True,  # Enable equalized learning rate?
        use_pixel_norm=False,  # Enable pixelwise feature vector normalization?
        use_instance_norm=True,  # Enable instance normalization?
        dtype=th.float32,  # Data type to use for activations and outputs.
        blur_filter=[1, 2, 1],  # Low-pass filter to apply when resampling activations. None = no filtering.
    ):

        super().__init__()

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.dlatent_size = dlatent_size
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4

        act, gain = {"relu": (th.relu, np.sqrt(2)), "lrelu": (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2)),}[
            nonlinearity
        ]
        blocks = []
        for res in range(2, resolution_log2 + 1):
            channels = nf(res - 1)
            name = "{s}x{s}".format(s=2 ** res)
            if res == 2:
                blocks.append(
                    (
                        name,
                        InputBlock(
                            channels,
                            dlatent_size,
                            const_input_layer,
                            gain,
                            use_wscale,
                            use_noise,
                            use_pixel_norm,
                            use_instance_norm,
                            use_styles,
                            act,
                        ),
                    )
                )

            else:
                blocks.append(
                    (
                        name,
                        GSynthesisBlock(
                            last_channels,
                            channels,
                            blur_filter,
                            dlatent_size,
                            gain,
                            use_wscale,
                            use_noise,
                            use_pixel_norm,
                            use_instance_norm,
                            use_styles,
                            act,
                        ),
                    )
                )
            last_channels = channels
        self.torgb = MyConv2d(channels, num_channels, 1, gain=1, use_wscale=use_wscale)
        self.blocks = nn.ModuleDict(OrderedDict(blocks))

    def forward(self, dlatents_in, noise):
        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
        # lod_in = tf.cast(tf.get_variable('lod', initializer=np.float32(0), trainable=False), dtype)
        for i, m in enumerate(self.blocks.values()):
            if i == 0:
                x = m(dlatents_in[:, 2 * i : 2 * i + 2], noise=noise)
            else:
                x = m(x, dlatents_in[:, 2 * i : 2 * i + 2], noise=noise)
        rgb = self.torgb(x)
        return rgb


class G_style(nn.Sequential):
    def __init__(self, output_size=1920, checkpoint=None):

        # TODO FIX THIS MONSTROSITY
        super().__init__()
        self.g_mapping = G_mapping()
        try:
            self.g_synthesis = G_synthesis(resolution=1024)
            if checkpoint is not None:
                self.load_state_dict(th.load(checkpoint), strict=False)
            network_resolution = 1024
        except:
            print("Trying 512px generator resolution...")
            try:
                self.g_synthesis = G_synthesis(resolution=512)
                if checkpoint is not None:
                    self.load_state_dict(th.load(checkpoint), strict=False)
                network_resolution = 512
            except:
                print("Trying 256px generator resolution...")
                try:
                    self.g_synthesis = G_synthesis(resolution=256)
                    if checkpoint is not None:
                        self.load_state_dict(th.load(checkpoint), strict=False)
                    network_resolution = 256
                except:
                    print("Trying 128px generator resolution...")
                    try:
                        self.g_synthesis = G_synthesis(resolution=128)
                        if checkpoint is not None:
                            self.load_state_dict(th.load(checkpoint), strict=False)
                        network_resolution = 128
                    except:
                        print("ERROR: Network too small or state_dict mismatch")
                        exit()

        const = getattr(self.g_synthesis.blocks, "4x4").const
        if network_resolution != 1024:
            means = th.zeros(size=(1, 512, int(4 * 1024 / network_resolution), int(4 * 1024 / network_resolution)))
            const = th.normal(mean=means, std=th.ones_like(means) * const.std(),)

        _, _, ch, cw = const.shape
        if output_size == 1920:
            layer0 = th.cat(
                [
                    const[:, :, :, [0]],
                    const[:, :, :, [0]],
                    # const[:, :, :, : cw // 2 + 1][:, :, :, list(range(cw // 2, 0, -1))],
                    const,
                    # const[:, :, :, cw // 2 :],
                    const[:, :, :, [-1]],
                    const[:, :, :, [-1]],
                ],
                axis=3,
            )
        elif output_size == 512:
            layer0 = const[:, :, ch // 4 : 3 * ch // 4, cw // 4 : 3 * cw // 4]
        else:
            layer0 = const
        getattr(self.g_synthesis.blocks, "4x4").const = th.nn.Parameter(layer0 + th.normal(0, const.std() / 2.0))
        _, _, height, width = getattr(self.g_synthesis.blocks, "4x4").const.shape

        for i in range(len(list(self.g_synthesis.blocks.named_parameters())) // 10):
            self.register_buffer(f"noise_{i}", th.randn(1, 1, height * 2 ** i, width * 2 ** i))

        self.truncation_latent = self.mean_latent(2 ** 14)

    def mean_latent(self, n_latent):
        latent_in = th.randn(n_latent, 512)
        latent = self.g_mapping(latent_in).mean(0, keepdim=True)
        return latent

    def forward(
        self,
        styles,
        noise=None,
        truncation=1,
        map_latents=False,
        randomize_noise=False,
        input_is_latent=True,
        transform_dict_list=None,
    ):
        if map_latents:
            return self.g_mapping(styles)

        if noise is None:
            noise = [None] * (len(list(self.g_synthesis.blocks.named_parameters())) // 10)
        for ns, noise_scale in enumerate(noise):
            if noise_scale is None:
                try:
                    noise[ns] = getattr(self, f"noise_{ns}")
                except:
                    pass

        if truncation != 1:
            interp = th.lerp(self.truncation_latent.to(styles.device), styles, truncation)
            do_trunc = (th.arange(styles.size(1)) < 8).view(1, -1, 1).to(styles.device)
            styles = th.where(do_trunc, interp, styles)

        # Input: Disentangled latents (W) [minibatch, num_layers, dlatent_size].
        # print(styles.shape, len(noise), len(self.g_synthesis.blocks.values()))
        for i, block in enumerate(self.g_synthesis.blocks.values()):
            if i == 0:
                x = block(styles[:, 2 * i : 2 * i + 2], noise=noise[i])
            else:
                x = block(x, styles[:, 2 * i : 2 * i + 2], noise=noise[i])
        img = self.g_synthesis.torgb(x)

        return img, None

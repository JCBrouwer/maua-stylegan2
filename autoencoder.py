import torch as th
from op import FusedLeakyReLU
from copy import copy


def info(x):
    print(x.shape, x.min(), x.mean(), x.max())


class PrintShape(th.nn.Module):
    def __init__(self):
        super(PrintShape, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class Flatten(th.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UnFlatten(th.nn.Module):
    def __init__(self, channels, size):
        super(UnFlatten, self).__init__()
        self.channels = channels
        self.size = size

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.size, self.size)


class LogCoshVAE(th.nn.Module):
    """
    Adapted from https://github.com/AntixK/PyTorch-VAE
    See LICENSE_AUTOENCODER
    """

    def __init__(self, in_channels, latent_dim, hidden_dims=None, alpha=10.0, beta=1.0, kld_weight=1):
        super(LogCoshVAE, self).__init__()

        my_hidden_dims = copy(hidden_dims)

        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta
        self.kld_weight = kld_weight

        modules = []
        if my_hidden_dims is None:
            my_hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in my_hidden_dims:
            modules.append(
                th.nn.Sequential(
                    th.nn.Conv2d(in_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1),
                    th.nn.BatchNorm2d(h_dim),
                    FusedLeakyReLU(h_dim),
                )
            )
            in_channels = h_dim

        self.encoder = th.nn.Sequential(*modules)
        self.fc_mu = th.nn.Linear(my_hidden_dims[-1] * 4, latent_dim)
        self.fc_var = th.nn.Linear(my_hidden_dims[-1] * 4, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = th.nn.Linear(latent_dim, my_hidden_dims[-1] * 4)

        my_hidden_dims.reverse()

        for i in range(len(my_hidden_dims) - 1):
            modules.append(
                th.nn.Sequential(
                    th.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    th.nn.Conv2d(my_hidden_dims[i], my_hidden_dims[i + 1], kernel_size=3, padding=1),
                    th.nn.BatchNorm2d(my_hidden_dims[i + 1]),
                    FusedLeakyReLU(my_hidden_dims[i + 1]),
                )
            )

        self.decoder = th.nn.Sequential(*modules)

        self.final_layer = th.nn.Sequential(
            th.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            th.nn.Conv2d(my_hidden_dims[-1], my_hidden_dims[-1], kernel_size=3, padding=1),
            th.nn.BatchNorm2d(my_hidden_dims[-1]),
            FusedLeakyReLU(my_hidden_dims[-1]),
            th.nn.Conv2d(my_hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            th.nn.Tanh(),
        )

    def encode(self, input):
        result = self.encoder(input)
        result = th.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.latent_dim, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def loss(self, real, fake, mu, log_var):
        t = fake - real

        recons_loss = self.alpha * t + th.log(1.0 + th.exp(-2 * self.alpha * t)) - th.log(th.tensor(2.0))
        recons_loss = (1.0 / self.alpha) * recons_loss.mean()

        kld_loss = th.mean(-0.5 * th.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + self.beta * self.kld_weight * kld_loss

        return {"Total": loss, "Reconstruction": recons_loss, "Kullback Leibler Divergence": -kld_loss}


class conv2DBatchNormRelu(th.nn.Module):
    def __init__(
        self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1, with_bn=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = th.nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if with_bn:
            self.cbr_unit = th.nn.Sequential(
                conv_mod, th.nn.BatchNorm2d(int(n_filters)), FusedLeakyReLU(int(n_filters))
            )
        else:
            self.cbr_unit = th.nn.Sequential(conv_mod, FusedLeakyReLU(int(n_filters)))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class segnetDown2(th.nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = th.nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetDown3(th.nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = th.nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetUp2(th.nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp2, self).__init__()
        self.unpool = th.nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class segnetUp3(th.nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp3, self).__init__()
        self.unpool = th.nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class SegNet(th.nn.Module):
    """
    Adapted from https://github.com/foamliu/Autoencoder
    See LICENSE_AUTOENCODER
    """

    def __init__(self, in_channels=3):
        super(SegNet, self).__init__()

        self.down1 = segnetDown2(in_channels, 64)
        self.down2 = segnetDown2(64, 128)
        self.down3 = segnetDown3(128, 256)
        self.down4 = segnetDown3(256, 512)
        self.down5 = segnetDown3(512, 512)

        self.up5 = segnetUp3(512, 512)
        self.up4 = segnetUp3(512, 256)
        self.up3 = segnetUp3(256, 128)
        self.up2 = segnetUp2(128, 64)
        self.up1 = segnetUp2(64, in_channels)

    def random_indices(self, shape):
        batch, channel, height, width = shape
        xy = th.randint(0, 2, size=[batch, channel, height, width, 2])
        grid = th.arange(height * width).reshape(height, width)
        indices = grid * 2 + (th.arange(height) * width * 2)[:, None] + xy[..., 0] + width * 2 * xy[..., 1]
        return indices.cuda()

    def encode(self, inputs):
        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)
        return down5

    def decode(self, inp):
        batch, _, height, width = inp.shape
        up5 = self.up5(inp, self.random_indices([batch, 512, height, width]), [batch, 512, height * 2, width * 2])
        up4 = self.up4(
            up5, self.random_indices([batch, 512, height * 2, width * 2]), [batch, 512, height * 4, width * 4]
        )
        up3 = self.up3(
            up4, self.random_indices([batch, 256, height * 4, width * 4]), [batch, 256, height * 8, width * 8]
        )
        up2 = self.up2(
            up3, self.random_indices([batch, 128, height * 8, width * 8]), [batch, 128, height * 16, width * 16]
        )
        up1 = self.up1(
            up2, self.random_indices([batch, 64, height * 16, width * 16]), [batch, 64, height * 32, width * 32]
        )
        return up1

    def forward(self, inputs):
        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)
        down3, indices_3, unpool_shape3 = self.down3(down2)
        down4, indices_4, unpool_shape4 = self.down4(down3)
        down5, indices_5, unpool_shape5 = self.down5(down4)

        up5 = self.up5(down5, indices_5.shape, unpool_shape5)
        up4 = self.up4(up5, indices_4.shape, unpool_shape4)
        up3 = self.up3(up4, indices_3.shape, unpool_shape3)
        up2 = self.up2(up3, indices_2.shape, unpool_shape2)
        up1 = self.up1(up2, indices_1.shape, unpool_shape1)

        return up1

    def init_vgg16_params(self, vgg16):
        blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())

        vgg_layers = []
        for _layer in features:
            if isinstance(_layer, th.nn.Conv2d):
                vgg_layers.append(_layer)

        merged_layers = []
        for idx, conv_block in enumerate(blocks):
            if idx < 2:
                units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
            else:
                units = [
                    conv_block.conv1.cbr_unit,
                    conv_block.conv2.cbr_unit,
                    conv_block.conv3.cbr_unit,
                ]
            for _unit in units:
                for _layer in _unit:
                    if isinstance(_layer, th.nn.Conv2d):
                        merged_layers.append(_layer)

        assert len(vgg_layers) == len(merged_layers)

        for l1, l2 in zip(vgg_layers, merged_layers):
            if isinstance(l1, th.nn.Conv2d) and isinstance(l2, th.nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data


class ConvSegNet(th.nn.Module):
    """
    Adapted from https://github.com/foamliu/Autoencoder
    See LICENSE_AUTOENCODER
    """

    def __init__(self, in_channels=3):
        super(ConvSegNet, self).__init__()

        self.encoder = th.nn.Sequential(
            conv2DBatchNormRelu(in_channels, 64, 3, 1, 1),
            conv2DBatchNormRelu(64, 64, 3, 1, 1),
            th.nn.MaxPool2d(2, 2),
            conv2DBatchNormRelu(64, 128, 3, 1, 1),
            conv2DBatchNormRelu(128, 128, 3, 1, 1),
            th.nn.MaxPool2d(2, 2),
            conv2DBatchNormRelu(128, 256, 3, 1, 1),
            conv2DBatchNormRelu(256, 256, 3, 1, 1),
            conv2DBatchNormRelu(256, 256, 3, 1, 1),
            th.nn.MaxPool2d(2, 2),
            conv2DBatchNormRelu(256, 512, 3, 1, 1),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            th.nn.MaxPool2d(2, 2),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            th.nn.MaxPool2d(2, 2),
            th.nn.Tanh(),
        )

        self.decoder = th.nn.Sequential(
            th.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            th.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            conv2DBatchNormRelu(512, 256, 3, 1, 1),
            th.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv2DBatchNormRelu(256, 256, 3, 1, 1),
            conv2DBatchNormRelu(256, 256, 3, 1, 1),
            conv2DBatchNormRelu(256, 128, 3, 1, 1),
            th.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv2DBatchNormRelu(128, 128, 3, 1, 1),
            conv2DBatchNormRelu(128, 64, 3, 1, 1),
            th.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv2DBatchNormRelu(64, 64, 3, 1, 1),
            conv2DBatchNormRelu(64, in_channels, 3, 1, 1),
        )

    def encode(self, inputs):
        return self.encoder(inputs)

    def decode(self, inputs):
        return self.decoder(inputs)

    def forward(self, inputs):
        z = self.encode(inputs)
        # print(z.min(), z.mean(), z.max(), z.shape)
        return self.decode(z)


class VariationalConvSegNet(th.nn.Module):
    """
    Adapted from https://github.com/foamliu/Autoencoder
    See LICENSE_AUTOENCODER
    """

    def __init__(self, in_channels=3):
        super(VariationalConvSegNet, self).__init__()

        self.encoder = th.nn.Sequential(
            conv2DBatchNormRelu(in_channels, 64, 3, 1, 1),
            conv2DBatchNormRelu(64, 64, 3, 1, 1),
            th.nn.MaxPool2d(2, 2),
            conv2DBatchNormRelu(64, 128, 3, 1, 1),
            conv2DBatchNormRelu(128, 128, 3, 1, 1),
            th.nn.MaxPool2d(2, 2),
            conv2DBatchNormRelu(128, 256, 3, 1, 1),
            conv2DBatchNormRelu(256, 256, 3, 1, 1),
            conv2DBatchNormRelu(256, 256, 3, 1, 1),
            th.nn.MaxPool2d(2, 2),
            conv2DBatchNormRelu(256, 512, 3, 1, 1),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            th.nn.MaxPool2d(2, 2),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            th.nn.MaxPool2d(2, 2),
            th.nn.Tanh(),
            Flatten(),
        )

        self.fc_mu = th.nn.Linear(512 * 4 * 4, 512 * 4 * 4)
        self.fc_var = th.nn.Linear(512 * 4 * 4, 512 * 4 * 4)

        self.decoder = th.nn.Sequential(
            UnFlatten(512, 4),
            th.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            th.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            conv2DBatchNormRelu(512, 512, 3, 1, 1),
            conv2DBatchNormRelu(512, 256, 3, 1, 1),
            th.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv2DBatchNormRelu(256, 256, 3, 1, 1),
            conv2DBatchNormRelu(256, 256, 3, 1, 1),
            conv2DBatchNormRelu(256, 128, 3, 1, 1),
            th.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv2DBatchNormRelu(128, 128, 3, 1, 1),
            conv2DBatchNormRelu(128, 64, 3, 1, 1),
            th.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            conv2DBatchNormRelu(64, 64, 3, 1, 1),
            conv2DBatchNormRelu(64, in_channels, 3, 1, 1),
            th.nn.Tanh(),
        )

    def reparameterize(self, mu, log_var):
        std = th.exp(0.5 * log_var)
        eps = th.randn_like(std)
        return eps * std + mu

    def encode(self, inputs):
        result = self.encoder(inputs)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return mu, log_var

    def decode(self, inputs):
        return self.decoder(inputs)

    def forward(self, inputs):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        return self.decode(z)

import torch
import torch.nn as nn
import torch.nn.functional as F
from op.fused_act import FusedLeakyReLU

## Encoder
def create_encoder_single_conv(in_chs, out_chs, kernel):
    assert kernel % 2 == 1
    return nn.Sequential(
        nn.Conv2d(in_chs, out_chs, kernel_size=kernel, padding=(kernel - 1) // 2),
        nn.BatchNorm2d(out_chs),
        FusedLeakyReLU(out_chs),
    )


class EncoderInceptionModuleSignle(nn.Module):
    def __init__(self, channels):
        assert channels % 2 == 0
        super().__init__()
        # put bottle-neck layers before convolution
        bn_ch = channels // 2
        self.bottleneck = create_encoder_single_conv(channels, bn_ch, 1)
        # bn -> Conv1, 3, 5
        self.conv1 = create_encoder_single_conv(bn_ch, channels, 1)
        self.conv3 = create_encoder_single_conv(bn_ch, channels, 3)
        self.conv5 = create_encoder_single_conv(bn_ch, channels, 5)
        self.conv7 = create_encoder_single_conv(bn_ch, channels, 7)
        # pool-proj(no-bottle neck)
        self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(5, stride=1, padding=2)

    def forward(self, x):
        # Original inception is concatenation, but use simple addition instead
        bn = self.bottleneck(x)
        out = self.conv1(bn) + self.conv3(bn) + self.conv5(bn) + self.conv7(bn) + self.pool3(x) + self.pool5(x)
        return out


class EncoderModule(nn.Module):
    def __init__(self, chs, repeat_num, use_inception):
        super().__init__()
        if use_inception:
            layers = [EncoderInceptionModuleSignle(chs) for i in range(repeat_num)]
        else:
            layers = [create_encoder_single_conv(chs, chs, 3) for i in range(repeat_num)]
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class Encoder(nn.Module):
    def __init__(self, use_inception, repeat_per_module):
        super().__init__()
        # stages
        self.upch1 = nn.Conv2d(3, 32, kernel_size=3)
        self.stage1 = EncoderModule(32, repeat_per_module, use_inception)
        self.upch2 = self._create_downsampling_module(32, 2)
        self.stage2 = EncoderModule(64, repeat_per_module, use_inception)
        self.upch3 = self._create_downsampling_module(64, 2)
        self.stage3 = EncoderModule(128, repeat_per_module, use_inception)
        self.upch4 = self._create_downsampling_module(128, 2)
        self.stage4 = EncoderModule(256, repeat_per_module, use_inception)

    def _create_downsampling_module(self, input_channels, pooling_kenel):
        return nn.Sequential(
            nn.AvgPool2d(pooling_kenel),
            nn.Conv2d(input_channels, input_channels * 2, kernel_size=1),
            nn.BatchNorm2d(input_channels * 2),
            FusedLeakyReLU(input_channels * 2),
        )

    def forward(self, x):
        # print(x.shape)
        out = self.stage1(self.upch1(x))
        # print(out.shape)
        out = self.stage2(self.upch2(out))
        # print(out.shape)
        out = self.stage3(self.upch3(out))
        # print(out.shape)
        out = self.stage4(self.upch4(out))
        # print(out.shape)
        out = F.avg_pool2d(out, 8)  # Global Average pooling
        # print(out.shape)
        return out.view(-1, 256)


## Decoder
def create_decoder_single_conv(in_chs, out_chs, kernel):
    assert kernel % 2 == 1
    return nn.Sequential(
        nn.ConvTranspose2d(in_chs, out_chs, kernel_size=kernel, padding=(kernel - 1) // 2),
        nn.BatchNorm2d(out_chs),
        FusedLeakyReLU(out_chs),
    )


class DecoderInceptionModuleSingle(nn.Module):
    def __init__(self, channels):
        assert channels % 2 == 0
        super().__init__()
        # put bottle-neck layers before convolution
        bn_ch = channels // 4
        self.bottleneck = create_decoder_single_conv(channels, bn_ch, 1)
        # bn -> Conv1, 3, 5
        self.conv1 = create_decoder_single_conv(bn_ch, channels, 1)
        self.conv3 = create_decoder_single_conv(bn_ch, channels, 3)
        self.conv5 = create_decoder_single_conv(bn_ch, channels, 5)
        self.conv7 = create_decoder_single_conv(bn_ch, channels, 7)
        # pool-proj(no-bottle neck)
        self.pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(5, stride=1, padding=2)

    def forward(self, x):
        # Original inception is concatenation, but use simple addition instead
        bn = self.bottleneck(x)
        out = self.conv1(bn) + self.conv3(bn) + self.conv5(bn) + self.conv7(bn) + self.pool3(x) + self.pool5(x)
        return out


class DecoderModule(nn.Module):
    def __init__(self, chs, repeat_num, use_inception):
        super().__init__()
        if use_inception:
            layers = [DecoderInceptionModuleSingle(chs) for i in range(repeat_num)]
        else:
            layers = [create_decoder_single_conv(chs, chs, 3) for i in range(repeat_num)]
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return self.convs(x)


class Decoder(nn.Module):
    def __init__(self, use_inception, repeat_per_module):
        super().__init__()
        # stages
        self.stage1 = DecoderModule(256, repeat_per_module, use_inception)
        self.downch1 = self._create_upsampling_module(256, 2)
        self.stage2 = DecoderModule(128, repeat_per_module, use_inception)
        self.downch2 = self._create_upsampling_module(128, 2)
        self.stage3 = DecoderModule(64, repeat_per_module, use_inception)
        self.downch3 = self._create_upsampling_module(64, 2)
        self.stage4 = DecoderModule(32, repeat_per_module, use_inception)
        self.downch4 = self._create_upsampling_module(32, 2)
        self.last = nn.ConvTranspose2d(16, 3, kernel_size=1)

    def _create_upsampling_module(self, input_channels, pooling_kenel):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channels, input_channels // 2, kernel_size=pooling_kenel, stride=pooling_kenel),
            nn.BatchNorm2d(input_channels // 2),
            FusedLeakyReLU(input_channels // 2),
        )

    def forward(self, x):
        out = F.upsample(x.view(-1, 256, 1, 1), scale_factor=8)
        out = self.downch1(self.stage1(out))
        out = self.downch2(self.stage2(out))
        out = self.downch3(self.stage3(out))
        out = self.downch4(self.stage4(out))
        return torch.sigmoid(self.last(out))


## VAE
class InceptionVAE(nn.Module):
    def __init__(self, latent_dim=512, repeat_per_block=1, use_inception=True):
        super(InceptionVAE, self).__init__()

        # # latent features
        self.n_latent_features = latent_dim

        # Encoder
        self.encoder = Encoder(use_inception, repeat_per_block)
        # Middle
        self.fc_mu = nn.Linear(256, self.n_latent_features)
        self.fc_logvar = nn.Linear(256, self.n_latent_features)
        self.fc_rep = nn.Linear(self.n_latent_features, 256)
        # Decoder
        self.decoder = Decoder(use_inception, repeat_per_block)

    def _reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).cuda()
        z = mu + std * esp
        return z

    def _bottleneck(self, h):
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self._reparameterize(mu, logvar)
        return z, mu, logvar

    def sampling(self):
        # assume latent features space ~ N(0, 1)
        z = torch.randn(24, self.n_latent_features).cuda()
        z = self.fc_rep(z)
        # decode
        return self.decoder(z)

    def forward(self, x):
        # Encoder
        h = self.encoder(x)
        # Bottle-neck
        z, mu, logvar = self._bottleneck(h)
        # decoder
        z = self.fc_rep(z)
        d = self.decoder(z)
        return d, mu, logvar

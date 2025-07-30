# model.py
import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from torch.nn.utils import spectral_norm

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(weights=VGG19_Weights.DEFAULT)
        # Use features before activation as in the original ESRGAN paper
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35]).eval()

    def forward(self, img):
        # VGG expects a 3-channel image. Repeat if necessary.
        if img.shape[1] == 1:
             img = img.repeat(1, 3, 1, 1)
        return self.feature_extractor(img)

class DenseResidualBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU(0.2, inplace=True)]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = nn.ModuleList([self.b1, self.b2, self.b3, self.b4, self.b5])

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters),
            DenseResidualBlock(filters),
            DenseResidualBlock(filters)
        )
    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x

class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=16, num_upsample=2):
        super(GeneratorRRDB, self).__init__()
        # First convolution
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual-in-Residual Dense Blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second convolution after RRDBs
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)

        # Output layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        in_channels, _, _ = self.input_shape

        # --- UPDATE: Added Spectral Normalization to each Conv2d layer ---
        # This stabilizes the discriminator's training, leading to more robust results.
        def discriminator_block(in_filters, out_filters):
            layers = [
                spectral_norm(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            return layers

        layers = []
        in_filters = in_channels
        for _, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters))
            in_filters = out_filters

        layers.append(spectral_norm(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1)))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
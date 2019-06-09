import torch
from torch.nn import LeakyReLU, PReLU, Conv2d, BatchNorm2d, PixelShuffle, Linear, Sigmoid
from src.loss_functions import mse_loss


class ResidualBlock(torch.nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2d(64, 64, 3)
        self.bn1 = BatchNorm2d(64)

        self.prelu = PReLU()

        self.conv2 = Conv2d(64, 64, 3)
        self.bn2 = BatchNorm2d(64)

    def forward(self, x):
        identity = x
        out = self.conv1()
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity

        res = self.prelu(out)
        return res


class Generator(torch.nn.Module):
    model = None
    def __init__(self, num_blocks=1):
        super(Generator, self).__init__()
        self.conv1 = Conv2d(1, 64, kernel_size=9)
        self.prelu = PReLU()

        self.layers = self.get_residual_blocks(num_blocks)

        self.conv2 = Conv2d(64, 64, kernel_size=3)
        self.bn2 = BatchNorm2d(64)

        self.conv3 = Conv2d(1, 256, kernel_size=3)
        self.pxshuffle = PixelShuffle()
        self.bn3 = BatchNorm2d(64)

        self.conv4 = Conv2d(1, 256, kernel_size=3)
        self.bn4 = BatchNorm2d(256)

        self.conv5 = Conv2d(256, 1, kernel_size=9)


    def get_residual_blocks(self, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(ResidualBlock())

        return torch.nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.prelu(out)

        out1 = out
        out = self.layers(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += out1
        out = self.prelu(out)

        out = self.conv3(out)
        out = self.pxshuffle(out)
        out = self.prelu(out)

        out = self.conv4(out)
        out = self.pxshuffle(out)
        out = self.prelu(out)

        res = self.conv5(out)
        return res


class DiscriminatorBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DiscriminatorBlock, self).__init__()

        self.model = torch.nn.Sequential([
            Conv2d(in_channels, out_channels, 3, stride),
            BatchNorm2d(out_channels),
            LeakyReLU(),
        ])

    def forward(self, x):
        return self.model(x)


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.num_blocks = 3
        self.model = torch.nn.Sequential([
            Conv2d(1, 64, 3),
            LeakyReLU(),
            DiscriminatorBlock(64, 64, 2),
            DiscriminatorBlock(64, 128, 1),
            DiscriminatorBlock(128, 128, 2),
            DiscriminatorBlock(128, 256, 1),
            DiscriminatorBlock(256, 256, 2),
            DiscriminatorBlock(256, 512, 1),
            DiscriminatorBlock(512, 512, 2),
            Flatten(),
            Linear(32*32*512, 1024),
            LeakyReLU(),
            Linear(1024, 1),
            Sigmoid()
        ])

    def forward(self, x):
        return self.model(x)


''' Super-Resolution GAN model class.'''


class SuperResolutionGAN(torch.nn.Module):
    batch_size = None
    num_generator_steps = None
    num_discriminator_steps = None
    num_epochs = 100000
    generator = None
    discriminator = None

    def __init__(self):
        super(SuperResolutionGAN, self).__init__()
        self.discriminator = Discriminator()
        self.generator = Generator()

    def forward(self, x):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError

    def run(self):
        pass


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

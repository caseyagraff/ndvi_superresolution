import torch
from torch.nn import LeakyReLU, PReLU, Conv2d, BatchNorm2d, PixelShuffle, Linear, Sigmoid
from src.utils.loss_functions import mse_loss

class ResidualBlock(torch.nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2d(64, 64, 3, padding=1)
        self.bn1 = BatchNorm2d(64)

        self.prelu = PReLU()

        self.conv2 = Conv2d(64, 64, 3, padding=1)
        self.bn2 = BatchNorm2d(64)

    def forward(self, x):
        identity = x
        out = self.conv1(identity)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity

        res = self.prelu(out)
        return res


class Generator(torch.nn.Module):

    def __init__(self, num_blocks=1):
        super(Generator, self).__init__()
        print("Initialized generator network..")
        self.conv1 = Conv2d(1, 64, kernel_size=9, padding=4)
        self.prelu = PReLU()

        self.layers = self.get_residual_blocks(num_blocks)

        self.conv2 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(64)

        self.conv3 = Conv2d(64, 256, kernel_size=3, padding=1)
        self.pxshuffle = PixelShuffle(upscale_factor=2)  # up-sampling

        # used in original SR-GAN paper, only for 4x up-sampling
        # self.conv4 = Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv5 = Conv2d(64, 1, kernel_size=9, padding=4)

    def get_residual_blocks(self, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(ResidualBlock())

        return torch.nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv1(x)
        out = self.prelu(out)

        identity = out
        out = self.layers(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.prelu(out)

        out = self.conv3(out)
        out = self.pxshuffle(out)
        out = self.prelu(out)

        res = self.conv5(out)
        print(res.shape)
        return res


class DiscriminatorBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super(DiscriminatorBlock, self).__init__()

        self.model = torch.nn.Sequential(*[
            Conv2d(in_channels, out_channels, 3, stride, padding),
            BatchNorm2d(out_channels),
            LeakyReLU()
        ])

    def forward(self, x):
        return self.model(x)


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        print("Initialized discriminator network..")
        self.num_blocks = 3
        self.model = torch.nn.Sequential(*[
            Conv2d(1, 64, 3, 1),
            LeakyReLU(),
            DiscriminatorBlock(64, 64, 2, 1),
            DiscriminatorBlock(64, 128, 1, 1),
            DiscriminatorBlock(128, 128, 2, 1),
            DiscriminatorBlock(128, 256, 1, 1),
            DiscriminatorBlock(256, 256, 2, 1),
            DiscriminatorBlock(256, 512, 1, 1),
            DiscriminatorBlock(512, 512, 2, 1),
            Flatten(),
            Linear(320000, 1024),
            LeakyReLU(),
            Linear(1024, 1),
            Sigmoid()
        ])

    def forward(self, x):
        return self.model(x)


''' Super-Resolution GAN model class.'''


class SuperResolutionGAN(torch.nn.Module):
    # default parameters used in original paper
    batch_size = None
    num_generator_steps = None
    num_discriminator_steps = None
    num_epochs = 100000
    generator = None
    discriminator = None
    learning_rate = 0.00001

    def __init__(self):
        super(SuperResolutionGAN, self).__init__()
        print("Initialized SR-GAN model..")

    def forward(self, x):
        generator = Generator()
        sample_high_res = generator.forward(x)

        discriminator = Discriminator()
        pred = discriminator.forward(sample_high_res)
        print("Discriminator predictions: {}".format(pred))

    def test(self):
        raise NotImplementedError

    def run(self):
        pass


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        print("batch size: {}".format(batch_size))
        return x.view(batch_size, -1)

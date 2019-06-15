import torch
from torch.nn import LeakyReLU, PReLU, Conv2d, BatchNorm2d, PixelShuffle, Linear, Sigmoid

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


class GeneratorFull(torch.nn.Module):

    def __init__(self, num_blocks=1):
        super().__init__()
        print("Initialized generator network..")
        self.conv1 = Conv2d(1, 64, kernel_size=9, padding=4)
        self.prelu = PReLU()

        self.layers = self._get_residual_blocks(num_blocks)

        self.conv2 = Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(64)

        self.conv3 = Conv2d(64, 256, kernel_size=3, padding=1)
        self.pxshuffle = PixelShuffle(upscale_factor=2)  # up-sampling

        # used in original SR-GAN paper, only for 4x up-sampling
        # self.conv4 = Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv5 = Conv2d(64, 1, kernel_size=9, padding=4)

    def _get_residual_blocks(self, num_blocks):
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
        return res


class DiscriminatorBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super().__init__()

        self.model = torch.nn.Sequential(*[
            Conv2d(in_channels, out_channels, 3, stride, padding),
            BatchNorm2d(out_channels),
            LeakyReLU()
        ])

    def forward(self, x):
        return self.model(x)


class DiscriminatorFull(torch.nn.Module):
    def __init__(self, high_res):
        super().__init__()
        print("Initialized discriminator network..")
        self.num_blocks = 3
        self.model = torch.nn.Sequential(*[
            Conv2d(1, 32, 3, 1),
            LeakyReLU(),
            DiscriminatorBlock(32, 32, 2, 1),
            DiscriminatorBlock(32, 64, 1, 1),
            DiscriminatorBlock(64, 64, 2, 1),
            DiscriminatorBlock(64, 128, 1, 1),
            DiscriminatorBlock(128, 128, 2, 1),
            DiscriminatorBlock(128, 256, 1, 1),
            DiscriminatorBlock(256, 256, 2, 1),
            Flatten(),
            Linear(1*high_res*high_res, 256),
            LeakyReLU(),
            Linear(256, 1),
            Sigmoid()
        ])

    def forward(self, x):
        return self.model(x)

class DiscriminatorFullSiamese(torch.nn.Module):
    def __init__(self, high_res):
        super().__init__()
        print("Initialized discriminator network..")
        self.num_blocks = 3
        self.model = torch.nn.Sequential(*[
            Conv2d(1, 32, 3, 1),
            LeakyReLU(),
            DiscriminatorBlock(32, 32, 2, 1),
            DiscriminatorBlock(32, 64, 1, 1),
            DiscriminatorBlock(64, 64, 2, 1),
            DiscriminatorBlock(64, 128, 1, 1),
            DiscriminatorBlock(128, 128, 2, 1),
            DiscriminatorBlock(128, 256, 1, 1),
            DiscriminatorBlock(256, 256, 2, 1),
            Flatten(),
            Linear(1*high_res*high_res, 256),
            LeakyReLU(),
            ])

        self.decider = torch.nn.Sequential(*[
            Linear(512, 1),
            Sigmoid(),
        ])

    def forward(self, x):
        x1, x2 = x
        a = self.model(x1)
        b = self.model(x2)

        return self.decider(torch.cat((a,b), dim=1))

class GeneratorSample(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torch.nn.Sequential(*[
            Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            PixelShuffle(upscale_factor=2),
            Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1),
        ])

    def forward(self, x):
        return self.model(x)

class DiscriminatorSample(torch.nn.Module):
    def __init__(self, high_res_dim):
        super().__init__()

        self.model = torch.nn.Sequential(*[
            Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1),
            Flatten(),
            Linear(high_res_dim**2, 1),
            Sigmoid(),
        ])

    def forward(self, x):
        return self.model(x)


''' Super-Resolution GAN model class.'''
class SuperResolutionGAN(torch.nn.Module):

    def __init__(self, low_resolution_dim, high_resolution_dim, generator, discriminator, **kwargs):
        super(SuperResolutionGAN, self).__init__()
        print("Initializing SR-GAN model.....")
        self.lr_dim = low_resolution_dim
        self.hr_dim = high_resolution_dim
        self.generator = generator
        self.discriminator = discriminator

        for k, v in kwargs.items():
            self.k = v

    def forward(self):
        raise NotImplementedError()

    def save(self, filename, epochs, generator_optimizer, discriminator_optimizer, loss):
        checkpoint = {
            'epoch': epochs,
            'model_state_dict': self.state_dict(),
            'generator_optimizer_state_dict': generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict() if discriminator_optimizer is not None else None,
            'loss': loss
        }
        torch.save(checkpoint, filename)

    def load(self, filename, generator_optimizer=None, discriminator_optimizer=None):
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['model_state_dict'])

        if generator_optimizer:
            generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])

        if discriminator_optimizer:
            discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])

        loss = checkpoint['loss']

        epochs = checkpoint['epoch']

        return epochs, generator_optimizer, discriminator_optimizer, loss

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1) 

'''
Loss functions used for training and evaluation.
'''

import torch
import torchvision
from torch.nn import MSELoss, BCELoss
from torch.autograd import Variable


def select_content_loss(loss_name, params):
    if loss_name is None:
        return None
    elif loss_name == 'l2':
        return mse_loss(params)
    elif loss_name == 'vgg':
        return vgg_loss(params)
    else:
        raise ValueError(f'Content loss "{loss_name}" is not valid.')


'''
Pixel-wise Mean Squared Error loss.
'''


def mse_loss(params=None):
    loss_fn = torch.nn.MSELoss()

    def _mse_loss(real_high_res, fake_high_res):
        return loss_fn(fake_high_res, real_high_res)

    return _mse_loss


'''
Pixel-wise Mean Squared Error based on VGG (trained) feature maps.
'''


def vgg_loss(params=None):
    vgg_net = torchvision.models.vgg19(pretrained=True, progress=True)  # VGG19 sans batch-normalization
    layer = 5
    if params is not None:
        layer = params.vgg_layer

    model = torch.nn.Sequential(*list(vgg_net.features.children())[:layer])

    for param in model.parameters():
            param.requires_grad = False

    loss_fn = MSELoss()

    def _vgg_loss(real_high_res, fake_high_res):
        model.eval()
        real_high_res_out = model(torch.cat((real_high_res, real_high_res, real_high_res), dim=1))
        fake_high_res_out = model(torch.cat((fake_high_res, fake_high_res, fake_high_res), dim=1))
        model.train()

        return loss_fn(real_high_res_out, fake_high_res_out)

    return _vgg_loss


def perceptual_loss(params=None):
    content_loss_fn = select_content_loss(params.loss_name)
    generator_loss = gan_loss()
    alpha = 0.001
    if params is not None:
        alpha = params.alpha

    def _perceptual_loss(real_high_res, fake_high_res):
        return content_loss_fn(real_high_res, fake_high_res) + alpha * generator_loss(real_high_res, fake_high_res)[0]

    return _perceptual_loss


'''
Takes in a batch of real and generated high resolution images, and returns a binary cross entropy loss.
Input: 
real_high_res: [batch_size, high_resolution_dim, high_resolution_dim], set of real high-res images.
fake_high_res: [batch_size, high_resolution_dim, high_resolution_dim], set of fake high-res images.
discriminator: Discriminator class object.

Output:
# loss: scalar, standard GAN loss.
'''


def gan_loss(params=None):
    loss_fn = BCELoss()

    def _gan_loss(real_high_res, fake_high_res, discriminator):
        num_real = len(real_high_res)
        num_fake = len(fake_high_res)

        batch_size = num_real + num_fake

        x = torch.cat((real_high_res, fake_high_res))
        y = torch.zeros(batch_size)

        # set y = 1 for real data-set images
        y[:num_real] = 1

        y_predict = discriminator(x)

        discriminator_loss = loss_fn(y_predict.squeeze(), y)

        # compute generator loss function only for fake images
        generator_loss = (-torch.log(y_predict[num_real:])).mean()

        return discriminator_loss, generator_loss

    return _gan_loss

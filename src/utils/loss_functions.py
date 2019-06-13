import torch
from torch.nn import MSELoss, BCELoss


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
    def _mse_loss(real_high_res, fake_high_res):
        loss = torch.nn.MSELoss(real_high_res, fake_high_res)
        return loss

    return _mse_loss


'''
Pixel-wise Mean Squared Error based on VGG (trained) feature maps.
'''


def vgg_loss(params=None):
    def _vgg_loss(real_high_res, fake_high_res, layer_num=2):
        raise NotImplementedError()

    return _vgg_loss


def perceptual_loss():
    raise NotImplementedError()


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
    def _gan_loss(real_high_res, fake_high_res, discriminator):
        num_real = len(real_high_res)
        num_fake = len(fake_high_res)

        batch_size = num_real + num_fake

        x = torch.cat(real_high_res, fake_high_res)
        y = torch.zeros(batch_size)
        # set y = 1 for real data-set images
        y[:num_real] = 1
        y_predict = discriminator(x)

        discriminator_loss = BCELoss()(y, y_predict)

        # compute generator loss function only for fake images
        generator_loss = -torch.log(y_predict[num_real:])

        return discriminator_loss, generator_loss

    return _gan_loss

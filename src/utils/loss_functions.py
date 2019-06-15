'''
Loss functions used for training and evaluation.
'''

import numpy as np
import torch
import torchvision
from torch.nn import MSELoss, BCELoss
from torch.autograd import Variable
try:
    from CannyEdgePytorch.net_canny import Net
except:
    print('CannyEdgePytorch.net_canny not found')


def select_content_loss(loss_name, params, device):
    if loss_name == 'None':
        return None
    elif loss_name == 'l2':
        return mse_loss(params, device)
    elif loss_name == 'vgg':
        return vgg_loss(params, device) 
    elif loss_name == 'edge_loss':
        return canny_edge_loss(params, device)
    else:
        raise ValueError(f'Content loss "{loss_name}" is not valid.')


def canny_edge_loss(params=None, device=None):
    loss_fn = torch.nn.MSELoss()
    net = Net(threshold=params.threshold, use_cuda=False)
    for param in net.parameters():
        param.requires_grad = False

    def _canny_edge_loss(real_high_res, fake_high_res):
        net.eval()
        batch_size = real_high_res.shape[0]
        losses = torch.zeros(batch_size)
        for i in range(batch_size):

            real_high_res_out = _canny(torch.cat((real_high_res[i], real_high_res[i], real_high_res[i]), dim=0), i)
            fake_high_res_out = _canny(torch.cat((fake_high_res[i], fake_high_res[i], fake_high_res[i]), dim=0), i)

            losses[i] = loss_fn(fake_high_res_out, real_high_res_out)

        return loss_fn(real_high_res, fake_high_res) + params.edge_loss_coef * torch.mean(losses)

    def _canny(raw_img, idx, use_cuda=False):
        batch = raw_img.unsqueeze(dim=0)

        if use_cuda:
            net.cuda()

        data = batch
        if use_cuda:
            data = batch.cuda()

        blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold = net(data)

        # # print(thresholded.shape)
        # filename = 'edge_detection_{}.png'.format(idx)
        # plt.subplot(2,1,1)
        # plt.imshow(raw_img.squeeze(dim=0)[0, :].detach().data.numpy(), cmap='gray')
        # plt.subplot(2, 1, 2)
        #
        # plt.imshow(thresholded.squeeze(dim=0)[0, :].squeeze(dim=0).detach().data.numpy(), cmap='gray')
        # plt.savefig(filename)

        return grad_mag.squeeze(dim=0)

    return _canny_edge_loss


'''
Pixel-wise Mean Squared Error loss.
'''


def mse_loss(params=None, device=None):
    loss_fn = torch.nn.MSELoss()

    def _mse_loss(real_high_res, fake_high_res):
        return loss_fn(real_high_res, fake_high_res)

    return _mse_loss


'''
Pixel-wise Mean Squared Error based on VGG (trained) feature maps.
'''


def vgg_loss(params=None, device=None):
    vgg_net = torchvision.models.vgg19(pretrained=True, progress=True)  # VGG19 sans batch-normalization
    layer = 35
    if params is not None:
        layer = params.vgg_layer

    model = torch.nn.Sequential(*list(vgg_net.features.children())[:layer]).to(device)

    #for param in model.parameters():
    #        param.requires_grad = False

    loss_fn = MSELoss()

    def _vgg_loss(real_high_res, fake_high_res):
        model.eval()
        real_high_res_out = model(torch.cat((real_high_res, real_high_res, real_high_res), dim=1)).detach()
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
def select_gan_loss(loss_name, params, device):
    if loss_name == 'normal':
        return gan_loss(params, device)
    elif loss_name == 'siamese':
        return siamese_gan_loss(params, device) 
    else:
        raise ValueError(f'Content loss "{loss_name}" is not valid.')


def gan_loss(params=None, device=None):
    def _gan_loss(real_high_res, fake_high_res, discriminator):
        g_loss = -torch.log(discriminator(fake_high_res)).mean()

        discrim_out_real = discriminator(real_high_res)
        discrim_out_fake = discriminator(fake_high_res.detach())

        d_loss = (-torch.log(discrim_out_real) - torch.log(1 - discrim_out_fake)).mean()

        return d_loss, g_loss

    return _gan_loss

def siamese_gan_loss(params=None, device=None):
    # Discriminator predicts 1 if right side (second input) is real and 0 if left side is real
    def _siamese_gan_loss(real_high_res, fake_high_res, discriminator):
        fake_is_right_side = np.random.randint(0,2)

        g_loss = -torch.log(discriminator((real_high_res, fake_high_res))).mean()

        if fake_is_right_side:
            d_loss = -torch.log(1 - discriminator((real_high_res, fake_high_res.detach()))).mean()
        else:
            d_loss = -torch.log(discriminator((fake_high_res.detach(), real_high_res))).mean()

        return d_loss, g_loss

    return _siamese_gan_loss

"""
def gan_loss(params=None, device=None):
    loss_fn = BCELoss()

    def _gan_loss(real_high_res, fake_high_res, discriminator):
        num_real = len(real_high_res)
        num_fake = len(fake_high_res)

        batch_size = num_real + num_fake

        x = torch.cat((real_high_res, fake_high_res))
        y = torch.zeros(batch_size).to(device)

        # set y = 1 for real data-set images
        y[:num_real] = 1

        # y_predict = discriminator(x)

        y_predict_real = discriminator(real_high_res).squeeze()
        y_predict_fake = discriminator(fake_high_res).squeeze()

        real_contr = -torch.log(y_predict_real).mean()
        fake_contr = -torch.log(1. - y_predict_fake).mean()
        discriminator_loss = real_contr + fake_contr

        # discriminator_loss = loss_fn(y_predict.squeeze(), y)
        # discriminator_loss = -y*torch.log(y_predict.squeeze()) - (1-y)*torch.log(1 - y_predict.squeeze())

        # compute generator loss function only for fake images
        # generator_loss = (-torch.log(y_predict.squeeze()[num_real:])).mean()
        generator_loss = torch.log(1-y_predict_fake).mean()

        return discriminator_loss, generator_loss

    return _gan_loss
"""

'''
Pixel-wise Mean Squared Error loss
'''

def select_perceptive_loss(loss_name, params):
    if loss_name is None:
        return None
    elif loss_name=='l2':
        raise NotImplementedError()
    elif loss_name=='vgg':
        raise NotImplementedError()
    else:
        raise ValueError(f'Perceptive loss "{loss_name}" is not valid.')


def mse_loss():
    raise NotImplementedError()


def perceptual_loss():
    raise NotImplementedError()


def adversarial_loss():
    raise NotImplementedError()

'''
Loss functions used for training and evaluation.
'''

def select_content_loss(loss_name, params):
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

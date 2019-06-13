"""
Instantiating models.
"""

from . import models

class GAN:
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator

class ModelFactory:
    @staticmethod
    def create_model(model_name, model_params):
        return GAN(None, None)

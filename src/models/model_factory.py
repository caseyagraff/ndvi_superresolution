"""
Instantiating models.
"""
from .models import SuperResolutionGAN


class ModelFactory:
    @staticmethod
    def create_model(model_name, model_params):

        if model_name == "SR-GAN":
            low_resolution_dim = model_params.low_resolution_dim
            high_resolution_dim = model_params.high_resolution_dim
            return SuperResolutionGAN(low_resolution_dim, high_resolution_dim)


"""
Instantiating models.
"""
from . import models


class ModelFactory:
    @staticmethod
    def create_model(model_name, model_params, low_res_dim, high_res_dim):
        if model_name == "sr_gan":
            generator= models.GeneratorFull(model_params.generator_blocks)
            discriminator  = models.DiscriminatorFull(high_res_dim)

            return models.SuperResolutionGAN(low_res_dim, high_res_dim, 
                    generator, discriminator)
        elif model_name == "siamese_gan":
            generator = models.GeneratorFull(model_params.generator_blocks)
            discriminator = models.DiscriminatorFullSiamese(high_res_dim)

            return models.SuperResolutionGAN(low_res_dim, high_res_dim, 
                    generator, discriminator)

        elif model_name == 'sample_gan':
            generator, discriminator = models.GeneratorSample(), models.DiscriminatorSample(high_res_dim)
            return models.SuperResolutionGAN(low_res_dim, high_res_dim, 
                    generator, discriminator)

        else:
            raise ValueError(f'Model Name "{model_name}" is invalid.')



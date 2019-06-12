"""
Instantiating models.
"""

from . import models

class ModelFactory:
    @staticmethod
    def create_model(model_name, model_params):
        raise NotImplementedError()

"""
Parsing config file parameters.
"""

from collections import namedtuple

import yaml

DEFAULT_PARAMS = {
        'general': {
            'seed': 123,
            'experiment_name': 'default',
        },

        'data': {
            'data_dir': './data/modis_ndvi_processed/',
        },

        'model': {
            'model_name': 'sr_gan',
        },

        'train': {
            'num_epochs': 1,
        },

        'results': {
            'results_dir': './results/',
            'overwrite': False,
        },

}

class Data(dict):
    __getattr__= dict.__getitem__
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__

class Parameters:
    default_params = DEFAULT_PARAMS

    @staticmethod
    def parse(file_name):
        with open(file_name, 'rb') as f_in:
            params = yaml.safe_load(f_in)

        params = Parameters.apply_defaults(params)
        params = Parameters.convert_to_class(params)

        return params

    @staticmethod
    def apply_defaults(params):
        combined_params = Parameters.default_params.copy()

        for param_type in combined_params.keys():
            if param_type in params:
                for k, v in params[param_type].items():
                    combined_params[param_type][k] = v
                    
        return combined_params

    @staticmethod
    def convert_to_class(params):
        new_dict = Data()

        for k, v in params.items():
            if isinstance(v, dict):
                new_dict[k] = Parameters.convert_to_class(v)
            else:
                new_dict[k] = v

        return new_dict



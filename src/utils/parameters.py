"""
Parsing config file parameters.
"""


class Parameters:
    config_dir = './configs/'

    @staticmethod
    def parse(file_name):
        with open(file_name, 'rb') as f_in:
            params = yaml.safe_load(f_in)

        return params


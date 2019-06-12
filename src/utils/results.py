"""
Saving model results.
"""

import os
import shutil

import torch

class Result:
    results_file_name = 'results.yaml'
    model_file_name = 'model.pt'

    def __init__(self, save_dir, overwrite=False):
        self.save_dir = save_dir
        self.overwrite = overwrite

    def create_save_directory(self):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        elif self.overwrite:
            shutil.rmtree(save_dir)
        else:
            raise Exception(f'Save directory "{self.save_dir}" already exists and overwrite is false.')

    def save_results(self, results):
        with open(os.path.join(self.save_dir, self.results_file_name, 'w') as f_out:
            yaml.dump(results, f_out, default_flow_style=False)

    def save_model(self, model):
        model_save_path = os.path.join(self.save_dir, self.model_file_name)
        model.save(model_save_path)

    def load_results(self):
        with open(results_path, 'rb') as f_in:
            return yaml.save_load(f_in)

    def load_model(self):
        raise NotImplementedError()


        




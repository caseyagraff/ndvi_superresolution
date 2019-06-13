"""
Saving model results.
"""

import os
import shutil

import torch

class Result:
    results_file_name = 'results.yaml'
    model_file_name = 'model.pt'

    def __init__(self, results_dir, experiment_name, overwrite=False):
        self.save_dir = os.path.join(results_dir, experiment_name)
        self.overwrite = overwrite

    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        elif self.overwrite:
            shutil.rmtree(self.save_dir)
        else:
            raise Exception(f'Save directory "{self.save_dir}" already exists and overwrite is false.')

    def save_results(self, results):
        results_save_path = os.path.join(self.save_dir, self.results_file_name)

        with open(results_save_path, 'wb') as f_out:
            yaml.dump(results, f_out, default_flow_style=False)

    def save_model(self, model):
        model_save_path = os.path.join(self.save_dir, self.model_file_name)
        model.save(model_save_path)

    def load_results(self):
        results_save_path = os.path.join(self.save_dir, self.results_file_name)

        with open(results_save_path, 'rb') as f_in:
            return yaml.save_load(f_in)

    def load_model(self, model_params):
        model = model_factory.ModelFactory.create_model(model_name.model_name, model_params)

        model_save_path = os.path.join(self.save_dir, self.model_file_name)
        model.load_state_dict(torch.load(model_save_path))

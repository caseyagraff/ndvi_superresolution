"""
Saving model results.
"""

import os
import shutil

import yaml
import torch
from src.models import model_factory 

from ..models import model_factory

class Result:
    results_file_name = 'results.yaml'
    model_file_name = 'model_%s.pt'
    sample_dir = 'samples/'
    eval_file_name = 'eval.yaml'
    model_dir = 'models/'

    def __init__(self, results_dir, experiment_name, overwrite=False):
        self.save_dir = os.path.join(results_dir, experiment_name)
        self.overwrite = overwrite

    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            pass
        elif self.overwrite:
            shutil.rmtree(self.save_dir)
        else:
            raise Exception(f'Save directory "{self.save_dir}" already exists and overwrite is false.')

        os.makedirs(self.save_dir)
        os.makedirs(os.path.join(self.save_dir, self.sample_dir))
        os.makedirs(os.path.join(self.save_dir, self.model_dir))

    def save_results(self, results, save_eval_results=False):
        if not(save_eval_results):
            results_save_path = os.path.join(self.save_dir, self.results_file_name)

            with open(results_save_path, 'w') as f_out:
                yaml.dump(results, f_out, default_flow_style=False)
        else:
            eval_save_path = os.path.join(self.save_dir, self.eval_file_name)
            with open(eval_save_path, 'w') as f_out:
                yaml.dump(results, f_out, default_flow_style=False)

    def save_model(self, model, model_train_state_dict):
        model_save_path = os.path.join(self.save_dir, self.model_dir, self.model_file_name % model_train_state_dict['epoch'])
        model.save(
                model_save_path, 
                model_train_state_dict['epoch'], 
                model_train_state_dict['gen_optimizer'], 
                model_train_state_dict['discrim_optimizer'], 
                model_train_state_dict['loss'], 
        )

    def load_results(self):
        results_save_path = os.path.join(self.save_dir, self.results_file_name)

        with open(results_save_path, 'rb') as f_in:
            return yaml.save_load(f_in)

<<<<<<< HEAD
    def load_model(self, model_params, low_res_data_dim, high_res_data_dim):
        model = model_factory.ModelFactory.create_model(model_params.model_name, model_params, low_res_data_dim, high_res_data_dim)
=======
    def load_model(self, model_params, low_res_dim, high_res_dim, epoch):
        model = model_factory.ModelFactory.create_model(model_params.model_name, model_params, low_res_dim, high_res_dim)
>>>>>>> 84286767b9b02e973b36d70a1d81f97748488d2d

        model_save_path = os.path.join(self.save_dir, self.model_dir, self.model_file_name % epoch)
        model.load(model_save_path)
<<<<<<< HEAD
    
=======

>>>>>>> 84286767b9b02e973b36d70a1d81f97748488d2d
        return model

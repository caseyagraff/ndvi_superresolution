"""
Training models.
"""
import os

from . import trainer
from . import evaluation

from ..utils import results, helper, normalization
from ..data import aggregate_data
from ..models import model_factory

def train_model(params):
    # Create save directory
    model_results = results.Result(
            params.results.results_dir,
            params.general.experiment_name,
            params.results.overwrite,
    )

    model_results.create_save_dir()
 
    # Set random seed
    helper.set_random_seed(params.general.seed)

    # Load data
    data = aggregate_data.AggregatedData.load(params.data.data_path)
    data_tr, data_te = data.data_tr, data.data_te

    # Normalize data (only fit params on training)
    data_tr, normalization_params = normalization.normalize_data_triple(data_tr, params=None)
    data_te, _ = normalization.normalize_data_triple(data_te, params=normalization_params)

    low_res_dim, high_res_dim = data_tr.low_res.shape[1], data_tr.high_res.shape[1]

    # Create model
    model = model_factory.ModelFactory.create_model(params.model.model_name, params.model, 
            low_res_dim, high_res_dim)

    # Create model trainer
    sample_dir = os.path.join(
            params.results.results_dir, 
            params.general.experiment_name, 
            results.Result.sample_dir
    )

    model_trainer = trainer.GanModelTrainer(
            model=model,
            params=params.train,
            sample_dir=sample_dir,
    )

    # Run trainer
    model_trainer.train(
            data=data_tr,
            num_epochs=params.train.num_epochs,
    )

    # Compute summary results
    # model_evaluator = evaluation.ModelEvaluator(model=model)
    # evaluation_results = model_evaluator.evaluate(data=data_te)
    evaluation_results = {'test_err': 0}

    # Save model and summary results
    model_train_state_dict = {
            'epoch': model_trainer.epoch, 
            'gen_optimizer': model_trainer.gen_optimizer,
            'discrim_optimizer': model_trainer.discrim_optimizer,
            'loss': None
    }

    model_results.save_model(model, model_train_state_dict)
    model_results.save_results(evaluation_results)


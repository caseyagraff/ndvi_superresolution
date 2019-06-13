"""
Training models.
"""

from ..utils import results, helper
from ..data import data_loader
from ..models import model_factory
from . import trainer
from . import evaluation

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
    x_train, x_test = data_loader.load_train_test(params.data.data_dir)

    # Create model
    model = model_factory.ModelFactory.create_model(params.model.model_name, params.model)

    # Create model trainer
    model_trainer = trainer.GanModelTrainer(
            model=model,
            params=params.train,
    )

    # Run trainer
    model_trainer.train(
            data=x_train,
            num_epochs=params.train.num_epochs,
    )

    # Compute summary results
    model_evaluator = evaluation.ModelEvaluator(model=model)
    evaluation_results = model_evaluator.evaluate(data=x_test)

    # Save model and summary results
    model_results.save_model(model)
    model_results.save_results(evaluation_results)


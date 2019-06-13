"""
Training models.
"""

from . import trainer
from . import evaluation

from ..utils import results, helper
from ..data import data_loader
from ..models import model_factory
from ..utils import normalization 

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

    # Normalize data (only fit params on training)
    data_tr, params = normalization.normalize_data_triple(data_tr, params=None)
    data_te, _ = normalization.normalize_data_triple(data_te, params=params)

    # Create model
    model = model_factory.ModelFactory.create_model(params.model.model_name, params.model)

    # Create model trainer
    model_trainer = trainer.GanModelTrainer(
            model=model,
            params=params.train,
            sample_dir=os.path.join(params.results.results_dir, Result.sample_dir),
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


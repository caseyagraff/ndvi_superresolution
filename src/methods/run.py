import torch
import torchvision

from ..utils import results, helper, normalization
from ..data import aggregate_data


def run_model(params):
    model_results = results.Result(
            params.results.results_dir,
            params.general.experiment_name,
            params.results.overwrite,
    )

    data = aggregate_data.AggregatedData.load(params.data.data_path)
    data_tr, data_te = data.data_tr, data.data_te

    data_tr, normalization_params = normalization.normalize_data_triple(data_tr, params=None)
    data_te, _ = normalization.normalize_data_triple(data_te, params=normalization_params)

    low_res_dim, high_res_dim = data_tr.low_res.shape[1], data_tr.high_res.shape[1]

    model = model_results.load_model(params.model, low_res_dim, high_res_dim, params.eval.checkpoint_epoch)

    device = torch.device(params.train.device) if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    data_image = torch.tensor(data_te.low_res[7]).float().view(1, 1, low_res_dim, low_res_dim).to(device)

    out = model.generator(data_image)

    torchvision.utils.save_image(out, 'test_out_image.png')

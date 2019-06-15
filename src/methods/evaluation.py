"""
Evaluation code for running model on data.
"""
import torch
import pytorch_ssim
from src.utils import results
from ..data import aggregate_data
from ..utils import results, helper, normalization
from torch.utils import data as t_data
from ..utils import torch_helpers


class ModelEvaluator:
    def __init__(self, model, params):
        self.model = model
        self.params = params

    def evaluate(self, data):
        self.model.generator.eval()
        x_low_data, x_high_data = data.low_res, data.high_res

        data_loader = t_data.DataLoader(torch_helpers.Dataset(x_low_data, x_high_data), 
                batch_size=self.params.train.batch_size, shuffle=self.params.train.shuffle)

        device = torch.device(self.params.train.device) if torch.cuda.is_available() else torch.device('cpu')

        tot_peak_snr = torch.tensor(0.).to(device)
        tot_ssim = torch.tensor(0.).to(device)

        with torch.no_grad():
            tot_num_images = 0.
            # Iterate over mini-batches
            for x_low, x_high in data_loader:
                x_high_gen = self.model.generator(x_low)
                
                peak_snr = self._peak_snr(x_high, x_high_gen)
                ssim_score = self._ssim_score(x_high, x_high_gen)
                tot_peak_snr += peak_snr
                tot_ssim += ssim_score
                tot_num_images += self.params.train.batch_size
        return (tot_peak_snr/tot_num_images).item(), (tot_ssim/tot_num_images).item()



#        print("Peak Signal-to-Noise Ratio evaluation score: {}".format(self._peak_snr(real_high_res, fake_high_res)))
#        print("Semantic Similarity (SSIM) evaluation score: {}".format(self.ssim_score(real_high_res, fake_high_res)))

    def _peak_snr(self, real_high_res, fake_high_res):
        #real_high_res = (real_high_res - torch.min(real_high_res, 0, keepdim=True).values) / ((torch.max(real_high_res, 0, keepdim=True).values - torch.min(real_high_res, 0, keepdim=True).values))
        #fake_high_res = (fake_high_res - torch.min(fake_high_res, 0, keepdim=True).values) / (torch.max(fake_high_res, dim=0, keepdim=True).values - torch.min(fake_high_res, 0, keepdim=True).values) 
        mse = torch.mean((real_high_res - fake_high_res)**2, dim=0)
        #max_pixel_value = torch.Tensor(1) #torch.max(real_high_res)
        #3psnr = - 10 * torch.log10(mse)
        #print(psnr)
        return torch.sum(mse)

    def _ssim_score(self, real_high_res, fake_high_res):
        return torch.sum(pytorch_ssim.ssim(real_high_res, fake_high_res, size_average=False))


def run_evaluation(params):
    # Create save directory
    model_results = results.Result(
            params.results.results_dir,
            params.general.experiment_name,
            params.results.overwrite,
    )

    #model_results.create_save_dir()


    # Load Data
    
    data = aggregate_data.AggregatedData.load(params.data.data_path)
    data_tr, data_te = data.data_tr, data.data_te

    # Normalize data (only fit params on training)
    data_tr, normalization_params = normalization.normalize_data_triple(data_tr, params=None)
    data_te, _ = normalization.normalize_data_triple(data_te, params=normalization_params)
    
    
    # Load Model
    model = model_results.load_model(params.model, data_tr.low_res.shape[1], data_tr.high_res.shape[1])

    # Initialize model evaluator and evaluate
    model_evaluator = ModelEvaluator(model, params)

    tr_psnr, tr_ssim  = model_evaluator.evaluate(data_tr)
    te_psnr, te_ssim = model_evaluator.evaluate(data_te)
    results_dict = {'tr_mse': tr_psnr, 'tr_ssim': tr_ssim, 'te_mse': te_psnr, 'te_ssim': te_ssim}

    model_results.save_results(results_dict, save_eval_results=True)
    

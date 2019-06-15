mode
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

    model = model_results.load_model(params.model)

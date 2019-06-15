# Entrance for runnning 
import os
import datetime as dt

import click

from src.data.modis_download import ModisDownloader, DEF_DOWNLOAD_CELLS, DEF_DOWNLOAD_DATES, MODIS_DOWNLOAD_DIR
from src.data.extract_patches import run_extract_patches, PATCH_DIR
from src.data.aggregate_data import AggregatedData, AGGREGATED_DIR, DEF_TRAIN_SPLIT_FRAC
from src.utils.parameters import Parameters
from src.methods import evaluation as eval 
from src.methods import train as tr, run as run_model

MODIS_NDVI_PRODUCT_DICT = {
    '250': ModisDownloader.DATA_PRODUCT_250M,
    '500': ModisDownloader.DATA_PRODUCT_500M,
}

@click.command()
@click.argument('product', type=click.Choice(MODIS_NDVI_PRODUCT_DICT.keys()))
def download(product):
    """
    Download MODIS NDVI product of specified resolution.
    """
    downloader = ModisDownloader(DEF_DOWNLOAD_CELLS, DEF_DOWNLOAD_DATES, MODIS_DOWNLOAD_DIR)
    downloader.download(MODIS_NDVI_PRODUCT_DICT[product])


@click.command()
@click.argument('patch_size', type=click.INT)
def extract_patches(patch_size):
    """
    Extract fixed size patches from data for low and high resolution (per year).
    """
    low_res_dir = os.path.join(MODIS_DOWNLOAD_DIR, ModisDownloader.DATA_PRODUCT_500M)
    high_res_dir = os.path.join(MODIS_DOWNLOAD_DIR, ModisDownloader.DATA_PRODUCT_250M)

    run_extract_patches(low_res_dir, high_res_dir, patch_size, PATCH_DIR)


@click.command()
@click.argument('patch_size', type=click.INT)
def aggregate_data(patch_size):
    """
    Aggregate yearly patches and perform train/test split.
    """
    patch_dir = os.path.join(PATCH_DIR, str(patch_size))
    aggregator = AggregatedData.create(patch_dir, AGGREGATED_DIR, DEF_TRAIN_SPLIT_FRAC)
    aggregator.save(os.path.join(AGGREGATED_DIR, str(patch_size)))

@click.command()
@click.argument('source_path', type=click.Path(exists=True))
@click.argument('dest_path', type=click.Path())
@click.argument('fraction', type=click.FLOAT)
def sample_data(source_path, dest_path, fraction):
    data = AggregatedData.load(source_path)
    sampled_data = data.sample(fraction)
    sampled_data.save(dest_path)


@click.command()
@click.argument('param_file', type=click.Path(exists=True))
def train(param_file):
    """
    Train model according to config.
    """
    params = Parameters.parse(param_file)

    print(f'Running Experiment: {params.general.experiment_name}')
    print('Params:', params)

    tr.train_model(params)

@click.command()
@click.argument('param_file', type=click.Path(exists=True))
<<<<<<< HEAD
def evaluate(param_file):
    params = Parameters.parse(param_file)
    print(f'Evaluating Experiment: {params.general.experiment_name}')

    eval.run_evaluation(params)
    
=======
def run(param_file):
    """
    Run model according to config.
    """
    params = Parameters.parse(param_file)

    print('Params:', params)

    run_model.run_model(params)

>>>>>>> 84286767b9b02e973b36d70a1d81f97748488d2d
    
# Setup Click command group
@click.group()
def cli():
    pass

cli.add_command(download)
cli.add_command(extract_patches)
cli.add_command(aggregate_data)
cli.add_command(train)
<<<<<<< HEAD
cli.add_command(evaluate)
=======
cli.add_command(sample_data)
cli.add_command(run)
>>>>>>> 84286767b9b02e973b36d70a1d81f97748488d2d

if __name__ == '__main__':
    cli()

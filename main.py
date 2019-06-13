# Entrance for runnning 
import os
import datetime as dt

import click

from src.data.modis_download import ModisDownloader, DEF_DOWNLOAD_CELLS, DEF_DOWNLOAD_DATES, MODIS_DOWNLOAD_DIR
from src.utils.parameters import Parameters
from src.methods import train as tr

@click.group()
def cli():
    pass

MODIS_NDVI_PRODUCT_DICT = {
    '250': ModisDownloader.DATA_PRODUCT_250M,
    '500': ModisDownloader.DATA_PRODUCT_500M,
}

@click.command()
@click.argument('product', type=click.Choice(MODIS_NDVI_PRODUCT_DICT.keys()))
def download(product):
    downloader = ModisDownloader(DEF_DOWNLOAD_CELLS, DEF_DOWNLOAD_DATES, MODIS_DOWNLOAD_DIR)
    downloader.download(MODIS_NDVI_PRODUCT_DICT[product])


@click.command()
@click.argument('param_file', type=click.Path(exists=True))
def train(param_file):
    params = Parameters.parse(param_file)

    print(f'Running Experiment: {params.general.experiment_name}\n')
    print('Params:', params)

    tr.train_model(params)
    

cli.add_command(download)
cli.add_command(train)

if __name__ == '__main__':
    cli()

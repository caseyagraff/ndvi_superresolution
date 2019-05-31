# Entrance for runnning 
import click
import datetime as dt

from src.data.modis_download import ModisDownloader, DEF_DOWNLOAD_CELLS, DEF_DOWNLOAD_DATES, MODIS_DOWNLOAD_DIR

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
def train():
    raise NotImplementedError()


cli.add_command(download)
cli.add_command(train)

if __name__ == '__main__':
    cli()

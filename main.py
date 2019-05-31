# Entrance for runnning 
import click
import datetime as dt

from src.data.modis_download import ModisDownloader, DEF_DOWNLOAD_CELLS, DEF_DOWNLOAD_DATES, MODIS_DOWNLOAD_DIR

@click.group()
def cli():
    pass

@click.command()
def download():
    downloader = ModisDownloader(DEF_DOWNLOAD_CELLS, DEF_DOWNLOAD_DATES, MODIS_DOWNLOAD_DIR)
    downloader.download(ModisDownloader.DATA_PRODUCT_500M)

@click.command()
def train():
    raise NotImplementedError()


cli.add_command(download)
cli.add_command(train)

if __name__ == '__main__':
    cli()

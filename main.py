# Entrance for runnning 
import click

from src.data import modis_download

@click.group()
def cli():
    pass

@click.command()
def download():
    modis_download.modis_download()

@click.command()
def train():
    raise NotImplementedError()


cli.add_command(download)
cli.add_command(train)

if __name__ == '__main__':
    cli()

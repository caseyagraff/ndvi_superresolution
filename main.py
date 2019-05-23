# Entrance for runnning 
import click

from src.data.modis_download import ModisDownloader

@click.group()
def cli():
    pass

@click.command()
def download():
    raise NotImplementedError()

@click.command()
def train():
    raise NotImplementedError()


cli.add_command(download)
cli.add_command(train)

if __name__ == '__main__':
    cli()

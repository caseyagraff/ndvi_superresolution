# Load necessary packages into Python
from netrc import netrc
import os
import requests

from ..utils.urls import get_url_paths


class ModisDownloader:
    NETRC_PATH = '~/.netrc'
    AUTHENTICATION_URL = 'urs.earthdata.nasa.gov'
    DATA_URL = 'https://e4ftl01.cr.usgs.gov/MOLA'

    DATA_PRODUCT_500M = 'MYD13A1.006'
    DATA_PRODUCT_250M = 'MYD13Q1.006'

    DATA_FMT = '%Y.%m.%d'

    def __init__(self, cells, dates, save_dir):
        self.cells = cells
        self.dates = dates
        self.save_dir = save_dir

    def download(self, data_product):
        file_list = self.generate_file_list(data_product)
        username, password = self.get_authentication_info()

        product_dir = self.create_save_dirs(data_product)
        product_url = os.path.join(ModisDownloader.DATA_URL, data_product)

        for file_name, date in file_list:
            file_url = os.path.join(product_url, date.strftime(ModisDownloader.DATE_FMT), file_name)
            save_name = os.path.join(product_dir, str(date.year), file_name)

            with requests.get(file_url, stream=True, auth=(username, password)) as response:
                if response.status_code != 200:
                    print(f'{save_name} not downloaded. Verify login information.')
                else:
                    response.raw.decode_content = True
                    content = response.raw
                    with open(save_name, 'wb') as d:
                        while True:
                            chunk = content.read(16 * 1024)
                            if not chunk:
                                break
                            d.write(chunk)

    def generate_file_list(self, data_product):
        file_list = []

        for date in self.dates:
            date_url = os.path.join(ModisDownloader.DATA_URL, data_product, data.strftime(ModisDownloader.DATE_FMT))
            files = get_url_paths(date_url, '.hdf')

            for cell in self.cells:
                matches = [fn for fn in files if cell in fn]
                
                if matches is None:
                    print(f'File for cell {cell} on date {date} was not found.')
                elif len(matches) > 1:
                    print(f'Skipping file becase more than one match for cell {cell} on date {date}:\n{matches}.')
                else:
                    file_list.append((matches[0], date))

        return file_list

    def get_authentication_info(self):
        try:
            netrcDir = os.path.expanduser(ModisDownloader.NETRC_PATH)
            netrc(netrcDir).authenticators(urs)[0]
        except Exception as e:
            print(f'Must have a valid netrc file at "{ModisDownloader.NETRC_PATH}" for url "{ModisDownloader.AUTHENTICATION_URL}".')
            raise e()

        return netrc(netrcDir).authenticators(urs)[0], netrc(netrcDir).authenticators(urs)[2]

    def create_save_dirs(self, data_product):
        # Create primary destination directory to save to
        product_dir = os.path.join(self.save_dir, data_product)

        if not os.path.exists(product_dir):
            os.makedirs(product_dir)

        # Create subdirectories for each year
        unique_years = {date.year for date in self.dates} 
        for year in unique_years:
            year_dir = os.path.join(product_dir, str(year))
            if not os.path.exists(year_dir):
                os.makedirs(year_dir)

        return product_dir

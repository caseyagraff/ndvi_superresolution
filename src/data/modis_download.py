# Load necessary packages into Python
from netrc import netrc
import os
import requests
import pickle
import datetime as dt

from tqdm import tqdm

from ..utils.urls import get_url_paths

def make_def_download_dates():
    leap_years = (2004, 2008, 2012, 2016)
    years = list(range(2004, 2013+1))
    #dates = ((1, 9), (4, 15), (7, 4), (10, 24))
    dates = ((1, 9), (7, 4))

    download_dates = []
    for year in years:
        for date in dates:
            is_leap = year in leap_years and date[0] > 2
            download_dates.append(dt.date(year, date[0], date[1]-is_leap))

    return download_dates

DEF_DOWNLOAD_CELLS = (
        #(8,6), 
        (8,5), (9,5), (10,5), (11,5), 
        (9,4), (10,4), (11,4), (12,4),
        #(10,3), (11,3), (12,3), (13,3)
        )

DEF_DOWNLOAD_DATES = make_def_download_dates()

MODIS_DOWNLOAD_DIR = './data/modis_ndvi/'


class ModisDownloader:
    NETRC_PATH = '~/.netrc'
    AUTHENTICATION_URL = 'urs.earthdata.nasa.gov'
    DATA_URL = 'https://e4ftl01.cr.usgs.gov/MOLA'

    DATA_PRODUCT_500M = 'MYD13A1.006'
    DATA_PRODUCT_250M = 'MYD13Q1.006'

    DATE_FMT = '%Y.%m.%d'

    FILE_LIST_NAME = 'file_list.pkl'

    def __init__(self, cells, dates, save_dir):
        self.cells = cells
        self.dates = dates
        self.save_dir = save_dir

    def download(self, data_product):
        product_dir = self.create_save_dirs(data_product)

        file_list = self.generate_file_list(data_product, product_dir)

        username, password = self.get_authentication_info()
        print('Got authentication info.')

        print('Created saved dirs.')
        product_url = os.path.join(ModisDownloader.DATA_URL, data_product)

        for file_name, date in tqdm(file_list):
            file_url = os.path.join(product_url, date.strftime(ModisDownloader.DATE_FMT), file_name)
            save_name = os.path.join(product_dir, str(date.year), file_name)

            # If file already downloaded, skip fetching
            if os.path.exists(save_name):
                continue

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

    def generate_file_list(self, data_product, product_dir):
        file_list_path = os.path.join(product_dir, ModisDownloader.FILE_LIST_NAME)

        # Check if file list already generated
        if os.path.exists(file_list_path):
            print(f'File list already exists at {file_list_path}.')
            with open(file_list_path, 'rb') as f_in:
                cells, dates, file_list =  pickle.load(f_in)

            # Check if previously generated file list matches current cells and dates
            if set(cells) == set(self.cells) and set(dates) == set(self.dates):
                return file_list
            else:
                print('Previously generated file list does not match current cells/dates, regenerating.')

        # If no generated list, build it
        print('Building file list.')
        file_list = []

        for date in tqdm(self.dates):
            date_url = os.path.join(ModisDownloader.DATA_URL, data_product, date.strftime(ModisDownloader.DATE_FMT))
            files = get_url_paths(date_url, '.hdf')

            for cell in self.cells:
                cell_name = 'h{:02d}v{:02d}'.format(*cell)
                matches = [fn for fn in files if cell_name in fn]
                
                if len(matches) == 0:
                    print(f'File for cell {cell} on date {date} was not found.')
                elif len(matches) > 1:
                    print(f'Skipping file becase more than one match for cell {cell} on date {date}:\n{matches}.')
                else:
                    file_list.append((matches[0], date))

        print(f'Built file list, length {len(file_list)}.')

        # Save file list
        with open(file_list_path, 'wb') as f_out:
            pickle.dump((self.cells, self.dates, file_list), f_out)

        return file_list

    def get_authentication_info(self):
        try:
            netrcDir = os.path.expanduser(ModisDownloader.NETRC_PATH)
            username, _, password = netrc(netrcDir).authenticators(ModisDownloader.AUTHENTICATION_URL)
        except Exception as e:
            print(f'Must have a valid netrc file at "{ModisDownloader.NETRC_PATH}" for url "{ModisDownloader.AUTHENTICATION_URL}".')
            raise e

        return username, password

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

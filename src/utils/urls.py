import requests
from bs4 import BeautifulSoup


def get_url_paths(url, ext=''):
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')

    return [node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]



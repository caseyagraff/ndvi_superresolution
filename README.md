# ndvi_superresolution
CS 295 Deep Generative Models project on GAN-based super-resolution of NDVI.

# MODIS NDVI Authentication
You must create an account at "urs.earthdata.nasa.gov". Then create a .netrc file in your home directory (~/.netrc) 
with the following format:

machine urs.earthdata.nasa.gov
  login {username}
  password {password}

# MODIS NDIV Download
To download the data, run "python main.py download {product}" where product is either 250 or 500 to specify the 
resolution to download.

# Package Setup
conda install -c conda-forge pyhdf
conda install -c pytorch pytorch
conda install matplotlib
conda install click
conda install beautifulsoup4
conda install requests
conda install tqdm
conda install -c pytorch torchvision
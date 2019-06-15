import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
try:
    from pyhdf.SD import SD, SDC
    from pyhdf import HDF
except:
    print('Warning - PyHdf not installed.')

import tqdm

PATCH_DIR = './data/modis_ndvi_processed/patches/'

'''
Load cell data from .hdf files. Automatically determines 250m vs 500m resolution.
Input:
filename: string
Returns:
data: returns as a numpy array of dimensions 4800x4800 for 250m resolution, and 2400x2400 for 500m resolution.
'''
def load_data_from_files(filename):
    if not os.path.exists(filename):
        print("File {} does not exist, cannot load data.".format(filename))
        return
    elif not HDF.ishdf(filename):
        print("File {} is not in hdf4 file format, cannot load data.".format(filename))
        return

    f = SD(filename, SDC.READ)
    data_field = None
    for i, d in enumerate(f.datasets()):
        # print("{0}. {1}".format(i+1,d))
        if "NDVI" in d:
            data_field = d

    ndvi_data = f.select(data_field)
    data = np.array(ndvi_data.get())
    return data

'''
Takes in a N by N numpy array of a single cell and outputs the patches for that cell
Params:
full_cell: the N by N numpy array of the entire cell to be split into patches
patch_size: the size of each patch, for higher res cells increase this parameter appropiately
date: the date this cell was collected
cell_id: the id of the cell from the MODI database

Returns:
patch_list: returns as a numpy array of length Num_patches by patch_size by patch_size
metadata_list: returns as a numpy array of length num_patches containing tuples of metadata for each cell
'''
def extract_patches_single_cell(full_cell, patch_size, date, cell_id):
    if full_cell.shape[0] % patch_size != 0:
        raise ValueError('Please use an evenly divisible patch size (check x dimension of cell)')
    elif full_cell.shape[1] % patch_size != 0:
        raise ValueError('Please use an evenly divisible patch size (check y dimension of cell)')
    number_patches_x = full_cell.shape[0]//patch_size
    number_patches_y = full_cell.shape[1]//patch_size
    patch_list = []
    metadata_list = []
    for patch_xid in range(number_patches_x):
        for patch_yid in range(number_patches_y):
            cur_patch = \
            full_cell[patch_xid * patch_size: (patch_xid + 1) * patch_size, patch_yid * patch_size: (patch_yid + 1) * patch_size ]
#            if not(is_mostly_water(cur_patch)): handled elsewhere
            patch_list.append(cur_patch)
            metadata_list.append((date, cell_id, patch_xid, patch_yid))
    return np.array(patch_list), np.array(metadata_list)

'''
filters out patches which are mostly water by using the low-res image only.
'''
def filter_water(low_res, high_res, metadata_list):
    if low_res.shape[0] != high_res.shape[0]:
        raise ValueError('Low and high res patch arrays do not have the same number of patches')
    idxs_mostly_land = np.array([i for i in range(low_res.shape[0]) if is_mostly_land(low_res[i])])
    filtered_low_res = low_res[idxs_mostly_land]
    filtered_high_res = high_res[idxs_mostly_land]
    filtered_metadata_list = metadata_list[idxs_mostly_land]
    return filtered_low_res, filtered_high_res, filtered_metadata_list

    
'''
TODO: Set threshold appropiately
'''
def is_mostly_land(patch, percent_threshold=50., water_threshold=-3000):
    plt.imshow(patch)
    tot_pixels = patch.shape[0] * patch.shape[1]
    num_water_pixels = np.sum(patch <= water_threshold)
    if num_water_pixels/tot_pixels * 100 >= percent_threshold:
        return False
    else: 
        return True

'''
makes patches for all cells and filters them. Saves a block for each cell in output dir, if you feed in data only from one year and
set save_per_year to True it will save all the data from a single year into one chunk
'''
def make_patches_and_filter_all_cells(low_res_cells, high_res_cells, patch_size, output_dir, 
        dates, cell_ids, save_per_year=False):
    year = dates[0][0:4]

    if save_per_year:
        for d,date in enumerate(dates):
            if date[0:4] != year:
                print(date, year)
                raise ValueError('If using save_per_year=True must have all data from same year')
        
        all_patches_low = []
        all_patches_high = []
        metadata_all = []

    assert(high_res_cells[0].shape[0] % low_res_cells[0].shape[0] == 0)
    high_res_factor = high_res_cells[0].shape[0] // low_res_cells[0].shape[0]

    for c, low_res_cell in enumerate(low_res_cells):
        low_res_patches, metadata_low = extract_patches_single_cell(low_res_cell, patch_size, dates[c], cell_ids[c])
        high_res_patches, metadata_high = extract_patches_single_cell(high_res_cells[c], high_res_factor * patch_size, dates[c], cell_ids[c])
        assert((metadata_low == metadata_high).all())

        low_res_patches, high_res_patches, metadata = filter_water(low_res_patches, high_res_patches, metadata_low)

        if not save_per_year:
            savename = cell_ids[c] + '_patch_size=' + str(patch_size) + '_date=' + str(dates[c])

            with open(os.path.join(output_dir, savename) + '.pkl', 'wb') as f:
                pickle.dump((low_res_patches, high_res_patches, metadata), f)

        else:
            all_patches_low.append(low_res_patches)
            all_patches_high.append(high_res_patches)
            metadata_all.append(metadata)

    if save_per_year:
        all_patches_low = np.concatenate(all_patches_low)
        all_patches_high = np.concatenate(all_patches_high)

        metadata_all = np.concatenate(metadata_all)

        # Create save path
        save_name = f'patch_size_{patch_size}_year_{dates[0][0:4]}.npz'
        save_path = os.path.join(output_dir, str(patch_size), save_name)

        # Create dir if it doesn't exist
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #with open(save_path, 'wb') as f:
        #    pickle.dump((all_patches_low, all_patches_high, metadata_all), f)
        np.savez(save_path, lo_res=all_patches_low, hi_res=all_patches_high, meta=metadata_all)

'''
TODO: make function which aggregates the per cell patch data into larger chunks containing data 
from multiple cells
'''
def aggregate_chunks():
    pass    
        
         
def run_extract_patches(low_res_cell_dir, high_res_cell_dir, patch_size, output_dir):
    high_res_cell_years = os.listdir(high_res_cell_dir)
    low_res_cell_years = os.listdir(low_res_cell_dir)

    for year in tqdm.tqdm(high_res_cell_years):
        high_res_cells = []
        low_res_cells = []
        dates = []
        cell_ids = []

        if not os.path.isdir(os.path.join(high_res_cell_dir, year)):
            continue

        sorted_files_high_res = sorted(os.listdir(os.path.join(high_res_cell_dir, year)))
        sorted_files_low_res = sorted(os.listdir(os.path.join(low_res_cell_dir, year)))

        assert(check_sorted(sorted_files_high_res, sorted_files_low_res))

        for high_res_cell, low_res_cell in zip(sorted_files_high_res, sorted_files_low_res):
            high_res_cells.append(load_data_from_files(os.path.join(high_res_cell_dir, year, high_res_cell)))
            low_res_cells.append(load_data_from_files(os.path.join(low_res_cell_dir, year, low_res_cell)))
            dates.append(high_res_cell.split('.')[1][1:])
            cell_ids.append(high_res_cell.split('.')[2])

        #now process all the cells per year using make_patches_and_filter_all_cells, also saves them
        #year_output_dir = os.path.join(output_dir, str(year))
        make_patches_and_filter_all_cells(np.array(low_res_cells), np.array(high_res_cells), patch_size, output_dir,
                dates, cell_ids, save_per_year=True)  

    #Aggregate chunks further if desired-currently grouped per year
    #aggregate_chunks(output_dir)



def view_patch(patch,savename='patch_example.png'):
    plt.imshow(patch)
    plt.colorbar()
    plt.savefig(savename)
    
def test_extract_patches_single_cell(path_to_cell, patch_sizes):
    cell = load_data_from_files(path_to_cell)
    date = path_to_cell.split('/')[-1].split('.')[1][1:]
    cell_id = path_to_cell.split('/')[-1].split('.')[2]

    for patch_size in patch_sizes:
        patches, metadata = extract_patches_single_cell(cell, patch_size, date, cell_id)
    print(patches)
    view_patch(patches[20])

def check_sorted(high_dir, low_dir):
    is_sorted = True
    for h, l in zip(high_dir, low_dir):
        if not(h.split('.')[1][1:] == l.split('.')[1][1:]):
            return False
        if not(h.split('.')[2] == l.split('.')[2]):
            return False
    return is_sorted
    
def test_make_patches_and_filter_all_cells(path_to_low_res_dir, path_to_high_res_dir):
    low_res_cells = []
    high_res_cells = []
    patch_size = 200
    low_res_amount = 500
    high_res_amount = 250
    dates = []
    cell_ids = []
    output_dir = '../../output/testing/'
    for low_res_file in sorted(os.listdir(path_to_low_res_dir)):
        low_res_cells.append(load_data_from_files(os.path.join(path_to_low_res_dir, low_res_file)))
        dates.append(low_res_file.split('.')[1][1:])
        cell_ids.append(low_res_file.split('.')[2])
    for f, high_res_file in enumerate(sorted(os.listdir(path_to_high_res_dir))):
        assert(cell_ids[f] == high_res_file.split('.')[2])
        assert(dates[f] == high_res_file.split('.')[1][1:])
        high_res_cells.append(load_data_from_files(os.path.join(path_to_high_res_dir, high_res_file)))
    high_res_cells = np.array(high_res_cells)
    low_res_cells = np.array(low_res_cells)
    make_patches_and_filter_all_cells(low_res_cells, high_res_cells, patch_size, low_res_amount, high_res_amount, output_dir, dates, cell_ids) 
    test_file = os.listdir(output_dir)[0]
    with open(os.path.join(output_dir, test_file), 'rb') as f:
        test_low_res_patches, test_high_res_patches, metadata = pickle.load(f)
    assert(test_low_res_patches.shape[1] == 200 and test_low_res_patches.shape[2] == 200)
    assert(test_high_res_patches.shape[1] == 400 and test_high_res_patches.shape[2] == 400)
    

if __name__ == '__main__':
     #test_extract_patches_single_cell('/lv_scratch/scratch/pputzel/test_processed_data/250_res/MYD13Q1.A2004009.h08v05.006.2015154150430.hdf', [400])
    #test_make_patches_and_filter_all_cells('/lv_scratch/scratch/pputzel/test_processed_data/500_res/',  '/lv_scratch/scratch/pputzel/test_processed_data/250_res/')
    high_res_cell_dir = '/extra/graffc0/ndvi_superresolution/data/modis_ndvi/MYD13Q1.006/'
    low_res_cell_dir = '/extra/graffc0/ndvi_superresolution/data/modis_ndvi/MYD13A1.006/'
    output_dir =       '/extra/graffc0/ndvi_superresolution/data/modis_ndvi_processed'
    patch_size = 150
    run_extract_patches(high_res_cell_dir, low_res_cell_dir, patch_size, output_dir, high_low_ratio=2)


import numpy as np
import os
import pickle
from extract_patches import view_patch


class DataLoader:
    def __init__(self, hparams):
        self.te_percent = hparams['te_percent']
        if hparams['years'] == 'all':
            self.years = ['2004', '2005', '2006', '2007' ,'2008', '2009', '2010', '2011', '2012', '2013']
        else:
            self.years = hparams['years']


    def load_aggregated_data(self, proprocessed_dir):
        with open(os.path.join(preprocessed_dir, 'high_res_te.pkl'), 'rb') as f:
            high_res_te = pickle.load(f)
        with open(os.path.join(preprocessed_dir, 'high_res_tr.pkl'), 'rb') as f:
            high_res_tr = pickle.load(f)

        with open(os.path.join(preprocessed_dir, 'low_res_te.pkl'), 'rb') as f:
            low_res_te = pickle.load(f)
        with open(os.path.join(preprocessed_dir, 'low_res_tr.pkl'), 'rb') as f:
            low_res_tr = pickle.load(f)

        return ((low_res_tr, high_res_tr), (low_res_te, high_res_te))
    
    def aggregate_years_and_split(self, by_year_data_dir, output_dir, preprocessed_dir=None, patch_size=150):
        if preprocessed_dir:
            self.load_aggregated_data(preprocessed_data(preprocessed_dir))
        low_res_by_year = []
        high_res_by_year = []
        metadata_by_year = []
        for year in self.years:
            
            filename = 'patch_size=' + str(patch_size) + '_year=' + str(year) + '.pkl'
            with open(os.path.join(by_year_data_dir, filename), 'rb') as f:
                low_res, high_res, metadata = pickle.load(f)
            low_res_by_year.append(low_res)
            high_res_by_year.append(high_res)
            metadata_by_year.append(metadata)
        tot_per_year = [low_res.shape[0] for low_res in low_res_by_year]
        
        low_res_te = []
        high_res_te = []
        metadata_te = []        
    
        low_res_tr = []
        high_res_tr = []
        metadata_tr = []
        for y, (low_res, high_res) in enumerate(zip(low_res_by_year, high_res_by_year)):
            num_te = round(self.te_percent * tot_per_year[y])
            shuffle_idxs = np.random.permutation(low_res.shape[0])
            shuffled_low_res = low_res[shuffle_idxs]
            shuffled_high_res = high_res[shuffle_idxs]
            shuffled_metadata = metadata_by_year[y][shuffle_idxs]
            
            low_res_te.append(shuffled_low_res[0:num_te])
            low_res_tr.append(shuffled_low_res[num_te:])
            
            metadata_te.append(shuffled_metadata[0:num_te])
            metadata_tr.append(shuffled_metadata[num_te:])

            high_res_te.append(shuffled_high_res[0:num_te])
            high_res_tr.append(shuffled_high_res[num_te:])
        
        low_res_te = np.concatenate(low_res_te)
        low_res_tr = np.concatenate(low_res_tr)
        
        metadata_te = np.concatenate(metadata_te)
        metadata_tr = np.concatenate(metadata_tr)
    
        high_res_te = np.concatenate(high_res_te)
        high_res_tr = np.concatenate(high_res_tr)

        low_res_tr, high_res_tr, low_res_te, high_res_te = self.normalize(low_res_tr, high_res_tr, low_res_tr, high_res_tr) 

        with open(os.path.join(output_dir, 'metadata_te.pkl'), 'wb') as f:
            pickle.dump(metadata_te, f)
        with open(os.path.join(output_dir, 'metadata_tr.pkl'), 'wb') as f:
            pickle.dump(metadata_tr, f)

        with open(os.path.join(output_dir, 'high_res_te.pkl'), 'wb') as f:
            pickle.dump(high_res_te, f)
        with open(os.path.join(output_dir, 'high_res_tr.pkl'), 'wb') as f:
            pickle.dump(high_res_tr, f)

        with open(os.path.join(output_dir, 'low_res_te.pkl'), 'wb') as f:
            pickle.dump(low_res_te, f)
        with open(os.path.join(output_dir, 'low_res_tr.pkl'), 'wb') as f:
            pickle.dump(low_res_tr, f)
        
         
        return ((low_res_tr, high_res_tr), (low_res_te, high_res_te))
   
    def normalize(self, low_res_tr, high_res_tr, low_res_te, high_res_te):
        max_val = np.max(low_res_tr) if np.max(low_res_tr) > np.max(high_res_tr) else np.max(high_res_tr)
        min_val = np.min(low_res_tr) if np.min(low_res_tr) < np.min(high_res_tr) else np.min(high_res_tr)
        
        return self.squash_range(low_res_tr, max_val, min_val), self.squash_range(high_res_tr, max_val, min_val), self.squash_range(low_res_te, max_val, min_val), self.squash_range(high_res_te, max_val, min_val)

    def squash_range(self, arr, max_val, min_val):
        return (arr - min_val)/(max_val - min_val) * 2. - 1.
         
def test_data_loader_part1():
    fake_hparams = {'te_percent':.10, 'years':[2006, 2012]}
    by_year_data_dir = '/extra/graffc0/ndvi_superresolution/data/modis_ndvi_processed/'
    output_dir = '/extra/graffc0/ndvi_superresolution/data/output_testing/'

    dl = DataLoader(fake_hparams)
    dl.aggregate_years_and_split(by_year_data_dir, output_dir)

def test_data_loader_part2():
    output_dir = '/extra/graffc0/ndvi_superresolution/data/output_testing/'
        
    with open(os.path.join(output_dir, 'low_res_tr.pkl'), 'rb') as f:
        low_res_tr = pickle.load(f)

    with open(os.path.join(output_dir, 'high_res_tr.pkl'), 'rb') as f:
        high_res_tr = pickle.load(f)
    

    with open(os.path.join(output_dir, 'low_res_te.pkl'), 'rb') as f:
        low_res_te = pickle.load(f)

    with open(os.path.join(output_dir, 'high_res_te.pkl'), 'rb') as f:
        high_res_te = pickle.load(f)

    
    with open(os.path.join(output_dir, 'metadata_tr.pkl'), 'rb') as f:
        metadata_tr = pickle.load(f)

    with open(os.path.join(output_dir, 'metadata_te.pkl'), 'rb') as f:
        metadata_te = pickle.load(f)
   
        
    print(metadata_tr.shape[0], low_res_tr.shape[0], high_res_tr.shape[0])
    print(metadata_tr)

    
    assert(metadata_tr.shape[0] == low_res_tr.shape[0] and low_res_tr.shape[0] == high_res_tr.shape[0])
    assert(metadata_te.shape[0] == low_res_te.shape[0] and low_res_te.shape[0] == high_res_te.shape[0])

    random_idx_tr = np.random.choice(low_res_tr.shape[0])
    random_idx_te = np.random.choice(low_res_te.shape[0])
    
    view_patch(low_res_tr[random_idx_tr], savename='random_low_res_tr.png')            
    view_patch(low_res_te[random_idx_te], savename='random_low_res_te.png')            
    view_patch(high_res_tr[random_idx_tr], savename='random_high_res_tr.png')
    view_patch(high_res_te[random_idx_te], savename='random_high_res_te.png')
     


if __name__ == '__main__':
    test_data_loader_part1()
    test_data_loader_part2() 

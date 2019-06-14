import numpy as np
import os
import pickle
from .extract_patches import view_patch
from collections import namedtuple
import numpy as np

import tqdm

AGGREGATED_DIR = './data/modis_ndvi_processed/aggregated'
DEF_YEARS = tuple(map(str, range(2004, 2013+1)))
DEF_TRAIN_SPLIT_FRAC = .8

DataTriple = namedtuple('DataTriple', ('low_res', 'high_res', 'metadata'))

class AggregatedData:
    def __init__(self, data_tr, data_te):
        self.data_tr = DataTriple(*data_tr)
        self.data_te = DataTriple(*data_te)

    @staticmethod
    def create(patch_dir, aggregated_dir, train_split_fraction, years=DEF_YEARS):
        """
        Aggregate yearly patches and split.
        """

        assert(0. < train_split_fraction < 1.)

        data_by_year = []

        files = os.listdir(patch_dir)
        print('Loading yearly data...')
        for file_name in tqdm.tqdm(files):
            #with open(os.path.join(patch_dir, filename), 'rb') as f_in:
            #    data_by_year.append(pickle.load(f_in))
            data = np.load(os.path.join(patch_dir, file_name))
            data_by_year.append( (data['lo_res'], data['hi_res'], data['meta']) )


        # Randomly shuffle and divide patches (per year) and split
        data_tr, data_te = [], []

        print('Aggregating...')
        for low_res, high_res, metadata in tqdm.tqdm(data_by_year):
            num_tr = round(train_split_fraction * low_res.shape[0])
            shuffle_idxs = np.random.permutation(low_res.shape[0])

            shuffled_low_res, shuffled_high_res, shuffled_metadata = (
                    low_res[shuffle_idxs], 
                    high_res[shuffle_idxs], 
                    metadata[shuffle_idxs],
                    )

            data_tr.append( (shuffled_low_res[:num_tr], shuffled_high_res[:num_tr], shuffled_metadata[:num_tr]) )
            data_te.append( (shuffled_low_res[num_tr:], shuffled_high_res[num_tr:], shuffled_metadata[num_tr:]) )

        # Concat years
        data_tr = tuple(map(np.concatenate, zip(*data_tr)))
        data_te = tuple(map(np.concatenate, zip(*data_te)))

        return AggregatedData(data_tr, data_te)

    def sample(self, fraction):
        """
        Sample a fraction of the train and test data.
        """
        num_samples_tr = int(fraction * len(self.data_tr.low_res))
        data_tr = DataTriple(
                self.data_tr.low_res[:num_samples_tr], 
                self.data_tr.high_res[:num_samples_tr],
                self.data_tr.metadata[:num_samples_tr]
        )

        num_samples_te = int(fraction * len(self.data_te.low_res))
        data_te = DataTriple(
                self.data_te.low_res[:num_samples_te], 
                self.data_te.high_res[:num_samples_te],
                self.data_te.metadata[:num_samples_te]
        )

        return AggregatedData(data_tr, data_te)

    def save(self, save_path):
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.savez(save_path, 
                lo_tr=self.data_tr.low_res, hi_tr=self.data_tr.high_res, meta_tr=self.data_tr.metadata,
                lo_te=self.data_te.low_res, hi_te=self.data_te.high_res, meta_te=self.data_te.metadata,
        )

    @staticmethod
    def load(save_path):
        data = np.load(save_path)

        data_tr = data['lo_tr'], data['hi_tr'], data['meta_tr']
        data_te = data['lo_te'], data['hi_te'], data['meta_te']

        return AggregatedData(data_tr, data_te)


# === Testing ===
def test_data_loader_part1():
    fake_hparams = {'te_percent':.10, 'years':[2006, 2012]}
    patch_dir = '/extra/graffc0/ndvi_superresolution/data/modis_ndvi_processed/'
    aggregated_dir = '/extra/graffc0/ndvi_superresolution/data/output_testing/'

    dl = DataLoader(fake_hparams)
    dl.aggregate_years_and_split(patch_dir, aggregated_dir)

def test_data_loader_part2():
    aggregated_dir = '/extra/graffc0/ndvi_superresolution/data/output_testing/'

    with open(os.path.join(aggregated_dir, 'low_res_tr.pkl'), 'rb') as f:
        low_res_tr = pickle.load(f)

    with open(os.path.join(aggregated_dir, 'high_res_tr.pkl'), 'rb') as f:
        high_res_tr = pickle.load(f)


    with open(os.path.join(aggregated_dir, 'low_res_te.pkl'), 'rb') as f:
        low_res_te = pickle.load(f)

    with open(os.path.join(aggregated_dir, 'high_res_te.pkl'), 'rb') as f:
        high_res_te = pickle.load(f)


    with open(os.path.join(aggregated_dir, 'metadata_tr.pkl'), 'rb') as f:
        metadata_tr = pickle.load(f)

    with open(os.path.join(aggregated_dir, 'metadata_te.pkl'), 'rb') as f:
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

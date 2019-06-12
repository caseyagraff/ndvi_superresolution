class DataLoader:
    def __init__(self, hparams):
        self.te_percent = hparams['te_percent']
        if hparams['years'] = 'all':
            hparams['years'] = []
        self.years = hparams['years']


    def load_aggregated_data(self, proprocessed_dir):
        with open(os.path.join(preprocessed_dir, 'metadata_te.pkl'), 'rb') as f:
            metadata_te = pickle.load(f)
        with open(os.path.join(preprocessed_dir, 'metadata_tr.pkl'), 'rb') as f:
            metadata_tr = pickle.load(f)

        with open(os.path.join(preprocessed_dir, 'high_res_te.pkl'), 'rb') as f:
            high_res_te = pickle.load(f)
        with open(os.path.join(preprocessed_dir, 'high_res_tr.pkl'), 'rb') as f:
            high_res_tr = pickle.load(f)

        with open(os.path.join(preprocessed_dir, 'low_res_te.pkl'), 'rb') as f:
            low_res_te = pickle.load(f)
        with open(os.path.join(preprocessed_dir, 'low_res_tr.pkl'), 'rb') as f:
            low_res_tr = pickle.load(f)
        return ((low_res_tr, high_res_tr, metadata_tr), (low_res_te, high_res_te, metadata_te))
    
    def aggregate_years_and_split(self, by_year_data_dir, output_dir, preprocessed_dir=None):
        if preprocessed_dir:
            self.load_aggregated_data(preprocessed_data(preprocessed_dir))
        low_res_by_year = []
        high_res_by_year = []
        for year in self.years:
            filename = year + 'fill me in with filenames'
            with open(os.path.join(by_year_data_dir, filename), 'rb') as f:
                low_res, high_res, metadata = pickle.load(f)
            low_res_by_year.append(low_res)
            high_res_by_year.append(highe_res)
        tot_per_year = [low_res.shape[0] for low_res in low_res_by_year]
        total_patches = np.sum(tot_per_year)
        num_patches_te = np.round(total_patches * self.te_percent)
        
        low_res_te = []
        high_res_te = []
        metatdata_te = []        
    
        low_res_tr = []
        high_res_tr = []
        metatdata_tr = []
        for y, low_res, high_res in enumerate(zip(low_res_by_year, high_res_by_year)):
            num_te = np.round(self.te_percent * tot_per_year[y])
            shuffled_low_res = np.random.permutation(low_res)
            shuffled_high_res = np.random.permutation(high_res)
            shuffled_metadata = np.random.permutation(metadata[y])
            
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

        with open(os.path.join(output_dir, 'metadata_te.pkl'), 'wb') as f:
            pickle.dump(metadata_te, f)
        with open(os.path.join(output_dir, 'metadata_tr.pkl'), 'wb') as f:
            pickle.dump(metatdata_tr, f)

        with open(os.path.join(output_dir, 'high_res_te.pkl'), 'wb') as f:
            pickle.dump(high_res_te, f)
        with open(os.path.join(output_dir, 'high_res_tr.pkl'), 'wb') as f:
            pickle.dump(high_res_tr, f)

        with open(os.path.join(output_dir, 'low_res_te.pkl'), 'wb') as f:
            pickle.dump(low_res_te, f)
        with open(os.path.join(output_dir, 'low_res_tr.pkl'), 'wb') as f:
            pickle.dump(low_res_tr, f)
           
         
        return ((low_res_tr, high_res_tr, metadata_tr), (low_res_te, high_res_te, metadata_te))
    
def test_data_loader(per_year_dir):
        fake_hparams = {'te_percent':.10, 'years':[2006, 2012]}
        
        dl = DataLoader()                

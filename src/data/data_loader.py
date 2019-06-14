import os

def load_train_test(data_dir):
    with open(os.path.join(data_dir, 'high_res_te.pkl'), 'rb') as f:
        high_res_te = pickle.load(f)

    with open(os.path.join(preprocessed_dir, 'high_res_tr.pkl'), 'rb') as f:
        high_res_tr = pickle.load(f)

    with open(os.path.join(preprocessed_dir, 'low_res_te.pkl'), 'rb') as f:

        low_res_te = pickle.load(f)
    with open(os.path.join(preprocessed_dir, 'low_res_tr.pkl'), 'rb') as f:
        low_res_tr = pickle.load(f)

    return ((low_res_tr, high_res_tr), (low_res_te, high_res_te))



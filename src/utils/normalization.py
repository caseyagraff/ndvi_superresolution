"""
Normalization functions.
"""
import numpy as np
from ..data.aggregate_data import DataTriple

def squash_range(arr, min_val, max_val):
    return (arr - min_val)/(max_val - min_val) * 2. - 1.

def normalize_data_triple(data_triple, params=None):
    if params:
        min_val, max_val = params
    else:
        min_val = min(np.min(data_triple[0]), np.min(data_triple[1])), 
        max_val = max(np.max(data_triple[0]), np.max(data_triple[1]))

    data_triple = DataTriple(
        squash_range(data_triple[0], min_val, max_val), 
        squash_range(data_triple[1], min_val, max_val),
        data_triple[2]
    )
    
    params = (min_val, max_val)

    return data_triple, params


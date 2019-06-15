"""
Other helper functions.
"""
import random
import numpy as np
import torch
from scipy.interpolate import interp2d

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



'Bicubic Interpolation takes in a numpy array, grid_to_sample is a np array with patch_size (in pixels) by patch_size dimensions, passing in the matching high_res_patch is fine'
def bicubic_interpolate_single_patch(low_res_patch, grid_to_sample, kind='cubic'):
    f_interpolated = interp2d(np.arange(low_res_patch.shape[0]) + 0.5, np.arange(low_res_patch.shape[1]) + 0.5, low_res_patch, kind=kind)
    interpolated_data = f_interpolated(np.arange(grid_to_sample.shape[0]), np.arange(grid_to_sample.shape[1]))
    return interpolated_data

def test_bicubic_interpolation():
    A = np.arange(36).reshape(6, 6)
    B = np.arange(144).reshape(12, 12)
    print(A)
    print(bicubic_interpolate_single_patch(A, B))

if __name__ == '__main__':
    test_bicubic_interpolation()

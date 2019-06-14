import numpy as np
import torch
import torchvision
from torch.utils import data as t_data

class Dataset(t_data.Dataset):
    def __init__(self, x_low, x_high):
        self.x_low = x_low.astype(np.float32)
        self.x_high = x_high.astype(np.float32)

        self.low_res = self.x_low.shape[1]
        self.high_res = self.x_high.shape[1]

    def __len__(self):
        return self.x_low.shape[0]

    def __getitem__(self, index):
        x_low, x_high = self.x_low[index], self.x_high[index]

        x_low, x_high = (
                torch.unsqueeze(torch.tensor(x_low).float(), 0),
                torch.unsqueeze(torch.tensor(x_high).float(), 0),
        )

        return x_low, x_high

def create_data_loader(x_low_data, x_high_data, batch_size, shuffle):
    dataset = Dataset(x_low_data, x_high_data)

    return t_data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

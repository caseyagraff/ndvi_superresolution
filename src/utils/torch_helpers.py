import torchvision
from torch.utils import t_data

class Dataset(t_data.Dataset):
    def __init__(self, x_low, x_high):
        self.x_low = x_low
        self.x_high = x_high
        self.transform = transform

    def __len__(self):
        return self.x_low.shape[0]

    def __getitem__(self, index):
        x_low, x_high = self.x_low[index], self.x_high[index]

        x_low, x_high = torch.tensor(x_low), torch.tensor(x_high)

        return x_low, x_high

def create_data_loader(x_low_data, x_high_data, batch_size, shuffle):
    dataset = torch_helpers.Dataset(x_low_data, x_high_data)

    return t_data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

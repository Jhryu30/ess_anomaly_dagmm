from pathlib import Path
import numpy as np
from torch.utils.data import Dataset

class ESSDataset(Dataset):
    def __init__(self, data_path, data_name, window_size, step, mode='train'):
        self.data_path = Path(data_path) / data_name
        self.window_size = window_size
        self.step = step
        self.mode = mode

        self.train = np.load(self.data_path / 'train.npy')
        self.test = np.load(self.data_path / 'test.npy')
        self.test_labels = np.load(self.data_path / 'test_labels.npy')

    def __len__(self):
        if self.mode == 'train':
            return (len(self.train) - self.window_size) // self.step + 1
        elif self.mode == 'val':
            return (len(self.test) - self.window_size) // self.step + 1
        else:
            return (len(self.test) - self.window_size) // self.window_size + 1
             
    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.train[idx*self.step:idx*self.step+self.window_size], self.test_labels[0:self.window_size]
        elif self.mode == 'val':
            return self.test[idx*self.step:idx*self.step+self.window_size], self.test_labels[idx*self.step:idx*self.step+self.window_size]
        else:
            return self.test[idx*self.window_size:idx*self.window_size+self.window_size], self.test_labels[idx*self.window_size:idx*self.window_size+self.window_size]
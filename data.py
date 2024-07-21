from pathlib import Path
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


BASE_DIR = Path(__file__).resolve().parents[2]


class MultiViewDataset(Dataset):
    def __init__(self, data_path=None, xy=None):
        super().__init__()
        if xy is not None:
            self.x, self.y = xy
        else:
            self.x, self.y = pickle.load(open(data_path, 'rb'))
        # noise
        # for k, v in self.x.items():
        #     if k in [1, 2]:
        #         # noise = np.random.normal(0, 1e9, v.shape)
        #         # self.x[k] += noise

    def __getitem__(self, index):
        x = dict()
        for v in self.x.keys():
            x[v] = self.x[v][index]
        return {
            'x': x,
            'y': self.y[index],
            'index': index
        }

    def __len__(self):
        return len(self.y)

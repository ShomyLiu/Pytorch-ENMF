# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
import numpy as np


NPY_PATH = "./data/ml-1m"


class RecData(Dataset):

    def __init__(self, mode):
        if mode == "Train":
            uids = np.load(f"{NPY_PATH}/u_train.npy", allow_pickle=True).tolist()
        else:
            uids = np.load(f"{NPY_PATH}/u_test.npy", allow_pickle=True).tolist()
        u_iid_dict = np.load(f"{NPY_PATH}/u_iid_list.npy", allow_pickle=True).tolist()

        self.x = list(zip(uids, u_iid_dict))

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)

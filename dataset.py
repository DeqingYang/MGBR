from torch.utils.data import Dataset
import torch
import numpy as np
from numpy import int64
import pandas as pd


class DealDataset(Dataset):

    def __init__(self, file_type: str):
        valid_record = pd.read_csv('data/' + file_type + '.txt', names=list("ABCDEFG"), sep='\t')
        item_sample = np.loadtxt('data/' + file_type + '_item_sampling.txt', delimiter='\t', dtype=int64)
        user_sample = np.loadtxt('data/' + file_type + '_user_sampling.txt', delimiter='\t', dtype=int64)

        self.target_user = torch.from_numpy(valid_record["A"].values)
        self.pos_item = torch.from_numpy(item_sample[:, 0])
        self.pos_user = torch.from_numpy(user_sample[:, 0])
        self.item_sample = torch.from_numpy(item_sample[:, :])
        self.user_sample = torch.from_numpy(user_sample[:, :])
        self.len = self.target_user.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.target_user[index], self.pos_item[index], self.pos_user[index], self.item_sample[index], \
               self.user_sample[index]
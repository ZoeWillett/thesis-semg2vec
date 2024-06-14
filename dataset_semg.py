import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset

class semgDataset(Dataset):
    samples = None
    def __init__(self, idx_file, labels_file, classification_type = 'gesture'):
        self.idx_list = pd.read_csv(idx_file)
        self.labels = pd.read_csv(labels_file, index_col='sample')
        if semgDataset.samples == None:
            semgDataset.samples = self.load_samples()
        self.classification_type = classification_type

    def load_samples(self):
        samples = []
        for i in range(1,20+1):
            print("loading subj ", i, "...")
            with open ('preprocessed/subj' + str(i) + '_samples', 'rb') as fp:
                data = pickle.load(fp)
            samples += data
        return samples

    # len of dataset (windows)
    def __len__(self):
        return len(self.idx_list)

    # gets index (0,len(dataset)) and returns window and label 
    def __getitem__(self, idx):
        window_start = self.idx_list.iloc[idx]['window_start']
        window_end = self.idx_list.iloc[idx]['window_stop']
        sample_idx = self.idx_list.iloc[idx]['sample']

        sample = self.samples[sample_idx]
        if self.classification_type == "bio_id":
            label = self.labels.loc[sample_idx]['subject']
        else:
            label = self.labels.loc[sample_idx]['gesture']

        # select correct window
        window = sample[window_start : window_end]
        
        return torch.tensor(window).float(), label 


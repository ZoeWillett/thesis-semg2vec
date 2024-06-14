import torch
from Prep_Func_git import loaddata_filt
import numpy as np
import time
import os
import pandas as pd
import pickle

from dataset_semg import semgDataset

def init_dataset():
    '''
    Initialize the datasets, samplers and dataloaders.
    dataloder_params = {'batch_size': 64,
                    'shuffle': True,
                    'num_workers': 3}
    '''
    # load datasets
    print("Load datasets...")
    train_dataset = semgDataset("preprocessed/new_split/reduced/idx_list_train_reduced_paper.csv", "preprocessed/new_split/reduced/train_labels_reduced.csv")
    test_dataset = semgDataset("preprocessed/new_split/reduced/idx_list_test_reduced_paper.csv", "preprocessed/new_split/reduced/test_labels_reduced.csv")
    val_dataset = semgDataset("preprocessed/new_split/reduced/idx_list_val_reduced_paper.csv", "preprocessed/new_split/reduced/val_labels_reduced.csv")
    
    print("Load dataloaders...")
    tr_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                batch_size=128,
                                                shuffle=True,
                                                num_workers=0) 
    
    ts_dataloader = torch.utils.data.DataLoader(test_dataset, 
                                                batch_size=128,
                                                shuffle=True,
                                                num_workers=0)

    tv_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                batch_size=128,
                                                shuffle=True,
                                                num_workers=0)
    
    
    return tr_dataloader,ts_dataloader,tv_dataloader

if __name__ == "__main__":
    tr_dataloader, val_dataloader, test_dataloader = init_dataset()
    print(tr_dataloader.__len__())
    print(val_dataloader.__len__())
    print(test_dataloader.__len__())
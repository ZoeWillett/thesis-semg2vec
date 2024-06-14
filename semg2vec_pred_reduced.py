import torch
from torch import nn as nn
import numpy as np
import time
import os
import pandas as pd
import pickle
from dataset_semg import semgDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import wandb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from helpers import to_2tuple
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):

        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x) # (n_samples, embed_dim, grid_size[0], grid_size[1]), BCHW
        x = x.flatten(2) # (n_samples, embed_dim, num_patches), BCN
        x = x.transpose(1, 2) # (n_samples, num_patches, embed_dim), BNC
        x = self.norm(x)
        return x

class Semg2vec(nn.Module):
    
    def __init__(self, img_size,
                       patch_size,
                       in_chans,  
                       embed_dim
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.patch_size = patch_size
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flat_patch = patch_size[0]*patch_size[1]*in_chans
        self.hidden_layer = nn.Linear(embed_dim, 2*self.flat_patch)

    def forward(self, x):

        n_samples = x.shape[0]
        #print("input shape: ", x.shape)
        # torch.Size([128, 8, 128, 16]) batch size, channel, window (128x16)
        x = self.patch_embed(x)
        # x.shape = 128, 16, 64 batch size, n patches, embed dim
        #print(x.shape)
        preds = []
        for i in range(1, x.shape[1]-1):
            pre_post = self.hidden_layer(x[:,i,:])
            x_pre, x_post = torch.split(pre_post, self.flat_patch, dim=1)
            preds += [x_pre, x_post]
        return preds


def batch_for_transformer(x, y):
    x = x.permute(0, 2,1, 3)
    x, y = x.cuda(), y.cuda()
    return x.float(), y.long()


def train_embed(tr_dataloader, model, optim, scheduler, val_dataloader=None):
    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    wandb.init(
    # set the wandb project where this run will be logged
    project="Master_final",
    name = "semg2vec_pred_reduced",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
    "architecture": "Semg2Vec_prediction",
    "dataset": "HDsemg",
    "number training samples": len(tr_dataloader),
    "epochs": 15,
    }
    )
    
    loss_fn = nn.MSELoss()
    for epoch in range(15): # 10
        count = 0
        print('=== Epoch: {} ==='.format(epoch+1))
        tr_iter = iter(tr_dataloader)
        model.train()
        model = model.cuda()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = batch_for_transformer(x, y)
            #patch_size = (8,16)
            patches = torch.nn.functional.unfold(x, model.patch_size, stride=model.patch_size).transpose(1, 2)
            model_output = model(x)
            loss = 0
            num_patches = patches.shape[1]
            j = 0
            for i in range(num_patches - 2):
                loss = loss + loss_fn(patches[:,i,:], model_output[j])
                j += 2
            j = 1
            for i in range(2, num_patches):
                loss = loss + loss_fn(patches[:,i,:], model_output[j])
                j += 2
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            count+=1
        assert count == len(tr_dataloader)
        avg_loss = np.mean(train_loss[-len(tr_dataloader):])
        print('Avg Train Loss: {}'.format(avg_loss))
        wandb.log({"avg loss": avg_loss, "epoch": epoch})
        for param_group in optim.param_groups:
            print(param_group['lr'])
        scheduler.step()
        
            
        val_iter = iter(val_dataloader)
        model.eval()
        with torch.no_grad():
            for batch in val_iter:
                x, y = batch
                x, y = batch_for_transformer(x, y)
                patches = torch.nn.functional.unfold(x, model.patch_size, stride=model.patch_size).transpose(1, 2)
                model_output = model(x)
                loss = 0
                num_patches = patches.shape[1]
                j = 0
                for i in range(num_patches - 2):
                    loss = loss + loss_fn(patches[:,i,:], model_output[j])
                    j += 2
                j = 1
                for i in range(2, num_patches):
                    loss = loss + loss_fn(patches[:,i,:], model_output[j])
                    j += 2
                val_loss.append(loss.item())
        avg_val_loss = np.mean(val_loss[-len(val_dataloader):])
        print('Avg Val Loss: {}'.format(avg_val_loss))
        wandb.log({"avg val loss": avg_val_loss, "epoch": epoch})

    print("Finished training")
    torch.save(model.state_dict(), 'models/paper_params/semg2vec_pred_reduced.pth')
    wandb.finish()
    return train_loss, train_acc, val_loss, val_acc

def init_dataset():
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
    print("initialising model...")
    model = Semg2vec(
        img_size= (64,16), #paper: (64,16)
        patch_size=(4,4), #paper: (4,4)
        in_chans=8,  
        embed_dim=64
    )
    print("initialising datasets...")
    tr_dataloader,ts_dataloader,tv_dataloader = init_dataset()
    
    optim = torch.optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=0.001)
    scheduler = StepLR(optim, step_size=10, gamma=0.07)
    
    res = train_embed(tr_dataloader=tr_dataloader,
                val_dataloader=tv_dataloader,
                model=model,
                optim=optim,
                scheduler=scheduler)
    train_loss, train_acc, val_loss, val_acc = res


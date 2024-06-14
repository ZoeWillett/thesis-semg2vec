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
        
class Semg2vecClassi(nn.Module):
    
    def __init__(self, img_size, patch_size, in_chans, embed_dim, n_classes):
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
        
        self.head = nn.Linear(self.num_patches * embed_dim, n_classes) # for flatten
        #self.head = nn.Linear(embed_dim, n_classes) # for mean

    def forward(self, x):

        n_samples = x.shape[0]
        #print("input shape: ", x.shape)
        # torch.Size([128, 8, 128, 16]) batch size, channel, window (128x16)
        x = self.patch_embed(x)
        # x.shape = 128, 16, 64 batch size, n patches, embed dim
        #print(x.shape)
        
        x = torch.flatten(x, start_dim = 1)
        #x = torch.mean(x, 1)
        
        x = self.head(x)
        return x


def batch_for_transformer(x, y):
    x = x.permute(0, 2,1, 3)
    x, y = x.cuda(), y.cuda()
    return x.float(), y.long()


def get_acc(last_model, last_targets):
    _, preds = last_model.max(1)
    acc = torch.eq(preds, last_targets).float().mean()
    return acc.item()


def train(tr_dataloader, model, optim, scheduler, val_dataloader=None):
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
    name = "semg2vec_gr",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
    "architecture": "Semg2Vec",
    "dataset": "HDsemg",
    "number training samples": len(tr_dataloader),
    "epochs": 15,
    }
    )
    
    loss_fn = nn.CrossEntropyLoss()
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
            model_output = model(x)
            loss = loss_fn(model_output, y)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(get_acc(model_output, y))
            count+=1
        assert count == len(tr_dataloader)
        avg_loss = np.mean(train_loss[-len(tr_dataloader):])
        avg_acc = np.mean(train_acc[-len(tr_dataloader):])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        wandb.log({"avg loss": avg_loss, "avg acc": avg_acc, "epoch": epoch})
        for param_group in optim.param_groups:
            print(param_group['lr'])
        scheduler.step()
        
            
        val_iter = iter(val_dataloader)
        model.eval()
        with torch.no_grad():
            for batch in val_iter:
                x, y = batch
                x, y = batch_for_transformer(x, y)
                model_output = model(x)
                loss = loss_fn(model_output, y)
                val_loss.append(loss.item())
                val_acc.append(get_acc(model_output, y))
        avg_val_loss = np.mean(val_loss[-len(val_dataloader):])
        avg_val_acc = np.mean(val_acc[-len(val_dataloader):])
        print('Avg Val Loss: {}, Avg Val Acc: {}'.format(avg_val_loss, avg_val_acc))
        wandb.log({"avg val loss": avg_val_loss, "avg val acc": avg_val_acc, "epoch": epoch})

    print("Finished training")
    torch.save(model.state_dict(), 'models/paper_params/semg2vec_gr.pth')
    return train_loss, train_acc, val_loss, val_acc

# Start of Testing!
def test(test_dataloader, model):
    test_acc_batch = list()
    test_iter = iter(test_dataloader)
    model.eval()
    y_pred=[]
    y_target=[]
    with torch.no_grad():
        for batch in test_iter:
            x, y = batch
            x, y = batch_for_transformer(x, y)
            model_output = model(x)
            test_acc_batch.append(get_acc(model_output, y))
            _,pred = model_output.max(1)
            y_pred.append(pred)
            y_target.append(y)
    assert len(test_dataloader) == len(test_acc_batch)
    test_acc = np.mean(test_acc_batch)
    test_std = np.std(test_acc_batch)
   
    cm = confusion_matrix(torch.cat(y_target).detach().cpu().numpy(), torch.cat(y_pred).detach().cpu().numpy())
    fig, ax = plt.subplots(figsize=(25, 25))
    ax = sns.heatmap(cm)
    ax.set(xlabel="Predicted", ylabel="Actual")
    confi = wandb.Image(ax)

    print('Test Acc: {}, Test Std: {}'.format(test_acc, test_std))
    print('len(test_acc_batch): {}'.format(len(test_acc_batch)))
    wandb.log({"test acc": test_acc, "test std": test_std, "confi": confi})
    
    wandb.finish()
    return test_acc, test_std

def init_dataset():
    # load datasets
    print("Load datasets...")
    train_dataset = semgDataset("preprocessed/new_split/idx_list_train_paper.csv", "preprocessed/new_split/train_labels.csv")
    test_dataset = semgDataset("preprocessed/new_split/idx_list_test_paper.csv", "preprocessed/new_split/test_labels.csv")
    val_dataset = semgDataset("preprocessed/new_split/idx_list_val_paper.csv", "preprocessed/new_split/val_labels.csv")
    
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
    model_a = Semg2vec(
        img_size = (64,16), #paper: (64,16)
        patch_size = (4,4), #paper: (4,4)
        in_chans = 8,  
        embed_dim = 64
    )
    model_b = Semg2vecClassi(
        img_size = (64,16), #paper: (64,16)
        patch_size = (4,4), #paper: (4,4)
        in_chans = 8,  
        embed_dim = 64,
        n_classes = 66
    )
    

    model_a.load_state_dict(torch.load('models/paper_params/semg2vec_gr_pred.pth'))

    model_b.patch_embed = model_a.patch_embed
   
    # freeze embed layer
    for param in model_b.patch_embed.parameters():
        param.requires_grad = False
    
    print("initialising datasets...")
    tr_dataloader,ts_dataloader,tv_dataloader = init_dataset()
    
    optim = torch.optim.Adam(params=model_b.parameters(), lr=0.0001, weight_decay=0.001)
    scheduler = StepLR(optim, step_size=10, gamma=0.07)
    
    res = train(tr_dataloader = tr_dataloader,
                val_dataloader = tv_dataloader,
                model = model_b,
                optim = optim,
                scheduler = scheduler)
    train_loss, train_acc, val_loss, val_acc = res

    test(test_dataloader = ts_dataloader,
         model = model_b)
    

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

        
class MLPclassi(nn.Module):
    
    def __init__(self, img_size, in_chans, n_classes):
        super().__init__()

        # MLP
        self.layers = nn.Sequential(
            nn.Linear(in_chans * img_size[0] * img_size[1], 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):
        x = torch.flatten(x, start_dim = 1)
        x = self.layers(x)
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
    name = "mlp_gr",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
    "architecture": "MLP",
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
    torch.save(model.state_dict(), 'models/paper_params/MLP_gr.pth')
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
   
    model = MLPclassi(
        img_size = (64,16), #paper: (64,16), default: (128,16)
        in_chans = 8,  
        n_classes = 66
    )
    
    print("initialising datasets...")
    tr_dataloader,ts_dataloader,tv_dataloader = init_dataset()
    
    optim = torch.optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=0.001)
    scheduler = StepLR(optim, step_size=10, gamma=0.07)
    
    res = train(tr_dataloader=tr_dataloader,
                val_dataloader=tv_dataloader,
                model=model,
                optim=optim,
                scheduler=scheduler)
    train_loss, train_acc, val_loss, val_acc = res

    test(test_dataloader = ts_dataloader,
         model = model)
    

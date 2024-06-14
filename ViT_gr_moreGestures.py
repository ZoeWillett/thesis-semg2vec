from models import VisionTransformer
from dataset_semg import semgDataset
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import wandb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
import pandas as pd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_dataset():
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


def batch_for_transformer(x, y):
    x = x.permute(0, 2,1, 3)
    x, y = x.cuda(), y.cuda()
    return x.float(), y.long()

def get_acc(last_model, last_targets):
    _, preds = last_model.max(1)
    acc = torch.eq(preds, last_targets).float().mean()
    return acc.item()

# Start of Training!
def train(tr_dataloader, model, optim, scheduler, val_dataloader=None):
    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0
    x=[]
    y=[]

    wandb.init(
    # set the wandb project where this run will be logged
    project="Master_final",
    name = "ViT_gr_moreGestures",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
    "architecture": "ViT_trans",
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
            model_output,cls_token = model(x)
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
                model_output,cls_token = model(x)
                loss = loss_fn(model_output, y)
                val_loss.append(loss.item())
                val_acc.append(get_acc(model_output, y))
        avg_val_loss = np.mean(val_loss[-len(val_dataloader):])
        avg_val_acc = np.mean(val_acc[-len(val_dataloader):])
        print('Avg Val Loss: {}, Avg Val Acc: {}'.format(avg_val_loss, avg_val_acc))
        wandb.log({"avg val loss": avg_val_loss, "avg val acc": avg_val_acc, "epoch": epoch})

    print("Finished training")
    torch.save(model.state_dict(), 'models/paper_params/ViT_gr_moreGestures.pth')
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
            model_output,cls_token = model(x)
            test_acc_batch.append(get_acc(model_output, y))
            _,pred=model_output.max(1)
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


if __name__ == "__main__":
    print("initialising model...")
    model = VisionTransformer(
        img_size= (64,16), #paper: (64,16), default: (128,16)
        patch_size=(4,4), #paper: (4,4), default: (8,16)
        in_chans=8, 
        n_classes=56, 
        embed_dim=64,
        depth=1, 
        n_heads=8, 
        mlp_ratio=1, 
        qkv_bias='store_true', 
        p=0.1, 
        attn_p=0.27, 
        qk_scale=None,
        norm_layer =nn.LayerNorm
    )
    model = model.cuda()
    model.load_state_dict(torch.load('models/paper_params/ViT_gr_reduced.pth'))
    
    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # create new linear head
    num_ftrs = model.head.in_features
    model.head = nn.Linear(num_ftrs, 66)
    
    model = model.to(device)
    
    
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
    
    #print('Testing...')
    test(test_dataloader=ts_dataloader, model=model)
    
    print('The number of parameters: {}'.format(get_n_params(model)))
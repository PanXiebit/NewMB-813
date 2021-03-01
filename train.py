import os
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
from netCDF4 import Dataset as ncDataset
from sklearn.model_selection import train_test_split
from src.models.main_model import MainInformer
from src.data.dataset import EarthDataSet
from torch.utils.data import DataLoader
from metrics import eval_score


def seed_everything(seed = 427):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
#     torch.set_deterministic(True)

seed_everything()

print("=" * 10 + " 1. Loading data " + "=" * 10)

SODA_train = ncDataset('enso_round1_train_20210201/CMIP_train.nc')
SODA_label = ncDataset('enso_round1_train_20210201/CMIP_label.nc')

total_label = SODA_label.variables['nino'][:].data[:, 12:36]
total_sst = SODA_train.variables['sst'][:].data[:, :12]
total_t300 = SODA_train.variables['t300'][:].data[:, :12]
total_ua = SODA_train.variables['ua'][:].data[:, :12]
total_va = SODA_train.variables['va'][:].data[:, :12]

total_ua = np.expand_dims(np.nan_to_num(total_ua), 2)
total_va = np.expand_dims(np.nan_to_num(total_va), 2)
total_t300 = np.expand_dims(np.nan_to_num(total_t300), 2)
total_sst = np.expand_dims(np.nan_to_num(total_sst), 2)

total_data = np.concatenate([total_sst, total_t300, total_ua, total_va], axis=2)

train_data, valid_data, train_label, valid_label = train_test_split(
    total_data, total_label, test_size=0.2, random_state=427)
print("train_data: ", train_data.shape)
print("valid_data: ", valid_data.shape)
print("train_label: ", train_label.shape)
print("valid_label: ", valid_label.shape)

train_dataset = EarthDataSet(train_data, train_label)
valid_dataset = EarthDataSet(valid_data, valid_label)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True)

print("=" * 10 + " 2. Loading model " + "=" * 10)
c_out=24*72
model = MainInformer(c_out=c_out, seq_len=12, label_len=1, out_len=24, mid_dim=128)
# print(model)
print('| num. module params: {} (num. trained: {})'.format(
    sum(p.numel() for p in model.parameters()),
    sum(p.numel() for p in model.parameters() if p.requires_grad),
))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
model = model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
# criterion = nn.L1Loss()
criterion = nn.MSELoss()

model.to(device)
criterion.to(device)
nums_epoch = 150

for i in range(nums_epoch):
    model.train()
    for step, batch in enumerate(train_loader):
        data = batch["data"].to(device).float()
        label = batch["label"].to(device).float()

        optim.zero_grad()
        preds = model(data)
        loss = criterion(preds, label)
        if step % 100 == 0:
            print("Epoch: {}, step: {}, train loss:{:.5f}".format(i, step, loss.item()))
        loss.backward()
        optim.step()
    model.eval()

    losses = []
    y_true, y_pred = [], []
    for valid_batch in tqdm(valid_loader):
        valid_data = valid_batch["data"].to(device).float()
        valid_label = valid_batch["label"].to(device).float()
        valid_preds = model(valid_data)

        valid_loss = criterion(valid_preds, valid_label)
        losses.append(valid_loss.cpu().detach().numpy())

        y_pred.append(valid_preds.cpu().detach().numpy())
        y_true.append(valid_label.cpu().detach().numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    print('========= Epoch: {}, valid losses: {:.5f}'.format(i, np.mean(losses)))
    sco = eval_score(y_true, y_pred)
    print('========= Epoch: {}, Valid Score {}'.format(i,sco))
    print("\n")
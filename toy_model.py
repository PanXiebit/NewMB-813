import torch
import numpy as np
import xarray as xr
import pandas as pd
import torch.nn as nn
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from netCDF4 import Dataset as ncDataset

import os
def seed_everything(seed = 427):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
#     torch.set_deterministic(True)
seed_everything()

SODA_train = ncDataset('enso_round1_train_20210201/CMIP_train.nc')
SODA_label = ncDataset('enso_round1_train_20210201//CMIP_label.nc')

print("Load data!")

train_label = SODA_label.variables['nino'][:].data[:, 12:36]
train_sst = SODA_train.variables['sst'][:].data[:, :12]
train_t300 = SODA_train.variables['t300'][:].data[:, :12]
train_ua = SODA_train.variables['ua'][:].data[:, :12]
train_va = SODA_train.variables['va'][:].data[:, :12]

# train_t300 = SODA_train['t300'][:, :12].values
# train_ua = SODA_train['ua'][:, :12].values
# train_va = SODA_train['va'][:, :12].values

# print("train_label: ", train_label.shape)
# print("train_sst: ", train_sst.shape)
# print("train_t300: ", train_t300.shape)
# print("train_ua: ", train_ua.shape)
# print("train_va: ", train_va.shape)

# exit()


class simpleSpatailTimeNN(nn.Module):
    def __init__(self, n_cnn_layer:int=1, kernals:list=[3], n_lstm_units:int=64):
        super(simpleSpatailTimeNN, self).__init__()
        self.conv1 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.conv2 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.conv3 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.conv4 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.max_pool = nn.AdaptiveMaxPool2d((22, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 128))
        self.batch_norm = nn.BatchNorm1d(12, affine=False)
        self.lstm = nn.LSTM(88, n_lstm_units, 2, bidirectional=True)
        self.linear = nn.Linear(128, 24)

    def forward(self, sst, t300, ua, va):
        # print("sst", sst.shape)
        # print("t300", t300.shape)
        # print("ua", ua.shape)
        # print("va", va.shape)

        for conv1 in self.conv1:
            sst = conv1(sst)
        for conv2 in self.conv2:
            t300 = conv2(t300)
        for conv3 in self.conv3:
            ua = conv3(ua)
        for conv4 in self.conv4:
            va = conv4(va)
        # print("sst", sst.shape)
        # print("t300", t300.shape)
        # print("ua", ua.shape)
        # print("va", va.shape)

        sst = self.max_pool(sst).squeeze(dim=-1)
        t300 = self.max_pool(t300).squeeze(dim=-1)
        ua = self.max_pool(ua).squeeze(dim=-1)
        va = self.max_pool(va).squeeze(dim=-1)

        # print("sst", sst.shape)
        # print("t300", t300.shape)
        # print("ua", ua.shape)
        # print("va", va.shape)
        # exit()

        x = torch.cat([sst, t300, ua, va], dim=-1)
        x = self.batch_norm(x)
        # print(x.shape)

        x, _ = self.lstm(x)
        x = self.avg_pool(x).squeeze(dim=-2)
        x = self.linear(x)
        return x



train_ua = np.nan_to_num(train_ua)
train_va = np.nan_to_num(train_va)
train_t300 = np.nan_to_num(train_t300)
train_sst = np.nan_to_num(train_sst)

train_index, valid_index = train_test_split(range(train_label.shape[0]),test_size=0.2, random_state=427)

class EarthDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['sst'])

    def __getitem__(self, idx):
        return {
            'sst':self.data['sst'][idx],
            't300':self.data['t300'][idx],
            'ua':self.data['ua'][idx],
            'va':self.data['va'][idx],
            'label':self.data['label'][idx]
        }

dict_train = {
    'sst':train_sst[train_index],
    't300':train_t300[train_index],
    'ua':train_ua[train_index],
    'va': train_va[train_index],
    'label': train_label[train_index]}

dict_valid = {
    'sst':train_sst[valid_index],
    't300':train_t300[valid_index],
    'ua':train_ua[valid_index],
    'va': train_va[valid_index],
    'label': train_label[valid_index]}

train_dataset = EarthDataSet(dict_train)
valid_dataset = EarthDataSet(dict_valid)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True)

model = simpleSpatailTimeNN()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
model = model.to(device)
optim = torch.optim.Adam(model.parameters(), lr=2e-6)
criterion = nn.L1Loss()

model.to(device)
criterion.to(device)
nums_epoch = 50
for i in range(nums_epoch):
    model.train()
    for data in tqdm(train_loader):
        sst = data['sst'].to(device).float()
        t300 = data['t300'].to(device).float()
        ua = data['ua'].to(device).float()
        va = data['va'].to(device).float()
        optim.zero_grad()
        label = data['label'].to(device).float()
        preds = model(sst, t300, ua, va)
        loss = criterion(preds, label)
        loss.backward()
        optim.step()
    model.eval()
    losses = []
    for data in tqdm(valid_loader):
        sst = data['sst'].to(device).float()
        t300 = data['t300'].to(device).float()
        ua = data['ua'].to(device).float()
        va = data['va'].to(device).float()
        label = data['label'].to(device).float()
        preds = model(sst, t300, ua, va)
        loss = criterion(preds, label)
        losses.append(loss.cpu().detach().numpy())
    print(f'losses: {np.mean(losses)}')



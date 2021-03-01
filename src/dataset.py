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
SODA_label = ncDataset('enso_round1_train_20210201/CMIP_label.nc')

print("=" * 5 + " 1. Loading data " + "=" * 5)
train_label = SODA_label.variables['nino'][:].data[:, 12:36]
train_sst = SODA_train.variables['sst'][:].data[:, :12]
train_t300 = SODA_train.variables['t300'][:].data[:, :12]
train_ua = SODA_train.variables['ua'][:].data[:, :12]
train_va = SODA_train.variables['va'][:].data[:, :12]

train_ua = np.nan_to_num(train_ua)
train_va = np.nan_to_num(train_va)
train_t300 = np.nan_to_num(train_t300)
train_sst = np.nan_to_num(train_sst)

print("train_label: ", train_label.shape)
print("train_sst: ", train_sst.shape)
print("train_t300: ", train_t300.shape)
print("train_ua: ", train_ua.shape)
print("train_va: ", train_va.shape)
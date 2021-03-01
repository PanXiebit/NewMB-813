# -*- coding: utf-8 -*-
from netCDF4 import Dataset
import numpy as np
import sys, time
import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.basemap import Basemap
from pandas import DataFrame
#数据读入

path = "enso_round1_train_20210201/"
train_nc=Dataset(path + 'CMIP_train.nc')
print(train_nc.variables.keys())
#取出各variable的数据看看,数据格式为numpy数组
for var in train_nc.variables.keys():
    data=train_nc.variables[var][:].data
    print(var,data.shape)
# sst = train_nc.variables["sst"][:].data
# t300 = train_nc.variables["t300"][:].data
# ua = train_nc.variables["ua"][:].data
# va = train_nc.variables["va"][:].data

test_nc=Dataset(path + 'CMIP_label.nc')
print(test_nc.variables.keys())
#取出各variable的数据看看,数据格式为numpy数组
for var in test_nc.variables.keys():
    data=test_nc.variables[var][:].data
    print(var,data.shape)
nino = test_nc.variables["nino"][:].data
print(nino.shape)
# t300 = train_nc.variables["t300"][:].data
# ua = train_nc.variables["ua"][:].data
# va = train_nc.variables["va"][:].data


# label = np.load("test_0144-01-12.npy")
# print(label.shape)
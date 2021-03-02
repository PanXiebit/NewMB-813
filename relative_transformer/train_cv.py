import os
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
from netCDF4 import Dataset as ncDataset
from sklearn.model_selection import train_test_split
from src.models.main_model_one_pass import MainModel
from src.data.dataset import EarthDataSet
from torch.utils.data import DataLoader
from metrics import eval_score
from sklearn.model_selection import KFold

def seed_everything(seed = 427):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
#     torch.set_deterministic(True)

seed_everything()

print("=" * 10 + " 1. Loading data " + "=" * 10)

SODA_train = ncDataset('../enso_round1_train_20210201/CMIP_train.nc')
SODA_label = ncDataset('../enso_round1_train_20210201/CMIP_label.nc')


total_sst = SODA_train.variables['sst'][:].data[:, :12]
total_t300 = SODA_train.variables['t300'][:].data[:, :12]
total_ua = SODA_train.variables['ua'][:].data[:, :12]
total_va = SODA_train.variables['va'][:].data[:, :12]

total_label = SODA_label.variables['nino'][:].data[:, 12:36]
sst_label = SODA_train.variables['sst'][:].data[:, 12:36]
t300_label = SODA_train.variables['t300'][:].data[:, 12:36]
ua_label = SODA_train.variables['ua'][:].data[:, 12:36]
va_label = SODA_train.variables['va'][:].data[:, 12:36]

total_ua = np.expand_dims(np.nan_to_num(total_ua), 2)
total_va = np.expand_dims(np.nan_to_num(total_va), 2)
total_t300 = np.expand_dims(np.nan_to_num(total_t300), 2)
total_sst = np.expand_dims(np.nan_to_num(total_sst), 2)

ua_label = np.expand_dims(np.nan_to_num(ua_label), 2)
va_label = np.expand_dims(np.nan_to_num(va_label), 2)
t300_label = np.expand_dims(np.nan_to_num(t300_label), 2)
sst_label = np.expand_dims(np.nan_to_num(sst_label), 2)

total_data = np.concatenate([total_sst, total_t300, total_ua, total_va], axis=2)
data_label = np.concatenate([sst_label, t300_label, ua_label, va_label], axis=2)

# train_data, valid_data, train_label, valid_label, train_data_label, valid_data_label = train_test_split(
#     total_data, total_label, data_label, test_size=0.1, random_state=427)
# print("train_data: ", train_data.shape)
# print("valid_data: ", valid_data.shape)
# print("train_label: ", train_label.shape)
# print("valid_label: ", valid_label.shape)
# print("train_data_label: ", train_data_label.shape)
# print("valid_data_label: ", valid_data_label.shape)
# k折交叉验证

print("=" * 10 + " 2. Loading model " + "=" * 10)
class Config():
    hidden_size = 512
    ff_size = 512
    num_heads = 8
    dropout = 0.3
    emb_dropout = 0.3
    cnn_dropout = 0.3
    num_layers = 3
    local_num_layers = 0
    use_relative = True
    max_relative_positions = 24
    embedding_dim = 512
opts = Config()
model = MainModel(opts)

# print(model)
print('| num. module params: {} (num. trained: {})'.format(
    sum(p.numel() for p in model.parameters()),
    sum(p.numel() for p in model.parameters() if p.requires_grad),
))


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
model = model.to(device)
model.to(device)
criterion = nn.MSELoss()
criterion_kl = nn.KLDivLoss()
criterion.to(device)
criterion_kl.to(device)

nums_epoch = 20


def adjust_learning_rate(optim, epoch, cur_loss, best_loss):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    if epoch > 5 and cur_loss > best_loss:
        print("Update learning rate!")
        for param_group in optim.param_groups:
            param_group['lr'] = param_group['lr'] * 0.8
            print("lr: {:.6f}".format(param_group["lr"]))


cv = 10
kf = KFold(n_splits=cv)
cv_sco = []
for cv_i, (train_index, valid_index) in enumerate(kf.split(total_data)) :
    # train_data, valid_data, train_label, valid_label, train_data_label, valid_data_label = train_test_split(
    #     total_data, total_label, data_label, test_size=0.1, random_state=cv_i)
    train_data = total_data[train_index]
    valid_data = total_data[valid_index]
    train_label = total_label[train_index]
    valid_label = total_label[valid_index]
    train_data_label = data_label[train_index]
    valid_data_label = data_label[valid_index]
    train_dataset = EarthDataSet(train_data, train_label, train_data_label)
    valid_dataset = EarthDataSet(valid_data, valid_label, valid_data_label)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True)

    # reset model and lr
    model.init()

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    cur_score = best_score = 0
    for epoch in range(nums_epoch):
        # adjust_learning_rate(optim, i, cur_loss, best_loss)
        model.train()
        for step, batch in enumerate(train_loader):
            data = batch["data"].to(device).float()
            label = batch["label"].to(device).float()
            data_y = batch["data_label"].to(device).float()
            # print("data: ", data.shape)
            # print("label: ", label.shape)
            # print("data_y: ", data_y.shape)
            optim.zero_grad()
            dec_out, prediction, y_true = model(data, data_y)
            mse_loss = criterion(prediction, label)
            kl_loss = criterion_kl(dec_out, y_true)
            loss = mse_loss
            if step % 100 == 0:
                print("Epoch: {}, step: {}, train kl loss: {:.5f}, mse loss: {:.5f}".
                      format(epoch, step, kl_loss.item(), mse_loss.item()))
            loss.backward()
            optim.step()
        model.eval()

        losses = []
        y_true, y_pred = [], []
        for valid_batch in tqdm(valid_loader):
            valid_data = valid_batch["data"].to(device).float()
            valid_label = valid_batch["label"].to(device).float()
            # valid_data_y = valid_batch["data_label"].to(device).float()

            valid_preds = model.decoder_one_pass(valid_data)
            valid_loss = criterion(valid_preds, valid_label)
            losses.append(valid_loss.cpu().detach().numpy())

            y_pred.append(valid_preds.cpu().detach().numpy())
            y_true.append(valid_label.cpu().detach().numpy())
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        print('========= Epoch: {}, valid losses: {:.5f}'.format(epoch, np.mean(losses)))
        sco = eval_score(y_true, y_pred, out_len=24)
        print('========= Epoch: {}, Valid Score {}'.format(epoch,sco))
        print("\n")

        # update cur_loss and best_loss
        cur_score = sco
        if cur_score > best_score:
            best_score = cur_score

            state_dict = model.state_dict()
            torch.save(state_dict, 'checkpoints/cv{}_epoch_{}_score_{:.4f}.pt'.format(cv_i, epoch, sco))

    cv_sco.append(sco)
    # del train_data, valid_data, train_label, valid_label, train_data_label, valid_data_label
    # gc.collect()
print(np.mean(cv_sco))
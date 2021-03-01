import os
import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
from netCDF4 import Dataset as ncDataset
from sklearn.model_selection import train_test_split
from src.models.main_model_one_pass import MainModel
from metrics import eval_score
from torch.utils.data import Dataset, DataLoader
import zipfile

cv=10

def seed_everything(seed = 427):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
#     torch.set_deterministic(True)

seed_everything()

print("=" * 10 + " 1. Loading data " + "=" * 10)

filespaths = "/tcdata/enso_round1_test_20210201/"

class TestEarthDataSet(Dataset):
    def __init__(self, filespaths):
        self.file_path_head = filespaths
        self.list_files = os.listdir(filespaths)
        for i, file_name in enumerate(self.list_files):
            if ".npy" not in file_name:
                self.list_files.pop(i)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        data = np.load(self.file_path_head+self.list_files[idx])
        data = data.transpose(0,3,1,2)

        return data, self.list_files[idx]

test_dataset = TestEarthDataSet(filespaths)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print("=" * 10 + " 2. Loading model " + "=" * 10)
class Config():
    hidden_size = 512
    ff_size = 512
    num_heads = 4
    dropout = 0.3
    emb_dropout = 0.3
    num_layers = 2
    local_num_layers = 0
    use_relative = True
    max_relative_positions = 24
    embedding_dim = 512
opts = Config()
model = MainModel(opts)

print('| num. module params: {} (num. trained: {})'.format(
    sum(p.numel() for p in model.parameters()),
    sum(p.numel() for p in model.parameters() if p.requires_grad),
))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

all_re = []

save_path_head = '/workspace/result/'

for i in range(cv):
    re = []
    weights = torch.load('/workspace/NewMB-813/relative_transformer_commit/weight/{}_19_weight.pt'.format(i))
    model.load_state_dict(weights)
    model.eval()

    for data, file_names in tqdm(test_loader):
        test_data = data.to(device).float()
        test_preds = model.decoder_one_pass(test_data)

        preds = test_preds.cpu().detach().numpy()
        re.append(preds)
        # for i, file_name in enumerate(file_names):
        #     np.save(save_path_head+file_name, preds[i, ...])
    re = np.concatenate(re, axis=0)
    re = np.expand_dims(re, axis=-1)
    all_re.append(re)

# 算出平均值
all_re = np.concatenate(all_re, axis=-1)
re_mean = np.mean(all_re, axis=-1)

# 按顺序保存
file_index = 0
for data, file_names in test_loader:
    for file_name in file_names:
        np.save(save_path_head+file_name, re_mean[file_index, ...])
        file_index += 1


print("==========zip file===========")
f = zipfile.ZipFile('/workspace/result.zip','w',zipfile.ZIP_DEFLATED)
for dirpath, dirnames, filenames in os.walk(save_path_head):
    for filename in filenames:
        f.write(os.path.join(dirpath,filename))
f.close()
print("==========done===========")


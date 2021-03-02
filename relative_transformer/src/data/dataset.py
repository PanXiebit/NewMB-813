
from torch.utils.data import Dataset, DataLoader

class EarthDataSet(Dataset):
    def __init__(self, data, label, data_label):
        self.data = data
        self.label = label
        self.data_label = data_label
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {"data": self.data[idx],
                "label": self.label[idx],
                "data_label": self.data_label[idx]}
        # {
        #     'sst':self.data['sst'][idx],
        #     't300':self.data['t300'][idx],
        #     'ua':self.data['ua'][idx],
        #     'va':self.data['va'][idx],
        #     'label':self.data['label'][idx]
        # }
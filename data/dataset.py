from torch.utils.data import Dataset,DataLoader
import pickle
import numpy as np

class MyDataset(Dataset):
    def __init__(self,):
        super(MyDataset,self).__init__()

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

if __name__ == '__main__':
    dataset_train = MyDataset()
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    for step,(data,label) in enumerate(dataloader_train):
        print(step,data.shape,label.shape)

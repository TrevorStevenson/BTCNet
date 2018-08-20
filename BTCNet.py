#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class BTCDataset(Dataset):
    def __init__(self, csv):
        self.btcFrame = pd.read_csv(csv)
        self.targets = self.btcFrame.iloc[:, 4:5]
        
    def __len__(self):
        return len(self.btcFrame)
    
    def __getitem__(self, i):
        item = self.btcFrame.iloc[i, 1:2].values
        item = item.astype("float")
        return item
    
    def getTarget(self, i):
        target = self.targets.iloc[i, :].values
        target = target.astype("float")
        return target
    
    def input_size(self):
        return len(self[0])
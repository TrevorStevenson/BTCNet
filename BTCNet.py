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

btcData = BTCDataset("bitcoinPrices.csv")
batch_size = 10
dataloader = DataLoader(btcData, batch_size=batch_size, shuffle=False, num_workers=2)

class BTCNet(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size, output_size):
        super(BTCNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
        self.hidden = (torch.zeros(1, batch_size, hidden_size), torch.zeros(1, batch_size, hidden_size))
       
    def forward(self, data):
        lstm_out, self.hidden = self.lstm(data.float(), self.hidden)
        lin_out = self.fc1(lstm_out)
        out = self.relu(lin_out)
        out = out.float()
        return out

btcModel = BTCNet(input_size=btcData.input_size(), batch_size=batch_size, hidden_size=5, output_size=1)

for i, sample in enumerate(dataloader):
    sample = sample.unsqueeze(0)
    out = btcModel(sample)
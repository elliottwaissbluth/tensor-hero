import torch
from torch import nn
import torch.optim as optim
import numpy as np
from pathlib import Path

class Dataset(torch.utils.data.Dataset):
    '''
    '''
    def __init__(self, list_of_files):
        self.files = list_of_files
        self.spec_slices = np.zeros(shape=(512, len(self.files)*400))
        print('Loading Files')
        for idx, file in enumerate(self.files):
            slice = np.load(file)
            self.spec_slices[:,idx*400:(idx+1)*400] = slice
        print('Files loaded!')
        print('Number of spectrogram slices: {}'.format(self.spec_slices.shape[1]))
      
    def __len__(self):
        return self.spec_slices.shape[1]
    
    def __getitem__(self, idx):
        return torch.tensor(self.spec_slices[:,idx], dtype=torch.float)

class VAE(nn.Module):
    def __init__(
        self,
        dropout,
    ):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(256, 30),
            nn.Sigmoid(),
       )
        self.decoder = nn.Sequential(
            nn.Linear(30, 256),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
       )
    
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z
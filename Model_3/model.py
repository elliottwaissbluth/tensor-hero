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

class LazyDataset(torch.utils.data.Dataset):
    '''
    '''
    def __init__(self, buckets):
        self.buckets = buckets
        self.num_buckets = len(buckets) # a bucket is a list of file names 
        self.examples_per_bucket = len(buckets[0]) # number of files per bucket
        self.count = 0 # number of batches drawn
        self.current_bucket = 0 # bucket currently being processed
        self.data = np.zeros(shape=(512, 400*self.examples_per_bucket))
        for idx, file in enumerate(self.buckets[0]):
            self.data[:,idx*400:(idx+1)*400] = np.load(file)

        print(f'num_buckets: {self.num_buckets}')
        print(f'examples_per_bucket: {self.examples_per_bucket}')

    def __len__(self):
        print('length {}'.format(self.examples_per_bucket*self.num_buckets))
        return self.examples_per_bucket*self.num_buckets
    
    def __getitem__(self, idx):
        print('entered __getitem__')
        # lazy load
        if self.count%self.examples_per_bucket == 0:
            print(f'changing buckets from {self.current_bucket} to {self.current_bucket+1}')
            del self.data
            self.current_bucket += 1
            self.data = np.zeros(shape=(512, 400*self.examples_per_bucket))
            for idx, file in enumerate(self.buckets[self.current_bucket]):
                self.data[:,idx*400:(idx+1)*400] = np.load(file)
        idx = idx - self.examples_per_bucket*self.current_bucket
        self.count += 1
        print('batch ready!')
        return torch.tensor(self.data[:,idx], dtype=torch.float)

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
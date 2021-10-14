import torch
from torch import nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import os

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

class LazierDataset(torch.utils.data.Dataset):
    '''
    Inspects the data at partition_path, creates a list (data_paths) and a dictionary (labels)
        - data_paths : list
            [<path to spectrogram frame> for _  in partition_path]
        - labels : dictionary
            - [<path to spectrogram frame> : <path to corresponding notes>]
    Loads each training example one at a time, as called by the dataloader

    ~~~~ ARGUMENTS ~~~~
    - partition_path : path or str
        - should be .../Training Data/Model 1 Training/<train, test, or val>
    - max_len : int
        - for padding, what the length of the notes arrays should be
    - pad_idx : int
        - value the notes tensors will be padded with
    '''
    def __init__(self, partition_path, max_len, pad_idx):
        # Construct list of spectrogram paths
        self.data_paths = [partition_path / 'spectrograms' / x for x in os.listdir(partition_path / 'spectrograms')]
        
        # Construct dictionary
        self.labels = {}
        for data_path in self.data_paths:
            self.labels[data_path] = data_path.parent.parent / 'notes' / (data_path.stem + '.npy')

        self.max_len = max_len
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.data_paths)

    def get_data_dict(self):
        return self.labels

    def pad_notes(self, notes):
        '''pads notes with pad_idx to length max_len'''
        notes = np.pad(notes, 
                       (0, self.max_len-notes.shape[0]),
                       'constant',
                       constant_values=self.pad_idx)
        return notes

    def __getitem__(self, idx):
        spec = np.load(self.data_paths[idx])
        notes = np.load(self.labels[self.data_paths[idx]])
        notes = self.pad_notes(notes)

        return torch.tensor(spec, dtype=torch.float), torch.tensor(notes)

class LazyDataset(torch.utils.data.Dataset):
    '''
    first attempt - failed
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
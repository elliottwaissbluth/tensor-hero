# Contains classes and functions related to dataloader and dataset
import torch
from torch import nn
import numpy as np
from pathlib import Path
import os
import random
import math
import torch.nn.functional as F
import torch.optim as optim


class DistributedFolderDataset(torch.utils.data.Dataset):
    def __init__(self, subroot_paths):
        '''
        Parses through data dispersed in distributed folders (subroots >> folders containing data >>>). 
        Shuffles and returns single training examples.
        subroot_paths => list containing string paths to subroot directories
        '''
        # self.data_size = data_size          # Amount of data to load
        self.X = np.zeros((3, 81, 0))       # src
        self.y = np.zeros(0)                # target
        self.length = 0                     # total length of all data
        self.chunk_length = 0               # length of individual chunks
        self.chunk_traversed = 0            # Number of samples in chunk that have already been processed
        self.subroot_idx = 0                # index of subroots directory to load into chunk
        self.chunk_loaded = False           # if true, the chunk has been loaded, if false it needs loading 
        self.subroot_paths = subroot_paths  # List of files in root
        self.total_chunk_length = 0         # Updated amount of chunk traversed to modify idx in __getitem__

        # Get length of dataset
        for path in self.subroot_paths:
            fileList = os.listdir(path)  
            if 'notes.npy' not in fileList or 'song.npy' not in fileList:
                continue

            # Load notes tensor to gather length for __len__
            notes = np.load(Path(path) / 'notes.npy')
            self.length += notes.shape[0]

    def load_chunk(self):
        '''
        Loads 1GB of data or the rest of the data in subroot_paths
        '''
        self.X = np.zeros((3, 81, 0))       # src
        self.y = np.zeros(0)                # target

        # Load data
        data_loaded = 0     # Amount of data loaded so far
        chunk_length = 0    # Number of samples in chunk
        while data_loaded < 1:

            if self.subroot_idx > (len(self.subroot_paths)-1):
                print('All subroots have been traversed')
                self.subroot_idx = 0
                break

            # Path to files, in this case src = song and target = notes
            notes_path = Path(self.subroot_paths[self.subroot_idx]) / 'notes.npy'
            song_path = Path(self.subroot_paths[self.subroot_idx]) / 'song.npy'
            self.subroot_idx += 1
            
            # Get data size
            try:
                data_loaded += notes_path.stat().st_size / 1e9  # Measure amount of data input
                data_loaded += song_path.stat().st_size / 1e9
            except WindowsError as err:  # If the files aren't all there
                print('Windows Error: Data in {} not found, skipping\n\n'.format(subroot))
                continue
            
            # Load numpy arrays
            notes = np.load(notes_path)
            song = np.load(song_path)

            # Put all the note and all the song data into one big array
            self.X = np.concatenate((self.X, song), axis=2)
            self.y = np.concatenate((self.y, notes), axis=0)

        chunk_length = self.y.shape[0]
        return chunk_length

            # print('{:3.2f} / {:3.2f} GB data loaded\n'.format(data_loaded, self.data_size))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.chunk_loaded == False:  # Time to load a new chunk
            print('Loading Chunk')
            self.chunk_length =  self.load_chunk()
            print('Chunk Length = {}\n'.format(self.chunk_length))
            self.chunk_traversed = 0
            self.chunk_loaded = True

        # Split song into 150 ms windows
        X = torch.from_numpy(np.take(self.X, range(idx-7-self.total_chunk_length,idx+8-self.total_chunk_length), axis=2, mode='wrap'))
        X = X.permute(0, 2, 1)
        y = 1 if self.y[idx-self.total_chunk_length] > 0 else 0  # Only care about onsets, losing note information

        # Check whether new chunk should be loaded
        self.chunk_traversed += 1
        if self.chunk_traversed > (self.chunk_length-1):
            self.total_chunk_length += self.chunk_length
            print('Full chunk traversed, {} / {} total samples traversed'.format(self.total_chunk_length, self.length))
            self.chunk_loaded = False
        return X, y


def train_val_test_split(root, data_amount, val = 0.1, test = 0.1, shuffle = False):
    '''
    Takes a directory input and outputs 3 lists of subdirectories of specified size.
    I'm going to operate under the assumption that songs converge on a mean length if you get enough of them.
    - root: root of subdirectories
    - data_amount: amount of data to load
    - val: validation split
    - test: test split
    - shuffle: shuffle names of directories
    '''

    subroot_paths = []
    data_loaded = 0

    # Generate list of song folders
    for dirName, subdirList, fileList in os.walk(root):  
        if 'notes.npy' not in fileList or 'song.npy' not in fileList:
            continue

        # Get data size
        notes_path = Path(dirName) / 'notes.npy'
        song_path = Path(dirName) / 'song.npy'
        try:
            data_loaded += notes_path.stat().st_size / 1e9  # Measure amount of data input
            data_loaded += song_path.stat().st_size / 1e9
        except WindowsError as err:  # If the files aren't all there
            print('Windows Error: Data in {} not found, skipping\n\n'.format(subroot))
            continue
        
        if data_loaded > data_amount:
            break

        subroot_paths.append(dirName)
        
    # Shuffle subroots if applicable
    if shuffle:
        random.shuffle(subroot_paths)

    # Split dataset
    num_val = math.floor(val * len(subroot_paths))
    num_test = math.floor(test * len(subroot_paths))

    train = subroot_paths[num_val:(len(subroot_paths)-num_test)]
    val = subroot_paths[0:num_val]
    test = subroot_paths[-num_test:]

    return train, val, test
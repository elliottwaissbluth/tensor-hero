# Defines a CNN and trains

import torch
from pathlib import Path
import sys
sys.path.insert(1, str(Path().cwd().parent) + r'\Audio Embeddings')
from loader import DistributedFolderDataset, train_val_test_split
from torch import nn
import numpy as np
import os
import random
import math
import torch.nn.functional as F
import torch.optim as optim


# Define CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(7,3))
        self.pool = nn.MaxPool2d(kernel_size=(1,3))
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3,3))
        self.fc1 = nn.Linear(7*8*20, 100)
        self.fc2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(7*8*20, -1)
        x = x.permute(1, 0)
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.squeeze(x)
        # x = self.sigmoid(x)  # Removed sigmoid because using nn.BCEWithLogitsLoss
        return x



# Parameters for dataloader
params = {'batch_size' : 15000,
          'shuffle' : False,
          'num_workers': 0}

model = Net()
criterion = nn.BCEWithLogitsLoss(pos_weight = torch.Tensor([10])) # weight = torch.Tensor([10]).repeat_interleave(params['batch_size']))
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

if torch.cuda.is_available():
    print('Using CUDA')
    model = model.cuda()
    criterion = criterion.cuda()
    device = torch.device('cuda:0')

print(model)

max_epochs = 2
root = Path(r'X:\Training Data\Processed')
data_size = 2  # Amount of data to load
train_paths, val_paths, test_paths = train_val_test_split(root, data_size, shuffle=True)

# Define datasets and loaders
trains = DistributedFolderDataset(train_paths)
train_loader = torch.utils.data.DataLoader(trains, drop_last=True, **params)
vals = DistributedFolderDataset(val_paths)
val_loader = torch.utils.data.DataLoader(vals, drop_last=True, **params)
tests = DistributedFolderDataset(test_paths)
test_loader = torch.utils.data.DataLoader(tests, drop_last=True, **params)

criterion = nn.BCEWithLogitsLoss(pos_weight = torch.Tensor([100])) # Start out with high loss for one epoch, then change afterwards
criterion = criterion.cuda()

train_accs =[]
val_accs = []

for epoch in range(max_epochs):
    model.train()
    num_true = 0

    # Training
    for batch_idx, (local_batch, local_labels) in enumerate(train_loader):
        #  Transfer to GPU
        local_batch, local_labels = local_batch.to(device, dtype = torch.float32), local_labels.to(device, dtype = torch.float32)

        # Model computations
        y_pred = model(local_batch)
        loss = criterion(y_pred, local_labels)
        preds = torch.argmax(y_pred, dim=-1).cpu().numpy()
        
        num_true += np.sum(preds == local_labels.cpu().numpy())

        loss.backward()
        optimizer.step()
        model.zero_grad()

        if batch_idx % 1000 == 0:
            # print training update
            print('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(local_batch), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        print('Training Accuracy: {}\n'.format(num_true / len(trains)))
        train_accs.append(num_true / len(trains))

        # Free up GPU memory
        

    criterion = nn.BCEWithLogitsLoss(pos_weight = torch.Tensor([10]))
    criterion = criterion.cuda()

    # Validation
    model.eval()  # Put model in evaluation mode
    num_true = 0

    for batch_idx, (local_batch, local_labels) in enumerate(val_loader):
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device, dtype = torch.float32), local_labels.to(device, dtype = torch.int64)

        # Model computations
        y_pred = model(local_batch)
        preds = torch.argmax(y_pred, dim=-1).cpu().numpy()
        num_true += np.sum(preds == local_labels.cpu().numpy()) 

    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Validation Accuracy: {}\n'.format(num_true / len(vals)))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    val_accs.append(num_true / len(vals))



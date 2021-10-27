import torch
from torch import nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import os
import matplotlib.pyplot as plt

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
    def __init__(self, partition_path, max_src_len, max_trg_len, pad_idx):
        # Construct list of spectrogram paths
        self.data_paths = [partition_path / 'spectrograms' / x for x in os.listdir(partition_path / 'spectrograms')]
        
        # Construct dictionary
        self.labels = {}
        for data_path in self.data_paths:
            self.labels[data_path] = data_path.parent.parent / 'notes' / (data_path.stem + '.npy')

        self.max_trg_len = max_trg_len
        self.max_src_len = max_src_len
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.data_paths)

    def get_data_dict(self):
        return self.labels

    def pad_notes(self, notes):
        '''pads notes with pad_idx to length max_trg_len'''
        notes = np.pad(notes, 
                       (0, self.max_trg_len-notes.shape[0]),
                       'constant',
                       constant_values=self.pad_idx)
        return notes
    
    def pad_spec(self, spec):
        '''pads spec with zeros to length max_src_len'''
        spec = np.pad(spec,
                      ((0, 0), (0, self.max_src_len-spec.shape[1])),
                      'constant',
                      constant_values=0)
        return spec

    def __getitem__(self, idx):
        spec = np.load(self.data_paths[idx])
        spec = self.pad_spec(spec)
        notes = np.load(self.labels[self.data_paths[idx]])
        assert notes.shape[0] < self.max_trg_len, 'ERROR: notes array is longer than max_len'
        notes = self.pad_notes(notes)

        return torch.tensor(spec, dtype=torch.float), torch.tensor(notes)

class RandomInputDataset(LazierDataset):
    '''
    Loads random spectrograms as input but actual notes
    '''
    def __init__(self, partition_path, max_src_len, max_trg_len, pad_idx):
        super().__init__(partition_path, max_src_len, max_trg_len, pad_idx)
        spec = np.load(self.data_paths[0])
        self.spec = self.pad_spec(spec)
        self.shape = [spec.shape[0], spec.shape[1]]
    def __getitem__(self,idx):
        random_spec = torch.rand(self.spec.shape[0], self.spec.shape[1])
        notes = np.load(self.labels[self.data_paths[idx]])
        assert notes.shape[0] < self.max_trg_len, 'ERROR: notes array is longer than max_len'
        notes = self.pad_notes(notes)

        return random_spec, torch.tensor(notes)

class InputEmbedding(nn.Module):
    '''
    This is a custom class which can be used in place of nn.Embedding for input embeddings.
    We can't use nn.Embedding for our input because our input is continuous, and nn.Embedding
    only works with discrete vocabularies such as words.

    When initializing the transformer, we define a nn.ModuleList (i.e. list of modules) containing
    400 of these layers. Each spectrogram frame goes through each one of these.
    '''
    def __init__(
        self,
        embedding_size,
    ):
        super(InputEmbedding, self).__init__()

        # Take the spectrogram frames and pass them through a FC layer
        self.linear = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
        )
    
    def forward(self, src):
        return self.linear(src)

class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,   # 512
        trg_vocab_size,   # 434
        num_heads,        # 8
        num_encoder_layers,  # 3
        num_decoder_layers,  # 3
        forward_expansion,   # 2048
        dropout,             # 0.1
        max_len,    # 400
        device,     # GPU or CPU?
    ):
        super(Transformer, self).__init__()

        # src_position_embedding and trg_position_embedding work the same, we can use nn.Embedding
        # for both
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        # trg_word_embeddings can also leverage nn.Embedding, since the target values come from a
        # discrete vocabulary (unlike the continuous spectrogram input)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)

        # The word embeddings need to come from a series of parallel linear layers, as defined in
        # this module list

        # inputs of dim 512, so we can't feed nn.Embedding an index
        # self.src_word_embedding = nn.ModuleList(
            # [
            # InputEmbedding(embedding_size) for _ in range(max_len)
            # ]
        # )
        self.src_word_embedding = InputEmbedding(embedding_size)

        self.device = device    # device will be used to move tensors from CPU to GPU 
        self.transformer = nn.Transformer(  # Define transformer
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)     # Output linear layer
        self.dropout = nn.Dropout(dropout)      # Dropout to avoid overfitting

        
     

    def make_src_mask(self, src):
        '''
        This src_mask can remain this hardcoded array of "False" values so long as we are only
        showing the transformer whole chunks of 4 second data. If we change this, we will need to
        create source masks that mask the padding on the input spectrogram.
        '''
        # Create tensor of all "False" values for single input
        src_mask = torch.zeros(src.shape[0], 400, dtype=torch.bool)
        # Create a tensor of all "True" values
        src_mask_true = torch.ones(src.shape[0], src.shape[2]-400, dtype=torch.bool)
        # Concatenate
        src_mask = torch.cat((src_mask, src_mask_true), 1)
        # shape = (N, src_len)

        return src_mask
    
    def make_embed_src(self, src):
        '''
        This function passes the spectrogram frames through the input embedding layer. Notice how it does
        so one layer at a time in a for loop. It may be possible to increase the speed of this and parallelize
        by doing a lil trick with nn.Conv1D. We'll investigate this in the future.
        '''
        # out = torch.zeros_like(src, requires_grad=True).to(self.device)
        out_list = []
        print('src shape : {}'.format(src.shape))

        # out is shape [1,512,400], just like the src.
        # "out" means the embedding of the input
        # when we loop, we access [1,512,0], [1,512,1], [1,512,2], ... , [1,512,399]
        # translating 400 slices of spectrogram data to 400 slices of embeddings
        for idx in range(src.shape[2]): # For each spectrogram frame
            if idx < 400:
                out_list.append(self.src_word_embedding(src[...,idx]).unsqueeze(-1))
            else:
                out_list.append(torch.zeros_like(out_list[0]))
        print(out_list[0].shape)
        return out_list


    def forward(self, src, trg):
        src_seq_length, N = src.shape[2], src.shape[0]  # This is always (400, 1) for now
        trg_seq_length, N = trg.shape[1], trg.shape[0]  # The target sequence length is 51 for our toy training example
                                                        # Originally 52, but we shift the target to mask the last value

        src_padding_mask = self.make_src_mask(src).to(self.device)
        
        # src_positions is just a sequence of increasing ints, expanded to the same shape as the batch of inputs used for
        # positional embedding calculation
        src_positions = (
            torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N)
            .to(self.device)
            )
        # [0, 1, 2, 3, ..., 399]

        # trg_positions is the same thing as src_positions, but with a slightly different shape since the trg and src inputs are
        # formatted in different ways (src is spectrogram frames, trg is a series of indices corresponding to note events)
        trg_positions = (
            torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N).permute(1,0)
            .to(self.device)
        )

        # [0, 1, 2, ..., 50] shape = (51) to shape = (N, 51)
        # if N = 2, AKA 2 spectrograms per batch
        # trg_positions = [[0, 1, 2, ..., 50], [0, 1, 2, ..., 50]]
        # trg without padding = [433, 70, 0, 300, 3, 434], Len = 5
        # trg with padding = [433, 70, 0, 300, 3, 434, 435, 435, 435, 435, ..., 435] Len = max_trg_seq_len


        # The permutations are just to get the embeddings into the right shape for the encoder
        # Notice how make_embed_src() is called, this is our custom function that passes the input through the parallel dense layers
        out_list = self.make_embed_src(src)
        src_embed = torch.cat(out_list, dim=2)
        print('src_embed shape: {}'.format(src_embed.shape))

        embed_src = self.dropout(
            (src_embed + self.src_position_embedding(src_positions).permute(1,2,0))
            .to(self.device)
        ).permute(0,2,1)
        # This is going into the transformer (final input after the pink blocks)

        # embed_trg uses "word" embeddings since trg is just a list of indices corresponding to "words" i.e. note events.
        # Positional embeddings are summed at this stage.
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
            .to(self.device)
        )
        # This is going into the decoder

        # This target mask ensures the decoder doesn't take context from future input while making predictions.
        # That would be useless for inference since our output is sampled autoregressively.
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask = src_padding_mask,
            tgt_mask=trg_mask,
        )

        # Pass the transformer output through a linear layer for a final prediction
        out = self.fc_out(out)

        return out
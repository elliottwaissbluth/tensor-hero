import torch
from torch import nn
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import sys

def check_notes_length(notes_path, max_len):
    '''Opens the processed notes array at notes_path and checks whether or not it is larger than max_len

    Args:
        notes_path (Path): path to notes array
        max_len (int): maximum length of notes
        
    Returns:
        bool: Whether the notes array at notes_path is >= max_len
    '''
    notes_array = np.load(notes_path)
    return notes_array.shape[0] < max_len

def note_dirs_from_spec_dirs(spec_file):
    '''
    Finds the note files corresponding to the spectrogram in spec_dir
    Helper function for ColabLazyDataset __init__()

    ~~~~ ARGUMENTS ~~~~
    - spec_file (Path): 
        - single path to a spectrogram in colab transformer training data
        - assumes file structure defined in tensor_hero/preprocessing/data.py
            -> preprocess_transformer_data() w/ COLAB=True
    
    ~~~~ RETURNS ~~~~
    Path: Path to notes array corresponding to spec in spec_file
    '''
    return Path(str(spec_file).replace('spectrograms', 'notes'))
    

class ColabMemoryDataset(torch.utils.data.Dataset):
    '''
    Inspects the data at partition_path, creates a list (data_paths) and a dictionary (labels) where the list contains
    paths to spectrogram slices (400ms) and the dictionarty contains path to spectrogram frames and corresponding notes
    arrays.
        - data_paths (list of Path): [<path to spectrogram frame> for _  in partition_path]
        - labels (dict) : [<path to spectrogram frame> : <path to corresponding notes>]

    Loads as many of these features into an array, self.specs and self.notes, as allowed by max_examples. max_examples should be
    chosen to be as large as possible without overloading RAM.

    One possible way to go about training with limited max_data_size:
    
    for epoch in num_epochs:
        for chunk in num_chunks: (where num_chunks is ceil([total num examples] / [max_examples]))
            <load chunk into dataloader>
            for idx, batch in dataloader:
                <train>
            <delete chunk from memory>
            <initialize next chunk>

    ~~~~ ARGUMENTS ~~~~
    - partition_path (Path): should be .../Training Data/Model 1 Training/<train, test, or val>
    - max_src_len (int): used for padding spectrograms, indicates what the length of the spectrograms should be in the time dimension
    - max_trg_len (int): used for padding notes, indicates what the length of the notes arrays should be
    - pad_idx (int): pad index, value the notes tensors will be padded with
    '''
    def __init__(self, partition_path, max_src_len, max_trg_len, pad_idx, max_examples):

        song_paths = [partition_path / x for x in os.listdir(partition_path)]
        specs_dirs = [x / 'spectrograms' for x in song_paths]

        specs_lists = []
        for dir_ in specs_dirs:
            for specs_dir, _, specs in os.walk(dir_):
                if not specs:
                    continue
                specs_lists.append([Path(specs_dir) / spec for spec in specs])
            
        specs_lists = [spec for spec_list in specs_lists for spec in spec_list]  # Flatten
        notes_lists = [note_dirs_from_spec_dirs(x) for x in specs_lists]
        
        # Construct dictionary where key:value is <path to spec>:<path to notes array>
        l = {}  # labels
        for i in range(len(specs_lists)):
            l[specs_lists[i]] = notes_lists[i]
            
        # Weed out bits of data that exceed the maximum length
        self.labels = {}
        self.data_paths = []
        self.too_long = 0
        print('Checking length of spectrograms and notes...')
        for x in tqdm(specs_lists):
            if check_notes_length(l[x], max_trg_len):
                self.data_paths.append(x)
                self.labels[x] = l[x]
            else:
                self.too_long += 1
        
        print(f'{self.too_long} datapoints removed due to exceeding maximum length')
        
        self.max_trg_len = max_trg_len
        self.max_src_len = max_src_len
        self.pad_idx = pad_idx
        self.max_examples = max_examples
        
        # Construct the data
        # Get the shapes of the current data 
        spec = np.load(self.data_paths[0])
        notes = np.load(self.labels[self.data_paths[0]])

        # Create and empty data matrix
        # Shape for self.specs = [max_examples, 512, max_src_len]
        # Shape for self.notes = [max_examples, max_trg_len]
        self.specs = np.empty(shape=(max_examples, spec.shape[0], max_src_len))
        self.notes = np.empty(shape=(max_examples, max_trg_len))
        
        print(f'Populating {max_examples} samples into memory')
        for idx in tqdm(range(max_examples)):
            spec = self.pad_spec(np.load(self.data_paths[idx]))
            notes = self.pad_notes(np.load(self.labels[self.data_paths[idx]]))
            self.specs[idx,...] = spec
            self.notes[idx,...] = notes
        print(f'self.specs is taking up {sys.getsizeof(self.specs) / (1024**2):.2f} MB')
        print(f'self.notes is taking up {sys.getsizeof(self.notes) / (1024**2):.2f} MB')
        
    def __len__(self):
        return self.max_examples

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
        return torch.tensor(self.specs[idx], dtype=torch.float), torch.tensor(self.notes[idx])

class ColabLazyDataset(torch.utils.data.Dataset):
    '''
    Designed as an implementation of LazierDataset for Colab. The main difference is the file structure
    of the data it interfaces with, which has been generated using tensor_hero.preprocessing.data ->
    preprocess_transformer_data() w/ COLAB=True
    
    Inspects the data at partition_path, creates a list (data_paths) and a dictionary (labels) where the list contains
    paths to spectrogram slices (400ms) and the dictionarty contains path to spectrogram frames and corresponding notes
    arrays.
        - data_paths (list of Path): [<path to spectrogram frame> for _  in partition_path]
        - labels (dict) : [<path to spectrogram frame> : <path to corresponding notes>]

    Lazy loads data, i.e. loads each training example one at a time, as the dataloader is called.

    ~~~~ ARGUMENTS ~~~~
    - partition_path (Path): should be .../Training Data/Model 1 Training/<train, test, or val>
    - max_src_len (int): used for padding spectrograms, indicates what the length of the spectrograms should be in the time dimension
    - max_trg_len (int): used for padding notes, indicates what the length of the notes arrays should be
    - pad_idx (int): pad index, value the notes tensors will be padded with
    '''
    def __init__(self, partition_path, max_src_len, max_trg_len, pad_idx):

        song_paths = [partition_path / x for x in os.listdir(partition_path)]
        specs_dirs = [x / 'spectrograms' for x in song_paths]

        specs_lists = []
        for dir_ in specs_dirs:
            for specs_dir, _, specs in os.walk(dir_):
                if not specs:
                    continue
                specs_lists.append([Path(specs_dir) / spec for spec in specs])
            
        specs_lists = [spec for spec_list in specs_lists for spec in spec_list]  # Flatten
        notes_lists = [note_dirs_from_spec_dirs(x) for x in specs_lists]
        
        # Construct dictionary where key:value is <path to spec>:<path to notes array>
        l = {}  # labels
        for i in range(len(specs_lists)):
            l[specs_lists[i]] = notes_lists[i]
            
        # Weed out bits of data that exceed the maximum length
        self.labels = {}
        self.data_paths = []
        self.too_long = 0
        for x in tqdm(specs_lists):
            if check_notes_length(l[x], max_trg_len):
                self.data_paths.append(x)
                self.labels[x] = l[x]
            else:
                self.too_long += 1
        
        print(f'{self.too_long} datapoints removed due to exceeding maximum length')
        
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
        assert notes.shape[0] < self.max_trg_len, 'ERROR: notes array is longer than max_trg_len, (notes length = {}), (max_len = {})'.format(notes.shape[0], self.max_trg_len)
        notes = self.pad_notes(notes)

        return torch.tensor(spec, dtype=torch.float), torch.tensor(notes)

class LazierDataset(torch.utils.data.Dataset):
    '''
    Inspects the data at partition_path, creates a list (data_paths) and a dictionary (labels) where the list contains
    paths to spectrogram slices (400ms) and the dictionarty contains path to spectrogram frames and corresponding notes
    arrays.
        - data_paths (list of Path): [<path to spectrogram frame> for _  in partition_path]
        - labels (dict) : [<path to spectrogram frame> : <path to corresponding notes>]

    Lazy loads data, i.e. loads each training example one at a time, as the dataloader is called.

    ~~~~ ARGUMENTS ~~~~
    - partition_path (Path): should be .../Training Data/Model 1 Training/<train, test, or val>
    - max_len (int): used for padding, indicates what the length of the notes arrays should be
    - pad_idx (int): pad index, value the notes tensors will be padded with
    '''
    def __init__(self, partition_path, max_src_len, max_trg_len, pad_idx):
        # Construct list of spectrogram paths
        dp = [partition_path / 'spectrograms' / x for x in os.listdir(partition_path / 'spectrograms')]
        
        # Construct dictionary
        l = {}  # labels
        for data_path in dp:
            l[data_path] = data_path.parent.parent / 'notes' / (data_path.stem + '.npy')
            
        # Weed out bits of data that exceed the maximum length
        self.labels = {}
        self.data_paths = []
        self.too_long = []
        for x in tqdm(dp):
            if check_notes_length(l[x], max_trg_len):
                self.data_paths.append(x)
                self.labels[x] = l[x]
            else:
                self.too_long.append(x)
        
        print(f'{len(self.too_long)} datapoints removed due to exceeding maximum length')
        
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
        assert notes.shape[0] < self.max_trg_len, 'ERROR: notes array is longer than max_trg_len, (notes length = {}), (max_len = {})'.format(notes.shape[0], self.max_trg_len)
        notes = self.pad_notes(notes)

        return torch.tensor(spec, dtype=torch.float), torch.tensor(notes)

class RandomInputDataset(LazierDataset):
    '''
    Loads random spectrograms as input but actual notes as output

    Designed to test a baseline for what the model learns given meaningless input
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
    '''
    def __init__(
        self,
        embedding_size,
    ):
        super(InputEmbedding, self).__init__()

        # Take the spectrogram frames and pass them through a FC layer
        self.linear = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.Sigmoid(),
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

        # src_position_embedding and trg_position_embedding work the same, we can use nn.Embedding for both
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        # trg_word_embeddings can also leverage nn.Embedding, since the target values come from a
        # discrete vocabulary of note events (unlike the continuous spectrogram input)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)

        # continuous inputs of dim 512, so we can't feed nn.Embedding an index
        self.src_spec_embedding = InputEmbedding(embedding_size)

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
        Creates a boolean array of shape [batch size, max_len] 
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

        # out is shape [1,512,400], just like the src.
        # "out" means the embedding of the input
        # when we loop, we access [1,512,0], [1,512,1], [1,512,2], ... , [1,512,399]
        # translating 400 slices of spectrogram data to 400 slices of embeddings
        for idx in range(src.shape[2]): # For each spectrogram frame
            if idx < 400:
                out_list.append(self.src_spec_embedding(src[...,idx]).unsqueeze(-1))
            else:
                out_list.append(torch.zeros_like(out_list[0]))
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
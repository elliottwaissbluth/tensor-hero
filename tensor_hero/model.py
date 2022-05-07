import torch
from torch import nn
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import math
import sys
try:
    sys.path.insert(1, str(Path.cwd()))
    from tensor_hero.preprocessing.data import encode_contour, notes_array_time_adjust
except:
    sys.path.insert(1, str(Path.cwd().parent))
    from tensor_hero.preprocessing.data import encode_contour, notes_array_time_adjust
    

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
# ---------------------------------------------------------------------------- #
#                                CHUNKS DATASET                                #
# ---------------------------------------------------------------------------- #

class Chunks():
    '''
    Designed to initialize a ColabMemoryDataset. Chunks can act as a context manager outside
    the actual ColabMemoryDataset object, allowing the training loop to release data from
    memory while still keeping track of the list of files.

    ~~~~ ARGUMENTS ~~~~
    '''
    def __init__(self, partition_path, max_trg_len, max_examples, CHECK_LENGTH=False):
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
        too_long = 0
        if CHECK_LENGTH:
            print('Checking length of spectrograms and notes...')
        for x in tqdm(specs_lists):
            if CHECK_LENGTH:
                if check_notes_length(l[x], max_trg_len):
                    self.data_paths.append(x)
                    self.labels[x] = l[x]
                else:
                    too_long += 1
            else:
                self.data_paths.append(x)
                self.labels[x] = l[x]
                
        print(f'{too_long} datapoints removed due to exceeding maximum length')

        self.max_examples = max_examples
        self.num_samples = len(self.labels)  # This could be lower than max_samples
        self.num_chunks = math.ceil(self.num_samples / self.max_examples)
        
        del too_long, l, song_paths, specs_dirs, specs_lists, notes_lists
        
    def get_chunk(self, chunk_idx):
        '''
        Returns a portion of self.data_paths and self.labels which can be leveraged in
        the initialization of ColabMemoryDataset to populate.
        
        self.labels is returned in full regardless of the chunk_idx. The data should be 
        constructed by feeding chunked_data_paths into self.labels iteratively
        
        ~~~~ ARGUMENTS ~~~~
        - chunk_idx (int): refers to the section of the training data being returned
        
        ~~~~ RETURNS ~~~~
        - list: paths to spectrogram data in chunk
        - dict: keys are paths to spectrograms, values are paths to notes
        '''
        if chunk_idx+1 < self.num_chunks:
            return self.data_paths[chunk_idx*self.max_examples:(chunk_idx+1)*self.max_examples], self.labels
        else:
            return self.data_paths[chunk_idx*self.max_examples:], self.labels
                
class ColabChunksDataset(torch.utils.data.Dataset):
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
    - max_examples (int): the maximum number of samples per chunk
    '''
    def __init__(self, max_src_len, max_trg_len, pad_idx, data_paths, labels):

        self.max_trg_len = max_trg_len
        self.max_src_len = max_src_len
        self.pad_idx = pad_idx
        self.num_samples = len(data_paths)
        
        # Construct the data
        # Get the shapes of the current data 
        spec = np.load(data_paths[0])
        notes = np.load(labels[data_paths[0]])

        # Create and empty data matrix
        # Shape for self.specs = [max_examples, 512, max_src_len]
        # Shape for self.notes = [max_examples, max_trg_len]
        self.specs = np.empty(shape=(self.num_samples, spec.shape[0], max_src_len))
        self.notes = np.empty(shape=(self.num_samples, max_trg_len))
        
        self.num_samples = len(data_paths) 
        for idx in tqdm(range(self.num_samples)):
            spec = self.pad_spec(np.load(data_paths[idx]))
            notes = self.pad_notes(np.load(labels[data_paths[idx]]))
            self.specs[idx,...] = spec
            self.notes[idx,...] = notes
        print(f'self.specs is taking up {sys.getsizeof(self.specs) / (1024**2):.2f} MB')
        print(f'self.notes is taking up {sys.getsizeof(self.notes) / (1024**2):.2f} MB')
        del spec, notes
        
    def __len__(self):
        return self.num_samples

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
        return torch.tensor(self.specs[idx], dtype=torch.float), torch.tensor(self.notes[idx], dtype=torch.long)

# ---------------------------------------------------------------------------- #
#                                MEMORY DATASET                                #
# ---------------------------------------------------------------------------- #

class ColabMemoryDataset(torch.utils.data.Dataset):
    '''
    Inspects the data at partition_path, creates a list (data_paths) and a dictionary (labels) where the list contains
    paths to spectrogram slices (400ms) and the dictionarty contains path to spectrogram frames and corresponding notes
    arrays.
        - data_paths (list of Path): [<path to spectrogram frame> for _  in partition_path]
        - labels (dict) : [<path to spectrogram frame> : <path to corresponding notes>]

    Loads as many of these features into an array, self.specs and self.notes, as allowed by max_examples. max_examples should be
    chosen to be as large as possible without overloading RAM.

    ~~~~ ARGUMENTS ~~~~
    -   partition_path (Path): should be .../Training Data/Model 1 Training/<train, test, or val>
    -   max_src_len (int): used for padding spectrograms, indicates what the length of the spectrograms should be in the time dimension
    -   max_trg_len (int): used for padding notes, indicates what the length of the notes arrays should be
    -   max_examples (int): the maximum number of samples per chunk. If set to -1, uses whole dataset
    -   pad_idx (int): pad index, value the notes tensors will be padded with
    -   CHECK_LENGTH (bool): if True, will check each notes training example against max_trg_len to make sure there are no notes that
                             exceed this length. If one of these notes exists, it will trigger CUDA device side asset
    '''

    def __init__(self, partition_path, max_src_len, max_trg_len, max_examples, pad_idx, CHECK_LENGTH=False):
        song_paths = [partition_path / x for x in os.listdir(partition_path)]
        specs_dirs = [x / 'spectrograms' for x in song_paths]

        specs_lists = []
        print('Loading list of notes and spectrogram files')
        for dir_ in tqdm(specs_dirs):
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
        self.labels = {}        # holds spec paths as keys, note paths as values
        self.data_paths = []    # list of spec paths
        too_long = 0            # how many of the notes have more elements than max_trg_len
        if CHECK_LENGTH:
            print('Checking length of spectrograms and notes...')
            for x in tqdm(specs_lists):
                if check_notes_length(l[x], max_trg_len):
                    self.data_paths.append(x)
                    self.labels[x] = l[x]
                else:
                    too_long += 1
                print(f'{too_long} datapoints removed due to exceeding maximum length')
        else:
            self.data_paths = specs_lists
            self.labels = l
            print('Notes were not checked against max_trg_len')
                
        self.num_samples = len(self.labels)  # This could be lower than max_samples
        self.max_examples = max_examples if max_examples > 0 else self.num_samples
        self.max_trg_len = max_trg_len
        self.max_src_len = max_src_len
        self.pad_idx = pad_idx
        del too_long, l, song_paths, specs_dirs, specs_lists, notes_lists
        
        # Create and empty data matrix
        spec = np.load(self.data_paths[0])  # Load single examples to get shape
        notes = np.load(self.labels[self.data_paths[0]])
        # Shape for self.specs = [max_examples, 512, max_src_len]
        # Shape for self.notes = [max_examples, max_trg_len]
        self.specs = np.empty(shape=(self.max_examples, spec.shape[0], max_src_len))
        self.notes = np.empty(shape=(self.max_examples, max_trg_len))
        
        # Populate data into memory
        print('Populating data into memory')
        for idx in tqdm(range(self.max_examples)):
            spec = self.pad_spec(np.load(self.data_paths[idx]))
            notes = self.pad_notes(np.load(self.labels[self.data_paths[idx]]))
            self.specs[idx,...] = spec      # Final data
            self.notes[idx,...] = notes     # Final data
        print(f'self.specs (shape = {self.specs.shape}) is taking up {(sys.getsizeof(self.specs) / (1024**2))/1000:.2f} GB')
        print(f'self.notes (shape = {self.notes.shape}) is taking up {(sys.getsizeof(self.notes) / (1024**2)):.2f} GB')
        del spec, notes
        
    def __len__(self):
        return self.max_examples

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
        return torch.tensor(self.specs[idx], dtype=torch.float), torch.tensor(self.notes[idx], dtype=torch.long)

# ---------------------------------------------------------------------------- #
#                               CONTOUR DATASETS                               #
# ---------------------------------------------------------------------------- #

# LIFTED FROM tensor_hero.inference TO AVOID CIRCULAR IMPORT
def __single_prediction_to_notes_array(prediction):
    '''
    Takes a single prediction from the transformer and translates it to a notes array
    of length 400.

    ~~~~ ARGUMENTS ~~~~
    -   prediction (Numpy Array, shape=(<max_trg_len>,)):
            -   Prediction from the transformer, should be a single list of max indices
                from transformer prediction. Expected to be in (time, note, time, note, etc.)
                format.
        
    ~~~~ RETURNS ~~~~
    -   notes_array (Numpy Array, shape = (400,)):
            -   The prediction translated into a 4 second simplified notes array
    '''
    note_vals = list(range(32))     # Values of output array corresponding to notes
    time_vals = list(range(32,432)) # Corresponding to times

    # Loop through the array two elements at a time
    pairs = []
    for i in range(prediction.shape[0]-1):
        pair = (prediction[i], prediction[i+1]) # Take predicted notes as couples
        if pair[0] in time_vals and pair[1] in note_vals:
            pairs.append(pair)  # Append if pair follows (time, note) pattern

    # Create notes array
    notes_array = np.zeros(400)
    for pair in pairs:
        notes_array[pair[0]-32] = pair[1]
    return notes_array

def contour_vector_from_notes(notes, tbps, include_time=True):
    '''Captures original transformer output notes arrays and translates them to
    contour vectors

    Args:
        notes (1D numpy array): original transformer output formatted notes
        tbps (int): time bins per second represented in output array
    Returns:
        contour_vector (1D numpy array): transformer formatted contour array
            - [time, note plurality, motion, time, note plurality, motion, ...]
    '''
    notes_array = __single_prediction_to_notes_array(notes)

    # Reduce time bins per second from 100 to tbps
    notes_array, _ = notes_array_time_adjust(notes_array, time_bins_per_second=tbps)
    
    # Create contour
    contour = encode_contour(notes_array)
    
    # Convert to vector representation
    #      index         information
    #  0            | <sos> 
    #  1            | <eos> 
    #  2            | <pad> 
    #  3-15         | <note pluralities 0-13>
    #  16-24        | <motion [-4, 4]>
    #  25-(tbps+25) | <time bin 1-tbps>
    contour_vector = contour_to_transformer_output(contour, tbps, include_time)
    
    return contour_vector

def contour_to_transformer_output(contour, tbps, include_time=True):
    '''Generates transformer output version of contour array
    
    ~~~~ ARGUMENTS ~~~~
        contour (2D numpy array): contour array, note plurality is first row, motion
                                    is second row 
        tbps (int): time bins per second. Determines dimensionality of output_vector
        include_time (bool): Determines whether onset times are included in output dimension
            if True, contour_vectors are in following format:
                [<sos>, <time 0>, <note plurality 0>, <motion 0>, <time 1>, ..., <eos>]
            if False, 
                [<sos>, <note plurality 0>, <motion 0>, <note plurality 1>, ..., <eos>]
    ~~~~ RETURNS ~~~~
        contour_vector (1D numpy array):
            [time, note plurality, motion, time, note plurality, motion, ...]

        The values of contour_vector are detailed below

            value           information
        ____________________________________
        0            | <sos> 
        1            | <eos> 
        2            | <pad>
        3-15         | <note pluralities    contour (_type_): _description_
        16-24        | <motion [-4, 4]>    tbps (_type_): _description_
        25-(tbps+24) | <time bin 1-tbps>
    '''
    # These lambda functions translate contour encoded notes, motions, and times into their
    # respective vector indices
    motion_idx = lambda motion: motion + 20     # motion in [-4, 4] -> [16, 24]
    time_idx = lambda time: time + 25           # time bin in [0,tbps*4] -> [25, tbps*4+24]
    np_idx = lambda note_p: note_p + 2          # note plurality in [1, 13] -> [3, 15]

    # Find indices with note events and create empty vector for contour
    note_events = np.where(contour[0,:] > 0)[0].astype(int)
    if include_time:
        contour_vector = np.zeros(shape=(2+note_events.shape[0]*3))
    else:
        contour_vector = np.zeros(shape=(2+note_events.shape[0]*2))
    
    for idx, ne in enumerate(list(note_events)):
        if include_time:
            contour_vector[1+(3*idx)] = time_idx(ne)
            contour_vector[2+(3*idx)] = np_idx(contour[0, ne])
            contour_vector[3+(3*idx)] = motion_idx(contour[1, ne])
        else:
            contour_vector[1+(2*idx)] = np_idx(contour[0, ne])
            contour_vector[2+(2*idx)] = motion_idx(contour[1, ne])
    
    # Populate eos, sos is already encoded as 0 at contour_vector[0]
    contour_vector[-1] = 1
    
    return contour_vector

class ContourMemoryDataset(ColabMemoryDataset):
    '''Implementation of ColabMemoryDataset but transforms output into contour_vectors
    '''
    def __init__(self, partition_path, max_src_len, max_trg_len, max_examples, 
                 pad_idx, CHECK_LENGTH=False, tbps=25, include_time=True):
        '''

        Args:
            partition_path (Path): _description_
            max_src_len (int): _description_
            max_trg_len (int): _description_
            max_examples (int): _description_
            pad_idx (int): _description_
            CHECK_LENGTH (bool, optional): _description_. Defaults to False.
            tbps (int, optional): _description_. Defaults to 25.
            include_time (bool, optional): Determines format of transformer output,
                if True:
                    contour_vectors include onset times
                if False:
                    contour_vectors are only note pluralities and motions
        '''
        self.max_trg_len = max_trg_len
        self.max_src_len = max_src_len
        self.pad_idx = pad_idx
        self.tbps = 25
        
        # Construct list of spectrogram file paths and list of note file paths
        song_paths = [partition_path / x for x in os.listdir(partition_path)]
        specs_dirs = [x / 'spectrograms' for x in song_paths]
        specs_lists = []
        print('Loading list of notes and spectrogram files')
        for dir_ in tqdm(specs_dirs):
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
        self.labels = {}        # holds spec paths as keys, note paths as values
        self.data_paths = []    # list of spec paths
        too_long = 0            # how many of the notes have more elements than max_trg_len
        if CHECK_LENGTH:
            print('Checking length of spectrograms and notes...')
            for x in tqdm(specs_lists):
                if check_notes_length(l[x], max_trg_len):
                    self.data_paths.append(x)
                    self.labels[x] = l[x]
                else:
                    too_long += 1
                print(f'{too_long} datapoints removed due to exceeding maximum length')
        else:
            self.data_paths = specs_lists
            self.labels = l
            print('Notes were not checked against max_trg_len')
        
        # Restrict max samples in Dataset to min(max_examples, num_samples)        
        self.num_samples = len(self.labels)  # This could be lower than max_samples
        self.max_examples = max_examples if max_examples > 0 else self.num_samples
        self.max_examples = min(self.max_examples, self.num_samples)
        del too_long, l, song_paths, specs_dirs, specs_lists, notes_lists
        
        # Create and empty data matrix
        spec = np.load(self.data_paths[0])  # Load single examples to get shape
        notes = np.load(self.labels[self.data_paths[0]])
        # Shape for self.specs = [max_examples, 512, max_src_len]
        # Shape for self.notes = [max_examples, max_trg_len]
        self.specs = np.empty(shape=(self.max_examples, spec.shape[0], max_src_len))
        self.notes = np.empty(shape=(self.max_examples, max_trg_len))
        
        # Populate data into memory
        for idx in tqdm(range(self.max_examples)):
            spec = self.pad_spec(np.load(self.data_paths[idx]))
            # Transform notes into contour_vectors
            # contour_vectors are formatted to be transformer output
            notes = np.load(self.labels[self.data_paths[idx]])
            notes = contour_vector_from_notes(notes, tbps, include_time)
            notes = self.pad_notes(notes)
            self.specs[idx,...] = spec      # Final data
            self.notes[idx,...] = notes     # Final data
        print(f'self.specs (shape = {self.specs.shape}) is taking up {sys.getsizeof(self.specs) / (1024**2):.2f} MB')
        print(f'self.notes (shape = {self.notes.shape}) is taking up {sys.getsizeof(self.notes) / (1024**2):.2f} MB')
        del spec, notes

# ---------------------------------------------------------------------------- #
#                                 LAZY DATASET                                 #
# ---------------------------------------------------------------------------- #

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
    
    This dataset is good for training with a large amount of data on machines that are not file-read throttled.

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
        device,     # GPU or CPU
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


# ---------------------------------------------------------------------------- #
#                               ONSET FORMULATION                              #
# ---------------------------------------------------------------------------- #

def check_notes_length_and_emptiness(notes_path, max_len):
    '''Opens the processed notes array at notes_path and checks whether or not it is larger than max_len
       and whether the contour notes are empty
    Args:
        notes_path (Path): path to notes array
        max_len (int): maximum length of notes
        
    Returns:
        bool: Whether the notes array at notes_path is >= max_len
    '''
    good = True
    notes_array = np.load(notes_path)
    if (notes_array.shape[0] > max_len) or (notes_array.shape[0] == 2):
        good=False
    return good

class ContourOnsetMemoryDataset(ColabMemoryDataset):
    '''Implementation of ColabMemoryDataset but transforms output into contour_vectors
    '''
    def __init__(self, partition_path, max_src_len, max_trg_len, max_examples, 
                 pad_idx, CHECK_LENGTH=False, tbps=25, include_time=True, remove_empty=False,
                 max_spec_len = 400):
        '''

        Args:
            partition_path (Path): _description_
            max_src_len (int): _description_
            max_trg_len (int): _description_
            max_examples (int): _description_
            pad_idx (int): _description_
            CHECK_LENGTH (bool, optional): _description_. Defaults to False.
            tbps (int, optional): _description_. Defaults to 25.
            include_time (bool, optional): Determines format of transformer output,
                if True:
                    contour_vectors include onset times
                if False:
                    contour_vectors are only note pluralities and motions
        '''
        self.max_trg_len = max_trg_len
        max_trg_len_with_time = 160
        self.max_src_len = max_src_len
        self.pad_idx = pad_idx
        self.tbps = 25
        
        # Construct list of spectrogram file paths and list of note file paths
        song_paths = [partition_path / x for x in os.listdir(partition_path)]
        specs_dirs = [x / 'spectrograms' for x in song_paths]
        specs_lists = []
        print('Loading list of notes and spectrogram files')
        for dir_ in tqdm(specs_dirs):
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
        self.labels = {}        # holds spec paths as keys, note paths as values
        self.data_paths = []    # list of spec paths
        too_long = 0            # how many of the notes have more elements than max_trg_len
        print('Checking length of spectrograms and notes...')
        for x in tqdm(specs_lists):
            if check_notes_length_and_emptiness(l[x], max_trg_len):
                self.data_paths.append(x)
                self.labels[x] = l[x]
            else:
                too_long += 1 # NOTE: This includes too short now too
        print(f'{too_long} datapoints removed due to exceeding maximum length')
        
        # Restrict max samples in Dataset to min(max_examples, num_samples)        
        self.num_samples = len(self.labels)  # This could be lower than max_samples
        self.max_examples = max_examples if max_examples > 0 else self.num_samples
        self.max_examples = min(self.max_examples, self.num_samples)
        del too_long, l, song_paths, specs_dirs, specs_lists, notes_lists
        
        # Create and empty data matrix
        spec = np.load(self.data_paths[0])  # Load single examples to get shape
        notes = np.load(self.labels[self.data_paths[0]])
        # Shape for self.specs = [max_examples, 512, max_src_len]
        # Shape for self.notes = [max_examples, max_trg_len]
        self.specs = np.empty(shape=(self.max_examples, spec.shape[0], max_spec_len))
        self.notes = np.empty(shape=(self.max_examples, max_trg_len_with_time))
        self.notes_no_time = np.empty(shape=(self.max_examples, max_trg_len))
        
        # Populate data into memory
        print('Populating data into memory')
        for idx in tqdm(range(self.max_examples)):
            spec = self.pad_spec(np.load(self.data_paths[idx]), max_spec_len)
            # Transform notes into contour_vectors
            # contour_vectors are formatted to be transformer output
            notes = np.load(self.labels[self.data_paths[idx]])
            notes_no_time = contour_vector_from_notes(notes, tbps, include_time=False)
            notes = contour_vector_from_notes(notes, tbps, include_time=True)

            # To be used to extract slices from spectrograms
            notes = self.pad_notes(notes, max_trg_len_with_time)
            self.notes[idx,...] = notes     # Final data

            # To be used for actual training
            notes_no_time = self.pad_notes(notes_no_time, self.max_trg_len)
            self.notes_no_time[idx,...] = notes_no_time

            self.specs[idx,...] = spec      # Final data

        print(f'self.specs (shape = {self.specs.shape}) is taking up {sys.getsizeof(self.specs) / (1024**2):.2f} MB')
        print(f'self.notes (shape = {self.notes.shape}) is taking up {sys.getsizeof(self.notes) / (1024**2):.2f} MB')
        del spec, notes
        
    def pad_spec(self, spec, max_len):
        '''pads spec with zeros to length max_src_len'''
        spec = np.pad(spec,
                      ((0, 0), (0, max_len-spec.shape[1])),
                      'constant',
                      constant_values=0)
        return spec
    
    def pad_notes(self, notes, max_len):
        '''pads notes with pad_idx to length max_trg_len'''
        notes = np.pad(notes, 
                    (0, max_len-notes.shape[0]),
                    'constant',
                    constant_values=self.pad_idx)
        return notes

    def spec_slices_from_notes(self, spec, notes, window=3):
        '''Takes timesteps where notes are present and returns a tensor of spectrogram
        slices at those timesteps, plus window frames on either side

        Args:
            spec (torch.Tensor, dtype=torch.float, shape=(512,400)):
                - log-mel spectrogram, 512 frequency bins, 400 time bins
            notes (1D numpy array): 
                - Contains the contour encoded notes from contour_vector_from_notes()
                - Assumes 25 tbps
            window (int):
                - The number of time frames on each side of an onset to pull from the spectrogram
                - Default is 3, 30ms on each side.
        
        Returns:
            spec (torch.Tensor, dtype=torch.float, shape = (512, 2*window+1, (max_trg_len-3)/2)):
                - (max_trg_len-3)/2 is the maximum number of notes in a sequence
                
        '''
        # Get onsets by leveraging fact that onsets are encoded between [25, tbps*4+25] 
        onsets = np.delete(notes, np.argwhere(notes < 25))
        onsets -= 25  # Subtract 25 to get exact time index
        onsets *= 4   # multiply by 4 to get corresponding indices
        
        # Pad spec on either side to avoid window overflow
        big_spec = torch.zeros(size=(512, 400+2*window+1))
        big_spec[:,window:(400+window)] = spec[:,:400]
        
        # Create empty tensor for spectrogram slices
        specs = torch.zeros(size=(512, 2*window+1, self.max_src_len), dtype=torch.float)
        for idx, onset in enumerate(onsets):
            onset = int(onset+window)  # To index properly in big_spec
            spec_slice = torch.zeros(size=(512, 2*window+1), dtype=torch.float)
            spec_slice[:,:window] = big_spec[:,(onset-window):onset]
            spec_slice[:,(window+1):] = big_spec[:,(onset+1):(onset+window+1)]
            spec_slice[:,window] = big_spec[:,onset]
            specs[:,:,idx] = spec_slice 
        
        return specs

        
    def __getitem__(self, idx):
        # Get spectrogram slices
        notes = self.notes[idx]
        spec = torch.tensor(self.specs[idx], dtype=torch.float)
        spec = self.spec_slices_from_notes(spec, notes)

        notes = list(notes)
        # if 125 in [int(x) for x in notes]:
            # for i in range(spec.size()[2]):
                # plt.figure()
                # specshow(spec[:,:,i].numpy())

        return spec, torch.tensor(self.notes_no_time[idx], dtype=torch.long)

class OnsetInputEmbedding(nn.Module):
    '''
    This is a custom class which can be used in place of nn.Embedding for input embeddings.
    We can't use nn.Embedding for our input because our input is continuous, and nn.Embedding
    only works with discrete vocabularies such as words.
    '''
    def __init__(
        self,
        embedding_size,
    ):
        super(OnsetInputEmbedding, self).__init__()

        # Take the spectrogram frames and pass them through a FC layer
        self.linear = nn.Sequential(
            nn.Linear(512*7, embedding_size),
            nn.Sigmoid(),
        )
    
    def forward(self, src):
        return self.linear(src)
    
class OnsetTransformer(nn.Module):
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
        super(OnsetTransformer, self).__init__()
        self.max_len = max_len
        # src_position_embedding and trg_position_embedding work the same, we can use nn.Embedding for both
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        # trg_word_embeddings can also leverage nn.Embedding, since the target values come from a
        # discrete vocabulary of note events (unlike the continuous spectrogram input)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)

        # continuous inputs of dim 512, so we can't feed nn.Embedding an index
        self.src_spec_embedding = OnsetInputEmbedding(embedding_size)

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

    def make_src_mask(self, src, trg):
        '''
        Finds the non-padded indices of trg and creates a mask to that length
        '''
        num_trg = (torch.sum(trg != 2, dim=1)-2)/2  # -2 to remove <sos>, <eos>
                                                    # /2 because trg encodes each note as two indices
        src_mask = torch.arange(self.max_len).expand(num_trg.size()[0], self.max_len).to(self.device) 
        src_mask = src_mask < num_trg.unsqueeze(1)
        
        return src_mask
    
    def make_embed_src(self, src):
        '''
        This function passes the spectrogram frames through the input embedding layer. Notice how it does
        so one layer at a time in a for loop. It may be possible to increase the speed of this and parallelize
        by doing a lil trick with nn.Conv1D. We'll investigate this in the future.
        '''
        # out = torch.zeros_like(src, requires_grad=True).to(self.device)
        out_list = []

        for idx in range(src.shape[-1]): # For each spectrogram frame of width 7
            if idx < self.max_len: # 50
                src = torch.reshape(src, (src.shape[0], 512*7, self.max_len))
                out_list.append(self.src_spec_embedding(src[...,idx]).unsqueeze(-1))
            else:
                out_list.append(torch.zeros_like(out_list[0]))
        return out_list

    def forward(self, src, trg):
        src_seq_length, N = src.shape[-1], src.shape[0]  # Should be 50, N
        trg_seq_length, N = trg.shape[1], trg.shape[0]  # The target sequence length is 51 for our toy training example
                                                        # Originally 52, but we shift the target to mask the last value

        src_padding_mask = self.make_src_mask(src, trg).to(self.device)
        
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
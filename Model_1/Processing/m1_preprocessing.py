from pathlib import Path 
import sys
# NOTE: You will have to change this to run it on your local machine
sys.path.insert(1, r'/Users/ewaissbluth/Documents/GitHub/tensor-hero/Shared_Functionality/Data_Viz')    # NEEDSCHANGE
sys.path.insert(1, r'/Users/ewaissbluth/Documents/GitHub/tensor-hero/Shared_Functionality/Preprocessing/Preprocessing Functions')   # NEEDSCHANGE
import numpy as np
from data_viz_functions import *
from preprocess_functions import *
import torch
import os
import pickle
import math
from tqdm import tqdm

def notes_to_output_space(notes):
    '''
    Takes a notes array as input, and outputs a matrix of numpy arrays in the output format specified
    by sequence to sequence piano transcription

    ~~~~ ARGUMENTS ~~~~
    - notes : numpy array

    ~~~~ RETURNS ~~~~
    - formatted : numpy array
        - Y axis is sequential note events
        - X axis is one hot encoded arrays
    '''
    # Get number of notes in array
    num_notes = np.count_nonzero(notes)

    # Convert "218" i.e. open notes to
    notes = np.where(notes == 218, 32, notes)

    # Construct a numpy array of the proper dimensionality
    # 32 positions for the one hot encoded notes, 400 for the absolute time
    formatted = np.zeros(shape=(num_notes*2, 32 + 400))

    # Loop through notes array and populate formatted

    i = 0
    for time_pos, x in enumerate(notes):
        if x != 0:
            formatted[2*i+1, int(x)-1] = 1   # subtract 1 for python indexing
            formatted[2*i, time_pos+32] = 1  # Add 32 to put at end of arrays
            i += 1

    return formatted

def formatted_notes_to_indices(notes):
    '''
    Takes formatted notes and returns a 1D array of indices, reverse one hot operation.
    Helper function for prepare_notes_tensor()

    ~~~~ ARGUMENTS ~~~~
    notes : numpy array
        - formatted notes, as output by notes_to_output_space()
    
    ~~~~ RETURNS ~~~~
    indices : numpy array
        - de-one hot encoded indices of note event series.
        - the format is [time, note, time, note, etc...] 
    '''
    # Loop through each row
    indices = np.argwhere(notes == 1)
    indices = indices[:,-1].flatten()
    return indices

def prepare_notes_tensor(notes):
    '''
    Takes formatted notes and converts them to the format suitable for PyTorch's transformer model.
    Helper function for populate_model_1_training_data()

    ~~~~ ARGUMENTS ~~~~
    - notes : numpy array
        - formatted notes, as output by notes_to_output_space()

    ~~~~ RETURNS ~~~~
    - notes : numpy array
        - format is [<sos>, time, note, time, note, etc..., <eos>]
    '''
    # Concatenate two extra dimensions for SOS and EOS to self.notes
    notes_append = np.zeros(shape=(notes.shape[0], 2))
    notes = np.c_[notes, notes_append]
    # Add a row at the beginning and end of note for <sos>, <eos>
    notes_append = np.zeros(shape=(1,notes.shape[1]))
    notes = np.vstack([notes_append, notes, notes_append])
    # Add proper values to self.notes
    notes[0,-2] = 1  # <sos>
    notes[-1,-1] = 1 # <eos>
    notes = formatted_notes_to_indices(notes)
    # Note: pytorch tensors don't compress as well as numpy arrays
    # notes = torch.tensor(notes, dtype=torch.float)
    return notes

def process_spectrogram(spec):
    '''
    Removes padding from spectrogram and normalizes in [0,1]

    ~~~~ ARGUMENTS ~~~~
    - spec : numpy array
        - padded spectrogram, loaded from spectrogram.npy in processed folder
    
    ~~~~ RETURNS ~~~~
    - spec : numpy array
        - Normalized and 70ms padding removed from beginning and end of time dimension
    '''
    spec = spec[:, 7:-7]    # Take off the padding
    spec = (spec+80) / 80   # Regularize
    return spec

def populate_model_1_training_data(training_data_path, model_1_training_path, REPLACE=False):
    '''
    Takes the spectrogram.npy files and the notes.npy files from the processed training data and
    slices them into torch tensors representing 400ms of data. Creates train, val, and test
    directories in model_1_training_path and populates them with numpy files of training data.
    These can be leveraged by a dataloader during training

    TO USE:
        - Within your training data folder, create a folder for this data, call it
        "Model 1 Training" for simplicity
        - Change the paths in the __main__ script and in the sys.path.insert lines to match your local directory
            - things that need changed are tagged with "NEEDSCHANGE"
        - Run this script

    NOTES:
        - Documentation on the format of the training data is available in the README within this directory
    
    ~~~~ ARGUMENTS ~~~~
    - training_data_path : Path object
        - Path to your training data, most likely ./tensor-hero/Training Data
    - model_1_training_path : Path object
        - Path to the directory where this data will be saved
        - suggested: ./tensor-hero/Training Data/Model 1 Training
    - REPLACE : bool
        - If true, will replace the  files already present in the directory

    ~~~~ RETURNS ~~~~
    N/A

    ~~~~ SAVES ~~~~
    - train_key.pkl : dict
        -  k,v = [index of saved pytorch file : information about where it came from]
        - "information" is another dictionary with
            - k,v = 'origin' : Path to origin, 'slice' : which 400ms section it came from
    '''


    # Get list of processed song paths
    unprocessed_path = training_data_path / 'Unprocessed'
    _, processed_list = get_list_of_ogg_files(unprocessed_path)

    # Get paths of notes and corresponding paths of spectrograms
    spec_paths = [song / 'spectrogram.npy' for song in processed_list]
    notes_paths = [song / 'notes_simplified.npy' for song in processed_list]

    def process_spectrogram(spec):
        spec = spec[:, 7:-7]    # Take off the padding
        spec = (spec+80) / 80   # Regularize
        return spec

    # Used to create the outfile names of the saved slices
    # Will also be able to use these in conjunction with "train_key", "test_key", and "val_key"
    # to determine which indices go to which song
    train_count = 0
    val_count = 0
    test_count = 0

    # Create the directory if it doesn't exist
    if not os.path.isdir(model_1_training_path):
        os.mkdir(model_1_training_path)

    # These paths are used to save the data once it is processed
    train_path = model_1_training_path / 'train'
    if not os.path.isdir(train_path):
        os.mkdir(train_path)
        os.mkdir(train_path / 'spectrograms')
        os.mkdir(train_path / 'notes')
    else:
        print('Deleting current files...')
        if REPLACE:
            for f in os.listdir(train_path / 'spectrograms'):
                os.remove(os.path.join(train_path / 'spectrograms', f))
            for f in os.listdir(train_path / 'notes'):
                os.remove(os.path.join(train_path / 'notes', f))
    val_path = model_1_training_path / 'val'
    if not os.path.isdir(val_path):
        os.mkdir(val_path)
        os.mkdir(val_path / 'spectrograms')
        os.mkdir(val_path / 'notes')
    else:
        if REPLACE:
            for f in os.listdir(val_path / 'spectrograms'):
                os.remove(os.path.join(val_path / 'spectrograms', f))
            for f in os.listdir(val_path / 'notes'):
                os.remove(os.path.join(val_path / 'notes', f))
    test_path = model_1_training_path / 'test'
    if not os.path.isdir(test_path):
        os.mkdir(test_path)
        os.mkdir(test_path / 'spectrograms')
        os.mkdir(test_path / 'notes')
    else:
        if REPLACE:
            for f in os.listdir(test_path / 'spectrograms'):
                os.remove(os.path.join(test_path / 'spectrograms', f))
            for f in os.listdir(test_path / 'notes'):
                os.remove(os.path.join(test_path / 'notes', f))

    # These dictionaries can act like keys if you'd like to find the origin of a piece of training data
    train_key = {}
    test_key = {}
    val_key = {}

    for i in tqdm(range(len(notes_paths))):
        # Process spectrogram
        try:
            spec = np.load(spec_paths[i])
        except FileNotFoundError:
            print('There is no spectrogram at {}'.format(spec_paths[i]))
            continue
        except ValueError as err:
            print(err)
            continue
        spec = process_spectrogram(spec)

        # Process notes
        try:
            notes = np.load(notes_paths[i])
        except FileNotFoundError:
            print('There is no notes_simplified at {}'.format(notes_paths[i]))
            continue
        notes = notes[7:-7]  # Eliminate padding

        assert notes.shape[0] == spec.shape[1], 'ERROR: Spectrogram and notes shape do not match'
        
        # Get number of 4 second slices
        num_slices = math.floor(spec.shape[1]/400)
        
        # Split notes and spectrogram into bins
        spec_bins = np.array([spec[:,i*400:(i+1)*400] for i in range(num_slices)])
        notes_bins = np.array([notes[i*400:(i+1)*400] for i in range(num_slices)])
        
        # This list will hold the pytorch note representations
        torch_notes = []
        for i in range(num_slices):
            t_notes = notes_to_output_space(notes_bins[i,:])
            t_notes = prepare_notes_tensor(t_notes)
            torch_notes.append(t_notes)
        
        # Convert the spectrogram to pytorch tensor
        # NOTE: This was too expensive to save, converting to numpy
        # torch_spec = torch.tensor(spec_bins, dtype=torch.float)
        torch_spec = spec_bins

        # Generate random integers to determine which folder the data shall be put into
        train_test_split_key = np.random.randint(10, size=num_slices)

        for idx, k in enumerate(train_test_split_key):
            # Define prepend path and count based on value of train_test_split_key
            if k == 9:
                prepend_path = test_path
                count = test_count
            elif k == 8:
                prepend_path = val_path
                count = val_count
            else:
                prepend_path = train_path
                count = train_count
            
            # Save file
            # NOTE: Converted to numpy files for space saving
            spec_outfile = prepend_path / 'spectrograms' / (str(count) + '.npy')
            notes_outfile = prepend_path / 'notes' / (str(count) + '.npy')
            # torch.save(spec_to_save, spec_outfile)
            # torch.save(torch_notes[idx], notes_outfile)
            np.save(spec_outfile, torch_spec[idx,...])
            np.save(notes_outfile, torch_notes[idx])

            # Increment counter, populate key
            if k == 9:
                test_key[test_count] = {'origin' : processed_list[idx], 'slice' : idx}
                test_count += 1
            elif k == 8:
                val_key[test_count] = {'origin' : processed_list[idx], 'slice' : idx}
                val_count += 1
            else:
                train_key[test_count] = {'origin' : processed_list[idx], 'slice' : idx}
                train_count += 1
        
    # Save train_key, test_ley, val_key
    with open('train_key.pkl', 'wb') as f:
        pickle.dump(train_key, f)
    f.close()
    with open('val_key.pkl', 'wb') as f:
        pickle.dump(val_key, f)
    f.close()
    with open('test_key.pkl', 'wb') as f:
        pickle.dump(test_key, f)
    f.close()
    return

def preprocess(training_data_path):
    '''
    Run this function to preprocess the entire training data. Creates a folder in the main training data directory
    called "Model 1 Training" and populates with processed data.

    ~~~~ ARGUMENTS ~~~~
    - training_data_path - path or string
        - Path to main training data directory, .../Training Data
    '''
    model_1_training_path = training_data_path / 'Model 1 Training'
    populate_model_1_training_data(training_data_path, model_1_training_path, REPLACE=True)
    return

if __name__ == '__main__':
    training_data_path = Path.cwd() / 'Training Data' / 'Training Data' # NEEDSCHANGE
    # print(training_data_path)
    preprocess(training_data_path)
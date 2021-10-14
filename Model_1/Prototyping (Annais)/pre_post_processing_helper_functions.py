# Bugs/todos: Turn no_notes into range 0 to 32, with mod & > 218 is 32
# Bugs/todos for later: Currently if song not exactly divisible by 400, last snippet will be removed - so in & out will not be exactly equal mostly


import numpy as np
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
import math, copy, time
#from torch.autograd import Variable
#import matplotlib.pyplot as plt
#import seaborn
#seaborn.set_context(context="talk")
#%matplotlib inline


# Global variables (should be passed in?)
seq_len = 400
no_notes = 32
freq_range = 512



def spec_to_in_spec_list(spec: np.array):
    """
    Split full spectrogram song into list of sub-spectrograms, each of length 400ms

    :param spec:
        spec (np.array): 2D input numpy array spectrogram of given song (512 frequency bins x length of song (in 10ms)
    :return:
        in_spec_list (list): List of numpy 2D numpy array spectrograms containing the song split up into 400ms bins
    """

    # Create empty list of sub-spectrograms
    in_spec_list = []

    no_seq = int(np.floor(spec.shape[1]/seq_len))

    # Iterate through full spectrogram & split up into 400ms bins
    for n in range(no_seq):
        # BUG: potentially last snippet not exactly 400s and not 400 columns
        in_spec_list.append(spec[:, n*400:n*400+400])

    return in_spec_list




def out_seq_list_to_notes(out_seq_list: list):
    """
    Convert seuqence to notes array (where 1 x length of song (in ms)

    :param out_seq_list:
        out_seq_list (list): Model output, where len(list) == length of song (in ms) / sequence length (400ms) * 2 for
        each time and note array; each sequence in list alternates between time position indication and note on a 1x431
        (number of notes + sequence interval bin in ms) array
    :return:
        Notes array for full song, where dimensions are 1 x length of song (in ms) and values are the notes (0-31)
    """
    max_list = []
    for n in out_seq_list:
        max_list.append(np.argmax(n))
    out_notes_len = int(np.ceil(max(max_list)/seq_len))*seq_len

    notes = np.zeros(out_notes_len)

    for i in range(0, len(max_list), 2):
        time_pos = max_list[i] - no_notes
        not_val = max_list[i+1]

        notes[time_pos] = not_val

    return notes




def notes_to_out_seq_list(notes: np.array):
    """
    Turn 1D note array into list out single sequence, where 1D array element alternately represents time of note and value of note

    :param notes:
        Notes array for full song, where dimensions are 1 x length of song (in ms) and values are the notes (0-31)
    :return:
        out_seq_list (list): Model output, where len(list) == length of song (in ms) / sequence length (400ms) * 2 for
        each time and note array; each sequence in list alternates between time position indication and note on a 1x431
        (number of notes + sequence interval bin in ms) array
    """

    # Create empty list for sub-sequence
    out_seq_list = []

    no_seq = int(np.floor(len(notes)/seq_len))
    rel_notes = notes[0:(no_seq*seq_len)]

    # Get all positions of non-zero values
    non_zero_pos = np.where(rel_notes > 0)

    for pos in non_zero_pos[0]:
        time_pos_arr = np.zeros(seq_len + no_notes)
        note_val_arr = np.zeros(seq_len + no_notes)

        note_val = int(rel_notes[pos])
        time_pos_arr[pos + no_notes] = 1
        note_val_arr[note_val] = 1

        out_seq_list.append(time_pos_arr)
        out_seq_list.append(note_val_arr)


    return out_seq_list




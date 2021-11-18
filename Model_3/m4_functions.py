import numpy as np
import librosa
import random

import sys
sys.path.insert(1, r'C:\Users\ewais\Documents\GitHub\tensor-hero\Model_1\Processing')
from m1_postprocessing import *
from pathlib import Path
import shutil

from numpy.lib.utils import source

def onset_time(song_path):
    '''
    Loads the song at song_path, computes onsets, returns array of times

    ~~~~ ARGUMENTS ~~~~
    - song_path : Path or str
        - path to song 
    '''
    # Load the songs and the notes arrays one at a time
    # for idx in range (len(song_paths)):
    # Load the song
    y, sr = librosa.load(song_path)

    # resample the song if it isn't sr=22050 (for consistent sizing)
    if not sr == 22050:
        y = librosa.resample(y, sr, 22050)
        sr = 22050

    #source seperation, margin can be tuned 
    y_harmonic, _ = librosa.effects.hpss(y, margin=2.0)

    # Set Hop_len
    hop_len = 512

    onset_frame_backtrack = librosa.onset.onset_detect(y_harmonic, sr = sr, hop_length = hop_len, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frame_backtrack)

    return y_harmonic, onset_times

def onset_time_bins(onset_times):
    otb = [int(x) for x in onset_times*100]
    return otb

def create_notes_array(onsets, notes):
    '''
    Takes equal length sequences of onsets and notes and creates a notes array which can be used to create
    final song output (leveraging Model_1/Processing/m1_postprocessing.py -> write_song_from_notes_array())

    ~~~~ ARGUMENTS ~~~~
    - onsets : list
        - contains onset times as 10ms bins
        - should be output of onset_time_bins()
    
    - notes : list or numpy array
        - notes corresponding to the onsets
    '''
    if type(notes) is np.ndarray:
        notes = notes.tolist()

    assert len(onsets) <= len(notes), 'ERROR: There are more onsets than notes'

    # Cut down notes if there are more notes than onsetse
    if len(notes) > len(onsets):
        notes = notes[:len(onsets)]
    
    notes_array = np.zeros(onsets[-1])
    np.put(notes_array, onsets, notes)

    return notes_array
    
def generate_notes(onset, interval_length=100):
    '''
    Function that takes in onsets and generates notes based on the time interval difference
    User can change the interval
    Shorter intervals will be a scale, longer intervals will be chords
    Interval length: 100 equals 1 sec
    Output is same length as onset and a numpy array
    '''
    # Notes: 1-5 are single notes
    # 6 - 31 are chords
    # 32 is open note
    # conditions:
    # check onset and onset + 1
    # if < 2 then it should be single note
    curr_note = 0
    note_array = []
    # need to generate first note
    if onset[1] - onset[0] < interval_length:
        curr_note = random.randint(1,5)
        note_array.append(curr_note)
    else:
        curr_note = random.randint(6,31)
        note_array.append(curr_note)
    for i in range(1,len(onset)-1): # since we’ll be forward looking by one
        curr_note = calc_note(i,onset, curr_note)
        note_array.append(curr_note)
    return note_array

def calc_note(idx, onset, curr_note, interval_length=100):
    # if short interval and current note is a single note
    n = random.random()
    if (onset[idx+1] - onset[idx] < interval_length) & (curr_note <= 5):
        if n < (1/3): # note repeats
            curr_note = curr_note
        elif n < (2/3): # note goes up (unless current note is 5)
            if curr_note == 5:
                curr_note = curr_note - 1
            else:
                curr_note = curr_note + 1
        else: # note goes down (unless current note is 1)
            if curr_note == 1:
                curr_note = curr_note + 1
            else:
                curr_note = curr_note - 1
    # if short interval but current note is a chord, goes back to single note
    elif (onset[idx+1] - onset[idx] < interval_length):
        curr_note = random.randint(1,5)
    elif (onset[idx+1] - onset[idx] >= interval_length) & (5 < curr_note < 32):
        n = random.random()
        if n > .25:
            curr_note = curr_note
        else:
            curr_note = random.randint(6,31)
    else:
        curr_note = random.randint(6,31)
    return curr_note

def generate_song(song_path, 
                  note_generation_function, 
                  onset_computation_function = onset_time,
                  generation_function_uses_song_as_input = False, 
                  source_separated_path = None, 
                  outfile_song_name = 'Model 4', 
                  artist = 'Forrest'):
    '''
    Takes the song present at song_path, uses onset_computation_function to compute onsets, uses note_generation_function
    to generate notes, then writes the song to an outfolder at ~/Model_3/Generated Songs - TO BE DELETED/<outfile_song_name>

    ~~~~ ARGUMENTS ~~~~
    - song_path : Path or str
        - path to the original song
        - this will be used to create the folder ingested by Clone Hero
    - note_generation_function : function
        - takes onset indices as input, formatted as an array of onset times in 10ms time bins
        - can optionally take song data as input, should set generation_function_uses_song_as_input = True in this case
        - outputs a sequence of notes that has the same length as the onsets array
        - uses source separated audio by default
    - onset_computation_function : function
        - given song_path (or source_separated_path if it is not None), generates onsets
        - onsets should be formatted as an array with entries being onsets in seconds, these will later be converted to
          the 10ms time bines required by note_generation_function
    - generation_function_uses_song_as_input : bool
        - if True, note_generation_function will be called as note_generation_function(onset_indices, song_path)
        - if False, note_generation_function will be called as note_generation_function(onset_indices)
    - source_separated_path : Path or str
        - Path to source separated file if it exists
    - outfile_song_name : str
        - this will determine the name of the folder the song will be saved under
        - it also determines the song name that will appear in Clone Hero once the folder is transferred
    - artist : str
        - determines the artist written to the .chart file
    '''
    if source_separated_path is not None:
        path = source_separated_path
    else:
        path = song_path
    
    _, onset_times = onset_computation_function(path)
    onset_indices = onset_time_bins(onset_times)

    if generation_function_uses_song_as_input:
        dense_notes = note_generation_function(onset_indices, path)
    else:
        dense_notes = note_generation_function(onset_indices)
    
    notes_array = create_notes_array(onset_indices, dense_notes)

    song_metadata = {'Name' : outfile_song_name,
                    'Artist' : artist,
                    'Charter' : 'tensorhero',
                    'Offset' : 0,
                    'Resolution' : 192,
                    'Genre' : 'electronic',
                    'MediaType' : 'cd',
                    'MusicStream' : 'song.ogg'}

    outfolder = Path(r'C:\Users\ewais\Documents\GitHub\tensor-hero\Model_3\Generated Songs\\'+ outfile_song_name)
    write_song_from_notes_array(song_metadata, notes_array, outfolder)
    shutil.copyfile(str(song_path), str(outfolder / 'song.ogg'))

# ---------------------------------------------------------------------------- #
#                                  DEPRECATED                                  #
# ---------------------------------------------------------------------------- #

# def redistribute_trans_table(A):
    # '''
    # We found the original transition table to be too heavy on the single notes
    # This caused the output from viterbi decode to be a constant back and forth
    # between two single notes. This function redistributes the values of the table
    # to mitigate this behavior.
    # '''
    # for i in range(5,A.shape[1]):
        # current_col = A[:,i]
        # idx = (-current_col).argsort()[:2]
        # second_max_val = current_col[idx[1]]
        # current_col[idx[0]] = second_max_val
        # new_array = (current_col/current_col.sum(axis=0))
        # A[:,i] = new_array
    
    # for i in range(0,5):
        # current_col = A[:,i]
        # idx = (-current_col).argsort()[:1]
        # max_val = current_col[idx[0]]
        # new_array = np.zeros_like(A[5:,i])
        # new_array = np.random.uniform(low=0,high=max_val,size=new_array.shape[0])
        # A[5:,i] = new_array
        # A[:,i] = A[:,i]/(A[:,i].sum(axis=0))
    # return A

# def viterbi(A, B, sequence, B_weight=1):
    # '''
    # Viterbi decodes a note sequence from labels (sequence), transition and emission probability table
    # ~~~~ ARGUMENTS ~~~~
    # sequence : list
        # - a sequence of labels
    # A : numpy array
        # - transition table
        # - shape = [number of notes + 1 for <start>, number of notes]
        # - NOTE: the <start> token should be indexed at the last row
    # B : numpy array
        # - emission table
        # - shape = [number of notes, number of possible labels]
    # '''
    # # let's work in log space
    # A = np.log(A)
    # B = np.log(B*B_weight)

    # num_notes = B.shape[0]

    # # create empty viterbi matrix and backpointer matrix
    # viterbi = np.full((num_notes, len(sequence)), None)
    # bp = np.full((num_notes, len(sequence)), None)

    # # Compute the first column
    # first_label = sequence[0]
    # start_token = A.shape[0]
    # for n in range(num_notes):
        # viterbi[n,0] = A[-1,n] + B[n,first_label]
        # bp[n,0] = start_token

    # for w in range(1, len(sequence)):
        # for n in range(num_notes):
            # viterbi[n,w], bp[n,w] = compute_viterbi_val(n=n, w=w, viterbi=viterbi, A_prev=A[:,n], B_prev=B[n,w]) #transitions from previous note to current note

    # # Find maximum value of last column of viterbi
    # max_idx = np.argmax(viterbi[:,-1])
    # # Trace back maximum indices in backpointer table
    # note_sequence = [max_idx]
    # next_note = bp[max_idx,-1]
    # for i in range(1,len(sequence)):
        # reverse_i = len(sequence)-i
        # # print('reverse_i : {}'.format(reverse_i))
        # note_sequence.append(bp[next_note, reverse_i])
        # next_note = bp[next_note, reverse_i]
        # # print('next_note: {}'.format(next_note))


    # # for i in range(0,len(sequence)):
        # # reverse_i = len(sequence)-i-1
        # # print(f'[{i-1},{reverse_i}]')
        # # note_sequence.append(bp[note_sequence[i-1], reverse_i])
    
    # note_sequence.reverse()
    # return note_sequence, viterbi, bp

# def compute_viterbi_val(n, w, viterbi, A_prev, B_prev):
    # '''
    # Helper function for viterbi(), computes single viterbi val
    # '''
    # # Compute first viterbi value
    # current_val = viterbi[0,w-1] + A_prev[0] + B_prev
    # max_val = current_val
    # bp = 0

    # # Loop through rest of values
    # for i, v in enumerate(list(viterbi[:,w-1])):
        # current_val = v + A_prev[i] + B_prev
        # if current_val > max_val:
            # max_val = current_val
            # bp = i
    
    # return max_val, bp

# def make_emission_table():
    # '''
    # Makes emission table as numpy array
    # '''
    # # Numpy array
    # em_table = np.random.uniform(0,1,(32,12000))
    # new_array = (em_table/em_table.sum(axis=0))
    # return new_array

# def onset_label(onset, spectrogram=None):
    # '''
    # Function that takes in 1D onset array and spectrogram and labels each onset with the spectrogram that most closely
    # matches the emission probability table

    # INPUTS: 1D Onset Array , computed spectrogram
    # OUTPUTS: 1D Array of same length as onset array corresponding to column indices of emission probability table 
    # '''
    # # X will probably need to be determined by spectrogram clusters , set locally just to highlight
    # X = 12000
    # return np.random.randint(0,X,len(onset))
import numpy as np
import librosa
import random
from tensor_hero.inference import write_song_from_notes_array
import shutil

'''
This script was previously implemented as m4_functions.py
'''

def __onset_time(song_path):
    '''
    Loads the song at song_path, computes onsets, returns array of times and harmonic component
    of source separated audio

    ~~~~ ARGUMENTS ~~~~
    - song_path (Path or str): path to audio file
    
    ~~~~ RETURNS ~~~~
    - y_harmonic (1D numpy array): numpy representation of audio, sr=22050, with percussive
                                   components of audio removed
    - onset_times (list of float): list of onset times corresponding to audio in seconds
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

def __onset_time_bins(onset_times):
    '''
    Converts the onset times (returned from __onset_time()) to 10ms time bins

    ~~~~ ARGUMENTS ~~~~
    - onset_times (list of float): list of onset times, returned from __onset_time()

    ~~~~ RETURNS ~~~~
    - list of ints: 10ms time bins corresponding to input
    '''
    otb = [int(x) for x in onset_times*100]
    return otb

def __create_notes_array(onsets, notes):
    '''
    Converts raw notes and onsets into a notes array, where the notes are placed at the onsets

    ~~~~ ARGUMENTS ~~~~
    - onsets (list):
        - contains onset times as 10ms bins, should have already been run through __onset_time_bins()
        - should be output of __onset_time_bins()
    
    - notes (list or 1D numpy array): notes corresponding to the onsets
    
    ~~~~ RETURNS ~~~~
    - 1D numpy array: notes array
    '''
    if type(notes) is np.ndarray:
        notes = notes.tolist()

    assert len(onsets) <= len(notes), 'ERROR: There are more onsets than notes'

    # Cut down notes if there are more notes than onsets
    if len(notes) > len(onsets):
        notes = notes[:len(onsets)]
    
    notes_array = np.zeros(onsets[-1])
    np.put(notes_array, onsets[:-1], notes)

    return notes_array
    
def __calc_note(idx, onset, curr_note, interval_length=100):
    '''
    Helper function for generate_notes()
    
    Calculates which note to play given the context of the current note, interval, and onset time
    '''
    # if short interval and current note is a single note
    n = random.random()
    
    # short and short = short - no change in state
    if (onset[idx] - onset[idx-1] < interval_length) & (onset[idx+1] - onset[idx] < interval_length): # changed conditional to or
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
    
    # short and long = short - no change
    elif (onset[idx] - onset[idx-1] < interval_length) & (onset[idx+1] - onset[idx] > interval_length):
        if n < (1/3): # note repeats
            curr_note = random.randint(6,31)
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
                
    # long and short = short - change 
    elif (onset[idx] - onset[idx-1] > interval_length) & (onset[idx+1] - onset[idx] < interval_length):
        curr_note = random.randint(1,5)
        
    # Long and Long = Long - no change
    else:
        curr_note = random.randint(6,31)
    return curr_note

def generate_notes(onset, interval_length=100):
    '''
    - Takes in onsets and generates notes based on the time interval difference
    - Shorter intervals will be a scale, longer intervals will be chords
    - Interval length: 100 equals 1 sec
    - Output is same length as onset and a numpy array
    
    ~~~~ ARGUMENTS ~~~~
    - onset (list of ints): onset times, should have already been run through __onset_time_bins()
    - interval_length (int): this interval controls the distance at which a single note vs chord is selected
    
    ~~~~ RETURNS ~~~~
    - list of ints: 
        - notes corresponding to each onset as defined in onset
        - this is NOT a notes array, it still needs to be processed afterward by __create_notes_array()
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
    
    
    for i in range(1,len(onset)-1): # since we'll be forward looking by one
        curr_note = __calc_note(i,onset, curr_note)
        note_array.append(curr_note)

    # Looks at last note and compares will repeat 
    if (note_array[-1] < 6) :
        curr_note = random.randint(1,5)
        note_array.append(curr_note)
    else:
        curr_note = random.randint(6,31)
        note_array.append(curr_note)
   
    return note_array

def generate_song(song_path, 
                  note_generation_function = generate_notes, 
                  onset_computation_function = __onset_time,
                  generation_function_uses_song_as_input = False, 
                  source_separated_path = None, 
                  outfile_song_name = 'Model 4', 
                  artist = 'Forrest',
                  outfolder = None,
                  original_song_path = None):
    '''
    Takes the song present at song_path, uses onset_computation_function to compute onsets, uses note_generation_function
    to generate notes, then writes the song to an outfolder at ~/Model_3/Generated Songs/<outfile_song_name>

    ~~~~ ARGUMENTS ~~~~
    - song_path : Path or str
        - path to the original song, i.e. song.ogg
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
    - outfolder : Path
        - Determines the folder where the generated song will be held
    - original_song_path : Path
        - if not none, will be used to save the original song rather than source separated song
          to outfolder
    '''
    if source_separated_path is not None:
        path = source_separated_path
    else:
        path = song_path

    print('Computing onsets...')
    _, onset_times = onset_computation_function(str(path))
    onset_indices = __onset_time_bins(onset_times)

    print('Generating Notes...\n')
    if generation_function_uses_song_as_input:
        dense_notes = note_generation_function(onset_indices[:-1], path)
    else:
        dense_notes = note_generation_function(onset_indices[:-1])
    
    notes_array = __create_notes_array(onset_indices[:len(dense_notes)], dense_notes)

    song_metadata = {'Name' : outfile_song_name,
                    'Artist' : artist,
                    'Charter' : 'tensorhero',
                    'Offset' : 0,
                    'Resolution' : 192,
                    'Genre' : 'electronic',
                    'MediaType' : 'cd',
                    'MusicStream' : 'song.ogg'}

    write_song_from_notes_array(song_metadata, notes_array, outfolder)
    if original_song_path is not None:
        shutil.copyfile(str(original_song_path), str(outfolder / 'song.ogg'))
    else:
        shutil.copyfile(str(path), str(outfolder / 'song.ogg'))
    return notes_array
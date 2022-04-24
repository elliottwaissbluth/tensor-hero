from pathlib import Path
import os
import sys
import shutil
import traceback
from tqdm import tqdm
import numpy as np
import math
from tensor_hero.preprocessing.chart import chart2tensor
from tensor_hero.preprocessing.audio import compute_mel_spectrogram
from tensor_hero.preprocessing.resources.resources import simplified_note_dict
__release_keys = list(np.arange(187, 218))
__release_keys.append(224)

'''
Contains functions related to...
- Audio and chart preprocessing
- File navigation
- Note simplification
'''

def __check_multiple_audio_files(file_list):
    '''
    Checks if there are separate audio files for single song. This is useful because some
    songs in the training data come source separated and have multiple audio tracks like 
    "guitar.ogg", "bass.ogg", etc.

    ~~~~ ARGUMENTS ~~~~
    - file_list (list of str): list of files in song directory
        - should be generated using os.listdir(<path to song folder>)

    ~~~~ RETURNS ~~~~
    - bool: True if there are multiple audio files in file_list, False otherwise
    '''

    # Get audio file list
    num_files = 0
    multiple_files = False
    for track in file_list:
        if track.endswith('.ogg'):
            num_files += 1
        if track.endswith('.mp3'):
            num_files += 1
        if track.endswith('.wav'):
            num_files += 1
    if num_files > 1:
        multiple_files = True
    return multiple_files

def __get_audio_file_name(file_list):
    '''
    Gets name of audio file from list of files in song folder

    ~~~~ ARGUMENTS ~~~~
    - file_list (list of str): list of files in song directory path`
        - should be generated using os.listdir(<path to song folder>)
    
    ~~~~ RETURNS ~~~~
    - str: name of audio file, if present
        - does not include full path, only stem
    '''
    for track in file_list:
        if track.endswith('.ogg') or track.endswith('.mp3') or track.endswith('.wav'):
            return track
    raise NameError('Error: audio file not present')

def __match_tensor_lengths(notes_array, song):
    '''
    Each notes_array has some space at the end where the song is still playing but there is no
    note data. This function appends zeros to the notes_tensor so that it matches song length.

    ~~~~ ARGUMENTS ~~~~
    - notes_array (1D numpy array): notes_array
    - song (2D numpy array): spectrogram of song

    ~~~~ RETURNS ~~~~
    - 1D numpy array: notes array matched to length of song
    '''
    
    length = song.shape[1]
    notes = np.zeros(length)
    notes[0:(notes_array.shape[0])] = notes_array

    return notes

def __check_for_sub_packs(unprocessed_path):
    '''
    Checks whether the track packs in unprocessed_path link directly to songs or if they are organized
    into sub-packs. Returns list of paths of track packs with sub-packs
    
    ~~~~ ARGUMENTS ~~~~
    - unprocessed_path (Path): Path to unprocessed training data
    
    ~~~~ RETURNS ~~~~
    - list of Paths: Paths to track pack directories that contain sub-packs
    '''
    sub_packs = []
    for track_pack in os.listdir(unprocessed_path):
        if track_pack in ['Thumbs.db', '.DS_Store']:
            continue
        for sub_pack in os.listdir(unprocessed_path / track_pack):
            if sub_pack in ['Thumbs.db', '.DS_Store']:
                continue
            for song in os.listdir(unprocessed_path / track_pack / sub_pack):
                if song in ['Thumbs.db', '.DS_Store']:
                    continue
                if os.path.isdir(unprocessed_path / track_pack / sub_pack / song):
                    sub_packs.append(unprocessed_path / track_pack)
                    break
                break
            break
    return sub_packs

def __pop_sub_packs(sub_packs):
    '''
    Takes the songs within the sub-packs and copies them outside their sub-pack directory to the track pack directory
    
    ~~~~ ARGUMENTS ~~~~
    - sub_packs (list of Paths): Path to track pack directories that contain sub-packs
                                 Should be the output of __check_for_sub_packs
    '''
    # Parse each sub-pack and move the songs to the track pack directory, then delete the sub-pack
    for track_pack in sub_packs:
        for sub_pack in [track_pack / x for x in os.listdir(track_pack) if x not in ['Thumbs.db', '.DS_Store']]:
            for song in [track_pack / sub_pack / y for y in os.listdir(track_pack / sub_pack) if y not in ['Thumbs.db', '.DS_Store']]:
                # Move song directory outside sub_pack
                if os.path.exists(track_pack / song.stem):
                    shutil.rmtree(track_pack / song.stem)
                shutil.move(song, track_pack / song.stem)
            # Delete sub_pack folder
            shutil.rmtree(sub_pack)

def populate_processed_folder(unprocessed_data_path, processed_data_path, REPLACE_NOTES = False,
                              PRINT_TRACEBACK=True, SUB_PACKS=False, TRACK_PACKS=True):
    '''
    Converts raw downloaded Clone Hero data into processed spectrograms and notes arrays.
    Populates processed_data_path by processing data in unprocessed_data_path.
    
    ~~~~ ARGUMENTS ~~~~
    - unprocessed_data_path (Path): directory containing track packs of downloaded Clone Hero songs
    - processed_data_path (Path): empty directory where processed data should be placed
    - REPLACE_NOTES (bool): if True, will replace any existing notes_arrays and spectrograms in processed
    - PRINT_TRACEBACK (bool): if True, will print full traceback when excepted errors are thrown
    - SUB_PACKS (bool): if True, will check for sub-packs within the parent unprocessed_data_path
    - TRACK_PACKS (bool): if True, file structure expects the unprocessed folder contains track packs
    
    ~~~~ RETURNS ~~~~
    - dict: metadata about processed data
        - 'wrong_format_charts' (list of Path): list of unprocessed song paths with .chart files in wrong format
        - 'multiple_audio_songs' (list of Path): list of unprocessed song paths with multiple audio files present
        - 'processed' (list of Path): list of successfully processed song paths
        - 'song_size' (float): spectrogram data size in GB
        - 'notes_size' (float): notes arrays data size in GB
    '''
    # Extract songs from sub-packs, if the track pack includes sub-packs
    if SUB_PACKS:
        sub_packs = __check_for_sub_packs(unprocessed_data_path)
        __pop_sub_packs(sub_packs)

    wrong_format_charts = []    # Holds paths to charts not in .chart format
    multiple_audio_songs = []   # Holds paths to charts with multiple audio files
    processed = []              # Holds paths to song folders that were successfully processed
    song_size = 0               # Total audio data size, in gigabytes
    notes_size = 0              # Total note data size, in gigabytes
   
    for dirName, _, fileList in tqdm(os.walk(unprocessed_data_path)):  # Walk through training data directory
        if not fileList or fileList in [['.DS_Store'], ['Thumbs.db']]:
            track_pack_ = Path(dirName).parent.stem
            continue

        track_pack_ = Path(dirName).parent.stem     # track pack name
        song_ = Path(dirName).stem                  # song name
        
        if TRACK_PACKS:
            processed_path = processed_data_path / track_pack_ / song_          # processed song folder
            unprocessed_path = unprocessed_data_path / track_pack_ / song_      # unprocessed song folder
        else:
            processed_path = processed_data_path / track_pack_ / song_          # processed song folder
            unprocessed_path = unprocessed_data_path / song_      # unprocessed song folder
        processed_song_path = processed_path / 'spectrogram.npy'            # spectrogram
        processed_notes_path = processed_path / 'notes.npy'                 # notes array

        audio_file_name = __get_audio_file_name(fileList)
        unprocessed_song_path = unprocessed_path / audio_file_name
        
        if REPLACE_NOTES:
            if processed_notes_path.exists():
                os.remove(processed_notes_path)  # Delete because I accidentally saved the same array hundreds of times lol

        # Skip creating the directory if there is more than one audio file
        # (some songs are divided into guitar.ogg, drums.ogg, etc.)
        if __check_multiple_audio_files(fileList):
            multiple_audio_songs.append(unprocessed_song_path)
            if processed_path.exists():
                if len(os.listdir(processed_path)) == 0: # If the folder exists but is empty
                    os.rmdir(processed_path)
            continue

        # Make folder for directory in 'Processed' if it doesn't already exist
        if not processed_path.parent.exists():
            os.mkdir(processed_path.parent)
        if not processed_path.exists():
            os.mkdir(processed_path)

        # Create note array for song
        try:
            notes_array = np.array(chart2tensor(unprocessed_path / 'notes.chart', print_release_notes = False)).astype(int)
        except TypeError as err:
            print('{}, {} .chart file is in the wrong format, skipping'.format(track_pack_, song_))
            if PRINT_TRACEBACK:
                print("Type Error: {0}".format(err))
                print(traceback.format_exc()) 
            wrong_format_charts.append(unprocessed_song_path)
            if processed_path.exists():
                # if len(os.listdir(processed_path)) == 0: # If the folder exists but is empty
                os.rmdir(processed_path)
            continue
        except:
            print('{}, {} .chart file is in the wrong format, skipping'.format(track_pack_, song_))
            if PRINT_TRACEBACK:
                print('Unknown Error: {0}'.format(sys.exc_info()[0]))
                print(traceback.format_exc())
            wrong_format_charts.append(unprocessed_song_path)
            if processed_path.exists():
                # if len(os.listdir(processed_path)) == 0: # If the folder exists but is empty
                os.rmdir(processed_path)
            continue
        
        # Check if song has already been processed
        # If it has, load it, because the notes array will be matched to its length
        if processed_song_path.exists():
            print('{} audio has already been processed'.format(processed_song_path.stem))
            song = np.load(processed_song_path)
        else:
            try:
                song = compute_mel_spectrogram(unprocessed_song_path)
                np.save(processed_song_path, song)
            except:
                print('{}, {} .chart file is in the wrong format, skipping'.format(track_pack_, song_))
                if PRINT_TRACEBACK:
                    print(traceback.format_exc()) 
                wrong_format_charts.append(unprocessed_song_path)
                if processed_path.exists():
                    # if len(os.listdir(processed_path)) == 0: # If the folder exists but is empty
                    os.rmdir(processed_path)
                continue
        # Check if notes have already been processed
        if processed_notes_path.exists():
            pass
        else:
            try:
                notes_array = __match_tensor_lengths(notes_array, song)
                np.save(processed_notes_path, notes_array)
            except ValueError as err:
                print('Value Error: {0}'.format(err))
                print('notes_array shape:', notes_array.shape)
                print('spectrogram shape:', song.shape)
                continue
            

        song_size += (processed_song_path.stat().st_size) / 1e9
        notes_size += (processed_notes_path.stat().st_size) / 1e9

    # create dict of data information
    data_info = {'wrong_format_charts': wrong_format_charts,
                 'multiple_audio_songs': multiple_audio_songs,
                 'processed': processed,
                 'song_size': song_size,
                 'notes_size': notes_size}

    return data_info

def get_list_of_ogg_files(unprocessed_path, prefix = None):
    '''
    Takes the root directory (unprocessed_path) and returns a list of the full file paths
    to all the .ogg files in that directory, provided there is only one audio file per folder (i.e. it
    skips folders that have source separated music files). Also returns processed_paths, the "processed"
    directory analog to each item in ogg_file_paths (just a path to the folder, not the files inside).
    
    A vast majority of the training data is .ogg, in the future it might be useful to expand this to
    .wav and .mp3 as well.

    ~~~~ ARGUMENTS ~~~~
    -   unprocessed_path (Path): path object or string to root unprocessed folder. probably ./Training Data/Unprocessed
    -   prefix (str or None): if string, will look for a specific prefix on the .ogg (e.g. "separated.ogg")

    ~~~~ RETURNS ~~~~
    -   ogg_file_paths (list of Path): list of paths to all .ogg files in unprocessed_path
    -   processed_paths (list of Path): list of paths to the processed folders corresponding to every item
                                        in ogg_file_paths.  
    '''
    ogg_file_paths = []
    processed_paths = []
    for track_pack in [unprocessed_path / x for x in os.listdir(unprocessed_path)]:
        # Ignore DS_Store if running on mac
        if 'DS_Store' in str(track_pack):
            continue
        for song_dir in [track_pack / y for y in os.listdir(track_pack)]:
            if 'DS_Store' in str(song_dir):
                continue
            if prefix is None:
                if __check_multiple_audio_files(os.listdir(song_dir)): # Check for multiple audio files
                    continue
            else:
                for f in os.listdir(song_dir):
                    if f.endswith('.ogg'):
                        if prefix is not None:
                            if f.startswith(prefix):
                                ogg_file_paths.append(song_dir / f)
                                processed_paths.append(Path(str(song_dir).replace('Unprocessed', 'Processed', 1)))
                        else:
                            ogg_file_paths.append(song_dir / f)
                            processed_paths.append(Path(str(song_dir).replace('Unprocessed', 'Processed', 1)))
                        
    return ogg_file_paths, processed_paths

def populate_processed_folder_with_spectrograms(unprocessed_path, spectrogram_name = 'spectrogram.npy', REPLACE=True):
    '''
    Takes all the .ogg files in unprocessed_path, computes their spectrogram, then saves that spectrogram to the 
    processed_path analog. This function assumes that the processed folder structure already exists, i.e. 
    populate_processed_folder() has already been run.

    Note that processed_path is assumed by taking processed_path and replacing "Unprocessed" with "Processed"

    ~~~~ ARGUMENTS ~~~~
    - unprocessed_path (path): path to root unprocessed folder (probably ./Training Data/Unprocessed)
    - spectrogram_name (str): name to save each spectrogram file under. This name will be checked if REPLACE=False
    - REPLACE (bool): if True, will replace any existing spectrogram.npy files with newly computed spectrograms, 
                      else,  it does not replace.
    '''
    ogg_file_paths, processed_paths = get_list_of_ogg_files(unprocessed_path)
    APPEND_ARR_CREATED = False
    for i in tqdm(range(len(ogg_file_paths))):

        if not os.path.exists(processed_paths[i]): # if the processed folder doesn't exist
            continue
        if not REPLACE:     # if processed folder has already been populated
            if os.path.exists(processed_paths[i] / spectrogram_name):
                continue

        spec = compute_mel_spectrogram(ogg_file_paths[i])  # Get spectrogram
        
        # ?silence?
        # # NOTE: Maybe get rid of this, but make sure it doesn't break anything down the pipeline
        # # Append 70ms of silence at beginnign and end
        # if not APPEND_ARR_CREATED:  # Only define once
            # append_arr = np.ones((spec.shape[0],7)) * np.min(spec)
            # APPEND_ARR_CREATED = True
        # spec = np.c_[append_arr, spec, append_arr]

        # Save to appropriate processed folder
        np.save(str(processed_paths[i] / spectrogram_name), spec)  # NOTE: Change to something like 'source_separated_spec.npy'

    return

# ---------------------------------------------------------------------------- #
#                              NOTES SIMPLIFICATION                            #
# ---------------------------------------------------------------------------- #

def __remove_release_keys(notes_array):
    '''
    Removes all notes corresponding to releases from notes_array
   
   ~~~~ ARGUMENTS ~~~~
   - notes_array (1D numpy array): notes array including release keys
   
   ~~~~ RETURNS ~~~~
   - 1D numpy array: new notes array with release keys removed (converted to zeros)
    '''
    new_notes = np.zeros(notes_array.size)
    changed_notes = []
    for x in range(notes_array.size):
        if notes_array[x] not in __release_keys:
            new_notes[x] = notes_array[x]
        else:
            changed_notes.append(x)
    return new_notes

def __remove_modifiers(notes_array):
    '''
    Maps modified and held notes to their regular counterparts
    '''
    new_notes = np.zeros(notes_array.size)
    for x in range(notes_array.size):
        if notes_array[x]:
            new_notes[x] = simplified_note_dict[notes_array[x]]
    return new_notes

def populate_with_simplified_notes(processed_directory, SUB_PACKS=False):
    '''
    Takes all the processed notes arrays (notes.npy) in processed_directory and adds simplified versions 
    (notes_simplified.npy) to the same subdirectories.
    
    Simplified notes are free from modifiers and held notes.
    
    ~~~~ ARGUMENTS ~~~~
    - processed_directory (Path): Path to processed training data directory (probably ./Training Data/Processed)
    '''
    
    # Parse through directory
    for x in tqdm(os.listdir(processed_directory)):
        # Check to see if the .DS_Store file is in the directory, skip if so
        if '.DS_Store' in x:
            continue
        if SUB_PACKS:
            for y in os.listdir(processed_directory / x):
                # Check to see if the simplified file already exists, continue if so
                if os.path.isfile(processed_directory / x / y / 'notes_simplified.npy'):
                    continue

                try:
                    notes_array = np.load(processed_directory / x / y / 'notes.npy')
                except FileNotFoundError:
                    print('Error: There is no notes.npy file at {}'.format(str(processed_directory / x / y)))
                    print('Continuing')
                    continue
                except NotADirectoryError as err:
                    print('ERROR: {}'.format(err))
                    if str(y) == '.DS_Store':
                        print('DS_Store error, continuing')
                        continue
                    else:
                        break

                new_notes = __remove_release_keys(notes_array)
                new_notes = __remove_modifiers(new_notes)

                # Save to same directory
                with open(str(processed_directory / x / y / 'notes_simplified.npy'), 'wb') as f:
                    np.save(f, new_notes)
                f.close()
        else:
            
            # Check to see if the simplified file already exists, continue if so
            if os.path.isfile(processed_directory / x / 'notes_simplified.npy'):
                continue

            try:
                notes_array = np.load(processed_directory / x / 'notes.npy')
            except FileNotFoundError:
                print('Error: There is no notes.npy file at {}'.format(str(processed_directory / x)))
                print('Continuing')
                continue
            except NotADirectoryError as err:
                print('ERROR: {}'.format(err))
                if str(y) == '.DS_Store':
                    print('DS_Store error, continuing')
                    continue
                else:
                    break

            new_notes = __remove_release_keys(notes_array)
            new_notes = __remove_modifiers(new_notes)

            # Save to same directory
            with open(str(processed_directory / x / 'notes_simplified.npy'), 'wb') as f:
                np.save(f, new_notes)
            f.close()


# ---------------------------------------------------------------------------- #
#                        TRANSFORMER DATA PREPROCESSING                        #
# ---------------------------------------------------------------------------- #


def __process_spectrogram(spec):
    '''
    Normalizes spectrogram in [0,1]
    
    ~~~~ ARGUMENTS ~~~~
    - spec (2D numpy array):padded spectrogram, loaded from spectrogram.npy in processed folder
    
    ~~~~ RETURNS ~~~~
    - 2D numpy array : Normalized spectrogram
    '''
    spec = (spec+80) / 80   # Regularize
    return spec

def __notes_to_output_space(notes):
    '''
    Takes a notes array as input, and outputs a matrix of numpy arrays in the output format specified
    by sequence to sequence piano transcription.
    
    ~~~~ ARGUMENTS ~~~~
    - notes (1D numpy array) : notes array
    
    ~~~~ RETURNS ~~~~
    - 2D numpy array:
        - Y axis is sequential note events, with each new row being a time event, then note event
        - X axis is one hot encoded arrays, where the index will be fed into the transformer as output
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
            formatted[2*i, time_pos+32] = 1  # One hot encode the time step
            formatted[2*i+1, int(x)-1] = 1   # One hot encode the note
            i += 1
            
    return formatted

def __formatted_notes_to_indices(notes):
    '''
    Takes formatted notes and returns a 1D array of indices, reverse one hot operation.
    Helper function for __prepare_notes_tensor()
    ~~~~ ARGUMENTS ~~~~
    notes (1D numpy array): formatted notes, as output by __notes_to_output_space()
    
    ~~~~ RETURNS ~~~~
    indices (1D numpy array): 
        - de-one hot encoded indices of note event series.
        - the format is [time, note, time, note, etc...] 
    '''
    indices = np.argwhere(notes == 1)
    indices = indices[:,-1].flatten()
    return indices

def __prepare_notes_tensor(notes):
    '''
    Takes formatted notes and converts them to the format suitable for PyTorch's transformer model.
    Helper function for populate_model_1_training_data()
    
    ~~~~ ARGUMENTS ~~~~
    - notes (1D numpy array): formatted notes, as output by __notes_to_output_space()
    
    ~~~~ RETURNS ~~~~
    - 1D numpy array : notes with format [<sos>, time, note, time, note, etc..., <eos>]
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
    notes = __formatted_notes_to_indices(notes)
    return notes

def preprocess_transformer_data(segment_length, training_data_path, train_val_test_probs, model_training_directory_name, COLAB=False, SEPARATED=True):
    '''
    Takes a directory of processed song level training data and slices each song into 4 second segments.
    Processed folder should already exist with simplified notes
    Splits train, test, and validation at the song level.

    If COLAB=False, file structure looks like

    Training_Data
    |----<model_training_directory_name>
        |----train
        |   |----<song name 1>
        |   |   |----notes
        |   |   |   |----1.npy
        |   |   |   |----2.npy (etc)
        |   |   |----spectrograms
        |   |   |   |----1.npy
        |   |   |   |----2.npy (etc)
        |   |----<song name 2> (etc)
        |----test
        |----val
    
    If COLAB=True, file structure looks like

    Training_Data
    |----<model_training_directory_name>
        |----train
        |   |----<song name 1>
        |   |   |----notes
        |   |   |   |----1
        |   |   |   |   |----1.npy
        |   |   |   |   |----2.npy (etc)
        |   |   |   |----2 (etc)
        |   |   |----spectrograms
        |   |   |   |----1
        |   |   |   |   |----1.npy
        |   |   |   |   |----2.npy (etc)
        |   |   |   |----2 (etc)
        |   |----<song name 2> (etc)
        |----test
        |----val
    
    ~~~~ ARGUMENTS ~~~~
    - segment_length (int): 
        - The number of elements in the time dimension of the notes arrays and spectrograms (typically 400)
        - Each element corresponds to 10ms of time
    - training_data_path (Path): Path to top level training data directory (probably tensor-hero/Training_Data)
    - train_val_test_probs (3 element list):
        - Determines how the dataset is split
        - [Probability of song being placed in train, val, test]
        - Must sum to 1
    - model_training_directory_name (str):
        - Within training_data_path, a folder of this name will be created to hold the processed transformer training data
    - COLAB (bool): 
        - If True, segments training data so folders do not exceed 40 total files
        - This is necessary because COLAB does not work well with directories with large numbers of files
    - SEPARATED (bool):
        - If True, will seek out separated spectrograms (spectrogram_separated.npy) instead of unseparated ones
    '''
    assert sum(train_val_test_probs) == 1, 'ERROR: train_val_test_probs does not sum to 1'
    
    unprocessed_path = training_data_path / 'Unprocessed'
    train_path = training_data_path / model_training_directory_name / 'train'
    val_path = training_data_path / model_training_directory_name / 'val'
    test_path = training_data_path / model_training_directory_name / 'test'

    # Make directories if they don't exist
    if not os.path.isdir(training_data_path / model_training_directory_name):
        os.mkdir(training_data_path / model_training_directory_name)
        print(f'made directory {str(training_data_path / model_training_directory_name)}')
    if not os.path.isdir(train_path):
        os.mkdir(train_path)
        print(f'made directory {str(train_path)}')
    if not os.path.isdir(val_path):
        os.mkdir(val_path)
        print(f'made directory {str(val_path)}')
    if not os.path.isdir(test_path):
        os.mkdir(test_path)
        print(f'made directory {str(test_path)}')
        
    if not SEPARATED:
        # _, processed_list = get_list_of_ogg_files(unprocessed_path)
        processed_path = training_data_path
        processed_list = []
        for x in [processed_path / x for x in os.listdir(processed_path)]:
            if os.path.exists(x / 'spectrogram.npy') and os.path.exists(x / 'notes_simplified.npy'):
                processed_list.append(x)
    else:
        _, processed_list = get_list_of_ogg_files(unprocessed_path, prefix='separated')
    
    # Get paths of notes and corresponding paths of spectrograms
    if not SEPARATED:
        spec_paths = [song / 'spectrogram.npy' for song in processed_list]
    else:
        spec_paths = [song / 'spectrogram_separated.npy' for song in processed_list]
    notes_paths = [song / 'notes_simplified.npy' for song in processed_list]

    for i in tqdm(range(len(processed_list))):
        # Process spectrogram
        try:
            spec = np.load(spec_paths[i])
        except FileNotFoundError:
            print('There is no spectrogram at {}'.format(spec_paths[i]))
            continue
        except ValueError as err:
            print(err)
            print(traceback.format_exc())
            continue
        spec = __process_spectrogram(spec)

        # Process notes
        try:
            notes = np.load(notes_paths[i])
        except FileNotFoundError:
            print('There is no notes_simplified at {}'.format(notes_paths[i]))
            continue
        
        # print(f'notes shape: {notes.shape}')
        # print(f'spec shape: {spec.shape}')
        
        # if notes.shape[0] != spec.shape[1]:  #TODO: FIXME
            # continue
        assert notes.shape[0] == spec.shape[1], 'ERROR: Spectrogram and notes shape do not match'
        
        # Get number of segment_length second slices
        num_slices = math.floor(spec.shape[1]/segment_length)
        
        # Split notes and spectrogram into bins
        spec_bins = np.array([spec[:,j*segment_length:(j+1)*segment_length] for j in range(num_slices)])
        notes_bins = np.array([notes[j*segment_length:(j+1)*segment_length] for j in range(num_slices)])
        
        # This list will hold the final note representations ready for the transformer
        final_notes = []
        for j in range(num_slices):
            t_notes = __notes_to_output_space(notes_bins[j,:])
            t_notes = __prepare_notes_tensor(t_notes)
            final_notes.append(t_notes)
        
        # Randomly select whether it goes in train, val, or test based on the desired split
        train_val_test_selection = np.random.choice(3, 1, p=train_val_test_probs)[0]
        if train_val_test_selection == 0:
            prepend_path = train_path
        elif train_val_test_selection == 1:
            prepend_path = val_path
        else:
            prepend_path = test_path

        # Create a folder for the outfile
        if not os.path.isdir(prepend_path / processed_list[i].stem): # Makes a folder with song name in correct subdirectory
            os.mkdir(prepend_path / processed_list[i].stem)

        if not COLAB:
            for j in range(len(final_notes)):
                spec_outfile = prepend_path / processed_list[i].stem / 'spectrograms' / (str(j) + '.npy')
                if not os.path.isdir(spec_outfile.parent):
                    os.mkdir(spec_outfile.parent)

                notes_outfile = prepend_path / processed_list[i].stem / 'notes' / (str(j) + '.npy') 
                if not os.path.isdir(notes_outfile.parent):
                    os.mkdir(notes_outfile.parent)

                np.save(spec_outfile, spec_bins[j,...].astype('float16')) # change to float16 to reduce hard disk memory
                np.save(notes_outfile, final_notes[j])
        
        else:
            for j in range(len(final_notes)):
                spec_outfile = prepend_path / processed_list[i].stem / 'spectrograms' / str(math.floor(j/40)) / (str(j) + '.npy')
                if not os.path.isdir(spec_outfile.parent.parent):
                    try:
                        os.mkdir(spec_outfile.parent.parent)
                    except:
                        print(f'Can\'t locate directory {spec_outfile.parent}, continuing...')
                        break
                if not os.path.isdir(spec_outfile.parent):
                    os.mkdir(spec_outfile.parent)
                
                notes_outfile = prepend_path / processed_list[i].stem / 'notes' / str(math.floor(j/40)) / (str(j) + '.npy') 
                if not os.path.isdir(notes_outfile.parent.parent):
                    os.mkdir(notes_outfile.parent.parent)
                if not os.path.isdir(notes_outfile.parent):
                    os.mkdir(notes_outfile.parent)

                np.save(spec_outfile, spec_bins[j,...]) # change to float16 to reduce hard disk memory
                np.save(notes_outfile, final_notes[j])

    return


# ~~~~~~~~~~~~~~~ NOTES CONTOUR ~~~~~~~~~~~~~~~~~~ #
    
# Describes an easier way to index specific notes for the purpose of note category grouping
# keys = more organized grouping for notes
# values = original simplified note array representation
note_category_grouping = {
    1 : 1,
    2 : 2,
    3 : 3,
    4 : 4,
    5 : 5,
    6 : 6,
    7 : 10,
    8 : 13,
    9 : 15,
    10 : 7,
    11 : 11,
    12 : 14,
    13 : 8,
    14 : 12,
    15 : 9,
    16 : 16,
    17 : 22,
    18 : 25,
    19 : 17,
    20 : 23,
    21 : 19,
    22 : 24,
    23 : 18,
    24 : 20,
    25 : 21,
    26 : 26,
    27 : 30,
    28 : 27,
    29 : 29,
    30 : 28,
    31 : 31,
    32 : 218
}
notes_to_note_category_grouping = dict([(v,k) for k,v in note_category_grouping.items()])

# keys = note category grouping organized notes
# values = note category encoding
notes_to_note_category = {
    1 : 1, # s
    2 : 1,
    3 : 1,
    4 : 1,
    5 : 1,
    6 : 2, # d0
    7 : 2,
    8 : 2,
    9 : 2,
    10 : 3, # d1
    11 : 3,
    12 : 3,
    13 : 4, # d2
    14 : 4,
    15 : 5, # d3
    16 : 6, # t0
    17 : 6,
    18 : 6,
    19 : 7, # t1
    20 : 7,
    21 : 8, # t2
    22 : 8,
    23 : 9, # t3
    24 : 9,
    25 : 9,
    26 : 10, # q0
    27 : 10,
    28 : 11, # q1
    29 : 11,
    30 : 11,
    31 : 12, # p
    32 : 13  # o
}

cardinality_of_note_categories = {
    1 : 5,
    2 : 4,
    3 : 3,
    4 : 2,
    5 : 1,
    6 : 3,
    7 : 2,
    8 : 2,
    9 : 3,
    10 : 2,
    11 : 3,
    12 : 1,
    13 : 1
}

# We use note category grouping of note to describe it
note_category_to_note = {
    1 : [1, 2, 3, 4, 5],
    2 : [6, 7, 8, 9],
    3 : [10, 11, 12],
    4 : [13, 14],
    5 : [15],
    6 : [16, 17, 18],
    7 : [19, 20],
    8 : [21, 22],
    9 : [23, 24, 25],
    10 : [26, 27],
    11 : [28, 29, 30],
    12 : [31],
    13 : [32]
}

# The note category encoded notes and their corresponding anchors
notes_to_anchor = {
    1 : 0,  # s[0]
    2 : 1,  # s[1]
    3 : 2,
    4 : 3,
    5 : 4,
    6 : 0,  # d0[0]
    7 : 1,  # d0[1]
    8 : 2,
    9 : 3,
    10 : 0,
    11 : 1,
    12 : 2,
    13 : 0,
    14 : 1,
    15 : 0,
    16 : 0,
    17 : 1,
    18 : 2,
    19 : 0,
    20 : 1,
    21 : 0,
    22 : 1,
    23 : 0,
    24 : 1,
    25 : 2,
    26 : 0,
    27 : 1,
    28 : 0,
    29 : 1,
    30 : 2,
    31 : 0,
    32 : 0,
}

# Includes open, pent, quad, and d3 notes
back_to_prev_anchor_notes = [15, *list(range(26,33))]

def encode_contour(notes_array):
    '''
    Takes a notes array and encodes the contour as described in ../Documentation/contour.md

    ~~~~ ARGUMENTS ~~~~
    - notes_array (1D numpy array): Simplified notes array
        - shape = (1, length of song in 10ms bines)
        
    ~~~~ RETURNS ~~~~
    - contour (2D numpy array): Encoded notes_array
        - shape = (2, length of song in 10ms bines)
        - contour[0,:] contains a key for the note category, C
        - contour[1,:] contains the corresponding motions at each onset
        - Time bins without note events are filled with 0s
    '''
    contour = np.zeros(shape=(2, notes_array.shape[0]))
    note_indices = np.where(notes_array > 0)[0]
    
    prev_anchor = 0   # We set the initial anchor at the green note, a=0
    for note_idx in note_indices:
        # Populate contour with note categories
        contour[0, note_idx] = notes_to_note_category[notes_to_note_category_grouping[int(notes_array[note_idx])]]

        # Populate contour with relative motion
        if notes_to_note_category_grouping[int(notes_array[note_idx])] in back_to_prev_anchor_notes:
            contour[1, note_idx] = 0
        else:
            new_anchor = notes_to_anchor[notes_to_note_category_grouping[int(notes_array[note_idx])]]
            contour[1, note_idx] = new_anchor - prev_anchor
            prev_anchor = new_anchor

    return contour


def decode_contour(contour):
    '''
    Takes an encoded contour array and decodes it into a simplified notes array
    
    ~~~~ ARGUMENTS ~~~~
    - contour (2D numpy array): Encoded notes_array
        - shape = (2, length of song in 10ms bins)
        - contour[0,:] contains a key for the note category, C
        - contour[1,:] contains the corresponding motions at each onset
        - Time bins without note events are filled with 0s
    
    ~~~~ RETURNS ~~~~
    - notes_array (1D numpy array): Decoded simplified notes_array
        - shape = (1, length of song in 10ms bins)
    '''
    # print(f'contour shape: {contour.shape}')
    notes_array = np.zeros(shape=(contour.shape[1]))
    
    # Loop through note categories in contour, adjusting anchor along the way
    note_indices = np.where(contour[0] > 0)[0].astype(int)
    anchor = 0  # Initialize anchor
    for note_idx in note_indices:
        if not int(contour[0,note_idx]) in [5, 10, 11, 12, 13]:  # If not d3, quad, pent, or open
            anchor = anchor+int(contour[1,note_idx])
            if anchor > cardinality_of_note_categories[int(contour[0,note_idx])]-1:  # Wrap around
                anchor = 0
            elif anchor < 0:
                anchor = cardinality_of_note_categories[int(contour[0,note_idx])]-1
            anchor = min(anchor, cardinality_of_note_categories[int(contour[0,note_idx])]-1)
            notes_array[note_idx] = note_category_grouping[note_category_to_note[int(contour[0,note_idx])][anchor]]
        else:
            # choose at random for these note categories
            temp_anchor = np.random.randint(0, cardinality_of_note_categories[int(contour[0,note_idx])])  
            notes_array[note_idx] = note_category_grouping[note_category_to_note[int(contour[0,note_idx])][temp_anchor]]

    return notes_array

def notes_array_time_adjust(notes_array, time_bins_per_second, reverse=False):
    '''
    Takes a notes array with 100 ticks per second and adjusts it to have time_bins_per_second
    time bins per second. Will snap to the nearest integer during division.

    ~~~~ ARGUMENTS ~~~~
    - notes_array (1D numpy array): notes_array with 100 time bins per second
    - time_bins_per_second (int): desired number of time bins per second 
    - reverse (bool): if True, will expand from time_bins_per_second to 100 time bins per second

    ~~~~ RETURNS ~~~~
    - notes_array_reduced (1D numpy array): note_array with time_bins_per_second time bins per second
    - reduction_factor (float): the factor the time bins were reduced by
    '''
    if not reverse:
        reduction_factor = 100/time_bins_per_second
    else:
        reduction_factor = time_bins_per_second/100
    onset_indices = np.where(notes_array > 0)[0]
    notes_array_reduced = np.zeros(shape=(math.ceil(notes_array.shape[0]/reduction_factor)+1))
   
    # Maps original onset times to reduced onset times 
    onset_mapping = dict([(x, round(x/reduction_factor)) for x in onset_indices])

    # Populate notes_array_reduced with notes
    for onset, reduced_onset in onset_mapping.items():
       notes_array_reduced[reduced_onset] = notes_array[onset] 

    return notes_array_reduced, reduction_factor
from pathlib import Path
import os
import sys
import shutil
import traceback
from tqdm import tqdm
import numpy as np
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

def populate_processed_folder(unprocessed_data_path, processed_data_path, REPLACE_NOTES = False):
    '''
    Converts raw downloaded Clone Hero data into processed spectrograms and notes arrays.
    Populates processed_data_path by processing data in unprocessed_data_path.
    
    ~~~~ ARGUMENTS ~~~~
    - unprocessed_data_path (Path): directory containing track packs of downloaded Clone Hero songs
    - processed_data_path (Path): empty directory where processed data should be placed
    - REPLACE_NOTES (bool): if True, will replace any existing notes_arrays and spectrograms in processed
    
    ~~~~ RETURNS ~~~~
    - dict: metadata about processed data
        - 'wrong_format_charts' (list of Path): list of unprocessed song paths with .chart files in wrong format
        - 'multiple_audio_songs' (list of Path): list of unprocessed song paths with multiple audio files present
        - 'processed' (list of Path): list of successfully processed song paths
        - 'song_size' (float): spectrogram data size in GB
        - 'notes_size' (float): notes arrays data size in GB
    '''
    # Extract songs from sub-packs, if the track pack includes sub-packs
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
        
        processed_path = processed_data_path / track_pack_ / song_          # processed song folder
        unprocessed_path = unprocessed_data_path / track_pack_ / song_      # unprocessed song folder
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
            print("Type Error: {0}".format(err))
            print(traceback.format_exc()) 
            wrong_format_charts.append(unprocessed_song_path)
            if processed_path.exists():
                if len(os.listdir(processed_path)) == 0: # If the folder exists but is empty
                    os.rmdir(processed_path)
            continue
        except:
            print('{}, {} .chart file is in the wrong format, skipping'.format(track_pack_, song_))
            print('Unknown Error: {0}'.format(sys.exc_info()[0]))
            print(traceback.format_exc())
            wrong_format_charts.append(unprocessed_song_path)
            if processed_path.exists():
                if len(os.listdir(processed_path)) == 0: # If the folder exists but is empty
                    os.rmdir(processed_path)
            continue
        
        # Check if song has already been processed
        # If it has, load it, because the notes array will be matched to its length
        if processed_song_path.exists():
            print('{} audio has already been processed'.format(processed_song_path.stem))
            song = np.load(processed_song_path)
        else:
            song = compute_mel_spectrogram(unprocessed_song_path)
            np.save(processed_song_path, song)
        
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

def get_list_of_ogg_files(unprocessed_path):
    '''
    Takes the root directory (unprocessed_path) and returns a list of the full file paths
    to all the .ogg files in that directory, provided there is only one audio file per folder (i.e. it
    skips folders that have source separated music files). Also returns processed_paths, the "processed"
    directory analog to each item in ogg_file_paths (just a path to the folder, not the files inside).
    
    A vast majority of the training data is .ogg, in the future it might be useful to expand this to
    .wav and .mp3 as well.

    ~~~~ ARGUMENTS ~~~~
    -   unprocessed_path (Path): path object or string to root unprocessed folder. probably ./Training Data/Unprocessed

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
            if __check_multiple_audio_files(os.listdir(song_dir)): # Check for multiple audio files
                continue
            else:
                for f in os.listdir(song_dir):
                    if f.endswith('.ogg'):
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

def populate_with_simplified_notes(processed_directory):
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
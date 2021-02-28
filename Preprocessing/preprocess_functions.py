# Contains functions to preprocess notes and audio

from pathlib import Path
import sys
sys.path.insert(1, str(Path().resolve().parent) + r'\Preprocessing')
import os
from chart_functions import chart2tensor, get_configuration, chart2onehot
from audio_functions import music2tensor
from tqdm import tqdm
import numpy as np

unprocessed_data_path = Path(r'X:\Training Data\Unprocessed')
processed_data_path = Path(r'X:\Training Data\Processed')

def check_multiple_audio_files(fileList):
    '''Checks if there are separate audio files for single song'''

    # Get audio file list
    num_files = 0
    multiple_files = False
    for track in fileList:
        if track.endswith('.ogg'):
            num_files += 1
        if track.endswith('.mp3'):
            num_files += 1
        if track.endswith('.wav'):
            num_files += 1
    if num_files > 1:
        multiple_files = True
    return multiple_files

def get_audio_file_name(fileList):
    '''Gets name of audio file'''
    for track in fileList:
        if track.endswith('.ogg') or track.endswith('.mp3') or track.endswith('.wav'):
            return track
    raise NameError('Error: audio file not present')

def match_tensor_lengths(notes_tensor, song):
    '''Each note_tensor has some space at the end where the song is still playing but there is no
       note data. This function appends zeros to the notes_tensor so that it matches song length '''
    
    length = song.shape[2]
    notes = np.zeros(length)
    notes[0:(notes_tensor.shape[0])] = notes_tensor

    return notes


def populate_processed_folder(unprocessed_data_path, processed_data_path, replace_notes = False):
    wrong_format_charts = []    # Holds paths to charts not in .chart format
    multiple_audio_songs = []   # Holds paths to charts with multiple audio files
    processed = []              # Holds paths to song folders that were successfully processed
    song_size = 0               # Total audio data size, in gigabytes
    notes_size = 0              # Total note data size, in gigabytes
    
    for dirName, _, fileList in tqdm(os.walk(unprocessed_data_path)):  # Walk through training data directory
        if not fileList:  # If file list is empty
            continue

        # Define a few paths and names of things
        track_pack_ = str(Path(dirName).parent).split('\\')[3]
        song_ = str(Path(dirName)).split('\\')[4]
        processed_path = processed_data_path / track_pack_ / song_
        unprocessed_path = unprocessed_data_path / track_pack_ / song_
        processed_song_path = processed_path / 'song.npy'
        processed_notes_path = processed_path / 'notes.npy'

        print('\n\nProcessing {}, {}'.format(track_pack_, song_))

        if replace_notes:
            if processed_notes_path.exists():
                os.remove(processed_notes_path)  # Delete because I accidentally saved the same array hundreds of times lol

        # Skip creating the directory if there is more than one audio file
        if check_multiple_audio_files(fileList):
            multiple_audio_songs.append(unprocessed_song_path)
            print('{}, {} contains multiple audio files, skipping'.format(track_pack_, song_))
            if processed_path.exists():
                if len(os.listdir(processed_path)) == 0: # If the folder exists but is empty
                    os.rmdir(processed_path)
            continue
        else:
            audio_file_name = get_audio_file_name(fileList)
            unprocessed_song_path = unprocessed_path / audio_file_name

        # Create note tensor for song
        try:
            note_tensor = np.array(chart2tensor(unprocessed_path / 'notes.chart', print_release_notes = False)).astype(int)
        except TypeError as err:
            print('{}, {} .chart file is in the wrong format, skipping'.format(track_pack_, song_))
            print("Type Error: {0}".format(err))
            wrong_format_charts.append(unprocessed_song_path)
            if processed_path.exists():
                if len(os.listdir(processed_path)) == 0: # If the folder exists but is empty
                    os.rmdir(processed_path)
            continue
        except:
            print('{}, {} .chart file is in the wrong format, skipping'.format(track_pack_, song_))
            print('Unknown Error: {0}'.format(sys.exc_info()[0]))
            wrong_format_charts.append(unprocessed_song_path)
            if processed_path.exists():
                if len(os.listdir(processed_path)) == 0: # If the folder exists but is empty
                    os.rmdir(processed_path)
            continue
        
        # Make folder in 'Processed' if it doesn't already exist
        if not processed_path.exists():
            os.mkdir(processed_path)

        # Check if song has already been processed
        if processed_song_path.exists():
            print('{} audio has already been processed'.format(str(Path(processed_path)).split('\\')[-1]))
            song = np.load(processed_song_path)
        else:
            song = music2tensor(unprocessed_song_path)
            np.save(processed_song_path, song)
        
        # Check if notes have already been processed
        if processed_notes_path.exists():
            print('{} notes have already been processed'.format(str(Path(processed_path)).split('\\')[-1]))
        else:
            try:
                note_tensor = match_tensor_lengths(note_tensor, song)
                np.save(processed_notes_path, note_tensor)
            except ValueError as err:
                print('Value Error: {0}'.format(err))
                print('note_tensor shape:',note_tensor.shape)
                print('song_shape:', song.shape)
                continue
            

        song_size += (processed_song_path.stat().st_size) / 1e9
        notes_size += (processed_notes_path.stat().st_size) / 1e9

        

    # Create dict of data information
    data_info = {'wrong_format_charts' : wrong_format_charts,
                'multiple_audio_songs' : multiple_audio_songs,
                'processed' : processed,
                'song_size' : song_size,
                'notes_size' : notes_size}
    
    return data_info




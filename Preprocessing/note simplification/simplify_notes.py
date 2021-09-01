# Change PROCESSED_DIRECTORY to match the location of training data on your machine
# Run this file to parse through all notes.npy files within PROCESSED_DIRECTORY, simplify them
# then add that file to the same directory as 'simplified_notes.npy'

from pathlib import Path
import  numpy as np
import pickle
import os
from tqdm import tqdm

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ CONSTANTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

with open(str(Path.cwd() / 'Resources' / 'simplified_note_dict.pkl'), 'rb') as f:
    simplified_note_keys = pickle.load(f)
f.close()

release_keys = list(np.arange(187, 218))
release_keys.append(224)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FUNCTION DEFITIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def remove_release_keys(notes_array):
    '''Removes all notes corresponding to releases from notes_array'''
    # new_notes = [x for x in notes_array if x not in release_keys]
    new_notes = np.zeros(notes_array.size)
    changed_notes = []
    for x in range(notes_array.size):
        if notes_array[x] not in release_keys:
            new_notes[x] = notes_array[x]
        else:
            changed_notes.append(x)
    return new_notes

def simplify_notes(notes_array):
    '''Maps modified and held notes to their regular counterparts'''
    new_notes = np.zeros(notes_array.size)
    for x in range(notes_array.size):
        if notes_array[x]:
            new_notes[x] = simplified_note_keys[notes_array[x]]
    return new_notes

def populate_with_simplified_notes(processed_directory):
    '''Takes all the processed notes.npy files in processed_directory (probably ./Training Data/Processed) and
       adds simplified versions (notes_simplified.npy) to the same subdirectories'''
    
    # Parse through directory
    for x in tqdm(os.listdir(processed_directory)):
        # Check to see if the .DS_Store file is mysteriously in the directory, skip if so
        if str(x)[-9:] == '.DS_Store':
            print('.DS_Store present in directory, continuing')
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

            new_notes = remove_release_keys(notes_array)
            new_notes = simplify_notes(new_notes)

            # Save to same directory
            with open(str(processed_directory / x / y / 'notes_simplified.npy'), 'wb') as f:
                np.save(f, new_notes)
            f.close()

if __name__ == '__main__':
    
    # Swap PROCESSED_DIRECTORY with the path to your local version of ./Training Data/Processed
    PROCESSED_DIRECTORY = Path(r'/Users/ewaissbluth/Documents/GitHub/tensor-hero/Training Data/Training Data/Processed')
    populate_with_simplified_notes(PROCESSED_DIRECTORY)
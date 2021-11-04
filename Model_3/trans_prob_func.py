import numpy as np
from pathlib import Path


def transition_table_valueAdder(note_array, trans_table=trans_table):
    """ 
    Function to update the first note table and the transition probability table
    Function will remove zeros for the note_array to simplify sorting
    Changes the open note (218) to 32 so that the table range is from 0-32
    
    INPUTS: note_array of the song, trans_table
    OUTPUTS: trans_table 
    """
    # Remove zeros from song note array
    note_array = note_array[note_array != 0]
    note_array[note_array == 218] = 32
    
    # update first_noteArray
    trans_table[32][int(note_array[0])-1] += 1 # adding first note to transition table

    # loop through note array and update trans_table - start with index 1
    for i in range(1, len(note_array)):
        trans_table[int(note_array[i-1]-1)][int(note_array[i])-1] += 1
    
    return trans_table


def prob_calc(array):
    """
    Function to calculate the probablilty of an event happening
    INPUT: array (transition table)
    OUTPUT: array (transition probability table)
    """
    new_array = np.nan_to_num(array/(array.sum(axis=0)))
    return (new_array)



def create_trans_table(track_pack_path, trans_table):
    """
    Function that takes in a file path (in this case starting at Condensed Notes Folder - Track Pack) and an empty table 
    that navigates through the files and pulls data only from the note_simplified.npy file
    Function calls transition_table_valueAdder

    INPUTS: File path, empty table (needs to be generated before)
    OUTPUTS: returns transition table
    """
    for album in track_pack_path.iterdir():
        album_path = track_pack_path / album.name
        
        if album.is_file():
            if album.name == 'notes_simplified.npy':
                notes = np.load(album_path / 'notes_simplified.npy')
                trans_table = transition_table_valueAdder(notes, trans_table)
        if album.is_dir():
            for song in album.iterdir():
                notes_path = album_path / song.name
                if song.is_file():
                    if song.name == 'notes_simplified.npy':
                        notes = np.load(notes_path / 'notes_simplified.npy')
                        trans_table = transition_table_valueAdder(notes, trans_table)
                if song.is_dir():
                    for piece in song.iterdir():
                        piece_path = notes_path / piece.name
                        if piece.is_file():
                            if piece.name == 'notes_simplified.npy':
                                notes = np.load(notes_path / 'notes_simplified.npy')
                                trans_table = transition_table_valueAdder(notes, trans_table)
    return trans_table



track_pack_path = Path('/Users/forrestbrandt/Documents/Berkeley/Fall_2021/TensorHero/Condensed Notes')
trans_table = np.zeros((33,33))
trans_table = create_trans_table(track_pack_path,trans_table)
prob_table = prob_calc(trans_table)
prob_table.sum(axis=0)
np.save('Model_3/transition_table_rawNums.npy',trans_table)
np.save('Model_3/trans_prob_table.npy',prob_table)

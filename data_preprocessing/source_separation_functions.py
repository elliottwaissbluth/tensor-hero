import shutil
import os
from pathlib import Path


def update_song_names_clonehero(parent_folder, new_folder):
    '''
    Takes folder of song folders (folders where each song is a folder in that folder), 
    iterates through and extracts the .ogg file from each folder
    renames the song to include the artists/song from the folder name, then moves file into new folder

    inputs
    parent_folder: is a path to the folder of all songs
    new_folder: path to desired location for newly named songs

    outputs
    no outputs, just completes task.
    '''
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if ('.ogg' or '.mp3') in str(file):
                
                p = Path(os.path.join(root, file))
                song_artist = p.parts[-2]
                new_name = song_artist.replace(' ','_') + "_" +file
                new_spot = new_folder + new_name
                shutil.copy(p, new_spot)
        
    return True


def update_song_names_guitarhero(parent_folder, new_folder):
    '''
    Takes folder of song folders (folders where each song is a folder in that folder), 
    iterates through and extracts the .ogg file from each folder
    renames the song to include the artists/song from the folder name, then moves file into new folder

    inputs
    parent_folder: is a path to the folder of all songs
    new_folder: path to desired location for newly named songs

    outputs
    no outputs, just completes task.
    '''
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            if 'layered.wav' in file:
                p = Path(os.path.join(root, file))
                song_artist = p.parts[-2]
                new_name = song_artist.replace(' ','_') + "_" +file
                new_spot = Path(new_folder) / str(new_name)
                shutil.copy(p, new_spot)
        
    return True



def demucs_source_separation(folder):
    """
    takes in folder with renamed songs and completes demucs separation for each song 
    
    input: folder (file path to folder, likely the new_folder path from update_song_names function)
        a list of song file paths
        each item in list needs to be in parenthesis
    """
    for song in [folder / x for x in os.listdir(folder)]:
        print(song)
        os.system( f'demucs "{str(song)}"')
    # for root, dirs, files in os.walk(folder):
        # for name in files:
            # p = os.path.join(root, name)
            # os.system( f'demucs "{p}"')
    
    return True


def move_finished_songs(source_dir, destination_dir):
    '''
    takes in parent folder of source separated songs as produces by demucs function, 
    and sends guitar track file to destination directory for each song

    input:
    source_dir: parent directory of demucs source separated songs
        will looks something like "/.../.../.../separated/mdx_extra_q"
        demucs creates all separated songs under a folder/subfolder called "separated/mdx_extra_q"
    destination_dir: folder path of where guitar tracks should be sent

    '''
    for root, dirs, files in os.walk(source_dir):
        for name in files:
            if 'other.wav' in name:
                my_song = os.path.join(root, name)
                parts = root.split('/')
                
                song_name = parts[-1]
                old_name = song_name.replace('_', ' ')
                old_name = old_name.replace(' song', '')
                print(old_name)
                destination = os.path.join(destination_dir, old_name)
                print(destination)
                shutil.copy(my_song, destination)

    return True
import sys
sys.path.insert(1, r'C:\Users\ewais\Documents\GitHub\tensor-hero\Model_1\Processing')
sys.path.insert(1, r'C:\Users\ewais\Documents\GitHub\tensor-hero\Model_3')
from m4_functions import *
from m1_postprocessing import *
import os
from pathlib import Path
import pickle
import numpy as np
from tqdm import tqdm

# For Model 1

def get_test_song_paths(test_songs_dir):
    '''
    Parses the subdirectories of test_songs_dir and returns a list of paths to test_songs
    '''
    return [test_songs_dir / x for x in os.listdir(test_songs_dir)]



def load_model(model_dir):
    '''
    Loads the PyTorch model and relevant information from model directory
    '''
    with open(str(model_dir / 'params.pkl'), 'rb') as f:
        params = pickle.load(f)
    f.close()

    # Define model hyperparameters from params dictionary
    trg_vocab_size = params['trg_vocab_size']
    embedding_size = params['embedding_size']
    num_heads = params['num_heads']
    num_encoder_layers = params['num_encoder_layers']
    num_decoder_layers = params['num_decoder_layers']
    dropout = params['dropout']
    max_len = params['max_src_len']
    forward_expansion = params['embedding_size'] * params['forward_expansion']

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        raise NameError('ERROR: cuda is not available')

    model = Transformer(
            embedding_size,
            trg_vocab_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
            max_len,
            device,
        ).to(device)

    # Load trained model    
    model_file = os.listdir(model_dir)
    model_file.remove('params.pkl')
    model_file = model_file[0]
    model.load_state_dict(torch.load(str(model_dir / model_file)))

    return model, params, device


def model_1_pre_sep():
    test_songs = get_test_song_paths(Path.cwd() / 'Experiments' / 'Test_Songs')
    song_paths = [x / 'song.ogg' for x in test_songs]

    model_dir = Path(r'C:\Users\ewais\Documents\GitHub\tensor-hero\Model_1\saved models\model10')
    model, params, device = load_model(model_dir)

    print(params)

    model_output_folder = 'm1_pre_sep'
    model_output_folder = Path(r'C:\Users\ewais\Documents\GitHub\tensor-hero\Experiments\Generated_Songs') / model_output_folder
    # Loop through the songs in song_path and do inference
    for song in song_paths:
        metadata = {
            'path_to_original_chart' : song.parent / 'notes.chart',
            'path_to_original_notes_array' : song.parent / 'notes_simplified.npy'
        }
        song_name = str(song.parent).split('\\')[-1] 
        if not os.path.isdir(model_output_folder / song_name):
            os.mkdir(model_output_folder / song_name)
        if not os.path.isdir(model_output_folder / song_name / song_name):
            os.mkdir(model_output_folder / song_name / song_name)
        
        song_metadata = {'Name' : params['model_name'] + song_name,
                        'Artist' : 'Forrest',
                        'Charter' : 'tensorhero',
                        'Offset' : 0,
                        'Resolution' : 192,
                        'Genre' : 'electronic',
                        'MediaType' : 'cd',
                        'MusicStream' : 'song.ogg'}

        notes_array = full_song_prediction(song_path=song,
                                        model=model,
                                        device=device,
                                        sos_idx=432,
                                        max_len=params['max_src_len'],
                                        song_metadata=song_metadata,
                                        outfolder=(model_output_folder / song_name / song_name))
        
        # Save notes array and metadata
        with open(str(model_output_folder / song_name / 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        f.close()
        np.save(str(model_output_folder / song_name / 'notes_array.npy'), notes_array)

def model_4_pre_sep():
    test_songs = get_test_song_paths(Path.cwd() / 'Experiments' / 'Test_Songs')
    song_paths = [x / 'song.ogg' for x in test_songs]


    model_output_folder = 'm4_pre_sep'
    model_output_folder = Path(r'C:\Users\ewais\Documents\GitHub\tensor-hero\Experiments\Generated_Songs') / model_output_folder
    # Loop through the songs in song_path and do inference
    for song in tqdm(song_paths):
        metadata = {
            'path_to_original_chart' : song.parent / 'notes.chart',
            'path_to_original_notes_array' : song.parent / 'notes_simplified.npy'
        }
        song_name = str(song.parent).split('\\')[-1] 
        if not os.path.isdir(model_output_folder / song_name):
            os.mkdir(model_output_folder / song_name)
        if not os.path.isdir(model_output_folder / song_name / song_name):
            os.mkdir(model_output_folder / song_name / song_name)

        # Generate notes array here
        notes_array = generate_song(song_path=song,
                    outfile_song_name = 'Model 4 - ' + song_name,
                    outfolder = model_output_folder / song_name / song_name)

        # Save notes array and metadata
        with open(str(model_output_folder / song_name / 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        f.close()
        np.save(str(model_output_folder / song_name / 'notes_array.npy'), notes_array)
        
def model_4_post_sep():
    test_songs = get_test_song_paths(Path.cwd() / 'Experiments' / 'Test_Songs')
    song_paths = [x / 'separated.ogg' for x in test_songs]


    model_output_folder = 'm4_post_sep'
    model_output_folder = Path(r'C:\Users\ewais\Documents\GitHub\tensor-hero\Experiments\Generated_Songs') / model_output_folder
    # Loop through the songs in song_path and do inference
    for song in tqdm(song_paths):
        metadata = {
            'path_to_original_chart' : song.parent / 'notes.chart',
            'path_to_original_notes_array' : song.parent / 'notes_simplified.npy'
        }
        song_name = str(song.parent).split('\\')[-1] 
        if not os.path.isdir(model_output_folder / song_name):
            os.mkdir(model_output_folder / song_name)
        if not os.path.isdir(model_output_folder / song_name / song_name):
            os.mkdir(model_output_folder / song_name / song_name)

        # Generate notes array here
        notes_array = generate_song(song_path=song,
                    outfile_song_name = 'Model 4 sep - ' + song_name,
                    outfolder = model_output_folder / song_name / song_name,
                    original_song_path = song.parent / 'song.ogg')

        # Save notes array and metadata
        with open(str(model_output_folder / song_name / 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        f.close()
        np.save(str(model_output_folder / song_name / 'notes_array.npy'), notes_array)
    

if __name__ == '__main__':
    # model_1_pre_sep()
    # model_4_pre_sep()
    model_4_post_sep()
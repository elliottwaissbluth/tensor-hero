import torch
import numpy as np
from tqdm import tqdm
import os
import shutil
from tensor_hero.preprocessing.audio import compute_mel_spectrogram
from tensor_hero.preprocessing.data import decode_contour
from tensor_hero.model import Transformer
from pathlib import Path

'''
This script was previously implemented as m1_postprocessing.py

# NOTE
# Some things in the functions are hardcoded. This is a rough outline so be wary of using these functions

# General heuristic for use:
    # 1) Train model
    # 2) Define model with same parameters in __main__
    # 3) Change some of the filepaths related to saving song output
    # 4) Run the file 
'''

# Convert notes array to strings representing note events
# This is useful for writing .chart files from notes arrays
notes_to_chart_strings = {
    1 : ['0'],
    2 : ['1'],
    3 : ['2'],
    4 : ['3'], 
    5 : ['4'],
    6 : ['0','1'],
    7 : ['0','2'],
    8 : ['0','3'],
    9 : ['0','4'],
    10 : ['1','2'], 
    11 : ['1','3'],
    12 : ['1','4'],
    13 : ['2','3'],
    14 : ['2','4'],
    15 : ['3','4'],
    16 : ['0','1','2'],
    17 : ['0','1','3'],
    18 : ['0','1','4'],
    19 : ['0','2','3'],
    20 : ['0','2','4'],
    21 : ['0','3','4'],
    22 : ['1','2','3'],
    23 : ['1','2','4'],
    24 : ['1','3','4'],
    25 : ['2','3','4'],
    26 : ['0','1','2','3'],
    27 : ['0','1','2','4'],
    28 : ['0','1','3','4'],
    29 : ['0','2','3','4'],
    30 : ['1','2','3','4'],
    31 : ['0','1','2','3','4'],
    32 : ['7'],
    218 : ['7']
}

def __single_prediction_to_notes_array(prediction):
    '''
    Takes a single prediction from the transformer and translates it to a notes array
    of length 400.

    ~~~~ ARGUMENTS ~~~~
    -   prediction (Numpy Array, shape=(<max_trg_len>,)):
            -   Prediction from the transformer, should be a single list of max indices
                from transformer prediction. Expected to be in (time, note, time, note, etc.)
                format.
        
    ~~~~ RETURNS ~~~~
    -   notes_array (Numpy Array, shape = (400,)):
            -   The prediction translated into a 4 second simplified notes array
    '''
    note_vals = list(range(32))     # Values of output array corresponding to notes
    time_vals = list(range(32,432)) # Corresponding to times

    # Loop through the array two elements at a time
    pairs = []
    for i in range(prediction.shape[0]-1):
        pair = (prediction[i], prediction[i+1]) # Take predicted notes as couples
        if pair[0] in time_vals and pair[1] in note_vals:
            pairs.append(pair)  # Append if pair follows (time, note) pattern

    # Create notes array
    notes_array = np.zeros(400)
    for pair in pairs:
        notes_array[pair[0]-32] = pair[1]
    return notes_array

def __contour_prediction_to_notes_array(prediction, tbps=25):
    '''
    Takes a single contour prediction from the transformer and translates it to a notes array
    of length 400.

    ~~~~ ARGUMENTS ~~~~
    -   prediction (numpy Array, shape=(<max_trg_len>,)):
            -   Prediction from the transformer, should be a single list of max indices
                from transformer prediction. 
            -   Expected to be formatted as
                [<sos>, time, note plurality, motion, etc., <eos>, <pad>, <pad>, etc.]
    -   tbps (int): time bins per second of predicted notes
        
    ~~~~ RETURNS ~~~~
    -   notes_array (Numpy Array, shape = (400,)):
            -   The prediction
    '''
    if type(prediction) == torch.Tensor:
        prediction = prediction.detach().cpu().numpy()

    #     value           information
    # ____________________________________
    # 0            | <sos> 
    # 1            | <eos> 
    # 2            | <pad>
    # 3-15         | <note pluralities>    
    # 16-24        | <motion [-4, 4]>    
    # 25-(tbps*4+24) | <time bin 1-tbps*4>

    note_vals = list(range(3, 16))            # Note pluralities
    time_vals = list(range(25, tbps*4+24))    # Corresponding to times
    motion_vals = list(range(16, 25))         # Motion in [-4,4]
    
    # Loop through the array 3 elements at a time
    pairs = []
    for i in range(prediction.shape[0]-2):
        pair = (prediction[i], prediction[i+1], prediction[i+2]) # Take predicted notes as couples
        if pair[0] in time_vals and pair[1] in note_vals and pair[2] in motion_vals:
            pairs.append(pair)  # Append if pair follows (time, note) pattern

    # Create contour from pairs
    expansion_factor = 100/tbps
    contour = np.zeros(shape=(2, 400))
    for pair in pairs:
        index = min(round((pair[0]-25)*expansion_factor), 400)
        contour[0, index] = pair[1]-2    # note plurality
        contour[1, index] = pair[2]-20   # motion
    
    # Create notes array from contour
    notes_array = decode_contour(contour)
     
    return notes_array


def transformer_output_to_notes_array(output, PROBABILITIES=True, contour_encoded=False):
    '''
    Takes a batch of output or input to the transformer and converts it to notes arrays.
    Converts the "model 1 data" as described in ./Documentation/'format of model 1 data.txt'
    Returns a matrix of notes arrays, the 0th dimension corresponding to the batch dimension
    of output

    ~~~~ ARGUMENTS ~~~~
    -   output (Torch Tensor, shape=[<batch_size>, <max_trg_len-1>, 435]):
            - Can either be the raw output from the transformer or the ground truth input
            - The difference between GT input and output is that GT input does not need to be
            collapsed along the probability dimension. If GT input, set PROBABILITES=True
    
    -   PROBABILITIES (bool, optional):
            - True if raw transformer output, False if ground truth transformer input 
            - If True, assumes that output has a probability dimension

    ~~~~ RETURNS ~~~~
    -   notes_arrays (Numpy Array, shape = (<batch_size>, 400)):
            - A batch of notes arrays corresponding to the input batch of transformer output
    '''
    
    output = output.detach().cpu().numpy()   # detach from computation graph, convert to numpy
    if PROBABILITIES:   # If the output has an extra dimension for probabilities
        assert len(list(output.shape)) == 3
        output = np.argmax(output, axis=-1)      # go from probabilities to index predictions
    else:
        assert len(list(output.shape)) == 2
    
    # Initialize matrix of notes arrays
    notes_arrays = np.empty(shape=(output.shape[0], 400))
    if not contour_encoded:
        for idx in range(output.shape[0]):      # Each element of the batch gets computed individually
            prediction = np.delete(output[idx], np.where(output[idx] >= 432))  # delete <sos>, <eos>, <pad>
            notes_arrays[idx] = __single_prediction_to_notes_array(prediction)
    else:
        for idx in range(output.shape[0]):      # Each element of the batch gets computed individually
            prediction = np.delete(output[idx], np.where(output[idx] <= 2))  # delete <sos>, <eos>, <pad>
            notes_arrays[idx] = __contour_prediction_to_notes_array(prediction)

    return  notes_arrays

def m1_tensor_to_note_array(output, PRINT=False):
    '''
    m1 tensors are returned from model 1.
    not currently designed to handle batches.

    ~~~~ ARGUMENTS ~~~~
    output : torch tensor
    -   output from model 1, e.g. tensor([])

    ~~~~ RETURNS ~~~~
    notes_array : numpy array
    -   shape = (<song length in seconds * 100>,)
    -   indices represent time in 10ms bins, values represent notes
    '''
    output = torch.argmax(output, dim=1)
    output = output.detach().cpu().numpy()
    # Remove padding, <sos>, <pad>
    output = np.delete(output, np.where(output == 432)) # <sos>
    '''
    # NOTE: The padding and EOS are removed in the inference loop now
    # May depracate soon

    output = np.delete(output, np.where(output == 434)) # <pad>
    # Find <eos> and only consider output before it
    eos_idx = np.where(output == 433)[0][0]             # <eos>
    print('END OF SEQUENCE INDEX: {}'.format(eos_idx))
    output = output[:eos_idx]
    '''

    # Detect the properly formatted pairs of output, i.e. (time, note)
    note_vals = list(range(32))     # Values of output array corresponding to notes
    time_vals = list(range(32,432)) # Corresponding to times

    # Loop through the array two elements at a time
    pairs = []
    for i in range(output.shape[0]-1):
        pair = (output[i], output[i+1])
        if pair[0] in time_vals and pair[1] in note_vals:
            pairs.append(pair)  # Append if pair follows (time, note) pattern

    # Create notes array
    notes_array = np.zeros(400)
    for pair in pairs:
        notes_array[pair[0]-32] = pair[1]
    
    if PRINT:
        print(notes_array)


    return notes_array

def m1_song_preprocessing(song_path):
    '''
    Loads the song (song.ogg) at song_path and converts it to an array of spectrograms with shape = (<song length in seconds / 4>, frequence (512), time (400)).
    If song length is not divisible by 4, pads the end of the song

    ~~~~ ARGUMENTS ~~~~
    - song_path : Path or String
        - path to song
    
    ~~~~ RETURNS ~~~~
    - full_spec : numpy array
        - full spectrogram split into 4 second segments
    '''
    spec = compute_mel_spectrogram(song_path)

    # Pad so the length is divisible by 400
    spec = np.pad(spec, ((0,0),(0,400-(spec.shape[1]%400))), mode='constant', constant_values=-80.0)
    spec = (spec+80)/80  # normalize

    # Populate full spectrogram
    full_spec = np.zeros(shape=(int(spec.shape[1]/400), 512, 400))
    assert (spec.shape[1]/400)%1 < 1e-8, 'Error: Spectrogram has been padded to the wrong length'
    for i in range(int(spec.shape[1]/400)):
        full_spec[i,...] = spec[:,(i*400):((i+1)*400)]

    return full_spec

def predict(model, device, input, sos_idx, max_len, eos_idx = 433):
    '''
    Predicts the output sequence for a single input spectrogram

    ~~~~ ARGUMENTS ~~~~
    - model : PyTorch model
        - Transformer architecture, takes len 400 spectrograms as input
        - Model should already be on device
    - device : str
        - cuda or cpu
    - input : numpy array
        - spectrogram
        - shape = (512, 400)
    - sos_idx : int
        - start of sequence index
    - max_len : int
        - max output sequence length

    ~~~~ RETURNS ~~~~
    - prediction : torch tensor
        - predicted output from model
    '''
    model.eval()

    # Pad spectrogram
    input = np.pad(input,
                   ((0, 0), (0, max_len-input.shape[1])),
                   'constant',
                   constant_values = 0)

    # Convert input to torch tensor
    input = torch.tensor(input, dtype=torch.float).to(device)
    input = input.unsqueeze(0) # Add batch dimension

    # Create initial input sequence, i.e. [<sos>]
    prediction = torch.tensor(np.array([sos_idx])).to(device)
    prediction = prediction.unsqueeze(0) # Add batch dimension

    # Get model output and construct prediction
    for i in tqdm(range(max_len)):

        # Get output
        output = model(input, prediction)
        pred = torch.tensor([torch.argmax(output[0,-1,:]).item()]).unsqueeze(0).to(device)
        
        # Stop predicting once <eos> is output
        if pred == eos_idx:
            break
        prediction = torch.cat((prediction, pred), dim=1)
    
    return prediction, output

def write_song_from_notes_array_to_string(song_metadata, notes_array):
    '''
    Takes song_metadata as well as notes_array and writes notes.chart file to outfolder

    ~~~~ ARGUMENTS ~~~~
    - song_metadata : dict
        - populates [Song] portion of .chart file
    - notes_array : numpy array
        - array of notes with each element corresponding to a 10ms time bin
    - outfolder : Path
        - folder to save the chart to
        - should already exist
    '''
    notes_array = list(notes_array.astype(int))
    chart_string = """""" 
    # populate '[Song]' portion of file
    chart_string += ('[Song]\n')
    chart_string += ('{\n')
    for k, v in song_metadata.items():
        if k in ['Name', 'Artist', 'Charter', 'Album', 'Year', 'Genre', 'MediaType', 'MusicStream']:
            chart_string += ('  ' + k + ' = "' + str(v) + '"\n')
        else:
            chart_string += ('  ' + k + ' = ' + str(v) + '\n')
    chart_string += ('}\n')

    # Populate '[SyncTrack]' portion of file, skip [Events]
    chart_string += ('[SyncTrack]\n{\n  0 = TS 1\n  0 = B 31250\n}\n[Events]\n{\n}\n')

    # Populate notes in '[ExpertSingle]'
    chart_string += ('[ExpertSingle]\n{\n')

    # Fill in notes from notes array
    for idx, note in enumerate(notes_array):
        if note == 0: # ignore no note is present
            continue
        for n in notes_to_chart_strings[note]:
            chart_string += ('  ' + str(idx) + ' = ' + 'N ' + n + ' 0\n')
    chart_string += ('}')
    
    return chart_string 


def write_song_from_notes_array(song_metadata, notes_array, outfolder):
    '''
    Takes song_metadata as well as notes_array and writes notes.chart file to outfolder

    ~~~~ ARGUMENTS ~~~~
    - song_metadata : dict
        - populates [Song] portion of .chart file
    - notes_array : numpy array
        - array of notes with each element corresponding to a 10ms time bin
    - outfolder : Path
        - folder to save the chart to
        - should already exist
    '''
    f = open(str(outfolder / 'notes.chart'), 'w')
    notes_array = list(notes_array.astype(int))
    
    # populate '[Song]' portion of file
    f.writelines(['[Song]\n', '{\n'])
    for k, v in song_metadata.items():
        if k in ['Name', 'Artist', 'Charter', 'Album', 'Year', 'Genre', 'MediaType', 'MusicStream']:
            f.writelines('  ' + k + ' = "' + str(v) + '"\n')
        else:
            f.writelines('  ' + k + ' = ' + str(v) + '\n')
    f.writelines('}\n')

    # Populate '[SyncTrack]' portion of file, skip [Events]
    f.writelines('[SyncTrack]\n{\n  0 = TS 1\n  0 = B 31250\n}\n[Events]\n{\n}\n')

    # Populate notes in '[ExpertSingle]'
    f.writelines('[ExpertSingle]\n{\n')

    # Fill in notes from notes array
    for idx, note in enumerate(notes_array):
        if note == 0: # ignore no note is present
            continue
        for n in notes_to_chart_strings[note]:
            f.writelines('  ' + str(idx) + ' = ' + 'N ' + n + ' 0\n')
    f.writelines('}')
    f.close()
        

def full_song_prediction(song_path, model, device, sos_idx, max_len, song_metadata, outfolder, 
                         PRINT=False, RETURN_RAW_OUTPUT=False, contour_encoded=False, eos_idx=433):
    '''
    Reads the song at song_path, uses model to predict notes over time, saves .chart to outfolder
    and copies song there as well. This outfolder can then be dropped into Clone Hero's song dir.
    
    NOTE:
        - This function is in rough shape right now, needs refined
        - Model does not predict song in batches, but rather one at a time
            - Should change this eventually
    
    ~~~~ ARGUMENTS ~~~~
    - song_path : Path
        - Path to .ogg file
    - model : PyTorch Model
        - Model used to predict notes
        - Currently expected to be Transformer
    - device : str
        - cuda or cpu
    - sos_idx : int
        - start of sequence index
        - 432 for simplified notes
    - max_len : int
        - max output sequence length
    - song_metadata : dict
        - populates [Song] portion of chart file
    - outfolder : Path
        - folder to save song and .chart file to
    '''
    # First, get the song split up into spectrograms 4 second segments
    full_spec = m1_song_preprocessing(song_path)
    # for i in range(full_spec.shape[0]):
        # specshow(full_spec[i,...])
        # plt.show()
    
    # Predict each 4 second segment, populate notes array along the way
    notes_array = np.zeros(full_spec.shape[0]*full_spec.shape[2])
    for i in range(full_spec.shape[0]):
        print(f'predicting segment {i}/{full_spec.shape[0]}')
        prediction, raw_output = predict(model, device, full_spec[i,...], sos_idx, max_len, eos_idx=eos_idx)
        if PRINT:
            print('m1 notes tensor: {}'.format(prediction))
        if not contour_encoded:
            notes_array[(i*full_spec.shape[2]):((i+1)*full_spec.shape[2])] = m1_tensor_to_note_array(prediction)
        else:
            notes_array[(i*full_spec.shape[2]):((i+1)*full_spec.shape[2])] = __contour_prediction_to_notes_array(prediction)

    # Write the outfolder
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)
    # Write the song into the outfolder
    write_song_from_notes_array(song_metadata, notes_array, outfolder)
    # Copy the audio file into the outfolder
    shutil.copyfile(str(song_path), str(outfolder / 'song.ogg'))
    if not RETURN_RAW_OUTPUT:
        return notes_array
    if RETURN_RAW_OUTPUT:
        return notes_array, raw_output

if __name__ == '__main__':

    song_path = Path(r'X:\Training Data\Unprocessed\Angevil Hero II\1. Andy McKee - Ouray\song.ogg')
    outfolder = Path(r'C:\Users\ewais\Documents\GitHub\tensor-hero\Model_1\test_song')

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device = {device}')

    # Model hyperparameters
    trg_vocab_size = 435  # <output length>
    embedding_size = 512
    num_heads = 8
    num_encoder_layers = 2
    num_decoder_layers = 2
    dropout = 0.1
    max_len = 500
    forward_expansion = 2048

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

    # just some dummy data for now
    song_metadata = {'Name' : 'model9',
                 'Artist' : 'some artist',
                 'Charter' : 'tensorhero',
                 'Offset' : 0,
                 'Resolution' : 192,
                 'Genre' : 'electronic',
                 'MediaType' : 'cd',
                 'MusicStream' : 'song.ogg'}

    model.load_state_dict(torch.load(r'C:\Users\ewais\Documents\GitHub\tensor-hero\Model_1\saved models\model11\model11.pt'))

    _ = full_song_prediction(song_path, model, device, 432, 500, song_metadata, outfolder)

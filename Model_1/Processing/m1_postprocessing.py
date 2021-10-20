
import sys
sys.path.insert(1, r'C:\Users\ewais\Documents\GitHub\tensor-hero\Model_1')
sys.path.insert(1,r'C:\Users\ewais\Documents\GitHub\tensor-hero\Shared_Functionality\Preprocessing\Preprocessing Functions')
import torch
import numpy as np
from model import *
from audio_functions import *
from tqdm import tqdm
import os
import shutil
from librosa.display import specshow

# Convert notes array to strings representing note events
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
    32 : ['7']
}

def m1_tensor_to_note_array(output):
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
    output = output.detach().cpu().numpy()
    # Remove padding, <sos>, <eos>
    output = np.delete(output, np.where(output == 432))
    output = np.delete(output, np.where(output == 434))

    # Detect the properly formatted pairs of output, i.e. (time, note)
    note_vals = list(range(32))
    time_vals = list(range(32,432))
    # Loop through the array two elements at a time
    pairs = []
    for i in range(output.shape[0]-1):
        pair = (output[i], output[i+1])
        if pair[0] in time_vals and pair[1] in note_vals:
            pairs.append(pair)

    # Create notes array
    notes_array = np.zeros(400)
    for pair in pairs:
        notes_array[pair[0]-32] = pair[1]
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

def predict(model, device, input, sos_idx, max_len):
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
    # Add batch dimension
    input = input.unsqueeze(0)

    # Create initial input sequence, i.e. [<sos>]
    prediction = torch.tensor(np.array([sos_idx])).to(device)
    # Add batch dimension
    prediction = prediction.unsqueeze(0)

    for i in tqdm(range(max_len)):

        # Get output
        output = model(input, prediction)
        pred = torch.tensor([torch.argmax(output[0,-1,:]).item()]).unsqueeze(0).to(device)
        prediction = torch.cat((prediction, pred), dim=1)
        # print(torch.argmax(output[0,-1,:]).item())

        # if i > 0 and torch.argmax(output[0,-1,:]).item() == 433:
            # print('final predicted output:')
            # print(prediction)
            # break
    
    return prediction

def write_song_from_notes_array(song_metadata, notes_array, outfolder):
    f = open(str(outfolder / 'notes.chart'), 'w')

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
        

def full_song_prediction(song_path, model, device, sos_idx, max_len, song_metadata, outfolder,):
    # First, get the song split up into spectrograms 4 second segments
    full_spec = m1_song_preprocessing(song_path)
    # for i in range(full_spec.shape[0]):
        # specshow(full_spec[i,...])
        # plt.show()
    
    # Predict each 4 second segment, populate notes array along the way
    notes_array = np.zeros(full_spec.shape[0]*full_spec.shape[2])
    for i in range(full_spec.shape[0]):
        print(f'predicting segment {i}/{full_spec.shape[0]}')
        prediction = predict(model, device, full_spec[i,...], sos_idx, max_len)
        print('m1 notes tensor: {}'.format(prediction))
        notes_array[(i*full_spec.shape[2]):((i+1)*full_spec.shape[2])] = m1_tensor_to_note_array(prediction)

    # Write the outfolder
    if not os.path.isdir(outfolder):
        os.mkdir(outfolder)
    # Write the song into the outfolder
    write_song_from_notes_array(song_metadata, notes_array, outfolder)
    # Copy the audio file into the outfolder
    shutil.copyfile(str(song_path), str(outfolder / 'song.ogg'))

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

    song_metadata = {'Name' : 'test song 2',
                 'Artist' : 'some artist',
                 'Charter' : 'tensorhero',
                 'Offset' : 0,
                 'Resolution' : 192,
                 'Genre' : 'electronic',
                 'MediaType' : 'cd',
                 'MusicStream' : 'song.ogg'}

    model.load_state_dict(torch.load(r'C:\Users\ewais\Documents\GitHub\tensor-hero\Model_1\model5.pt'))

    full_song_prediction(song_path, model, device, 432, 500, song_metadata, outfolder)

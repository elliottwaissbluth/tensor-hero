import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def compute_mel_spectrogram(song_path):
    '''
    Computes the log-mel spectrogram of a .ogg file

    ~~~~ INPUTS ~~~~
    -   song_path : path to .ogg file, either string or path object
    
    ~~~~ OUTPUTS ~~~~
    -   spec : 2D numpy array containing log-mel spectrogram.
        -   dimensions = [frequency, time], frequency dimension is always 512
        -   each time slice represents 10 milliseconds
        -   log-scale, so max(spec) = 0, min(spec) = -80
        -   70ms of silence appended to beginning and end of spectrogram
    '''
    data, sr = librosa.load(str(song_path))
    resampled = librosa.resample(data, sr, 44100)
    spec = librosa.feature.melspectrogram(resampled, 44100, n_fft=2048*2, hop_length=441, n_mels=512, power=2, fmax = sr/2)
    spec = librosa.power_to_db(spec, ref=np.max)
    return spec

def compute_source_separated_mel_spectrogram(song_path):
    '''    
    TODO: This function is not yet implemented
    
    Computes the log-mel spectrogram of a .ogg file that has been processed through source separation
    Specifically, the vocals and drums are removed.

    ~~~~ INPUTS ~~~~
    -   song_path : path to .ogg file, either string or path object
    
    ~~~~ OUTPUTS ~~~~
    -   spec : 2D numpy array containing log-mel spectrogram of source separated song.
        -   dimensions = [frequency, time], frequency dimension is always 512
        -   each time slice represents 10 milliseconds
        -   log-scale, so max(spec) = 0, min(spec) = -80
        -   70ms of silence appended to beginning and end of spectrogram
    '''
    data, sr = librosa.load(str(song_path))
    resampled = librosa.resample(data, sr, 44100)
    raise NotImplementedError('ERROR: IMPLEMENT SOURCE SEPARATION')
    # TODO: Source separate the song before computing the spectrogram
    spec = librosa.feature.melspectrogram(resampled, 44100, n_fft=2048*2, hop_length=441, n_mels=512, power=2, fmax = sr/2)
    spec = librosa.power_to_db(spec, ref=np.max)
    return spec
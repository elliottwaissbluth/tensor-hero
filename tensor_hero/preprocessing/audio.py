import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import math

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
    '''
    data, sr = librosa.load(str(song_path))
    resampled = librosa.resample(data, sr, 44100)
    spec = librosa.feature.melspectrogram(resampled, 44100, n_fft=2048*2, hop_length=441, n_mels=512, power=2, fmax = sr/2)
    spec = librosa.power_to_db(spec, ref=np.max)
    return spec

def ninos(audio, sr, gamma=0.94):
    '''Calculates Normalized Identifying Note Onsets based on Spectral Sparsity (NINOS)
    over time for audio.
   
    Implementation as described in 
        https://www.eurasip.org/Proceedings/Eusipco/Eusipco2016/papers/1570256369.pdf 
        
    Each time bin in the returned frame corresponds to approx 4.6ms of audio data.

    Args:
        audio (1D numpy array): Raw audio samples
        sr (int): sample rate
        gamma (float in (0,1]): Proportion of frequency bins to keep
    
    Returns:
        ninos (1D numpy array): Normalized inverse-sparsity measure
        J (int): Number of retained frequency bins
        hop_length (int): Hop length used to compute spectrogram
        
    '''
    # Define spectrogram parameters
    if sr == 22050:
        n_fft = 1024
        hop_length = 102
    elif sr == 44100:
        n_fft = 2048
        hop_length = 205
    else:
        raise ValueError(f'ERROR: sr = {sr}, sr must be either 22050 or 44100')    

    # Compute spectrogram
    spec = np.abs(librosa.stft(audio, n_fft, hop_length))
    
    # Order by magnitude within each time bin
    spec = np.sort(spec, axis=0)
    plt.figure()
    librosa.display.specshow(spec)
    
    # Remove the highest energy frames, cut down by factor of gamma
    J = math.floor(spec.shape[0]*gamma)
    print(f'old spec shape: {spec.shape}')
    spec = spec[:J,:]
    print(f'J: {J}')
    print(f'new spec shape: {spec.shape}')
    plt.figure()
    librosa.display.specshow(spec)
    
    # Compute squared l2 norm and l4 norm of spec along time axis
    l2_squared = np.square(np.linalg.norm(spec, ord=2, axis=0))
    print(f'l2_squared shape: {l2_squared.shape}')
    l4 = np.linalg.norm(spec, ord=4, axis=0)
    print(f'l4 shape: {l4.shape}')
    
    # Convert to NINOS
    ninos = l2_squared / ((J**(1/4))*l4)
    plt.figure()
    plt.plot(ninos)

    return ninos, J, hop_length

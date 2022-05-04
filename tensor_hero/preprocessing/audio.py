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
    
    # Remove the highest energy frames, cut down by factor of gamma
    J = math.floor(spec.shape[0]*gamma)
    spec = spec[:J,:]
    
    # Compute squared l2 norm and l4 norm of spec along time axis
    l2_squared = np.square(np.linalg.norm(spec, ord=2, axis=0))
    l4 = np.linalg.norm(spec, ord=4, axis=0)
    
    # Convert to NINOS
    ninos = l2_squared / ((J**(1/4))*l4)

    return ninos, J, hop_length

def squeeze_idx(idx, min, max):
    '''Helper function that ensures the indices of the windows stays between 0 and max.
   
    Args:
        idx (int): Candidate index
        min (int): Minimum value to be assigned (typically 0)
        max (int): Maximum value to be assigned (typically len(arr) - 1)
    
    Returns:
        idx (int): Correct index to be used for window bounds
        
    '''
    if idx < min:
        return min
    elif idx > max:
        return max
    else:
        return idx

def onset_select(odf_arr, w1=3, w2=3, w3=7, w4=1, w5=0, delta=0, plot=False):
    '''Implements peak-picking for the results of ninos ODF data. 
   
    Implementation as described in 
        https://ismir2012.ismir.net/event/papers/049_ISMIR_2012.pdf 
        
    Args:
        odf_arr (1D numpy array): Values of ninos ODF function
        w1 (int): Hyperparameter for left boundary of window for condition 1
        w2 (int): Hyperparameter for right boundary of window for condition 1
        w3 (int): Hyperparameter for left boundary of window for condition 2
        w4 (int): Hyperparameter for right boundary of window for condition 2
        w5 (int): Hyperparameter for onset threshold (how many windows to use as buffer before selecting a new onset)
        delta (float in [0,infinity)): Threshold for condition 2
        plot (bool): Whether or not to plot onsets overlaid on ninos data
    
    Returns:
        onsets (1D numpy array): Frame indices for onsets
        
    '''
    onsets = []
    plt_onsets = []

    for frame in range(len(odf_arr)):
        # Determine whether candidate frame is a local maximum
        idx1 = squeeze_idx(frame-w1, 0, len(odf_arr)-1)
        idx2 = squeeze_idx(frame+w2, 0, len(odf_arr)-1)
        max_frame = idx1 + np.argmax(odf_arr[idx1:idx2])
        cond1 = frame == max_frame
        # Determine whether candidate frame surpasses local average by delta
        idx1 = squeeze_idx(frame-w3, 0, len(odf_arr)-1)
        idx2 = squeeze_idx(frame+w4, 0, len(odf_arr)-1)
        mean_frame = np.mean(odf_arr[idx1:idx2]) + delta
        cond2 = odf_arr[frame] >= mean_frame
        # True by default if onsets is empty
        cond3 = True

        if len(onsets) > 0:
            # Determine whether candidate frame surpasses a threshold since last onset
            cond3 = frame - onsets[len(onsets) - 1] > w5
        if cond1 and cond2 and cond3:
            onsets.append(frame)
            
    onsets = np.array(onsets)
    if plot:
        plt.figure(figsize=(20, 15))
        plt.plot(odf_arr[:1000])
        plt.vlines(onsets[np.where(onsets < 1000)[0]], ymin=0, ymax=np.max(odf_arr[:1000]), colors=['red'])
        plt.show()
    return onsets
    
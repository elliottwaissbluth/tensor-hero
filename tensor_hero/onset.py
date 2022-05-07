import sys
import librosa
from mir_eval.onset import f_measure
import numpy as np
import matplotlib.pyplot as plt
import math
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

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
    
def onset_frames_to_time(onsets, sr, hop_len):
    '''Converts a list of onset frames to corresponding time

    Args:
        onsets (1D numpy array): Frames of onsets, as determined by hop_len parameter from ninos()
        sr (int): sample rate
        hop_len (int): hop length, determines how frames are converted to times

    Returns:
        _type_: _description_
    '''
    time_per_frame = hop_len/sr
    return [x*time_per_frame for x in onsets]

def onset_times_to_bins(onset_times):
    '''Converts onset times in seconds to onset times in 10ms bins

    Args:
        onset_times (1D numpy array): array of onset times in seconds        

    Returns:
        onset_times (1D numpy array): array of onset times in rounded 10ms bins
    '''
    onset_times = [round(x*100) for x in onset_times]
    return onset_times

def get_10ms_onset_frames(audio, sr, w1=10, w2=1, w3=1, w4=8, w5=10, delta=1.0,
                          start=-1, end=-1):
    '''Takes raw audio and uses spectral sparsity onset detection to predict onsets, which
    are returned as 10ms time frames relative to start and end (full audio if start and end
    aren't specified)

    The default onset_select() hyperparameters were chosen as they performed the best on average
    for the tested segments during grid search

    Args:
        audio (1D numpy array): Raw audio waveform
        sr (int): sample rate
        w1 (int, optional): see onset_select(). Defaults to 10.
        w2 (int, optional): see onset_select(). Defaults to 1.
        w3 (int, optional): see onset_select(). Defaults to 1.
        w4 (int, optional): see onset_select(). Defaults to 8.
        w5 (int, optional): see onset_select(). Defaults to 10.
        delta (float, optional): see onset_select(). Defaults to 1.0.
        start (int, optional): start of portion of song to compute in seconds. Defaults to -1.
                               - If negative or zero, will assume start is beginning
        end (int, optional): end of portion of song to compute in seconds. Defaults to -1.
                             - If negative or zero, will assume end is end of audio.

    Returns:
        onset_time_bins (list of ints): predicted 10ms time bins corresponding to onsets
    '''
    # Get odf
    if start <= 0 and end <= 0:
        odf, _, hop_len = ninos(audio, sr)
    else:
        assert start >= 0 and end >= start and end*sr <= audio.shape[0]
        odf, _, hop_len = ninos(audio[sr*start:sr*end], sr)
    
    # Peak pick
    onsets = onset_select(odf, w1, w2, w3, w4, w5, delta, plot=False)
    onset_times = onset_frames_to_time(onsets, sr, hop_len)
    onset_time_bins = onset_times_to_bins(onset_times)  # convert to 10ms time bins
    
    return onset_time_bins
    
def compare_onsets(audio, sr, notes_array, start, end,
                   w1=3, w2=3, w3=7, w4=1, w5=0, delta=0,
                   plot= False):
    '''Takes onsets from ground truth notes array, computes them from corresponding
    audio, then compares using f1 measure. Plot optional
    
    Args:
        audio (1D numpy array): Raw waveform of audio
        sr (int): sample rate of audio
        notes_array (1D numpy array): notes_array corresponding to audio
        start (int): start of section to measure in seconds
        end (int): end of section to measure in seconds
        [w1:w5, delta] (ints): hyperparameters of onset_select
        plot (bool): if True, will print plot of compared onsets
        
    Returns:
        f1 (float): f1 score of predicted onsets vs ground truth onsets
    ''' 
    # Measure onsets using spectral sparsity
    odf, _, hop_len = ninos(audio[sr*start:sr*end], sr)
    onsets = onset_select(odf, w1, w2, w3, w4, w5, delta, plot=False)
    onset_times = onset_frames_to_time(onsets, sr, hop_len)
    onset_time_bins = onset_times_to_bins(onset_times)

    # Get ground truth clone hero onsets
    ch_onsets = np.where(notes_array[start*100:end*100] > 0)[0]
    ch_onset_times = [x/100 for x in ch_onsets]
    
    # Compare with f_measure
    f1, _, _ = f_measure(np.array(ch_onset_times), np.array(onset_times))
    
    # plot
    if plot:
        plt.figure(figsize=(15,5))
        for o in ch_onsets:
            plt.axvline(x=o, ymin=0, ymax=0.5, color='r')
        for o in onset_time_bins:
            plt.axvline(x=o, ymin=0.5, ymax=1, color='g')
    
    return f1

def notes_array_onset_f1(ground_truth, candidate):
    '''Generates the onset f1 score between a ground truth and predicted notes array

    Args:
        ground_truth (1D numpy array): ground truth notes array
        candidate (1D numpy array): candidate notes array 

    Returns:
        _type_: _description_
    '''
    gt_onset_times = np.where(ground_truth>0)[0]/100
    candidate_onset_times = np.where(candidate>0)[0]/100
    f1, _, _ = f_measure(gt_onset_times, candidate_onset_times)
    return f1
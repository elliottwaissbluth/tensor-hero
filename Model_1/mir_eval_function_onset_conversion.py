import numpy as np
import mir_eval

def notes_to_onset(note_array):
    '''
   This function intakes a numpy array of notes and outputs the onset values in seconds. 
   Created for the mir_eval functions that require an onset array of second values.
    ~~~~ ARGUMENTS ~~~~
    notes : numpy array
        - formatted note arrays
        - format is [0,0,4,0..13,0] - values range from 0-32
    
    ~~~~ RETURNS ~~~~
    seconds array : numpy array
        - second value of note events
        - the format is [second, second, second, ... second] 
        '''

    song_length = len(note_array)+1
    millisecond_array = np.arange(10,song_length*10, 10)
    seconds = millisecond_array/100
    note_indices = np.nonzero(note_array)
    onsets = seconds[note_indices]
    
    return onsets


def eval_fmeas_precision_recall(onset_true, onset_estimate, window = .05):
    '''
   This function intakes a numpy array of ground truth onset values, and predicted onset values. 
   Default window for metrics is .05 seconds. Function also includes checkpoint to make sure onset arrays are valid. 
   If not, will return exception.

   Function outputs the mir_eval evalation metrics of f_measure, precision, and recall. 
    
    ~~~~ ARGUMENTS ~~~~
    notes : numpy array
        - formatted note arrays
        - format is [0,0,4,0..13,0] - values range from 0-32
    
    ~~~~ RETURNS ~~~~
    array of format [f_measure, precision, recall]

    f_measure : float
        = 2*precision*recall/(precision + recall)
    precision : float
        = (# true positives)/(# true positives + # false positives)
    recall : float
        = (# true positives)/(# true positives + # false negatives)

    '''

    mir_eval.onset.validate(onsets, onsets)
    error_metrics = mir_eval.onset.f_measure(onset_true, onset_estimate, window=0.05)
    
    return error_metrics
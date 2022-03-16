import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mir_eval
from tensor_hero.inference import transformer_output_to_notes_array


def freq_saturation(truth,output):
    """
    provides saturation value - how many total notes did our model predict
    requires notes array be of value 0-32

    ~~~~ ARGUMENTS ~~~~
    - truth : np.array
        - should be values 0-32 notes array
        - true note values
    - output : np.array
        - should be values 0-32 notes array
        - predicted note values
    
    ~~~~ RETURNS ~~~~
    - float : ratio of output to truth 
        - could be more or less than 1
        - predicted / truth
    """
    truth_nonzero = np.count_nonzero(truth)
    output_nonzero = np.count_nonzero(output)
    if truth_nonzero:
        return output_nonzero/truth_nonzero
    else:
        if output_nonzero:
            return 3
        else:
            return 1

def freq_histogram(truth, output):
    """
    #returns plotted histogram of distribution of notes 0-32
    #requires notes array be of value 0-32

    ~~~~ ARGUMENTS ~~~~
    - truth : np.array
        - should be values 0-32 notes array
        - true note values
    - output : np.array
        - should be values 0-32 notes array
        - predicted note values
    
    ~~~~ RETURNS ~~~~
    - plotted histogram of distributions
    """
    true = np.histogram(truth, bins = np.arange(0,34))
    observed = np.histogram(output, bins = np.arange(0,34))
    # Position of bars on x-axis
    ind = np.arange(0,33)

    ticks = [str(val) for val in ind]
    # Figure size
    plt.figure(figsize=(10,5))

    # Width of a bar 
    width = 0.3       

    # Plotting
    plt.bar(ind, true[0] , width,log = True, label='Truth')
    plt.bar(ind + width, observed[0], width,log = True, label='Observed')

    plt.xticks(ind + width / 2, ticks)
    plt.legend()
    plt.show();


def freq_table(truth, output):
    """
    #returns dataframe of distribution of notes
    #requires notes array be of value 0-32

    ~~~~ ARGUMENTS ~~~~
    - truth : np.array
        - should be values 0-32 notes array
        - true note values
    - output : np.array
        - should be values 0-32 notes array
        - predicted note values
    
    ~~~~ RETURNS ~~~~
    - Pandas Dataframe of distribution of true vs. predicted notes
    """
    
    true = np.histogram(truth, bins = np.arange(0,34))
    observed = np.histogram(output, bins = np.arange(0,34))
    data = np.array((true[0], true[0]/np.sum(true[0]), observed[0], observed[0]/np.sum(true[0])))
    df = pd.DataFrame(data = data.T, columns= ['Truth', '% of total','Observed','% of total'])

    return df


#returns plotted histogram of distribution of note GROUP types (ex: single note, double note)
#requires notes array be of value 0-32
def type_freq_hist(truth, output):
    """
    #returns plotted histogram of distribution of note types (none, single, double, .. , open)
    #requires notes array be of value 0-32

    ~~~~ ARGUMENTS ~~~~
    - truth : np.array
        - should be values 0-32 notes array
        - true note values
    - output : np.array
        - should be values 0-32 notes array
        - predicted note values
    
    ~~~~ RETURNS ~~~~
    - plotted histogram of distribution of note types
    """
    true = np.histogram(truth, bins = np.arange(0,34))
    observed = np.histogram(output, bins = np.arange(0,34))
    # Position of bars on x-axis
    ind = np.array([0,1,2,3,4,5,6])

    ticks = ['None', 'Single', 'Double', 'Triple', 'Quad', 'Five', 'Open']
    # Figure size
    plt.figure(figsize=(10,5))

    # Width of a bar 
    width = 0.3       

    # Plotting
    plt.bar(ind, true[0] , width,log = True, label='Truth')
    plt.bar(ind + width, observed[0], width,log = True, label='Observed')
    plt.legend()
    plt.xticks(ind,ticks)
    plt.show();


def freq_type_table(truth, output):
    """
    #returns dataframe of distribution of note types
    #requires notes array be of value 0-32

    ~~~~ ARGUMENTS ~~~~
    - truth : np.array
        - should be values 0-32 notes array
        - true note values
    - output : np.array
        - should be values 0-32 notes array
        - predicted note values
    
    ~~~~ RETURNS ~~~~
    - Pandas Dataframe of distributions of true vs. predicted notes
    """
    true = np.histogram(truth, bins = [0,1,6,16,26,31,31,33])
    observed = np.histogram(output, bins = [0,1,6,16,26,31,31,33])
    data = np.array((true[0], true[0]/np.sum(true[0]), observed[0], observed[0]/np.sum(true[0])))
    # Position of bars on x-axis
    df = pd.DataFrame(data = data.T, columns = ['Truth', '% of total','Observed','% of total'], index= ['None', 'Single', 'Double', 'Triple', 'Quadruple', 'Five', 'Open'])

    return df


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
    
    mir_eval.onset.validate(onset_true, onset_estimate)
    error_metrics = mir_eval.onset.f_measure(onset_true, onset_estimate, window=window)
    
    return error_metrics


def evaluate_model_run(model_output, ground_truth):
    """
    returns dictionary of dataframes
    dataframes include: Frequency Notes, Frequency Note Types, Model Metrics

    ~~~~ ARGUMENTS ~~~~
    - preds_tensor : prediction batch output
        - torch.Size([batch_size, 499, 435])
    - truth_tensor : true notes
        - torch.Size([batch_size, 499])
    
    ~~~~ RETURNS ~~~
        1. Frequency Table (% for each note value of total notes)
        2. Frequency Type Table (% for each type type of total notes)
        3. Metrics Table ('Saturation', 'F1', 'Precision', 'Recall') 
    """
    #dictionary for function return
    model_run_metrics = {}
 
    #convert tensor format to np arrays for both candidate tensor output, and truth tensor
    predicted_notes_batch = transformer_output_to_notes_array(model_output, PROBABILITIES=True)
    true_notes_batch = transformer_output_to_notes_array(ground_truth, PROBABILITIES=False)

    #create placeholder metric objects
    freq_saturation_arr = np.zeros(shape=(len(predicted_notes_batch), 1))
    freq_table_arr = np.zeros(((len(predicted_notes_batch), 33, 4)))
    freq_type_table_arr = np.zeros(((len(predicted_notes_batch), 7 , 4)))
    f1_arr = np.zeros(shape=(len(predicted_notes_batch), 1))
    precision_arr = np.zeros(shape=(1, len(predicted_notes_batch)))
    recall_arr = np.zeros(shape=(1, len(predicted_notes_batch)))


    # Iterate through each predicted index of training batch, get metrics against true notes values
    for i in range(len(predicted_notes_batch)):
        # Get true & predicted notes arrays for given song
        temp_pred_notes = predicted_notes_batch[i]
        temp_true_notes = true_notes_batch[i]

        # Convert our arrays to format required for mir_eval functions
        temp_pred_onset = notes_to_onset(note_array=temp_pred_notes)
        temp_true_onset = notes_to_onset(note_array=temp_true_notes)

        # Get all evaluation metric / output for given song
        # Get custom evaluation metrics
        temp_freq_saturation = freq_saturation(truth=temp_true_notes,output=temp_pred_notes)
        temp_freq_table = freq_table(truth=temp_true_notes,output=temp_pred_notes).to_numpy()
        temp_freq_type_table = freq_type_table(truth=temp_true_notes,output=temp_pred_notes).to_numpy()

        # Get mir_eval onset evaluation metrics
        temp_onset_metrics = eval_fmeas_precision_recall(onset_true=temp_true_onset, onset_estimate=temp_pred_onset,
                                                         window=.05)  # returns [f_measure, precision, recall]

        # For each evaluation metrics: Add all metric values for given song to array of evaluation metrics
        freq_saturation_arr[i] = temp_freq_saturation
        freq_table_arr[i, ...] = temp_freq_table
        freq_type_table_arr[i, ...] = temp_freq_type_table
        f1_arr = temp_onset_metrics[0]
        precision_arr = temp_onset_metrics[1]
        recall_arr = temp_onset_metrics[2]


    # Average each evaluation metric across all songs for given model run
    avg_model_run_freq_saturation = np.mean(freq_saturation_arr)
    avg_model_run_freq_table = np.mean(freq_table_arr, axis=0)
    avg_model_run_freq_type_table = np.mean(freq_type_table_arr, axis=0)
    avg_model_run_f1 = np.mean(f1_arr)
    avg_model_run_precision = np.mean(precision_arr)
    avg_model_run_recall = np.mean(recall_arr)

    #collect in single array
    array_metric_avgs = np.matrix((avg_model_run_freq_saturation, avg_model_run_f1, avg_model_run_precision, avg_model_run_recall))

    #convert values to readable amounts in dataframes
    avg_model_run_freq_table_df = pd.DataFrame(data = avg_model_run_freq_table[:,[1,3]], columns= ['% of truth AVG','% of pred AVG'])
    avg_model_run_freq_type_table_df = pd.DataFrame(data = avg_model_run_freq_type_table[:,[1,3]], columns = ['% of truth AVG','% of pred AVG'], index= ['None', 'Single', 'Double', 'Triple', 'Quadruple', 'Five', 'Open'])
    single_metrics_df = pd.DataFrame(data = array_metric_avgs,columns = ['Saturation', 'F1', 'Precision', 'Recall'], index = ['average'])

    # Add all dataframes to dictionary
    model_run_metrics['Frequency Table'] = avg_model_run_freq_table_df
    model_run_metrics['Frequency Type Table'] = avg_model_run_freq_type_table_df
    model_run_metrics['Metric Values'] = single_metrics_df

    return model_run_metrics



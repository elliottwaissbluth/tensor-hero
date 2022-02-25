import numpy as np
import pickle
import pandas as pd
import mir_eval
from pathlib import Path
import os
import sys
sys.path.insert(1, r'/Users/scarr/MIMS/tensor-hero/Prototyping/onset_detection/Model_Evaluation')
# from elliotts computer
#  sys.path.insert(1, r'C:\Users\ewais\Documents\GitHub\tensor-hero\Shared_Functionality\Model_Evaluation')
from model_metric_functions import *
from mir_eval_function_onset_conversion import *

# def evaluate_model_run(pred_notes_path_list, true_notes_path_list ):
def evaluate_model_run(predicted_notes_batch, true_notes_batch ):
    """
    #returns  dictionary of dataframes
    #dataframes include: Frequency Notes, Frequency Note Types, Model Metrics

    ~~~~ ARGUMENTS ~~~~
    - truth : batch of np.array
        - should be values 0-32 notes array
        - true note values
        - each array is member of batch input for training, should be 400 long
    - output : prediction batch of np.array
        - should be values 0-32 notes array
        - predicted note values
        - each array is member of batch output after training, should be 400 long

    
    ~~~~ RETURNS ~~~~
    - Dictionary of various dataframes with model performance information
    """
    #dictionary for function return
    model_run_metrics = {}
 

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



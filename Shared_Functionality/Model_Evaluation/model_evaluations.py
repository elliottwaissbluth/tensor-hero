import numpy as np
import pickle
import pandas as pd
import mir_eval
from pathlib import Path
import os
import sys
sys.path.insert(1, r'C:\Users\ewais\Documents\GitHub\tensor-hero\Shared_Functionality\Model_Evaluation')
from model_metric_functions import *
from mir_eval_function_onset_conversion import *




def evaluate_model_run(model_run_path: str):
    model_run_metrics = {}

    pred_notes_path_list = [] # todo
    true_notes_path_list = [] # todo

    path = Path(model_run_path)
    for song_folder in [path / x for x in os.listdir(path)]:
        # Add path to predicted notes array
        pred_notes_path_list.append(song_folder / 'notes_array.npy')
        # Add path to true notes array from metadata.pkl
        with open(song_folder / 'metadata.pkl', 'rb') as f:
            temp_metadata_dict = pickle.load(f)
        true_notes_path_list.append(temp_metadata_dict['path_to_original_notes_array'])


    freq_saturation_arr = np.zeros(shape=(len(pred_notes_path_list), 1))
    freq_table_arr = np.zeros(((len(pred_notes_path_list), 33, 4)))
    freq_type_table_arr = np.zeros(((len(pred_notes_path_list), 7 , 4)))
    f1_arr = np.zeros(shape=(len(pred_notes_path_list), 1))
    precision_arr = np.zeros(shape=(1, len(pred_notes_path_list)))
    recall_arr = np.zeros(shape=(1, len(pred_notes_path_list)))


    # Iterate through each predicted song
    for i in range(len(pred_notes_path_list)):
        # Get true & predicted notes arrays for given song
        temp_pred_notes = np.load(pred_notes_path_list[i])
        temp_true_notes = np.load(true_notes_path_list[i])

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
        f1_arr = temp_onset_metrics [0]
        precision_arr = temp_onset_metrics[1]
        recall_arr = temp_onset_metrics[2]


    # Average each evaluation metric across all songs for given model run
    avg_model_run_freq_saturation = np.mean(freq_saturation_arr)
    avg_model_run_freq_table = np.mean(freq_table_arr, axis=0)
    avg_model_run_freq_type_table = np.mean(freq_type_table_arr, axis=0)
    avg_model_run_f1 = np.mean(f1_arr)
    avg_model_run_precision = np.mean(precision_arr)
    avg_model_run_recall = np.mean(recall_arr)

    # Add all values to dictionary
    model_run_metrics['freq_saturation'] = avg_model_run_freq_saturation
    model_run_metrics['freq_table'] = avg_model_run_freq_table
    model_run_metrics['freq_type_table'] = avg_model_run_freq_type_table
    model_run_metrics['f1'] = avg_model_run_f1
    model_run_metrics['precision'] = avg_model_run_precision
    model_run_metrics['recall'] = avg_model_run_recall

    return model_run_metrics



if __name__ == '__main__':
    # todo: Must change for path for local machine
    model_run_metric_dict = evaluate_model_run(r'C:\Users\ewais\Documents\GitHub\tensor-hero\Experiments\Generated_Songs\m1_pre_sep')




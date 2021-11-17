import numpy as np
import pandas as pd
from model_metric_functions import *


# For each model
# For each run
# Get relevant folder with predicted notes for respective model & run
# Output list of paths with folder holding predicted notes for each model to be evaluated


# Create empty dict:
    # for each model, have three KPIs


# For each model run folder to be evaluated
    # Create empty freq_saturation_arr
    # Create empty freq_table_df
    # Create empty freq_type_table_df

    # Get list of paths to each song for predicted
    # Get list of paths to each song for truth

    # For each song: get matching predicted notes numpy array and true notes numpy array
        # Call freq_saturation for ratio
        # Add value to freq_saturation np array

        # Call freq_table
        # Add to df on third dimension

        # Call freq_type_table
        # Add to df on third dimension

        # Extension

# Add avg. freq_saturation to model run
# Add avg. freq_table to model run
# Add avg. freq_type_table to model run



def evaluate_model_run(model_run_path: str):
    model_run_metrics = {}

    pred_notes_path_list = [] # todo
    true_notes_path_list = [] # todo

    freq_saturation_arr = np.zeros(shape=(1, len(pred_notes_path_list)))
    freq_table_arr = np.zeros(shape=(1, len(pred_notes_path_list)))
    freq_type_table_arr = np.zeros(shape=(1, len(pred_notes_path_list)))


    # Iterate through each predicted song
    for i in range(len(pred_notes_path_list)):
        # Get true & predicted notes arrays for given song
        temp_pred_notes = pred_notes_path_list[i]
        temp_true_notes = true_notes_path_list[i]

        # Convert our arrays to format required for mir_eval functions
        # todo

        # Get all evaluation metric / output for given song
        # Get custom evaluation metrics
        temp_freq_saturation = freq_saturation(truth=temp_true_notes,output=temp_pred_notes)
        temp_freq_table = freq_table(truth=temp_true_notes,output=temp_pred_notes)
        temp_freq_type_table = freq_type_table(truth=temp_true_notes,output=temp_pred_notes)
        # Get mir_eval onset evaluation metrics
        # todo: Call mir_eval functions

        # For each evaluation metrics: Add all metric values for given song to array of evaluation metrics
        freq_saturation_arr[i] = temp_freq_saturation
        freq_table_arr[i] = temp_freq_table
        freq_type_table_arr[i] = temp_freq_type_table
        # todo: Add mir_eval values to collections for each metric


    # Average each evaluation metric across all songs for given model run
    avg_model_run_freq_saturation = np.mean(freq_saturation_arr)
    avg_model_run_freq_table = 0 #todo
    avg_model_run_freq_type_table = 0 #todo
    # todo: avg for mir_eval metrics

    # Add all values to dictionary
    model_run_metrics['freq_saturation'] = avg_model_run_freq_saturation
    model_run_metrics['freq_table'] = avg_model_run_freq_table
    model_run_metrics['freq_type_table'] = avg_model_run_freq_type_table
    #todo

    return model_run_metrics












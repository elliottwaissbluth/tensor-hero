'''
Takes raw, unprocessed data and processes expert .chart files into a specified processed folder
'''

# To process the training data, we will use the populated_processed_folder()
from pathlib import Path
import sys
sys.path.insert(1, str(Path.cwd()))
import os
from tensor_hero.preprocessing.data import populate_processed_folder, populate_with_simplified_notes

unprocessed_path = Path.cwd() / 'Training_Data' / 'fully_curated'
processed_path = Path.cwd() / 'Training_Data' / 'Processed' / 'fully_curated'

assert (os.path.isdir(unprocessed_path) and os.path.isdir(processed_path)), 'ERROR: Place "Training Data" folder in this directory'

print('PROCESSING DATA')
# processing_data = populate_processed_folder(unprocessed_data_path=unprocessed_path,
                                            # processed_data_path=processed_path,
                                            # PRINT_TRACEBACK=True, SUB_PACKS=False, TRACK_PACKS=False)

print('\nPOPULATING PROCESSED FOLDER WITH SIMPLIFIED NOTES')
populate_with_simplified_notes(processed_path)
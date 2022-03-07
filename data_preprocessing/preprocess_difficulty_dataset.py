# To process the training data, we will use the populated_processed_folder()
from pathlib import Path
import sys
sys.path.insert(1, str(Path.cwd()))
import os
from tensor_hero.preprocessing.data import create_difficulty_dataset, populate_difficulty_with_simplified_notes

unprocessed_path = r"C:\Users\cheny\OneDrive\Desktop\difficulty_dataset\unprocessed_charts"
processed_path = r"C:\Users\cheny\OneDrive\Desktop\difficulty_dataset\processed_notes"

assert (os.path.isdir(unprocessed_path) and os.path.isdir(processed_path)), 'ERROR: Place "Training Data" folder in this directory'

# print('PROCESSING DATA')
processing_data = create_difficulty_dataset(unprocessed_data_path=unprocessed_path,
                                            processed_data_path=processed_path,
                                            PRINT_TRACEBACK=True)

print('\nPOPULATING PROCESSED FOLDER WITH SIMPLIFIED NOTES')
populate_difficulty_with_simplified_notes(processed_path)
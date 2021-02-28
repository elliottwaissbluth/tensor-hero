# Run this script to preprocess note and audio data

from pathlib import Path
import sys
sys.path.insert(1, str(Path().resolve().parent) + r'\Preprocessing')
from preprocess_functions import populate_processed_folder

unprocessed_data_path = Path(r'X:\Training Data\Unprocessed')
processed_data_path = Path(r'X:\Training Data\Processed')

data_info = populate_processed_folder(unprocessed_data_path, processed_data_path, replace_notes = True)

print(data_info)
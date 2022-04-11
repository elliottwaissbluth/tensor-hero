'''
Once 1_preprocess_training_data.py has been run, this script will further process the data
to be ready for the transformer model.
'''

from pathlib import Path
import sys
sys.path.insert(1, str(Path.cwd()))
from tensor_hero.preprocessing.data import preprocess_transformer_data

segment_length = 400
training_data_path = Path.cwd() / 'Training_Data' / 'Processed' / 'fully_curated'
train_val_test_probs = [0.95, 0.05, 0]

# # If training w/ colab:
# model_training_directory_name = 'colab_training_data'
# preprocess_transformer_data(segment_length,
                            # training_data_path,
                            # train_val_test_probs,
                            # model_training_directory_name,
                            # COLAB=True)

# If not training w/ colab:
# model_training_directory_name = 'colab_training_data'
# preprocess_transformer_data(segment_length,
                            # training_data_path,
                            # train_val_test_probs,
                            # model_training_directory_name,
                            # COLAB=True)

# For generating source separated colab data:
model_training_directory_name = 'fully_curated_colab'
preprocess_transformer_data(segment_length,
                            training_data_path,
                            train_val_test_probs,
                            model_training_directory_name,
                            COLAB=True,
                            SEPARATED=False)
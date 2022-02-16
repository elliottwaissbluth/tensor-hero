'''
Once 1_preprocess_training_data.py has been run, this script will further process the data
to be ready for the transformer model.
'''

import sys
from pathlib import Path
sys.path.inser(1, Path.cwd())
from tensor_hero.preprocessing.data import preprocess_transformer_data

segment_length = 400
training_data_path = Path(r'X:\Training Data')
train_val_test_probs = [0.95, 0.025, 0.025]

# If training w/ colab:
model_training_directory_name = 'colab_training_data'
preprocess_transformer_data(segment_length,
                            training_data_path,
                            train_val_test_probs,
                            model_training_directory_name,
                            COLAB=True)

# If not training w/ colab:
# model_training_directory_name = 'colab_training_data'
# preprocess_transformer_data(segment_length,
                            # training_data_path,
                            # train_val_test_probs,
                            # model_training_directory_name,
                            # COLAB=True)

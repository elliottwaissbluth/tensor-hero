import sys
from pathlib import Path
sys.path.insert(1, str(Path.cwd()))
from tensor_hero.preprocessing.data import preprocess_transformer_data

segment_length = 400
training_data_path = Path(r'X:\Training Data')
train_val_test_probs = [0.95, 0.025, 0.025]

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
model_training_directory_name = 'separated_colab'
preprocess_transformer_data(segment_length,
                            training_data_path,
                            train_val_test_probs,
                            model_training_directory_name,
                            COLAB=True,
                            SEPARATED=True)
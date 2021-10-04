This folder is useful because it contains a working version of Model 1, which is trained to overfit on a single training example located in /toy training data/

~~~~ Files ~~~~
- model.py
    - Defines the transformer model and a Dataset with the purpose of loading that single training example
- train.ipynb
    - Defines the training loop for the model defined in model.py
- preprocess_toy_data.ipynb
    - Used to create the files in toy_training_data
    - Some of the functionality is productionized in /Model_1/Preprocessing/m1_preprocessing.py 


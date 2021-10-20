This folder contains resources related to transforming the training data into a format usable for training model 1

tldr: run m1_preprocessing.py to preprocess your data, you need to change a few lines and create a directory to get it to work with your computer

~~~~ Files ~~~~
- m1_preprocessing.py
    - See the docstring of populate_model_1_training_data() for instructions on how to use
    - Defines functions related to preprocessing data for model 1
    - If run as __main__ (i.e. just running the file), then it can be used to populate a folder
    which might be leveraged by a dataloader
    - There are a few things you need to change in this file to run it on your local machine, everything that needs
    changed is tagged with a comment: NEEDSCHANGE
        - ctl+f search this to find the things you need to modify

~~~~ Data Format ~~~~
Each song was split into segments of 4 seconds. The last segment was discarded as it usually didn't add up to 4 seconds
- processed spectrograms
    - found as <idx>.npy in ./Training Data/Model 1 Training/<train, test, or val>/spectrograms
    - shape = (512, 400)
        - (frequency bins, time)
    - normalized in [0,1]
    - 70ms silence padding removed from beginning and end
- processed notes
    - found as <idx>.npy in ./Training Data/Model 1 Training/<train, test, or val>/notes
    - shape = (400)
        - (time)
    - 70ms silence padding removed from beginning and end


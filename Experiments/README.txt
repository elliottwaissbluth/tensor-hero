This directory contains files and folders related to conducting experiments to compare models.

~~~~ Files ~~~~
- inference.py
    - contains functions related to gathering model output
- analysis.py
    - contains functions related to analyzing the models' output on the songs present in test_songs


~~~~ Folders ~~~~
- Test_Songs
    - Contains 15 songs that serve as the test set for the models.
- Generated_Songs
    - Contains folders which contain the output of the models
    FOLDERS WITHIN:
    - m1_post_sep
        - model 1 trained with source separation
    - m1_pre_sep
        - model 1 trained without source separated data
    - m4_post_sep
        - model 4 with onsets detected using source separated data
    - m4_pre_sep
        - model 4 with onsets detected without source separated data
    NOTE: Within each models' folder, the song folders have the following structure:
    - <song name>
        - metadata.pkl : dict
            {
                'path_to_original_chart' : Path, contains path to the original chart file (probably in Training Data)
                'path_to_original_notes_array' : Path, contains path to original notes array (probably in Training Data)
            }
        - notes_array.npy : numpy array
            - the predicted notes_array
        - <song name> : folder which can be dragged and dropped into Clone Hero
            - song.ogg
            - notes.chart
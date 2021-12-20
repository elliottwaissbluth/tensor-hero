# Import relevant libraries
# scipy is a signals processing toolkit
# pydub lets you import mp3 files as numpy arrays
# https://www.filehorse.com/download-ffmpeg-64/download/     -> to get ffmpeg
# https://www.youtube.com/watch?v=qjtmgCb8NcE                -> for how to install ffmpeg
# I also did a pip install (py -m pip install ffmpeg-python) before following the steps laid out above
#   not sure if it's necessary or not but my program works so idk...

from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import pydub
import numpy as np
import pandas as pd

# Define mp3 importing function
def mp3_read(f, normalized=False):
    """MP3 to numpy array"""
    # First column of mp3 data is left side audio, second column is right
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

# Lots of the guitar hero files are ogg. Runs into a memory error with long files
def ogg_read(f, normalized=False):
    """OGG to numpy array"""
    # First column of mp3 data is left side audio, second column is right
    a = pydub.AudioSegment.from_ogg(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y


import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def music2tensor(path, display_spectrogram = False):
    '''
    WARNING: SOMETHING IS WRONG WITH THIS FUNCTION AS OF 9/19/2021

    music2tensor takes a path to an audio file as input and returns a tensor of processed audio data.
    - path = path to audio file
    - display_spectrogram = if true, will print a figure showing the spectrograms

    returns a tensor with dimensions (channels, mel bins, 10ms time slice)
    '''
    # Load the song and get the sample rate, sr
    data, sr = librosa.load(path)

    # Upsample to 44100, mp3 files have this sample rate. 
    # It will be easiest to convert all the audio files to 44100 before processing
    data = librosa.resample(data, sr, 44100)

    # Get the STFT for three separate window lengths - 93ms, 46ms, 23ms
    # Absolute value is necessary because STFT returns complex numbers
    S_93 = np.abs(librosa.stft(data, n_fft = 2048, hop_length = 441))  # Hop length = stride
    S_46 = np.abs(librosa.stft(data, n_fft = 1024, hop_length = 441))
    S_23 = np.abs(librosa.stft(data, n_fft = 512, hop_length = 441))

    # Create mel filters
    mel93 = librosa.filters.mel(44100, n_fft = 2048, n_mels = 81)
    mel46 = librosa.filters.mel(44100, n_fft = 1024, n_mels = 81)
    mel23 = librosa.filters.mel(44100, n_fft = 512, n_mels = 81)

    # Transform the STFT matrix to the mel filterbank, reducing dimensionality of columns to 81
    S_93 = np.matmul(mel93,S_93)
    S_46 = np.matmul(mel46,S_46)
    S_23 = np.matmul(mel23,S_23)

    # Convert to dB for clearer spectrogram and to better represent human perception
    S93db = librosa.amplitude_to_db(S_93, ref=np.max)
    S46db = librosa.amplitude_to_db(S_46, ref=np.max)
    S23db = librosa.amplitude_to_db(S_23, ref=np.max)

    # Prepend and append seven columns of zeros (even the first and last 10ms 
    # windows should have 70ms of prior context and posterior context)
    append_arr = np.ones((81,7)) * np.min(S93db)
    S93arr = np.c_[append_arr, S93db, append_arr]
    S46arr = np.c_[append_arr, S46db, append_arr]
    S23arr = np.c_[append_arr, S23db, append_arr]

    # Convert each frequency band to zero mean and unit variance
    S93_mean = np.repeat(S93arr.mean(axis=1), S93arr.shape[1]).reshape(81, S93arr.shape[1])
    S93_std = np.repeat(S93arr.std(axis=1), S93arr.shape[1]).reshape(81, S93arr.shape[1])

    S46_mean = np.repeat(S46arr.mean(axis=1), S46arr.shape[1]).reshape(81, S46arr.shape[1])
    S46_std = np.repeat(S46arr.std(axis=1), S46arr.shape[1]).reshape(81, S46arr.shape[1])

    S23_mean = np.repeat(S23arr.mean(axis=1), S23arr.shape[1]).reshape(81, S23arr.shape[1])
    S23_std = np.repeat(S23arr.std(axis=1), S23arr.shape[1]).reshape(81, S23arr.shape[1])

    S93arr = np.nan_to_num(np.divide(np.subtract(S93arr, S93_mean), S93_std))
    S46arr = np.nan_to_num(np.divide(np.subtract(S46arr, S46_mean), S46_std))
    S23arr = np.nan_to_num(np.divide(np.subtract(S23arr, S23_mean), S23_std))

    S_tensor = np.stack((S93arr, S46arr, S23arr), axis=0)

    if display_spectrogram:
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10,5))
        img1 = librosa.display.specshow(S93db, sr = 44100, hop_length=441, 
                                        x_axis='time', y_axis='mel', ax=ax[0])
        ax[0].set(title='93ms Window')
        img2 = librosa.display.specshow(S46db, sr = 44100, hop_length=441, 
                                        x_axis='time', y_axis='mel', ax=ax[1])
        ax[1].set(title='46ms Window')
        img3 = librosa.display.specshow(S23db, sr = 44100, hop_length=441, 
                                        x_axis='time', y_axis='mel', ax=ax[2])
        ax[2].set(title='23ms Window');

    return S_tensor

def split_music_tensor(S_tensor):
    S_tensor = np.array([S_tensor[:,:, i-7 : i+8] for i in range(7, S_tensor.shape[2] - 7)])
    return S_tensor

def compute_mel_spectrogram(song_path):
    '''
    UPDATED: Computes the log-mel spectrogram of a .ogg file

    ~~~~ INPUTS ~~~~
    -   song_path : path to .ogg file, either string or path object
    
    ~~~~ OUTPUTS ~~~~
    -   spec : 2D numpy array containing log-mel spectrogram.
        -   dimensions = [frequency, time], frequency dimension is always 512
        -   each time slice represents 10 milliseconds
        -   log-scale, so max(spec) = 0, min(spec) = -80
        -   70ms of silence appended to beginning and end of spectrogram
    '''
    data, sr = librosa.load(str(song_path))
    if sr != 44100:
        resampled = librosa.resample(data, sr, 44100)
    else:
        resampled = data
    spec = librosa.feature.melspectrogram(resampled, 44100, n_fft=2048*2, hop_length=441, n_mels=512, power=2, fmax = sr/2)
    spec = librosa.power_to_db(spec, ref=np.max)
    return spec

def compute_source_separated_mel_spectrogram(song_path):
    '''    
    Computes the log-mel spectrogram of a .ogg file that has been processed through source separation
    Specifically, the vocals and drums are removed.

    ~~~~ INPUTS ~~~~
    -   song_path : path to .ogg file, either string or path object
    
    ~~~~ OUTPUTS ~~~~
    -   spec : 2D numpy array containing log-mel spectrogram of source separated song.
        -   dimensions = [frequency, time], frequency dimension is always 512
        -   each time slice represents 10 milliseconds
        -   log-scale, so max(spec) = 0, min(spec) = -80
        -   70ms of silence appended to beginning and end of spectrogram
    '''
    data, sr = librosa.load(str(song_path))
    resampled = librosa.resample(data, sr, 44100)
    raise NotImplementedError('ERROR: IMPLEMENT SOURCE SEPARATION')
    # TODO: Source separate the song before computing the spectrogram
    spec = librosa.feature.melspectrogram(resampled, 44100, n_fft=2048*2, hop_length=441, n_mels=512, power=2, fmax = sr/2)
    spec = librosa.power_to_db(spec, ref=np.max)
    return spec
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Try out a song from Training Data
# data = the song data
# sr = sample rate
data, sr = librosa.load('/Users/ewais/Documents/Github/tensor-hero/Training Data/Audioslave - Exploder (Chezy)/song.ogg')

# This is a 1D array, with as many elements as there are samples
print('The original data shape is', data.shape)
print('The original sampling rate is', sr)       # sr is 22050 for .ogg files I believe

# Let's upsample to 44.1k so that mp3 files can be handled and to make the 10ms window more accurate when computing STFT w/ hop length
data_new = librosa.resample(data, sr, 44100)
print('After resampling, data shape is', data_new.shape)
print('And the new sr is 44100')

# Take the STFT
S = np.abs(librosa.stft(data_new, n_fft = 2048, hop_length = 441))
print('After taking the STFT w/ 10 ms stride, the shape of the data is', S.shape)

# Create mel filter
melfilter = librosa.filters.mel(44100, n_fft = 2048, n_mels = 80)
print('The shape of the mel filter is', melfilter.shape)

# Let's transform the STFT matrix to the mel filterbank, reducing the dimensionality of the columns to 80
S_filtered = np.matmul(melfilter,S)
print('The shape of the new filtered data is', S_filtered.shape)

# Take the log of the data to better represent human perception
S_filtered = librosa.amplitude_to_db(S_filtered, ref=np.max)

# Prepend and append 7 columns of zeros (corresponding to 70ms of silence before and after song starts)
s_length = np.size(S_filtered, 1) + 14
S_for_parsing = np.zeros(80, s_length)  # Create an empty matrix with 14 extra columns
S_for_parsing[7,:-7] = S_filtered

#S_for_parsing = np.insert(S_for_parsing, range(np.size(S_for_parsing,1)-1,(np.size(S_for_parsing,1)+6)), 0)
print('Before appending zeros, the shape was', S_filtered.shape)
print('After appending zeros, the shape is', S_for_parsing.shape)

# Plot the new filtered dat
#fig, ax = plt.subplots()
#img = librosa.display.specshow(librosa.amplitude_to_db(S_filtered,
#                                                       ref=np.max),
#                               y_axis='log', 
#                               x_axis='time',
#                               ax=ax)
#ax.set_title('Power spectrogram')
#fig.colorbar(img, ax=ax, format="%+2.0f dB")
#plt.show()
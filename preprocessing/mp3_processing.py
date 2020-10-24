from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import pydub
import numpy as np
import pandas as pd
from music2array import mp3_read
from music2array import ogg_read

sr, x = ogg_read('guitar_small.ogg') #sr = sample rate, x = data
#print(x[20000:22500])
print('The sample rate is', sr)
#x = x*10

L = [] # Left audio
R = [] # Right audio
M = [] # Middle? audio, not sure about this one
for y in x:
    L.append(y[0])
    R.append(y[1])
    M.append((y[0]+y[1]/2))

# Convert to numpy
L = np.array(L)
R = np.array(R)
M = np.array(M)

print('for an ogg file of length', len(x)/sr, 'seconds, the length of the array is', len(L))

# Plot
f, t, Sxx = signal.spectrogram(M, sr) # f = frequency, t = time, Sxx = signal
fig, ax = plt.subplots(figsize = (15,10))
ax.pcolormesh(t, f, Sxx, shading='gouraud')
ax.set_ylabel('Frequency [Hz]')
ax.set_xlabel('Time [sec]')
plt.show()

song = pd.DataFrame({
    'Left' : L,
    'Right' : R,
    'Average' : M
})
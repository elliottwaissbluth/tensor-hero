from pathlib import Path
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf
import sys

GH_dir = Path.cwd() / 'Training_Data' / 'gold_standard_rock'
print(GH_dir)

# Get's a list of ogg files from a directory containing .ogg
find_oggs = lambda file_list: [x for x in file_list if (x.endswith('.ogg') 
                                                        and x != 'preview.ogg' 
                                                        and x != 'layered.ogg' 
                                                        and x != 'song.ogg')]

# Loop through all the songs, grabbing the component .ogg files
for song_dir in tqdm([GH_dir / x for x in os.listdir(GH_dir)]):
  print(f'song_dir: {song_dir}')
  ogg_files = [song_dir / x for x in find_oggs(os.listdir(song_dir))]
  print(f'num ogg files in song_dir: {len(ogg_files)}')
  print(f'ogg_files: {ogg_files}')
  # Load the songs in as a list of numpy arrays
  final_song, sr = sf.read(song_dir / 'song.ogg')
  
  for x in ogg_files:
    song, sr = sf.read(x)
    if song.shape[0] != final_song.shape[0]:
        del song
        continue
    print(f'song.shape: {song.shape}')
    print(f'song.shape[0]: {song.shape[0]}')
    print(f'sr: {sr}')
    final_song += song
    del song

  print(f'max_len: {final_song.shape[0]}')
  print(f'np.max(final_song): {np.max(final_song)}')
  
  # Write the song
  sf.write(str(song_dir / 'layered.ogg'), final_song, 22050, format='ogg')
  plt.plot(final_song)
  plt.show()
  break
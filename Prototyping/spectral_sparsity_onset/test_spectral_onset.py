import sys
from pathlib import Path
from tensor_hero.preprocessing.audio import ninos, onset_select
from tensor_hero.preprocessing.chart import chart2tensor
from tensor_hero.preprocessing.data import __remove_release_keys, __remove_modifiers
import librosa
from mir_eval.onset import f_measure
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def onset_frames_to_time(onsets, sr, hop_len):
    time_per_frame = hop_len/sr
    return [x*time_per_frame for x in onsets]

def onset_times_to_bins(onset_times):
    onset_times = [round(x*100) for x in onset_times]
    return onset_times

def compare_onsets(audio, sr, notes_array, start, end,
                   w1=3, w2=3, w3=7, w4=1, w5=0, delta=0,
                   plot= False):
    
    # Measure onsets using spectral sparsity
    odf, _, hop_len = ninos(audio[sr*start:sr*end], sr)
    onsets = onset_select(odf, w1, w2, w3, w4, w5, delta, plot=False)
    onset_times = onset_frames_to_time(onsets, sr, hop_len)
    onset_time_bins = onset_times_to_bins(onset_times)

    # Get ground truth clone hero onsets
    ch_onsets = np.where(notes_array[start*100:end*100] > 0)[0]
    ch_onset_times = [x/100 for x in ch_onsets]
    
    # Compare with f_measure
    f1, _, _ = f_measure(np.array(ch_onset_times), np.array(onset_times))
    
    # plot
    if plot:
        plt.figure(figsize=(15,5))
        for o in ch_onsets:
            plt.axvline(x=o, ymin=0, ymax=0.5, color='r')
        for o in onset_time_bins:
            plt.axvline(x=o, ymin=0.5, ymax=1, color='g')
    
    return f1

# Get a few 10 second song segments to try out (10 seconds defined below)
anberlin_path = Path.cwd() / 'Training_Data' / 'fully_curated' / 'full_curated' \
    / 'Anberlin - The Feel Good Drag'
notes_array_anberlin = chart2tensor(anberlin_path / 'notes.chart')
notes_array_anberlin = __remove_modifiers(__remove_release_keys(notes_array_anberlin))
anberlin, sr_anberlin = librosa.load(str(anberlin_path / 'other.wav'))

infected_path = Path.cwd() / 'Training_Data' / 'fully_curated' / 'full_curated' \
    / 'Bad Religion - Infected'
notes_array_infected = chart2tensor(infected_path / 'notes.chart')
notes_array_infected = __remove_modifiers(__remove_release_keys(notes_array_infected))
infected, sr_infected = librosa.load(str(infected_path / 'other.wav'))

number_path = Path.cwd() / 'Training_Data' / 'fully_curated' / 'full_curated' \
    / 'Ben Harper and Relentless7 - Number with No Name'
notes_array_number = chart2tensor(number_path / 'notes.chart')
notes_array_number = __remove_modifiers(__remove_release_keys(notes_array_number))
number, sr_number = librosa.load(str(number_path / 'other.wav'))

soothsayer_path = Path.cwd() / 'Training_Data' / 'fully_curated' / 'full_curated' \
    / 'Buckethead - Soothsayer (xX760Xx)'
notes_array_soothsayer = chart2tensor(soothsayer_path / 'notes.chart')
notes_array_soothsayer = __remove_modifiers(__remove_release_keys(notes_array_soothsayer))
soothsayer, sr_soothsayer = librosa.load(str(soothsayer_path / 'other.wav'))

misirlou_path = Path.cwd() / 'Training_Data' / 'fully_curated' / 'full_curated' \
    / 'Dick Dale - Misirlou'
notes_array_misirlou = chart2tensor(misirlou_path / 'notes.chart')
notes_array_misirlou = __remove_modifiers(__remove_release_keys(notes_array_misirlou))
misirlou, sr_misirlou = librosa.load(str(misirlou_path / 'other.wav'))

littlewing_path = Path.cwd() / 'Training_Data' / 'fully_curated' / 'full_curated' \
    / 'Jimi Hendrix - Little Wing'
notes_array_littlewing = chart2tensor(littlewing_path / 'notes.chart')
notes_array_littlewing = __remove_modifiers(__remove_release_keys(notes_array_littlewing))
littlewing, sr_littlewing = librosa.load(str(littlewing_path / 'other.wav'))

w_1 = [*range(1,11)]
w_2 = [*range(1,11)]
w_3 = [*range(1,11)]
w_4 = [*range(1,11)]
_delta = np.arange(2, 3, step=0.2)

best_hyperparams_anberlin = {
    'w1' : 0,
    'w2' : 0,
    'w3' : 0,
    'w4' : 0,
    'w5' : 0,
    'delta' : 0
}
best_hyperparams_infected = {
    'w1' : 0,
    'w2' : 0,
    'w3' : 0,
    'w4' : 0,
    'w5' : 0,
    'delta' : 0
}
best_hyperparams_number = {
    'w1' : 0,
    'w2' : 0,
    'w3' : 0,
    'w4' : 0,
    'w5' : 0,
    'delta' : 0
}
best_hyperparams_soothsayer = {
    'w1' : 0,
    'w2' : 0,
    'w3' : 0,
    'w4' : 0,
    'w5' : 0,
    'delta' : 0
}
best_hyperparams_misirlou = {
    'w1' : 0,
    'w2' : 0,
    'w3' : 0,
    'w4' : 0,
    'w5' : 0,
    'delta' : 0
}
best_hyperparams_littlewing = {
    'w1' : 0,
    'w2' : 0,
    'w3' : 0,
    'w4' : 0,
    'w5' : 0,
    'delta' : 0
}
best_hyperparams_mean = {
    'w1' : 0,
    'w2' : 0,
    'w3' : 0,
    'w4' : 0,
    'w5' : 0,
    'delta' : 0
}

best_f1_anberlin = 0
best_f1_infected = 0
best_f1_number = 0
best_f1_soothsayer = 0
best_f1_misirlou = 0
best_f1_littlewing = 0
best_f1_mean = 0

for w1 in tqdm(w_1, position=0, desc='w1', leave=False, colour='red', ncols=80):
    for w2 in tqdm(w_2, position=1, desc='w2', leave=False, colour='green', ncols=80):
        for w3 in tqdm(w_3, position=2, desc='w3', leave=False, colour='yellow', ncols=80):
            for w4 in tqdm(w_4, position=3, desc='w4', leave=False, colour='blue', ncols=80):
                for delta in tqdm(_delta, position=4, desc='delta', leave=False, colour='cyan', ncols=80, postfix={'best mean f1' : f'{best_f1_mean:.3f}'}):
                    
                    # 0:10 - 0:20
                    f1_anberlin = compare_onsets(anberlin, sr_anberlin, notes_array_anberlin, 10, 20,
                                        w1=w1, w2=w2, w3=w3, w4=w4, w5=0, delta=delta)
                    if f1_anberlin > best_f1_anberlin:
                        best_f1_anberlin = f1_anberlin
                        best_hyperparams_anberlin['w1'] = w1
                        best_hyperparams_anberlin['w2'] = w2
                        best_hyperparams_anberlin['w3'] = w3
                        best_hyperparams_anberlin['w4'] = w4
                        best_hyperparams_anberlin['w5'] = 0
                        best_hyperparams_anberlin['delta'] = delta
                        best_hyperparams_anberlin['best_f1'] = f1_anberlin
                        
                        with open(Path.cwd() / 'Prototyping' / 'spectral_sparsity_onset' / 'best_hyperparams_anberlin.pkl', 'wb') as f:
                            pickle.dump(best_hyperparams_anberlin, f)

                    # 1:00 - 1:10
                    f1_infected = compare_onsets(infected, sr_infected, notes_array_infected, 60, 70,
                                        w1=w1, w2=w2, w3=w3, w4=w4, w5=0, delta=delta)
                    if f1_infected > best_f1_infected:
                        best_f1_infected = f1_infected
                        best_hyperparams_infected['w1'] = w1
                        best_hyperparams_infected['w2'] = w2
                        best_hyperparams_infected['w3'] = w3
                        best_hyperparams_infected['w4'] = w4
                        best_hyperparams_infected['w5'] = 0
                        best_hyperparams_infected['delta'] = delta
                        best_hyperparams_infected['best_f1'] = f1_infected
                        
                        with open(Path.cwd() / 'Prototyping' / 'spectral_sparsity_onset' / 'best_hyperparams_infected.pkl', 'wb') as f:
                            pickle.dump(best_hyperparams_infected, f)

                    # 1:20 - 1:30
                    f1_number = compare_onsets(number, sr_number, notes_array_number, 100, 110,
                                        w1=w1, w2=w2, w3=w3, w4=w4, w5=0, delta=delta)
                    if f1_number > best_f1_number:
                        best_f1_number = f1_number
                        best_hyperparams_number['w1'] = w1
                        best_hyperparams_number['w2'] = w2
                        best_hyperparams_number['w3'] = w3
                        best_hyperparams_number['w4'] = w4
                        best_hyperparams_number['w5'] = 0
                        best_hyperparams_number['delta'] = delta
                        best_hyperparams_number['best_f1'] = f1_number
                        
                        with open(Path.cwd() / 'Prototyping' / 'spectral_sparsity_onset' / 'best_hyperparams_number.pkl', 'wb') as f:
                            pickle.dump(best_hyperparams_number, f)
                    
                    # 4:40 - 4:50
                    f1_soothsayer = compare_onsets(soothsayer, sr_soothsayer, notes_array_soothsayer, 280, 290,
                                        w1=w1, w2=w2, w3=w3, w4=w4, w5=0, delta=delta)
                    if f1_soothsayer > best_f1_soothsayer:
                        best_f1_soothsayer = f1_soothsayer
                        best_hyperparams_soothsayer['w1'] = w1
                        best_hyperparams_soothsayer['w2'] = w2
                        best_hyperparams_soothsayer['w3'] = w3
                        best_hyperparams_soothsayer['w4'] = w4
                        best_hyperparams_soothsayer['w5'] = 0
                        best_hyperparams_soothsayer['delta'] = delta
                        best_hyperparams_soothsayer['best_f1'] = f1_soothsayer
                        
                        with open(Path.cwd() / 'Prototyping' / 'spectral_sparsity_onset' / 'best_hyperparams_soothsayer.pkl', 'wb') as f:
                            pickle.dump(best_hyperparams_soothsayer, f)

                    # 1:10 - 1:20
                    f1_misirlou = compare_onsets(misirlou, sr_misirlou, notes_array_misirlou, 70, 80,
                                        w1=w1, w2=w2, w3=w3, w4=w4, w5=0, delta=delta)
                    if f1_misirlou > best_f1_misirlou:
                        best_f1_misirlou = f1_misirlou
                        best_hyperparams_misirlou['w1'] = w1
                        best_hyperparams_misirlou['w2'] = w2
                        best_hyperparams_misirlou['w3'] = w3
                        best_hyperparams_misirlou['w4'] = w4
                        best_hyperparams_misirlou['w5'] = 0
                        best_hyperparams_misirlou['delta'] = delta
                        best_hyperparams_misirlou['best_f1'] = f1_misirlou
                        
                        with open(Path.cwd() / 'Prototyping' / 'spectral_sparsity_onset' / 'best_hyperparams_misirlou.pkl', 'wb') as f:
                            pickle.dump(best_hyperparams_misirlou, f)
                    
                    # 1:30 - 1:40
                    f1_littlewing = compare_onsets(littlewing, sr_littlewing, notes_array_littlewing, 90, 100,
                                        w1=w1, w2=w2, w3=w3, w4=w4, w5=0, delta=delta)
                    if f1_littlewing > best_f1_littlewing:
                        best_f1_littlewing = f1_littlewing
                        best_hyperparams_littlewing['w1'] = w1
                        best_hyperparams_littlewing['w2'] = w2
                        best_hyperparams_littlewing['w3'] = w3
                        best_hyperparams_littlewing['w4'] = w4
                        best_hyperparams_littlewing['w5'] = 0
                        best_hyperparams_littlewing['delta'] = delta
                        best_hyperparams_littlewing['best_f1'] = f1_littlewing
                        
                        with open(Path.cwd() / 'Prototyping' / 'spectral_sparsity_onset' / 'best_hyperparams_littlewing.pkl', 'wb') as f:
                            pickle.dump(best_hyperparams_littlewing, f)
                    
                    f1_mean = (f1_anberlin + f1_infected + f1_number + f1_soothsayer + f1_misirlou + f1_littlewing) / 6         
                    if f1_mean > best_f1_mean:
                        best_f1_mean = f1_mean
                        best_hyperparams_mean['w1'] = w1
                        best_hyperparams_mean['w2'] = w2
                        best_hyperparams_mean['w3'] = w3
                        best_hyperparams_mean['w4'] = w4
                        best_hyperparams_mean['w5'] = 0
                        best_hyperparams_mean['delta'] = delta
                        best_hyperparams_mean['best_f1'] = f1_mean
                        
                        with open(Path.cwd() / 'Prototyping' / 'spectral_sparsity_onset' / 'best_hyperparams_mean.pkl', 'wb') as f:
                            pickle.dump(best_hyperparams_mean, f)
                    
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# viz_dict maps one hot indices to their corresponding .chart representation
viz_dict = {1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [1, 2], 7: [1, 3], 8: [1, 4], 9: [1, 5], 10: [2, 3], 11: [2, 4], 12: [2, 5], 13: [3, 4], 14: [3, 5], 15: [4, 5], 16: [1, 2, 3], 17: [1, 2, 4], 18: [1, 2, 5], 19: [1, 3, 4], 20: [1, 3, 5], 21: [1, 4, 5], 22: [2, 3, 4], 23: [2, 3, 5], 24: [2, 4, 5], 25: [3, 4, 5], 26: [1, 2, 3, 4], 27: [1, 2, 3, 5], 28: [1, 2, 4, 5], 29: [1, 3, 4, 5], 30: [2, 3, 4, 5], 31: [1, 2, 3, 4, 5], 32: [1], 33: [2], 34: [3], 35: [4], 36: [5], 37: [1, 2], 38: [1, 3], 39: [1, 4], 40: [1, 5], 41: [2, 3], 42: [2, 4], 43: [2, 5], 44: [3, 4], 45: [3, 5], 46: [4, 5], 47: [1, 2, 3], 48: [1, 2, 4], 49: [1, 2, 5], 50: [1, 3, 4], 51: [1, 3, 5], 52: [1, 4, 5], 53: [2, 3, 4], 54: [2, 3, 5], 55: [2, 4, 5], 56: [3, 4, 5], 57: [1, 2, 3, 4], 58: [1, 2, 3, 5], 59: [1, 2, 4, 5], 60: [1, 3, 4, 5], 61: [2, 3, 4, 5], 62: [1, 2, 3, 4, 5], 63: [1], 64: [2], 65: [3], 66: [4], 67: [5], 68: [1, 2], 69: [1, 3], 70: [1, 4], 71: [1, 5], 72: [2, 3], 73: [2, 4], 74: [2, 5], 75: [3, 4], 76: [3, 5], 77: [4, 5], 78: [1, 2, 3], 79: [1, 2, 4], 80: [1, 2, 5], 81: [1, 3, 4], 82: [1, 3, 5], 83: [1, 4, 5], 84: [2, 3, 4], 85: [2, 3, 5], 86: [2, 4, 5], 87: [3, 4, 5], 88: [1, 2, 3, 4], 89: [1, 2, 3, 5], 90: [1, 2, 4, 5], 91: [1, 3, 4, 5], 92: [2, 3, 4, 5], 93: [1, 2, 3, 4, 5], 94: [1], 95: [2], 96: [3], 97: [4], 98: [5], 99: [1, 2], 100: [1, 3], 101: [1, 4], 102: [1, 5], 103: [2, 3], 104: [2, 4], 105: [2, 5], 106: [3, 4], 107: [3, 5], 108: [4, 5], 109: [1, 2, 3], 110: [1, 2, 4], 111: [1, 2, 5], 112: [1, 3, 4], 113: [1, 3, 5], 114: [1, 4, 5], 115: [2, 3, 4], 116: [2, 3, 5], 117: [2, 4, 5], 118: [3, 4, 5], 119: [1, 2, 3, 4], 120: [1, 2, 3, 5], 121: [1, 2, 4, 5], 122: [1, 3, 4, 5], 123: [2, 3, 4, 5], 124: [1, 2, 3, 4, 5], 125: [1], 126: [2], 127: [3], 128: [4], 129: [5], 130: [1, 2], 131: [1, 3], 132: [1, 4], 133: [1, 5], 134: [2, 3], 135: [2, 4], 136: [2, 5], 137: [3, 4], 138: [3, 5], 139: [4, 5], 140: [1, 2, 3], 141: [1, 2, 4], 142: [1, 2, 5], 143: [1, 3, 4], 144: [1, 3, 5], 145: [1, 4, 5], 146: [2, 3, 4], 147: [2, 3, 5], 148: [2, 4, 5], 149: [3, 4, 5], 150: [1, 2, 3, 4], 151: [1, 2, 3, 5], 152: [1, 2, 4, 5], 153: [1, 3, 4, 5], 154: [2, 3, 4, 5], 155: [1, 2, 3, 4, 5], 156: [1], 157: [2], 158: [3], 159: [4], 160: [5], 161: [1, 2], 162: [1, 3], 163: [1, 4], 164: [1, 5], 165: [2, 3], 166: [2, 4], 167: [2, 5], 168: [3, 4], 169: [3, 5], 170: [4, 5], 171: [1, 2, 3], 172: [1, 2, 4], 173: [1, 2, 5], 174: [1, 3, 4], 175: [1, 3, 5], 176: [1, 4, 5], 177: [2, 3, 4], 178: [2, 3, 5], 179: [2, 4, 5], 180: [3, 4, 5], 181: [1, 2, 3, 4], 182: [1, 2, 3, 5], 183: [1, 2, 4, 5], 184: [1, 3, 4, 5], 185: [2, 3, 4, 5], 186: [1, 2, 3, 4, 5], 187: [1], 188: [2], 189: [3], 190: [4], 191: [5], 192: [1, 2], 193: [1, 3], 194: [1, 4], 195: [1, 5], 196: [2, 3], 197: [2, 4], 198: [2, 5], 199: [3, 4], 200: [3, 5], 201: [4, 5], 202: [1, 2, 3], 203: [1, 2, 4], 204: [1, 2, 5], 205: [1, 3, 4], 206: [1, 3, 5], 207: [1, 4, 5], 208: [2, 3, 4], 209: [2, 3, 5], 210: [2, 4, 5], 211: [3, 4, 5], 212: [1, 2, 3, 4], 213: [1, 2, 3, 5], 214: [1, 2, 4, 5], 215: [1, 3, 4, 5], 216: [2, 3, 4, 5], 217: [1, 2, 3, 4, 5], 218: [6], 219: [6], 220: [6], 221: [6], 222: [6], 223: [6], 224: [6]}
note_idx_to_y = { # Note index to y position on plots
    1 : 5,
    2 : 4,
    3 : 3,
    4 : 2,
    5 : 1
}
note_idx_to_c = { # Colors of notes
    1 : 'g',
    2 : 'r',
    3 : 'y',
    4 : '#4a68ff',
    5 : '#ff9100'
}

def __create_scatter_axes(notes, ax=None):
    '''
    Helper function for plot_chart(), creates a matplotlib axis from a notes array.

    ~~~ ARGUMENTS ~~~~
    - notes (list of int): notes arrays. The longer the notes array, the wider the plot.

    ~~~~ RETURNS ~~~~
    - ax (matplotlib ax): holds entire plot of input notes array
    '''
    x = [] # x position of scatter points for each note in notes
    y = [] # y position of scatter points for each note in notes
    c = [] # color array of scatter points
    x_lines = [] # x position of open notes
    scaler = 1 # scales the note placement on the y dimension up or down

    for idx, note in enumerate(notes):  # Parse notes to populate visualization data
        if not note: # skip zeros
            continue

        chord = viz_dict[note]
        for n in chord: # chords are arranged as multiple ints in list, GRY = [1, 2, 3]
            if n != 6:  # If not an open note
                x.append(idx)
                y.append(note_idx_to_y[n]*scaler)
                c.append(note_idx_to_c[n])
            else:
                x_lines.append(idx)

    # Generate coordinates for open notes, these coords map start and end point of purple bars
    coords = []
    for tick in x_lines:
        coords.append(([tick, tick], [0, 6]))

    # Plot GRYBO notes
    if ax is None:
        ax = plt.gca()

    # Plot and format GRYBO notes
    ax.set_ylim((0,6))
    ax.set_xlabel('Ticks', fontsize=15)
    ax.set_yticks(np.arange(0,7,1))
    ax.grid(axis= 'y', which='both')
    ax.set_axisbelow(True)
    ax.set_yticklabels([])
    ax.set_xlim(0, len(notes))
    ax.scatter(x, y, edgecolors='k', color=c, s=160)

    # Plot and format open notes
    for coord in coords:
        ax.plot(coord[0], coord[1], linewidth=3, color='#c000eb')

    return ax

def slice_notes(notes, start=0, end=2):
    '''
    Takes a notes array or spectrogram and slices it between start and end in seconds
    
    ~~~~ ARGUMENTS ~~~~
    - notes (1D numpy array or 2D numpy array): notes array or spectrogram
    - start (int): start of slice relative to beginning of song in seconds
    - end (int): end of slice relative to beginning of song in seconds
   
   ~~~~ RETURNS ~~~~
   1D numpy array or 2D numpy array: notes array or spectrogramsliced between start and end
    '''
    assert start < end, "Error: start value must be less than end value"
    start_tick, end_tick = start*100, end*100  # ticks are in 10ms bins
    if len(notes.shape) == 1:
        return notes[start_tick:end_tick]
    else:
        return notes[:, start_tick:end_tick]

def plot_chart(ground_truth=None, candidate=None, audio=None, SHOW=True):
    '''
    Plots Guitar Hero charts and spectrograms using matplotlib.
    
    Can also be used to plot spectrograms without notes, just fill in the audio arg without
    ground_truth or candidate.

    ~~~~ ARGUMENTS ~~~~
    - ground_truth (list of int): ground truth notes array
    - candidate (list of int): candidate notes array
    - audio (2D numpy matrix): spectrogram
    - SHOW (bool): If true, will show the generated plot in place

    ~~~~ RETURNS ~~~~
    - matplotlib figure: contains the full generated plot
    '''
    num_subplots = int(ground_truth is not None) + int(candidate is not None) + int(audio is not None)
    assert num_subplots > 0, 'ERROR, plot_chart was called without input'
    
    fig, axes = plt.subplots(num_subplots)
    if ground_truth is not None:
        fig.set_size_inches(min(len(ground_truth)/40, 900), 2*num_subplots)
    elif candidate is not None:
        fig.set_size_inches(min(len(candidate)/40, 900), 2*num_subplots)
    ax_idx = 0

    if num_subplots == 1:
        axes = [axes]

    # ground truth plot
    if ground_truth is not None:
        __create_scatter_axes(ground_truth, axes[ax_idx])
        axes[ax_idx].set_title('Ground Truth')
        ax_idx += 1

    # candidate plot
    if candidate is not None:
        __create_scatter_axes(candidate, axes[ax_idx])
        axes[ax_idx].set_title('Candidate')
        ax_idx += 1

    if audio is not None:
        axes[ax_idx].imshow(audio, aspect='auto', origin='lower')
    
    fig.align_xlabels(axes)
    
    if SHOW:
        plt.show()

    return fig
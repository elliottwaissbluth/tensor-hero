import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Define some resources
with open(str(Path.cwd() / 'Shared_Functionality' / 'Data Viz' / 'Resources' / 'viz_dict.pkl'), 'rb') as f:
    viz_dict = pickle.load(f)
f.close()
# Note index to y position on plots
note_idx_to_y = {
    1 : 5,
    2 : 4,
    3 : 3,
    4 : 2,
    5 : 1
}
# Colors of notes
note_idx_to_c = {
    1 : 'g',
    2 : 'r',
    3 : 'y',
    4 : '#4a68ff',
    5 : '#ff9100'
}
# String representation of notes to index
string_to_list = {
    'G' : 1,
    'R' : 2,
    'Y' : 3,
    'B' : 4,
    'O' : 5,
    'o' : 6
}

def slice_notes(notes, start=0, end=2):
    '''
    Takes a notes array and slices it between start and end in seconds
    '''
    assert start < end, "Error: start value must be less than end value"

    start_tick, end_tick = start*100, end*100
    return notes[start_tick:end_tick]

def create_scatter_axes(notes, ax=None):
    '''
    Takes a notes array and creates three vectors:
    -   x : x positions of scatter points
    -   x_lines : x position of open notes
    -   y : y positions of scatter points
    -   c : color array of scatter points
    sub-function for plot_chart() (defined below)
    '''
    x = []
    y = []
    c = []
    x_lines = []

    scaler = 1 # This scaler to scale the y dimension up or down

    # Loop through the whole notes array
    for idx, note in enumerate(notes):
        if not note: # Skip zeros
            continue

        chord = viz_dict[note]
        for n in chord:
            if n != 6:  # If not an open note
                x.append(idx)
                y.append(note_idx_to_y[n]*scaler)
                c.append(note_idx_to_c[n])
            else:
                x_lines.append(idx)

    # Generate coordinates for open notes
    coords = []
    for tick in x_lines:
        coords.append(([tick, tick], [0, 6]))

    # Plot GRYBO notes
    if ax is None:
        ax = plt.gca()
    # fig.set_size_inches(min(len(notes)/40, 900), 2)

    ax.set_ylim((0,6))
    ax.set_xlabel('Ticks', fontsize=15)
    ax.set_yticks(np.arange(0,7,1))
    ax.grid(axis= 'y', which='both')
    ax.set_axisbelow(True)
    ax.set_yticklabels([])
    ax.set_xlim(0, len(notes))
    ax.scatter(x, y, edgecolors='k', color=c, s=160)

    # Plot open notes
    for coord in coords:
        ax.plot(coord[0], coord[1], linewidth=3, color='#c000eb')

    return ax

def plot_chart(ground_truth, candidate=None, audio=None):
    num_subplots = 1 + int(candidate is not None) + int(audio is not None)

    fig, axes = plt.subplots(num_subplots)
    fig.set_size_inches(min(len(ground_truth)/40, 900), 2*num_subplots)

    # Create the ground truth plot
    create_scatter_axes(ground_truth, axes[0])
    axes[0].set_title('Ground Truth')

    # Compare the candidate
    if candidate is not None:
        create_scatter_axes(candidate, axes[1])
        axes[1].set_title('Candidate')

    if audio is not None:
        axes[2].imshow(audio, aspect='auto', origin='lower')
    
    fig.align_xlabels(axes)
    plt.show()

if __name__ == '__main__':

    # Run this script for a lil demo
    notes = np.load(str(Path.cwd() / 'Shared_Functionality' / 'Data Viz' / 'Prototyping' / 'notes.npy'))
    # Put path to some spectrogram here
    # audio = np.load(str(Path.cwd() / 'Shared_Functionality' / 'Data Viz' / 'Prototyping' / 'audio.npy'))

    start = 60
    end = 70

    sliced_notes = slice_notes(notes, start=start, end=end)

    plot_chart(sliced_notes, sliced_notes) # Add audio[:, start*100:end*100] to see spectrogram
    
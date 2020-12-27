# This file contains function definitions related to processing .chart files
import sys
import numpy as np
from pathlib import Path
from itertools import combinations

def chart2tensor(path, print_release_notes=False):
    '''
    Inputs a path to a chart and returns a tensor with 10ms ticks as indices and one hot
    representation as values.
    '''
    coded_notes = chart2onehot(path, print_release_notes)
    notes_tensor = np.zeros(max(coded_notes.keys()))
    for k, v in coded_notes.items():
        notes_tensor[k-1] = v
    return notes_tensor

    
def chart2dict(path):
    '''
    input:
        - path - a path to a chart
    output:
        - notes
        - song_metadata
        - time_signatures
        - BPMs
    '''

    # Read chart into array
    with open(path, 'r') as file:
        raw_chart = file.readlines()
    file.close()

    # Strip lines of \n character
    for i in range(len(raw_chart)):
        raw_chart[i] = raw_chart[i].replace('\n', '')

    # Create lists to hold sections of of the chart file
    song = []
    synctrack = []
    expertsingle = []

    # Parse chart file, populating lists
    i = 0
    for data in raw_chart:
        if data in ['[Song]', '[SyncTrack]', '[Events]', '[ExpertSingle]']:
            i += 1
        
        if data in ['{', '}', '[Song]', '[SyncTrack]', '[Events]', '[ExpertSingle]']:
            continue
        elif i == 1:
            song.append(data[2:])
        elif i == 2:
            synctrack.append(data[2:])
        elif i == 3:
            continue
        elif i == 4:
            expertsingle.append(data[2:])

    time_signatures = {'tick' : [],
                       'TS' : []}
    BPMs = {'tick' : [],
            'BPM' : []}

    for event in synctrack:
        line = event.split(' ')
        if line[2] == 'TS':
            time_signatures['tick'].append(int(line[0]))
            time_signatures['TS'].append(int(line[3]))
        elif line[2] == 'R':
            raise NameError('Error - Resolution changes during this song')
        else:
            BPMs['tick'].append(int(line[0]))
            BPMs['BPM'].append(int(line[3]))

    # Parse the 'expertsingle' section of the .chart file for note information
    notes = {'tick' : [],       # What tick the note is at
            'N_S' : [],         # Whether it is a note (N) or star power (S)
            'note' : [],        # What the note is 
            'duration': []}     # tick duration of the note or star power

    for note in expertsingle:
        line = note.split(' ')
        if line[2] == 'E':       # skip the lines that mark events
            continue
        else:
            notes['tick'].append(int(line[0]))
            notes['N_S'].append(line[2])
            notes['note'].append(line[3])
            notes['duration'].append(int(line[4]))

    # Parse the 'song' section of the .chart file to get relevent chart information
    song_metadata = {'Name' : '',
                    'Artist' : '',
                    'Charter' : '',
                    'Offset' : '',
                    'Resolution' : '',
                    'Genre' : '',
                    'MediaType' : '',
                    'MusicStream' : ''}

    for data in song:
        line = data.split(' ')
        if line[0] in song_metadata.keys():
            song_metadata[line[0]] = line[-1]
    song_metadata['Offset'] = int(song_metadata['Offset'])
    song_metadata['Resolution'] = int(song_metadata['Resolution'])

    return notes, song_metadata, time_signatures, BPMs


def get_configuration(path):
    '''
    input:
    - path - path to a chart folder
    output
    - configuration - returns as a dictionary of the config settings
    '''
    # Read chart into array
    with open(path + '\\song.ini', 'r') as file:
        raw_chart = file.readlines()
    file.close()

    # Configuration is held in a dictionary
    configuration = {}
    # Strip lines of \n character
    for i in range(1, len(raw_chart)-1):
        raw_chart[i] = raw_chart[i].replace('\n', '')
        split_string = raw_chart[i].split('=')
        configuration[split_string[0]] = split_string[-1]

        # Turn int values into ints if possible
        try:
            int(split_string[-1])
            configuration[split_string[0]] = int(split_string[-1])
        except:
            pass

    # Turn int values into ints
    return(configuration)


def chart2onehot(path, print_release_notes=False):
    '''
    Returns a dictionary with keys corresponding to ticks and values corresponding to
    one hot representations of note events. See ~/Prototypes/chart_to_one_hot.ipynb
    for more info.
    - path = path to the chart
    - print_release_notes = if true, will output a feed of release notes that had to be
                            bumped due to coincidence with the start of new notes.
    '''
    notes, _, _, _ = chart2dict(path)

    # Create a dictionary where the keys are the tick values and the values
    # are a list of notes corresponding to the data
    coded_notes_0 = {}

    # Loop through song one note at a time, processing along the way
    for i in range(len(notes['tick'])):
        #if notes['N_S'][i] == 'S':  # if the token is a star power indicator, skip it
        if notes['N_S'] == 'S':  # if the token is a star power indicator, skip it
            continue

        if notes['tick'][i] not in coded_notes_0:  # If the key is not in the dictionary
            coded_notes_0[notes['tick'][i]] = []                   # Create empty list

        if notes['duration'][i] == 0:  # If the note is not held
            coded_notes_0[notes['tick'][i]].append(int(notes['note'][i]))  # Add note to list
        else:  # If the note is held
            if (notes['tick'][i] + notes['duration'][i]) not in coded_notes_0:  # If the key is not in the dictionary
                coded_notes_0[notes['tick'][i] + notes['duration'][i]] = []     # Create empty list
            # Add the note with a hold key
            # Hold key is the code of the note + 10 (so 11 indicates release green, 12 is release red, etc.)
            coded_notes_0[notes['tick'][i]].append(int(notes['note'][i]) + 10)  # Add note to list

            # Add a release key at time step when note will be released.
            # Release key is the code of the note + 20 (so 21 indicates release green, 22 is release red, etc.)
            coded_notes_0[notes['tick'][i] + notes['duration'][i]].append(int(notes['note'][i]) + 20)  # Add note to list

    # coded_notes_1 will hold intermediate values of coded_notes
    coded_notes_1 = {}

    for x in coded_notes_0.keys():
        if 5 in coded_notes_0[x]:    # If a force note
            coded_notes_0[x].remove(5)
            coded_notes_1[x] = map_notes_0(coded_notes_0[x], 'force')
        elif 6 in coded_notes_0[x]:  # If a tap note
            coded_notes_0[x].remove(6)
            coded_notes_1[x] = map_notes_0(coded_notes_0[x], 'tap')
        else:                        # If a regular note
            coded_notes_1[x] = map_notes_0(coded_notes_0[x], 'regular')

    # coded_notes_2 will map the coded_notes_1 values into the syntax of the values described by all_combinations
    coded_notes_2 = {}

    for x in coded_notes_1.keys():
        notestring = ''
        for note_event in coded_notes_1[x]:
            notestring += str(note_event)
        coded_notes_2[x] = notestring

    combo_dictionary = generate_combo_dict()
    coded_notes_3 = {}
    replaced = {'x' : [],  # If notes need to be shuffled for errors, this will be populated
                'replacement_digits' : [],
                'y' : [],
                'release_digits' : []}
    for x in coded_notes_2.keys():
        try:
            coded_notes_3[x] = combo_dictionary[coded_notes_2[x]]  # If no error, insert combo into coded_notes_3
        except:
            if check_for_release_notes(x, coded_notes_2):  # If released note and new note coincide on a tick
                y = x+1
                while y in coded_notes_2:  # Choose an unoccupied tick in front of x
                    y+=1
                
                # Parse the string and strip away release indicators
                replacement_digits = ''
                release_digits = ''
                code = ''
                for digit in coded_notes_2[x]:
                    if not code:
                        code = digit
                        continue
                    if len(code) < 2:
                        code += digit
                    else:
                        code = digit
                        continue
                    if code in ['16', '23', '30', '37', '44']:
                        release_digits += code
                    else:
                        replacement_digits += code

                # Replace note
                replaced['x'].append(x)
                replaced['y'].append(y)
                replaced['release_digits'].append(release_digits)
                replaced['replacement_digits'].append(replacement_digits)
                if print_release_notes:            
                    print('Release Notes Coincided at tick', x,': bumped to tick', y)
                

    for i in range(len(replaced['x'])):
        coded_notes_2[replaced['x'][i]] = replaced['replacement_digits'][i]
        coded_notes_2[replaced['y'][i]] = replaced['release_digits'][i]
        try:
            coded_notes_3[replaced['x'][i]] = combo_dictionary[coded_notes_2[replaced['x'][i]]]
            coded_notes_3[replaced['y'][i]] = combo_dictionary[coded_notes_2[replaced['y'][i]]]
        except:
            raise NameError('Release notes are not in combination dictionary')
    
    return coded_notes_3


def map_notes_0(note_array, note_type):
    '''
    map_notes_0 maps a note array from the initial representation to an intermediate
    representation that can be processed later into a full one hot representation.
    The note_array should be preprocessed so that force and tap flags are removed.
    This function is utilized by chart2onehot()
    - note_array = array of notes
    - type = 'regular', 'force', or 'tap'
    '''
    assert note_type in ['regular', 'force', 'tap'], 'note_type should be "regular", "force", or "tap"'
    
    for i in range(len(note_array)):
        if note_array[i] == 0:
            note_array[i] = 10
        elif note_array[i] == 10:
            note_array[i] = 13
        elif note_array[i] == 20:
            note_array[i] = 16
        elif note_array[i] == 1:
            note_array[i] = 17            
        elif note_array[i] == 11:
            note_array[i] = 20
        elif note_array[i] == 21:
            note_array[i] = 23
        elif note_array[i] == 2:
            note_array[i] = 24
        elif note_array[i] == 12:
            note_array[i] = 27
        elif note_array[i] == 22:
            note_array[i] = 30
        elif note_array[i] == 3:
            note_array[i] = 31
        elif note_array[i] == 13:
            note_array[i] = 34
        elif note_array[i] == 23:
            note_array[i] = 37
        elif note_array[i] == 4:
            note_array[i] = 38
        elif note_array[i] == 14:
            note_array[i] = 41
        elif note_array[i] == 24:
            note_array[i] = 44
        elif note_array[i] == 7:
            note_array[i] = 45
        else:
            raise NameError('Error: note encoded incorrectly')

        if note_type == 'regular':
            continue
        elif note_type == 'force':
            note_array[i] += 1
        elif note_type == 'tap':
            note_array[i] += 2

    return note_array


def generate_combo_dict():
    '''
    Generates a dictionary mapping note events to simple one hot representations.
    This function is utilized by chart2onehot()
    - keys = combinations strings
    - values = condensed one hot representation
    '''
    # To one hot encode, we need to generate a list of all the possible unique values each note event could take on
    g = list(range(10,17))
    r = list(range(17,24))
    y = list(range(24,31))
    b = list(range(31,38))
    o = list(range(38,45))

    note_vals = [g, r, y, b, o]
    note_vals = np.array(note_vals)

    rr = note_vals[:,0]  # regular regular
    rf = note_vals[:,1]  # regular forced
    rt = note_vals[:,2]  # regular tapped
    hr = note_vals[:,3]  # held regular
    hf = note_vals[:,4]  # held forced
    ht = note_vals[:,5]  # held tapped
    release = note_vals[:,6]  # release
    note_combos = [rr, rf, rt, hr, hf, ht, release]

    all_combinations = []  # Will hold all possible note combinations
    for combo_class in note_combos:
        for combo_length in range(1, len(combo_class)+1):
            for combo in list(combinations(combo_class, combo_length)):
                keystring = ''
                for element in combo:
                    keystring += str(element)
                all_combinations.append(keystring)

    # Add open notes
    all_combinations.append('45')
    all_combinations.append('46')
    all_combinations.append('47')

    combo_dictionary = {}
    for i in range(1, len(all_combinations)+1):
        combo_dictionary[all_combinations[i-1]] = i

    return combo_dictionary


def check_for_release_notes(x, coded_notes_2):
    '''
    Checks to see if there is a released note at tick x
    Utilized by chart2onehot()
    '''
    r_notes = ['16', '23', '30', '37', '44']
    r_in_x = []
    for r in (r_notes):
        if r in coded_notes_2[x]:
            r_in_x.append(r)

    if not r_in_x:
        return False
    else:
        return r_in_x


# FOR TESTING
#____________#

# Test chart2onehot()
#d = str(Path().resolve().parent)
#chartpath = d+'\\tensor-hero\Preprocessing\Chart Files\degausser_notes.chart'
#coded_notes = chart2onehot(chartpath, print_release_notes = True)
#print(coded_notes)

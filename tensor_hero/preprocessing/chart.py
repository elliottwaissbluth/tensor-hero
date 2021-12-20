from pathlib import Path
import numpy as np
from itertools import combinations

'''
Contains functions related to processing .chart files
'''

def chart2tensor(path, print_release_notes=False):
    '''
    Inputs a path to a chart and returns a tensor with 10ms ticks as indices and one hot
    representation as values.
    '''
    coded_notes = chart2onehot(path, print_release_notes)

    if coded_notes == None:
        print('\nThe chart at {} is not in .chart format'.format(path))
        return None

    notes_tensor = np.zeros(max(coded_notes.keys()))
    for k, v in coded_notes.items():
        notes_tensor[k-1] = v       # This could potentially offset all notes by 10ms
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
    try:
        with open(path, 'r') as file:
            raw_chart = file.readlines()
        file.close()
    except:  # This will happen when the chart is not in .chart format
        return None, None, None, None

    # Strip lines of \n character
    for i in range(len(raw_chart)):
        raw_chart[i] = raw_chart[i].replace('\n', '')

    # Create lists to hold sections of the chart file
    song = []
    synctrack = []
    expertsingle = []

    # Parse chart file, populating lists
    i = 0
    for data in raw_chart:
        if '[Song]' in data:
            i = 1
        elif data == '[SyncTrack]':
            i = 2
        elif data in ['[Events]', '[EasySingle]', '[MediumSingle]','[HardSingle]']:
            i = 3
        elif data == '[ExpertSingle]':
            i = 4
        
        if data in ['{', '}', '[Song]', 'ï»¿[Song]', '[SyncTrack]', '[Events]', '[ExpertSingle]']:
            continue
        if data[0] == ' ':
            if i == 1:
                song.append(data[2:])
            elif i == 2:
                synctrack.append(data[2:])
            elif i == 3:
                continue
            elif i == 4:
                expertsingle.append(data[2:])
        else:
            if i == 1:
                song.append(data[0:].strip('\t'))
            elif i == 2:
                synctrack.append(data[0:].strip('\t'))
            elif i == 3:
                continue
            elif i == 4:
                expertsingle.append(data[0:].strip('\t'))

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
            try:
                BPMs['tick'].append(int(line[0]))
                BPMs['BPM'].append(int(line[3]))
            except:
                # print(line)
                # print(expertsingle)
                # print(time_signatures)
                raise NameError('Chart file may have improper indentation')

    # Parse the 'expertsingle' section of the .chart file for note information
    notes = {'tick' : [],       # What tick the note is at
            'N_S' : [],         # Whether it is a note (N) or star power (S)
            'note' : [],        # What the note is 
            'duration': []}     # tick duration of the note or star power

    for note in expertsingle:
        line = note.split(' ')
        if len(line) < 2:
            break
        if line[2] == 'E':       # skip the lines that mark events
            continue
        else:
            if "=" in line:
                notes['tick'].append(int(line[0]))
                notes['N_S'].append(line[2])
                notes['note'].append(line[3])
                notes['duration'].append(int(line[4]))
            else:
                notes['tick'].append(int(line[0]))
                notes['N_S'].append(line[1])
                notes['note'].append(line[2])
                notes['duration'].append(int(line[3]))

    # Parse the 'song' section of the .chart file to get relevent chart information
    song_metadata = {'Name' : '',
                        'Artist' : '',
                        'Charter' : '',
                        'Offset' : '',
                        'Resolution' : '',
                        'Genre' : '',
                        'MediaType' : '',
                        'MusicStream' : ''}
    if song:
        for data in song:
            line = data.split(' ')
            if line[0] in song_metadata.keys():
                song_metadata[line[0]] = line[-1]
        song_metadata['Offset'] = int(float(song_metadata['Offset']))
        song_metadata['Resolution'] = int(float(song_metadata['Resolution']))
    else:
        print('song metadata lost')
        print(song)
        song_metadata['Resolution'] = 192

    return notes, song_metadata, time_signatures, BPMs

def shift_ticks(notes, song_metadata, time_signatures, BPMs):
    '''
    shift_ticks converts the time signatures to 31.25 BPM w/ TS 4, i.e. the ticks are
    shifted to correspond to 10ms slots.
        - Parameters come from chart2dict 
    '''
    # Split the song into bins corresponding to particular time signatures and BPMs

    # First, assemble some lists from the preprocessing step
    note_keys = list(zip(notes['tick'], notes['N_S'], 
                    notes['note'], notes['duration']))                    # (tick, 'N_S', note)
    TS_events = list(zip(time_signatures['tick'], time_signatures['TS']))  # (tick, TS)
    BPM_events = list(zip(BPMs['tick'], BPMs['BPM']))                      # (tick, BPM)

    # Append None at the end of these lists so the loop knows where to stop
    TS_events.append(None)
    BPM_events.append(None)

    # Loop through all the notes in the song
    TS_index = 0
    BPM_index = 0

    cur_TS = TS_events[TS_index]                # Current time signature
    cur_BPM = BPM_events[BPM_index]             # Current BPM
    next_TS = None                              # Next time signature
    next_BPM = None                             # Next BPM
    if len(TS_events) > 1:
        next_TS = TS_events[TS_index + 1]
    if len(BPM_events) > 1:
        next_BPM = BPM_events[BPM_index + 1]


    # bins['TS'][0] corresponds to the time signature of bin 0
    # bins['notes'] is a list of lists of notes in each bin
    bins = {
        'TS' : [],              # time signature
        'BPM' : [],             # BPM
        'shift_tick' : [],      # The first tick where the TS / BPM combo starts
        'notes' : [[]],         # The notes in the bin
    }

    # Append the first element of each array before looping
    event_index = 0     # Counts how many times either BPM or TS change
    bins['TS'].append(cur_TS[1])
    bins['BPM'].append(cur_BPM[1])
    bins['shift_tick'].append(cur_BPM[0])
    bins['notes'][event_index].append(note_keys[0])

    # Initialize ticks
    cur_TS_tick = cur_TS[0]
    if next_TS != None:
        next_TS_tick = next_TS[0]
    else:
        next_TS_tick = None
    cur_BPM_tick = cur_BPM[0]
    if next_BPM != None:
        next_BPM_tick = next_BPM[0]
    else:
        next_BPM_tick = None

    for i in range(1, len(note_keys)):
        if next_BPM_tick == None and next_TS_tick == None:     # If in the last bin
            bins['notes'][-1].append(note_keys[i])             # Add notes until there are no more to add
            continue
        
        if next_TS_tick != None:                        # If there is a time signature change in the future
            if note_keys[i][0] >= next_TS_tick:         # If the current note is past that change                            
                if next_BPM_tick != None:                   # If there is a BPM change in the future
                    if note_keys[i][0] >= next_BPM_tick:    # If the current note is past that change
                        TS_index += 1                       # Update time signature and BPM, they changed at the same time
                        cur_TS = TS_events[TS_index]
                        cur_TS_tick = cur_TS[0]
                        next_TS = TS_events[TS_index + 1]
                        if next_TS != None:
                            next_TS_tick = next_TS[0]
                        else:
                            next_TS_tick = None

                        BPM_index += 1
                        cur_BPM = BPM_events[BPM_index]
                        cur_BPM_tick = cur_BPM[0]
                        next_BPM = BPM_events[BPM_index + 1]
                        if next_BPM != None:
                            next_BPM_tick = next_BPM[0]
                        else:
                            next_BPM_tick = None

                        bins['TS'].append(cur_TS[1])
                        bins['BPM'].append(cur_BPM[1])
                        bins['shift_tick'].append(min(cur_TS[0], cur_BPM[0]))
                        bins['notes'].append([])
                        bins['notes'][-1].append(note_keys[i])
                        continue

                    else:                                   # If the time signature changed but the BPM didn't
                        TS_index += 1                       # Update the time signature, but not the BPM
                        cur_TS = TS_events[TS_index]
                        cur_TS_tick = cur_TS[0]
                        next_TS = TS_events[TS_index + 1]
                        if next_TS != None:
                            next_TS_tick = next_TS[0]
                        else:
                            next_TS_tick = None

                        bins['TS'].append(cur_TS[1])
                        bins['BPM'].append(cur_BPM[1])
                        bins['shift_tick'].append(min(cur_TS[0], cur_BPM[0]))
                        bins['notes'].append([])
                        bins['notes'][-1].append(note_keys[i])
                        continue

                else:                               # If the next BPM tick = None but the note tick is past the time signature
                    TS_index += 1                   # Update the time signature, but not the BPM
                    cur_TS = TS_events[TS_index]
                    cur_TS_tick = cur_TS[0]
                    next_TS = TS_events[TS_index + 1]
                    if next_TS != None:
                        next_TS_tick = next_TS[0]
                    else:
                        next_TS_tick = None       

                    bins['TS'].append(cur_TS[1])
                    bins['BPM'].append(cur_BPM[1])
                    bins['shift_tick'].append(cur_TS[0])
                    bins['notes'].append([])
                    bins['notes'][-1].append(note_keys[i])
                    continue

            else:  # If there is a time signature change in the future but the note is not past it
                if next_BPM_tick != None:                   # If there is a BPM change in the future
                    if note_keys[i][0] >= next_BPM_tick:    # If the note is past that BPM change    
                        BPM_index += 1                      # Update the BPM but not the time signature
                        cur_BPM = BPM_events[BPM_index]
                        cur_BPM_tick = cur_BPM[0]
                        next_BPM = BPM_events[BPM_index + 1]
                        if next_BPM != None:
                            next_BPM_tick = next_BPM[0]
                        else:
                            next_BPM_tick = None

                        bins['TS'].append(cur_TS[1])
                        bins['BPM'].append(cur_BPM[1])
                        bins['shift_tick'].append(cur_BPM[0])
                        bins['notes'].append([])
                        bins['notes'][-1].append(note_keys[i])
                        continue

                    else:  # If the time signature did not change and the BPM also did not change
                        bins['notes'][-1].append(note_keys[i])  # Add note and continue
                        continue

        #-------------------------------------------------------------------------------------------------------#
        # The second half of the ifzilla:
        # If there is not a time signature change in the future

        else:                        # If there is NOT a time signature change in the future                              
            if next_BPM_tick != None:                   # If there is a BPM change in the future
                if note_keys[i][0] >= next_BPM_tick:    # If the current note is past that change
                    BPM_index += 1                      # Update the BPM
                    cur_BPM = BPM_events[BPM_index]
                    cur_BPM_tick = cur_BPM[0]
                    next_BPM = BPM_events[BPM_index + 1]
                    if next_BPM != None:
                        next_BPM_tick = next_BPM[0]
                    else:
                        next_BPM_tick = None

                    bins['TS'].append(cur_TS[1])
                    bins['BPM'].append(cur_BPM[1])
                    bins['shift_tick'].append(cur_BPM[0])
                    bins['notes'].append([])
                    bins['notes'][-1].append(note_keys[i])
                    continue

                else:  # If the current note is not past the BPM change
                    bins['notes'][-1].append(note_keys[i])  # Add note and continue
                    continue

            else:                               # If the next BPM tick = None and the next TS tick = None
                                                # Then the if statement at the beginning of this thing should have fired off
                raise NameError('Error: Tick Conversion Failure')

    bins['X'] = []                # Tick conversion factor
    bins['sync_tick'] = []        # The tick value that the first note in 'notes' should have
    res = song_metadata['Resolution']
    BPM_new = int(60000 / (res*0.01))

    # Populate 'X' and 'sync_tick' field of bins
    for i in range(len(bins['shift_tick'])):
        bins['X'].append(BPM_new / bins['BPM'][i])  # 31250 = 31.25 beats/minute * 1000, this BPM corresponds to 10ms per tick

        if i == 0:
            bins['sync_tick'].append(0)
        else:
            bins['sync_tick'].append(round(bins['notes'][i][0][0] * bins['X'][i-1]))

    # Create array in bins for 'shift'
    bins['shift'] = [0]  # Don't shift the first bin

    # Convert note ticks to using conversion factor X
    for i in range(len(bins['shift_tick'])):
        for j in range(len(bins['notes'][i])):
            bins['notes'][i][j] = list(bins['notes'][i][j])  # Convert to list
            
            # Construct shift length
            if j == 0 and i != 0:
                bins['shift'].append(round((bins['shift_tick'][i] * bins['X'][i])) - (round((bins['shift_tick'][i] * bins['X'][i-1])) - bins['shift'][i-1]))

            bins['notes'][i][j][0] = round(bins['notes'][i][j][0]*bins['X'][i])  # Scale tick mark
            bins['notes'][i][j][3] = round(bins['notes'][i][j][3]*bins['X'][i])  # Scale duration
            bins['notes'][i][j][0] -= bins['shift'][i]                           # Shift

    # Convert back to dictionary format
    shifted_notes = {'tick' : [],        # What tick the note is at
                    'N_S' : [],         # Whether it is a note (N) or star power (S)
                    'note' : [],        # What the note is 
                    'duration': []}     # tick duration of the note or star power

    for T_bin in bins['notes']:
        for n in T_bin:
            shifted_notes['tick'].append(n[0])
            shifted_notes['N_S'].append(n[1])
            shifted_notes['note'].append(n[2])
            shifted_notes['duration'].append(n[3])

    return shifted_notes


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
    notes, song_metadata, time_signatures, BPMs = chart2dict(path)
    notes = shift_ticks(notes, song_metadata, time_signatures, BPMs)

    if notes == None:  # If the chart file is not in .chart format
        return None

    # Create a dictionary where the keys are the tick values and the values
    # are a list of notes corresponding to the data
    coded_notes_0 = {}

    # Loop through song one note at a time, processing along the way
    for i in range(len(notes['tick'])):
        # note: This was originally implemented but star power indicators also have note information.
        # if notes['N_S'] == 'S':  # if the token is a star power indicator, skip it
        #     continue

        if notes['tick'][i] not in coded_notes_0:  # If the key is not in the dictionary
            coded_notes_0[notes['tick'][i]] = []                   # Create empty list

        if notes['duration'][i] == 0 or int(notes['note'][i]) == 5 or int(notes['note'][i]) == 6:  # If the note is not held
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
        
        if 5 in coded_notes_0[notes['tick'][i]] and 6 in coded_notes_0[notes['tick'][i]]:  # Sometimes there are notes with conflicting force/tap flags. Set these to tap flags
            # print('Force AND tap flag at tick {}'.format(notes['tick'][i]))
            coded_notes_0[notes['tick'][i]].remove(5)

        if coded_notes_0[notes['tick'][i]] != list(set(coded_notes_0[notes['tick'][i]])):  # Check for duplicate values
            if len(coded_notes_0[notes['tick'][i]]) > 2:
                # print('Duplicate notes present in chart at tick {}'.format(notes['tick'][i]))
                # print('Old notes: ', coded_notes_0[notes['tick'][i]])
                coded_notes_0[notes['tick'][i]] = list(set(coded_notes_0[notes['tick'][i]]))
                # print('New notes: ', coded_notes_0[notes['tick'][i]])

    # coded_notes_1 will hold intermediate values of coded_notes
    coded_notes_1 = {}

    for x in coded_notes_0.keys():
        if 5 in coded_notes_0[x]:    # If a force note
            # print('coded_notes_0[x]: ', coded_notes_0[x])
            coded_notes_0[x].remove(5)
            # print('coded_notes_0[x]: ', coded_notes_0[x])
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
        if sorted(coded_notes_1[x]) != coded_notes_1[x]:
            # print('Notes are unsorted, sorting now')
            # print('original: ', coded_notes_1[x])
            coded_notes_1[x] = sorted(coded_notes_1[x])
            # print('sorted_notes: ', coded_notes_1[x])
        for note_event in coded_notes_1[x]:
            notestring += str(note_event)  
            # print(notestring)
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
                    # print('coded_notes_2[x]: ', coded_notes_2[x])
                    if not code:
                        code = digit
                        continue
                    if len(code) < 2:
                        code += digit
                    else:
                        code = digit
                        continue
                    if code in ['16', '23', '30', '37', '44']:
                        if code not in release_digits:
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
                    print('Full replaced dictionary: ', replaced)
                

    for i in range(len(replaced['x'])):
        coded_notes_2[replaced['x'][i]] = replaced['replacement_digits'][i]
        coded_notes_2[replaced['y'][i]] = replaced['release_digits'][i]
        try:
            if replaced['replacement_digits'][i] and replaced['release_digits'][i]:  # If the error was flagged as a new note coinciding with a release
                coded_notes_3[replaced['x'][i]] = combo_dictionary[coded_notes_2[replaced['x'][i]]]
                coded_notes_3[replaced['y'][i]] = combo_dictionary[coded_notes_2[replaced['y'][i]]]
            if replaced['replacement_digits'][i] and not replaced['release_digits'][i]:  # If the error was flagged as a new note coinciding with another new note
                codes = []
                code = ''
                for digit in replaced['replacement_digits'][i]:
                    # print('replaced[\'replacement_digits\'][i]: ', replaced['replacement_digits'][i])
                    if not code:
                        code = digit
                        continue
                    if len(code) < 2:
                        code += digit
                        codes.append(int(code))
                    else:
                        code = digit
                        continue
                code = max(codes)
                # print('max code = ', code)
                replaced['replacement_digits'][i] = str(code)
                coded_notes_2[replaced['x'][i]] = replaced['replacement_digits'][i]
                coded_notes_3[replaced['x'][i]] = combo_dictionary[coded_notes_2[replaced['x'][i]]]
                # coded_notes_3[replaced['y'][i]] = combo_dictionary[coded_notes_2[replaced['y'][i]]]
        except:
            try:
                codes = []
                code = ''
                for digit in replaced['replacement_digits'][i]:
                    # print('replaced[\'replacement_digits\'][i]: ', replaced['replacement_digits'][i])
                    if not code:
                        code = digit
                        continue
                    if len(code) < 2:
                        code += digit
                        codes.append(int(code))
                    else:
                        code = digit
                        continue
                code = max(codes)
                # print('max code = ', code)
                replaced['replacement_digits'][i] = str(code)
                coded_notes_2[replaced['x'][i]] = replaced['replacement_digits'][i]
                coded_notes_3[replaced['x'][i]] = combo_dictionary[coded_notes_2[replaced['x'][i]]]
                # coded_notes_3[replaced['y'][i]] = combo_dictionary[coded_notes_2[replaced['y'][i]]]
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
    
    # Behold, the great elif statement that should have been a dictionary.
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
        elif note_array[i] == 17:  # Added these last two as a patch - need to check
            note_array[i] = 48
        elif note_array[i] == 27:
            note_array[i] = 51
        else:
            print(note_array)
            print('The erroneous note in note_array at index {} is {}'.format(i, note_array[i]))
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
    all_combinations.append('48')
    all_combinations.append('49')
    all_combinations.append('50')
    all_combinations.append('51')

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
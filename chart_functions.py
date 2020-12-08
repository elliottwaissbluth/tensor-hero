# This file contains function definitions related to processing .chart files


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

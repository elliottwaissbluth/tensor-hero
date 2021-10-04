''' samantha's helper functions'''

def simplify_ohc_notes(path):
   notes_array = np.load(path)
   notes_array = notes_array.astype(int)
   for i in range(0,len(notes)):
    if notes[i] in [31.62,93,124,155,186,217]:
        notes[i] = 31
   mod_array = notes_array % 31
   return mod_array 



#turn simplified 1d notes array into 31x400 matrix
#each row in matrix corresponds to note value 0-31, one hot encoded so that on each timestep we know which note is played
def create_matrix(array):
    nb_classes = 31
    targets = np.array([array]).reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]
    return one_hot_targets.T


#create consecutive 431 dimensional arrays for each timestep/note combination in 400ms, outputs 400 431-d arrays
def make_time_and_note_array(matrix):
    rotate = matrix.T.astype(int)
    output = []
    #create consecutive array inputs consisting of a 400 dim timestep one-hot array + 31 note one-hot
    for i in range(0,rotate.shape[0]):
        time = np.zeros(400).astype(int)
        time[i] = 1
        new = np.concatenate((time,rotate[i]))
        output.append(new)
    output = np.array(output)
    return output

#create list of paired values (time index and note index)
def create_pairs(input_array_seq):
    list_pairs = []
    for array in input_array_seq:
        pairs = []
        for i in len(array):
            if i != 0:
                pairs.append(i)
        list_pairs.append(pairs)
    return list_pairs


#going back

#takes output format (400 arrays of 431d) and returns 31x400 matrix of note,timestep
def output_to_matrix(output):
    #keep only the last 31 columns of the matrix (first 400 are timestep one-hots)
    return output[:,400:].T

#convert 31note x 400ms matrix and compress down into notes array 1 x 400 with values 0-31 
def matrix_to_notesarray(matrix):
    #multiply by its index to change the one-hot value to 0-31 for each row
    a = [matrix[i] * i for i in range(0,len(matrix))]
    a = np.array(a)
    #return the verticle sum across all columns
    return a.sum(axis = 0)
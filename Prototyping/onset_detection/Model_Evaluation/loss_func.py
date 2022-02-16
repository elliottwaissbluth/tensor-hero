import numpy as np


def weights_matrix(time_note_weight = 1.5, weights = [1.2,1.3,1.4,1.5]):

    """
    #returns  weight matrix needed for multiplication against output
    #weights incorporate punishment for predicting timestep on note, note on timestep
    #punishes wrong note-type prediction

    ~~~~ ARGUMENTS ~~~~
    - weights : list of weights
        - length 4
        - for single,double,triple, and quad degree incorrectness
    
    ~~~~ RETURNS ~~~~
    - 431x431 weights matrix
    """

    w = np.ones((431,431))

    #punish timesteps that were predicted as notes
    w[0:400, 400:] = time_note_weight
    
    #punish notes that were predicted as timesteps
    w[400:, 0:400] = time_note_weight

    #punish wrong notes
    #single degree
    w[400:405, 405:415] = weights[0]
    w[405:415, 400:405] = weights[0]
    w[405:415, 415:425] = weights[0]
    w[415:425, 405:415] = weights[0]
    w[415:425, 425:430] = weights[0]
    w[425:430, 415:425] = weights[0]
    w[430,     425:430] = weights[0]
    w[425:430,     430] = weights[0]    

    #second degree
    w[400:405, 415:425] = weights[1]
    w[415:425, 400:405] = weights[1]
    w[405:415, 425:430] = weights[1]
    w[425:430, 405:415] = weights[1]
    w[415:425,     430] = weights[1]
    w[430,     415:425] = weights[1]

    #third degree
    w[400:405, 425:430] = weights[2]
    w[425:430, 400:405] = weights[2]
    w[430,     405:415] = weights[2]
    w[405:415,     430] = weights[2]

    #fourth degree
    w[430,     400:405] = weights[3]
    w[400:405,     430] = weights[3]

    return w
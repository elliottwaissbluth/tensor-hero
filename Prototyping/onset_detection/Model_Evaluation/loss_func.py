import numpy as np
import torch

def weights_matrix(time_note_weight = 1.5, open_weight = 1, weights = [1.2,1.3,1.4,1.5]):

    """
    #returns  weight matrix needed for multiplication against output
    #weights incorporate punishment for predicting timestep on note, note on timestep
    #punishes wrong note-type prediction

    ~~~~ ARGUMENTS ~~~~
    - weights : list of weights
        - length 4
        - for single,double,triple, and quad degree incorrectness
    
    ~~~~ RETURNS ~~~~
    - 435x435 weights matrix
    """

    w = np.ones((435,435))

    #punish predictions for notes that should be times
    w[:32, 32:] = time_note_weight
    
    #punish predictions for times that should be notes
    w[32: , :32] = time_note_weight

    #punish wrong notes
    #single degree
    w[ :5  ,  5:15] = weights[0]
    w[5:415,   :5 ] = weights[0]
    w[5:15 , 15:25] = weights[0]
    w[15:25,  5:15] = weights[0]
    w[15:25, 25:30] = weights[0]
    w[25:30, 15:25] = weights[0]
    w[30   , 25:30] = weights[0]
    w[25:30,   30] = weights[0]    

    #second degree
    w[  : 5, 15:25] = weights[1]
    w[15:25,   :5 ] = weights[1]
    w[ 5:15, 25:30] = weights[1]
    w[25:30,  5:15] = weights[1]
    w[15:25,    30] = weights[1]
    w[30,    15:25] = weights[1]

    #third degree
    w[  : 5, 25:30] = weights[2]
    w[25:30,   : 5] = weights[2]
    w[30   ,  5:15] = weights[2]
    w[5:15,    430] = weights[2]

    #fourth degree
    w[30,     0:5] = weights[3]
    w[ :5,     30] = weights[3]

    #open note
    w[31, :31] = open_weight
    w[:31, 31] = open_weight
    return w


def calculate_loss(truth_tensor,preds_tensor,weights = weights_matrix()):
    """
    #returns  penalty for total loss in prediction

    ~~~~ ARGUMENTS ~~~~
    - truth_tensor : true notes
        - torch.Size([batch_size, 499])
    - preds_tensor : prediction output from batch
        - torch.Size([batch_size, 499, 435])

    ~~~~ RETURNS ~~~~
    - single loss value
    """
    #create one-hot of truth values
    truth_one_hot = torch.nn.functional.one_hot(truth_tensor, num_classes= -1)
    
    #subtract guesses from truth
    incorrectness = truth_one_hot - preds_tensor

    #construct weight matrix for true values
    weight_matrix = weights[truth_tensor,:]
    weight_matrix = torch.tensor(weight_matrix)

    #multiply weight matrix by incorrectness, take absolute value for total loss 
    loss = abs(torch.mul(weight_matrix,incorrectness))
    loss_avg = torch.mean(loss)
    
    return loss_avg








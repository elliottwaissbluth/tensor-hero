"""error metrics for model1"""
import numpy as np
import matplotlib.pyplot as plt

#saturation - how many total notes did our model predict
#returns ratio of output to truth (could be more or less than 1)
def freq_saturation(truth,output):
    truth_nonzero = np.count_nonzero(truth)
    output_nonzero = np.count_nonzero(output)
    return output_nonzero/truth_nonzero


#returns plotted histogram of distribution of note types
#requires notes array be of value 0-32
def freq_histogram(truth, output):
    true = np.histogram(truth, bins = np.arange(0,34))
    observed = np.histogram(output, bins = np.arange(0,34))
    # Position of bars on x-axis
    ind = np.arange(0,33)

    ticks = [str(val) for val in ind]
    # Figure size
    plt.figure(figsize=(10,5))

    # Width of a bar 
    width = 0.3       

    # Plotting
    plt.bar(ind, true[0] , width,log = True, label='Truth')
    plt.bar(ind + width, observed[0], width,log = True, label='Observed')

    plt.xticks(ind + width / 2, ticks)
    plt.legend()
    plt.show();


#returns plotted histogram of distribution of note GROUP types (ex: single note, double note)
#requires notes array be of value 0-32
def type_freq_hit(truth, output):
    true = np.histogram(truth, bins = np.arange(0,34))
    observed = np.histogram(output, bins = np.arange(0,34))
    # Position of bars on x-axis
    ind = np.array([0,1,2,3,4,5,6])

    ticks = ['None', 'Single', 'Double', 'Triple', 'Four', 'Five', 'Open']
    # Figure size
    plt.figure(figsize=(10,5))

    # Width of a bar 
    width = 0.3       

    # Plotting
    plt.bar(ind, true[0] , width,log = True, label='Truth')
    plt.bar(ind + width, observed[0], width,log = True, label='Observed')
    plt.legend()
    plt.xticks(ind,ticks)
    plt.show();
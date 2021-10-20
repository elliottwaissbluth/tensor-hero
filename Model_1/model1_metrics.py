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


def freq_table(truth, output):
    true = np.histogram(truth, bins = np.arange(0,34))
    observed = np.histogram(output, bins = np.arange(0,34))
    data = np.array((true[0], true[0]/np.sum(true[0]), observed[0], observed[0]/np.sum(true[0])))
    df = pd.DataFrame(data = data.T, columns= ['Truth', '% of total','Observed','% of total'])

    return df


#returns plotted histogram of distribution of note GROUP types (ex: single note, double note)
#requires notes array be of value 0-32
def type_freq_hit(truth, output):
    true = np.histogram(truth, bins = np.arange(0,34))
    observed = np.histogram(output, bins = np.arange(0,34))
    # Position of bars on x-axis
    ind = np.array([0,1,2,3,4,5,6])

    ticks = ['None', 'Single', 'Double', 'Triple', 'Quad', 'Five', 'Open']
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


def freq_type_table(truth, output):
    true = np.histogram(truth, bins = [0,1,6,16,26,31,31,33])
    observed = np.histogram(observed, bins = [0,1,6,16,26,31,31,33])
    data = np.array((true[0], true[0]/np.sum(true[0]), observed[0], observed[0]/np.sum(true[0])))
    # Position of bars on x-axis
    df = pd.DataFrame(data = data.T, columns = ['Truth', '% of total','Observed','% of total'], index= ['None', 'Single', 'Double', 'Triple', 'Quadruple', 'Five', 'Open'])

    return df
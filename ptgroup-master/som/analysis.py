from root_numpy import root2array, tree2array, array2root, testdata
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import datetime
from som import SOM

dimx = 10 # set the x dimension of the SOM
dimy = 10 # set the y dimension of the SOM

# get the data from the montecarlo (must be from root format and separated signal and bg)
sigroot = root2array('/home/ptgroup/pacordova/code/bin/Training_Trees.root', 'TreeS') # path to sig rootfile
bgroot = root2array('/home/ptgroup/pacordova/code/bin/Training_Trees.root', 'TreeB') # path to bg rootfile

##### TRAINING #####
### PREPARE THE SIGNAL ARRAY ###
# signal and background arrays should be of the same shape with the same variables, they will be concatenated later
print(str(datetime.datetime.now())) # Print time, useful for determining how long it takes to train the self-organizing map
sig = np.array(sigroot.tolist()) # convert to np array
np.random.shuffle(sig)
sig2 = sig
backupsig = sig2
sig = sig[:10000,:] # cut the np array to preferred size
flag_sig = sig[:,35] # array of ones for signal, flags each event as signal
sig = sig[:,[24,25,26,29,30,31,32,34]] # cuts the signal array to only include variables you want to train on
### PREPARE THE BACKGROUND ARRAY ###
bg = np.array(bgroot.tolist()) # convert to np array
np.random.shuffle(bg)
bg2 = bg
backupbg = bg2
bg = bg[:10000,:] # cut the np array to preferred size
flag_bg = bg[:,35] # array of zeros for background, flags each event as background 
bg = bg[:,[24,25,26,29,30,31,32,34]] # cuts the background array to only include variables you want to train on
# finish preparing the final dataset (one big numpy array)
data = np.concatenate((sig,bg),axis=0) # concatenate the sig and bg numpy arrays into one big array
flags = np.concatenate((flag_sig, flag_bg), axis=0) # concatenates the flag arrays into one array, each entry corresponding to the entry of the data array
### TRAINING ###
som = SOM(dimx, dimy, 8, 400) # Train a dimx X dimy SOM with 400 iterations, 8 is the number of variables in the data array
som.train(data) # trains on the dataset prepared
### THIS WILL TAKE AWHILE ###
### TO STORE THE TRAINING RESULTS INTERACTIVELY IN ipython DO: 
### weightages = som._weightages 
### %store weightages 
### THEN TO RECOVER DO: 
### %store -r 
### som = SOM(dimx, dimy, 8, 400)
### som._weightages = weightages
### som._trained = True
print(str(datetime.datetime.now())) # print the time to observe how long the training will take
mapped = np.array(som.map_vects(data)) # map each datapoint to its nearest neuron
# post training manipulations to the final dataset (one big numpy array)
data = np.append(data,mapped,axis=1) # append the mapped neurons to data
data = np.column_stack((data,flags)) # append the flags to the dataset 

##### APPLICATION #####
# get new data
### PREPARE THE SIGNAL ARRAY ###
sig2 = sig2[15000:,:] # make a different cut on the np array for different datapoints
flag_sig2 = sig2[:,35] # array of ones for signal, flags each event as signal
# VARS MUST BE SAME AS FROM TRAINING
sig2 = sig2[:,[24,25,26,29,30,31,32,34]] # cuts the signal array to only include variables being used
### PREPARE THE BACKGROUND ARRAY ###
bg2 = np.array(bgroot.tolist()) # convert to np array
bg2 = bg2[15000:,:] # make a different cut on the np array for different datapoints
flag_bg2 = bg2[:,35] # array of zeros for background, flags each event as background
# VARS MUST BE SAME AS FROM TRAINING
bg2 = bg2[:,[24,25,26,29,30,31,32,34]] # cuts the signal array to only include variables being used
# finish preparing the final dataset (one big numpy array)
data2 = np.concatenate((sig2,bg2),axis=0) # concatenate the sig and bg numpy arrays into one big array
flags2 = np.concatenate((flag_sig2, flag_bg2), axis=0) # concatenates the flag arrays into one array, each entry corresponding to the entry of the data array
mapped2 = np.array(som.map_vects(data2)) # map each datapoint to its nearest neuron
data2 = np.append(data2,mapped2,axis=1) # append the mapped neurons to data
data2 = np.column_stack((data2,flags2)) # append the flags to the dataset 
### CREATE THE COUTING LOOPS AND ARRAYS TO DRAW CHARTS ###
count = np.empty([dimx,dimy]) # create a count array of the number of data points associated with each neuron
count_flag = np.empty([dimx,dimy]) # create a signal count array, each element is equal to: #sig/#total
count_flag_bg = np.empty([dimx,dimy]) # create a background count array, each element is equal to: #bg/#total
# fill the count array with the counts
for i in range(dimx):
    for j in range(dimy):
        count[i,j] = data2[np.where((data2[:,8] == float(i)) & (data2[:,9] == float(j))),10].size
# fill the count_flag array with #signal/#total (it is actually #total+1 to avoid dividing by zero)
for i in range(dimx):
    for j in range(dimy):
        count_flag[i,j] = data2[np.where((data2[:,8] == float(i)) & (data2[:,9] == float(j))),10].sum().astype(int)/(count[i,j]+1)
# fill the count_flag_bg array with #background/#total (it is actually #total+1 to avoid dividing by zero)
for i in range(dimx):
    for j in range(dimy):
        count_flag_bg[i,j] = (count[i,j] - data2[np.where((data2[:,8] == float(i)) & (data2[:,9] == float(j))),10].sum().astype(int))/(count[i,j]+1)
### PLOTTING THE CHARTS ###
plt.imshow(count, cmap='hot', interpolation='nearest') # creates a heatmap of the data counts associated with each neuron
plt.colorbar() # plots the color bar on the heatmap
plt.show() # shows the map
### RANKING THE NEURONS FROM THE ORIGINAL DATASET ###
df = pd.DataFrame(data) # convert the np array data to a dataframe
df['coords'] = df[8].astype('str') + df[9].astype('str') # create a coords column in df, equal to the cartesian coordinate that corresponds to each neuron in 2d grid
df2 = pd.DataFrame(data2) # convert the np array data2 to a dataframe
df2['coords'] = df2[8].astype('str') + df2[9].astype('str') # create a coords column in df2, equal to the cartesian coordinate that corresponds to each neuron in 2d grid
neurons = pd.DataFrame(df[10].groupby(df['coords']).mean()) # finds the purity of each neuron trained on the OLD dataset
neurons.columns = ['purity'] # renames the column to 'purity'
neurons['signal'] = df2[10].groupby(df2['coords']).sum() # creates a signal count column, counting the number of signal entries in the NEW dataset corresponding to each neuron
neurons['background'] = (df2[10]*-1+1).groupby(df2['coords']).sum() # creates a background count column, counting the number of background entries in the NEW dataset corresponding to each neuron
neurons = neurons.fillna(value=0) # fills the NaN with zeroes
### PLOT THE PURITY VS CUT THRESHOLD ###
plt.hist(neurons['purity'], bins=50, weights=neurons['signal'], alpha=0.5, label='sig') # plots the signal counts in the corresponding x coordinate
plt.hist(neurons['purity'], bins=50, weights=neurons['background'], alpha=0.5, label='bg') # plots the background counts in the corresponding x coordinate
plt.legend(loc='upper left') # creates a legend
plt.show() # show the plot

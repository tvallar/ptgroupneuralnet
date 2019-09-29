# ptgroupneuralnet

In this repository are all the files needed to run the network
All the code is in the Compton_FF_Code folder. The most up to date network is Curve_Fitting_Network_Single.py, it is designed to be trained on a single set of kinematic parameter values to produce the correct ReH, ReE, and ReHTilde for those kinematic parameters and measured Cross Section values
To run the network, whether to load an older network or train a new one, run run_network.py.
It will prompt for input in the terminal, asking what kind of operation you would like done whether it be (1) training a new group of networks (will prompt for the number of networks to create) (2) load a group of older, already trained networks or (3) exit

# Important note:
The network that is being used is Curve_Fitting_Network_Single.py
This is the code that is commented.
The other versions of this such as Curve_Fitting_Network and Curve_fitting_network_2 are older and function in a similar way but are not commented

The way the data is formatted is that each set of kinematic parameters has 36 points in a curve. When training new networks, the code will ask for a line number. By line number it is referring to which set of 36 points, and thus the kinematic variable values, to use when training. Look at the data file DVCS_cross.csv to see this better.

# Saving Networks:
The networks will automatically save themselves once they are finished training. These will be saved into the networks folder.

# Loading Networks:
To load a network, enter the name of the network without any file extension when prompted. Also do not include the -# at the end as the loading process will append the network number on the end as it loads each of the N networks requested.
Example: the network name is network-v1. If you trained 10 then they would save as network-v1-0.txt, network-v1-1.txt, etc.
When loading these networks, you will just enter 10 when prompted for the number, and the name should be entered as network-v1

# Running the code:
To run the code run run_network.py
When loading the code into a workspace, make sure the top level directory is the ptgroupneuralnet folder, not the Compton_FF_Code folder. Otherwise the code will most likely not be able to open the saved networks folder.
The python libraries needed are scipy, pandas, numpy, and matplotlib (only for plotting).


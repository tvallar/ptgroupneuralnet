# Importing python packages
import numpy as np
import os
import pandas as pd
import math
from pandas.plotting import scatter_matrix # optional
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
# My classes
import Curve_Fitting_Network as CFN
import Curve_Fitting_Network_2 as CFN2
import Curve_Fitting_Network_Single as CFNS
import BHDVCS as BHDVCS


# This is the main file for the network


def rmse(estimated, actual):
    n = len(estimated)
    return (1/n)*np.sqrt(np.sum((estimated-actual)*(estimated-actual)))

def calculate_observable(data, par0, par1, par2):
    x, x_b, t, Q = data
    M_p = 0.938 #GeV
    #-1/(par[3]*par[3]*par[4]*(1+2*par[3]*M_p*2*par[3]*M_p/(par[5]*par[5]))*(1+2*par[3]*M_p*2*par[3]*M_p/(par[5]*par[5])))*(par[0]
    #+ par[1]*cos(x[0]) + par[2]*cos(x[0]*x[0]));
    return -1/(x_b*x_b*t*(1+2*x_b*M_p*2*x_b*M_p/(Q*Q))*(1+2*x_b*M_p*2*x_b*M_p/(Q*Q)))*(par0 + par1*np.cos(x) + par2*np.cos(x*x))

def uniform_sample(val, error):
    return np.random.uniform(low=val-error, high=val+error)


### Returns the results of network prediction in a format that is easily
# used in a matplotlib python plot
# The line value in this case is used to determine the set of points to be graphed, where
# num_points is the number of points per curve in the data file
# Example: line_value = 10, num_points = 36, then this function would return
# the predicted values for points between 360 and 396.
def get_graph_arrays(line_value, x_axis, model):
        line1 = line_value
        #x_axis = np.linspace(0, 6, num=100)
        x_b1 = np.zeros((len(x_axis))) + x_b[line1*num_points]
        t_1 = np.zeros((len(x_axis))) + t[line1*num_points]
        Q_1 = np.zeros((len(x_axis))) + Q[line1*num_points]
        k_vals_1 = np.zeros((len(x_axis)))+k_vals[line1*num_points]
        data1 = (x_axis, x_b1, t_1, Q_1)
        y_vals1 = []
        model_curve1 = []
        params_returns = []
        for i in range(len(x_axis)):
            params_tmp, y_tmp_val = model.predict(np.array([[x_b1[i]], [t_1[i]], [Q_1[i]], [k_vals_1[i]]]), [x_axis[i]])
            #print(params_tmp)
            #print(X[i+line1*num_points])
            model_curve1.append(y_tmp_val)
            if i==0:
                print(params_tmp[0][0],' ', params_tmp[1][0], ' ', params_tmp[2][0])
                params_returns = [params_tmp[0][0], params_tmp[1][0], params_tmp[2][0]]

        return data1, model_curve1, params_returns

def get_mean_and_std(values):
    sum_tmp = 0.0
    n = len(values)
    for i in range(n):
        sum_tmp+=values[i]
    mean = sum_tmp/n
    std_sum_tmp = 0.0
    for i in range(n):
        std_sum_tmp += (values[i]-mean)*(values[i]-mean)
    std_dev = np.sqrt(std_sum_tmp/n)

    return mean, std_dev


# Generate set takes the experimental data and creates
# data generated using a normal distribution with the F value and F error
def generate_set(X_data, F, errF, other_data, y_params, num=-1, shuffle=True,):
    data_set = []
    tot = 0
    x_tot = len(X_data)
    if num==-1:
        tot=len(X_data)
    else:
        tot=num
    for i in range(tot):
        F_tmp = np.random.normal(F[i%x_tot], errF[i%x_tot])
        data_set.append((np.reshape(X_data[i%x_tot],(num_inputs,1)), np.reshape(np.array([F_tmp]),(1)), other_data[i%x_tot], np.reshape(y_params[i%x_tot], (num_outputs, 1))))
    if shuffle:
        np.random.shuffle(data_set)
    return data_set


###----------------------------------------------------------------------####

# Loading data
colors = ['b', 'g', 'r', 'c', 'm', 'y']
data_file_name = './Compton_FF_Code/DVCS_cross.csv'
data = pd.read_csv(data_file_name)
attributes =['X', 'X_b', 'Q', 't', 'F']


k_vals = np.array(data['k'])
x_b = np.array(data['x_b'])
Q = np.array(data['QQ'])
t = np.array(data['t'])
X = np.array(data['phi_x'])#*math.pi/180.0


F1 = np.array(data['F1'])
F2 = np.array(data['F2'])
ReH = np.array(data['ReH'])
ReE = np.array(data['ReE'])
ReHT = np.array(data['ReHTilde'])

F = np.array(data['F'])
errF = np.array(data['errF'])

# Formatting data
# In here X_data is what is being put into the network
# Y_data is the expected output
# other_data is other data needed for each training example to be able to calculate the predicted
# cross section value
X_data = []
y_data = []
y_data_params = []
other_data=[]
for i in range(len(x_b)):
    X_data.append([x_b[i], t[i]])#, Q[i], k_vals[i]])
    other_data.append([X[i], Q[i], k_vals[i], F1[i], F2[i]])
    y_data_params.append([ReH[i], ReE[i], ReHT[i]])

X_data = np.array(X_data)
other_data = np.array(other_data)

total_examples = 2000
train_num = 1500 # number of training examples to generate
test_num = total_examples-train_num

# num points is how many data points per curve in the data file
num_points = 36

## Model Parameters
num_inputs = 2
num_outputs = 3
learning_rate = 0.001
regularization_rate = 0.05 # larger it is, the smaller the parameters in the network will  be
iterations = 20
batch_size = 60 # The batch size determines how many points will be used when determining the best
# parameters for a given backpropagation and training iteration
# the larger the batch size, the more iterations that will be needed to shrink the windows
layers = [num_inputs, 20, num_outputs]

initial_parameter_ranges = [0.0, 1.5] # the window min and maximum that the parameters will be held between
# as training progresses these will shrink until they are around the 


#               Data stuff 
### ---------------------------------------------- ###

while(True):
    res = input('Do you want to: \n(1) retrain the network(s)\n(2) graph previous networks\n(3) Exit?\n')
    
    if res=='3':
        print('Exiting..')
        break
    
    num = int(input('How many networks would you like to train / load? Enter Integer: '))
    filen = input('Enter name for network cluster: ')
    
    networks = []
    train_sets = []
    test_sets = []
    network_results = []
    
    if res == '1':
        line_num = int(input('Enter line(s) to train on or -1 to train on all: '))

        ### Data generation
        #file_name = './Compton_FF_Code/DVCS_cross.csv'
        print('Generating ', num, ' datasets from ', data_file_name)
        for j in range(num):
            if line_num!=-1:
                data = generate_set(X_data[line_num*num_points:(line_num+1)*num_points], F[line_num*num_points:(line_num+1)*num_points], errF[line_num*num_points:(line_num+1)*num_points], 
                    other_data[line_num*num_points:(line_num+1)*num_points], y_data_params[line_num*num_points:(line_num+1)*num_points], num=total_examples)
            else:
                data = generate_set(X_data, F, errF, other_data, y_data_params, num=40000)
                train_num = 35000
            training_data_tmp = data[:train_num]
            test_eval_data_tmp = data[train_num:]
            train_sets.append(training_data_tmp)
            test_sets.append(test_eval_data_tmp)

        ## Network training
        print('Data Generated.')
        print('Training Networks...')

        # Training the networks, trains number of networks specified in beginning
        for j in range(num):
            print('Network ', j,)
            tmp_network = CFNS.CurveFittingNetwork(layers)

            eval_cost, eval_acc, train_cost, train_acc = tmp_network.SGD(train_sets[j], iterations, batch_size, learning_rate, 
                                lmbda=regularization_rate, evaluation_data=test_sets[j], param_ranges=initial_parameter_ranges,
                                monitor_training_accuracy=True, monitor_training_cost=True, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True)

            network_results.append([eval_cost, eval_acc, train_cost, train_acc])

            networks.append(tmp_network)

            tmp_network.save('./networks/{0}-L={1}-n{2}-{3}.txt'.format(filen, line_num, num, j))
            print('---------')
        
        for j in range(num):
            eval_cost = network_results[j][0]
            train_cost = network_results[j][2]
            plt.title('Graph of Cost for Training and Eval Cost')
            plt.plot(np.arange(start=0, stop=iterations), eval_cost[0:], 'r--', label='Evaluation Accuracy')
            plt.plot(np.arange(start=0, stop=iterations), train_cost[0:], 'b--', label='Training Accuracy')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.legend()
            plt.savefig('./results/cost_graph-{0}-{1}.png'.format(filen, j))
            plt.clf()
    elif res=='2':
        for j in range(num):
            print('Network ', j,': ./networks/{0}-{1}.txt'.format(filen, j))
            tmp_network = CFNS.load('./networks/{0}-{1}.txt'.format(filen, j))

            networks.append(tmp_network)
            print('---------------')
    
    while(True):
        print('Graph model on lines (normal ones are 0, 60, 65)')
        line1 = int(input('Enter Line Number (0-65): '))
        print('Correct Parameters - ReH: ', ReH[line1*num_points], ' ReE: ', ReE[num_points*line1], ' ReHT: ', ReHT[num_points*line1])
        param1_estimates = []
        param2_estimates = []
        param3_estimates = []
        dex = line1*num_points
        for j in range(num):
            
            x_axis = np.linspace(0, 360, num=360)
            params_returns = networks[j].feedforward(np.array([[x_b[dex]], [t[dex]]]))
            param1_estimates.append(params_returns[0])
            param2_estimates.append(params_returns[1])
            param3_estimates.append(params_returns[2])
            true_curve1 = []
            # Set QQ, xB, t and k and calculate 4-vector products
            
        for p in range(num_points):
                
            pars = [Q[dex+p], x_b[dex+p], t[dex+p], k_vals[dex+p], ReH[dex+p], ReE[dex+p], ReH[dex+p], ReE[dex+p], ReHT[dex+p],  0.014863]
            angle = [X[dex+p]]
            #print(angle)
            true_curve1.append(networks[j].observable_equation(angle, pars))

            
            #true_curve2 = calculate_observable(data1, param0_list[line1], param1_list[line1], param2_list[line1])

        print('Parameter Means and Std Devations:')
        p1_m, p1_std = get_mean_and_std(param1_estimates)
        p1_dnn_bool = (p1_m - p1_std) < ReH[line1*num_points] and (p1_m+p1_std) > ReH[line1*num_points]
        print('P1 mean: ', p1_m, ' Std. Dev: ', p1_std, ' Correct in error: ', p1_dnn_bool)
        p2_m, p2_std = get_mean_and_std(param2_estimates)
        p2_dnn_bool = (p2_m - p2_std) < ReE[line1*num_points] and (p2_m+p2_std) > ReE[line1*num_points]
        print('P2 mean: ', p2_m, ' Std. Dev: ', p2_std, ' Correct in error: ', p2_dnn_bool)
        p3_m, p3_std = get_mean_and_std(param3_estimates)
        p3_dnn_bool = (p3_m - p3_std) < ReHT[line1*num_points] and (p3_m+p3_std) > ReHT[line1*num_points]
        print('P3 mean: ', p3_m, ' Std. Dev: ', p3_std, ' Correct in error: ', p3_dnn_bool)

        dnn_curve1 = []

        for p in range(len(x_axis)):
                
            pars = [Q[dex], x_b[dex], t[dex], k_vals[dex], F1[dex], F2[dex], p1_m, p2_m, p3_m, 0.014863]
            angle = [x_axis[p]]
            #print(angle)
            dnn_curve1.append(networks[j].observable_equation(angle, pars))   

           
            
            #plt.plot(x_axis, true_curve2, 'r--', alpha=0.5, label='Just These Points Curve fit') # plot the raw data
        plt.plot(x_axis, dnn_curve1, colors[0], label='Deep Network Average') # plot the raw data
        plt.title('Graph of observables vs X')
        

        ## Curve Fitting: 
        initialParameters = np.array([1.0, 1.0, 1.0])
        constraints = ((0.0, 0.0, 0.0), # Parameter Lower Bounds
               (1.5, 1.5, 1.5)) # Parameter upper bounds

        constants = np.zeros((num_points)) + 0.014863
        curve_fit_y = F[dex:dex+num_points]
        data = (X[dex:dex+num_points], Q[dex:dex+num_points], x_b[dex:dex+num_points], t[dex:dex+num_points], k_vals[dex:dex+num_points], constants)

        func = BHDVCS.BHDVCS()
        params_fit, pcov = curve_fit(func.TotalUUXS_curve_fit, data, curve_fit_y, bounds=constraints, sigma=errF[dex:dex+num_points])
        perr = np.sqrt(np.diag(pcov))
        print('Curve Fit Parameters: ')
        p1_fit_bool = (params_fit[0] - perr[0]) < ReH[line1*num_points] and (params_fit[0]+perr[0]) > ReH[line1*num_points]
        print('P1: ', params_fit[0], ' Std. Dev: ', perr[0], ' Correct in error: ', p1_fit_bool)
        p2_fit_bool = (params_fit[1] - perr[1]) < ReH[line1*num_points] and (params_fit[1]+perr[1]) > ReH[line1*num_points]
        print('P2: ', params_fit[1], ' Std. Dev: ', perr[1], ' Correct in error: ', p2_fit_bool)
        p3_fit_bool = (params_fit[2] - perr[2]) < ReH[line1*num_points] and (params_fit[2]+perr[2]) > ReH[line1*num_points]
        print('P3: ', params_fit[2], ' Std. Dev: ', perr[2], ' Correct in error: ', p3_fit_bool)

        curve_fit_points = []
        for p in range(num_points):
            
            pars = [Q[dex+p], x_b[dex+p], t[dex+p], k_vals[dex+p], params_fit[0], params_fit[1], params_fit[0], params_fit[1], params_fit[2], 0.014863]
            angle = [X[dex+p]]
            #print(angle)
            
            curve_fit_points.append(func.TotalUUXS(angle, pars))

        plt.plot(X[line1*num_points:(line1+1)*num_points], true_curve1, 'g--', alpha=0.5, label='My Equation Plotted') # plot the raw data
        plt.plot(X[line1*num_points:(line1+1)*num_points], curve_fit_points, 'r--', alpha=0.5, label='Curve Fit') # plot the raw data
        plt.errorbar(X[line1*num_points:(line1+1)*num_points], F[line1*num_points:(line1+1)*num_points], errF[line1*num_points:(line1+1)*num_points],  None, 'bo', label='t={0} x_b={1} Q={2}'.format(t[line1*7],x_b[line1*7], Q[line1*7]), alpha=0.8) # plot the raw data
        plt.xlabel('X value')
        plt.ylabel('Observables')

        plt.legend()
        plt.show()

        exit_check2 = input('Exit graphing? (Y/N)')
        if exit_check2=='Y' or exit_check2=='y':
            break



### ---------------------------------------------------- ####



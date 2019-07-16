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
from sklearn.neural_network import MLPRegressor  
from sklearn.ensemble import GradientBoostingRegressor

import Curve_Fitting_Network as CFN
import Curve_Fitting_Network_2 as CFN2

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

def get_graph_arrays(line_value, x_axis, model):
        line1 = line_value
        #x_axis = np.linspace(0, 6, num=100)
        x_b1 = np.zeros((len(x_axis))) + x_b[line1*7]
        t_1 = np.zeros((len(x_axis))) + t[line1*7]
        Q_1 = np.zeros((len(x_axis))) + Q[line1*7]
        data1 = (x_axis, x_b1, t_1, Q_1)

        model_curve1 = []
        for i in range(len(x_axis)):
            params_tmp = model.feedforward(np.array([[x_b1[i]], [t_1[i]], [Q_1[i]], [x_axis[i]], [np.cos(x_axis[i])], [np.cos(x_axis[i]*x_axis[i])]]))

            data1_tmp = (x_axis[i],x_b1[i], t_1[i], Q_1[i])
            model_curve1.append(calculate_observable(data1_tmp, params_tmp[0][0], params_tmp[1][0], params_tmp[2][0]))
            if i==0:
                print(params_tmp[0][0],' ', params_tmp[1][0], ' ', params_tmp[2][0])

        return data1, model_curve1

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

data = pd.read_csv('./Compton_FF_Code/data_ff.csv')
attributes =['X', 'X_b', 'Q', 't', 'F']
#scatter_matrix(data[attributes])
plt.show()

print(data.columns)

x_b = np.array(data['X_b'])
Q = np.array(data['Q'])
t = np.array(data['t'])
X = np.array(data['X'])

axis = np.arange(len(X))

F = np.array(data['F'])
errF = np.array(data['errF'])


tot_num = len(X)
train_num = 400
test_num = tot_num-train_num

#h = (X, x_b, t, Q)
y_data = F
X_data = []
for i in range(len(x_b)):
    X_data.append([x_b[i], t[i], Q[i]])
X_data = np.array(X_data)
#np.random.shuffle(X_data)




h = (X, x_b, t, Q)
y = F

# some initial parameter values, can be set to specific value or sign and constrained or set to 1.0 as default
initialParameters = np.array([1.0, 1.0, 1.0])
constraints = ((-np.inf, -np.inf, -np.inf), # Parameter Lower Bounds
               (np.inf, np.inf, np.inf)) # Parameter upper bounds




#Actually Very Real Parameters -------------------------------------------------- ##
fittedParameters, pcov = curve_fit(calculate_observable, h, y, initialParameters)#, bounds=constraints)



### getting parameters for each curve
curve_fit_parameters = []
curve_fit_pcov = []

param0_list=[]
param1_list=[]
param2_list=[]

for i in range(0, len(X), 7):
    h=(X[i:(i+7)], x_b[i:(i+7)], t[i:(i+7)], Q[i:(i+7)])
    y = F[i:(i+7)]
    t_p, p_cov = curve_fit(calculate_observable, h, y, initialParameters)#, bounds=constraints)
    curve_fit_parameters.append(t_p)
    curve_fit_pcov.append(p_cov)

    param0_list.append(t_p[0])
    param1_list.append(t_p[1])
    param2_list.append(t_p[2])

    print(t_p)
print(fittedParameters)

par0_m, par0_std = get_mean_and_std(param0_list)
print('Parameter 0 mean: {0} std dev: {1}'.format(par0_m, par0_std))
par1_m, par1_std = get_mean_and_std(param1_list)
print('Parameter 1 mean: {0} std dev: {1}'.format(par1_m, par1_std))
par2_m, par2_std = get_mean_and_std(param2_list)
print('Parameter 2 mean: {0} std dev: {1}'.format(par2_m, par2_std))

plt.subplot(3,1,1)
plt.hist(param0_list, label='Parameter 0')
#fit_0 = stats.norm.pdf(param0_list, par0_m, par0_std)
#plt.plot(param0_list, fit_0)
plt.title('Parameters Curve_fit Histogram')
plt.legend()

plt.subplot(3,1,2)
plt.hist(param1_list, label='Parameter 1')
plt.legend()

plt.subplot(3,1,3)
plt.hist(param2_list, label='Paramter 2')
plt.legend()
plt.show()

p0,p1,p2 = fittedParameters
output = calculate_observable(h, p0, p1, p2)



print()
print('Curve fit:')
print('Fitted ', fittedParameters)
print('Actual: ? ')
print('RMSE: ', rmse(y, output))


#-------------------------------------------------------------------------_#
#           Model               #


## Model Parameters
num_inputs = 6
num_outputs = 3
learning_rate = 0.03
regularization_rate = 0.1
F_error_scaling = np.array([[0.03], [0.03], [0.09]])
iterations = 1800
batch_size = 6
layers = [num_inputs, 12, num_outputs]


#model_gradient_boosting=GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, 
#    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=42, 
#    max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)


#normalizer = 0.2
#y_data = F
#np.zeros((len(x_b), 1)) + [p0/normalizer, p1/normalizer, p2/normalizer]
X_data = []
y_data = []
for i in range(len(x_b)):
    X_data.append([x_b[i], t[i], Q[i], X[i], np.cos(X[i]), np.cos(X[i]*X[i])])
    y_data.append([uniform_sample(F[i], errF[i])])
X_data = np.array(X_data)
y_data = np.array(y_data)

X_train = X_data[:train_num]
y_train = y_data[:train_num]

X_test = X_data[train_num:]
y_test = y_data[train_num:]
#np.random.shuffle(X_data)

print(np.shape(y_train))
print(np.shape(X_train))

training_data = []
for i in range(len(X_train)):
    training_data.append((np.reshape(X_train[i],(num_inputs,1)), np.reshape(y_train[i],(1))))

test_eval_data = []
for i in range(len(X_test)):
    test_eval_data.append((np.reshape(X_test[i],(num_inputs,1)), np.reshape(y_test[i],(1))))

while(True):
    


    res = input('Do you want to: \n(1) retrain the network \n(2) load from a saved model \n(3) graph multiple models or \n(4) Exit?\n')
    
    if res=='4':
        print('Exiting..')
        break
    
    model_type = input('Which model would you like to use?\n(1) Model using Derivatives \n(2) Model Without Derivatives\n')
    if res == '1':

        
        if model_type=='1':
            model_deep_network = CFN.CurveFittingNetwork(layers, parameter_scaling=0.0000000000001)
        elif model_type=='2':
            model_deep_network = CFN2.CurveFittingNetwork2(layers, parameter_scaling=0.0000000000001)

        filen = input('Enter Filename to save network under (e.g. saved_network.txt): ')
        eval_cost, eval_acc, train_cost, train_acc = model_deep_network.SGD(training_data, iterations, batch_size, learning_rate, 
                            lmbda=regularization_rate, scaling_value=F_error_scaling, shrinking_learn_rate=True, evaluation_data=test_eval_data,
                            monitor_training_accuracy=True, monitor_training_cost=True, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True)

        
        model_deep_network.save(filen)

        plt.title('Graph of Cost for Training and Eval Cost')
        plt.plot(np.arange(start=10, stop=iterations), eval_cost[10:], 'r--', label='Evaluation Accuracy')
        plt.plot(np.arange(start=10, stop=iterations), train_cost[10:], 'b--', label='Training Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')


        predicted_dnn =[]
        actual_dnn = []
        for (x,y) in test_eval_data:
            out = model_deep_network.feedforward(x)
            h_tmp = (x[3], x[0], x[1], x[2])
            predicted_dnn.append(calculate_observable(h, out[0], out[1], out[2]))
            actual_dnn.append(y)
        predicted_dnn=np.array(predicted_dnn)
        actual_dnn=np.array(actual_dnn)
        layers_string = ''
        for x in layers:
            layers_string+=''+str(x)+' '

        rmse_val = rmse(predicted_dnn, actual_dnn)
        fin_eval_cost = eval_cost[-1]
        fin_training_cost = train_cost[-1]
        with open("model_results.csv", "a") as myfile:
            myfile.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}\n'.format(layers_string, learning_rate, regularization_rate, F_error_scaling, 
                                            iterations, batch_size, rmse_val, fin_eval_cost, fin_training_cost, filen))



        plt.legend()
        plt.show()
    elif res=='2':
        filen = input('Enter Saved Network Name (e.g. saved_network.txt): ')
        if model_type=='1':
            model_deep_network = CFN.load(filen)
        elif model_type=='2':
            model_deep_network = CFN2.load(filen)
    elif res=='3':
        x_axis = np.linspace(0, 6, num=100)
        plt.title('Graph of observables vs X')
        print('Graph model on lines (normal ones are 0, 60, 65)')
        line1 = int(input('Enter Line Number (0-65): '))

        plt.errorbar(X[line1*7:(line1+1)*7], F[line1*7:(line1+1)*7], errF[line1*7:(line1+1)*7],  None, 'bo', label='t={0} x_b={1} Q={2}'.format(t[line1*7],x_b[line1*7], Q[line1*7])) # plot the raw data
        model_deep_network = CFN.load('saved_network1.txt')
        data1, dnn_curve1 = get_graph_arrays(line1, x_axis, model_deep_network)
        true_curve1 = calculate_observable(data1, p0, p1, p2)
        plt.plot(x_axis, true_curve1, 'b--', alpha=0.5, label='Curve fit', linewidth=0.8) # plot the raw data
            
        colors=['b-', 'r-', 'g-', 'c-', 'm-', 'y-', 'k-']
        plt.xlabel('X value')
        for i in range(8):
            model_deep_network = CFN.load('saved_network{0}.txt'.format(i+1))
            data1, dnn_curve1 = get_graph_arrays(line1, x_axis, model_deep_network)
            plt.plot(x_axis, dnn_curve1, colors[i%len(colors)], label='Deep Network fit', linewidth=0.8) # plot the raw data
    
        plt.ylabel('Observables')

        plt.legend()
        plt.show()
    predicted_dnn =[]
    actual_dnn = []
    for (x,y) in test_eval_data:
        out = model_deep_network.feedforward(x)
        h_tmp = (x[3], x[0], x[1], x[2])
        predicted_dnn.append(calculate_observable(h, out[0], out[1], out[2]))
        actual_dnn.append(y)
    predicted_dnn=np.array(predicted_dnn)
    actual_dnn=np.array(actual_dnn)
    #for xt, yt in zip(predicted[:10], y_test[:10]):
        #print (xt, ' | ', yt)


    print('Model Scoring Results')
    print('RMSE of Observable Predictions for DNN: ', )
    print('RMSE of Observable Values: ', )

    print('Correct Params: ')
    print(p0,' ', p1, ' ', p2)

   
    while(True):
        print('Graph model on lines (normal ones are 0, 60, 65)')
        line1 = int(input('Enter Line Number (0-65): '))

        print('Estimated Params for line: ')
        
        x_axis = np.linspace(0, 6, num=100)
        data1, dnn_curve1 = get_graph_arrays(line1, x_axis, model_deep_network)
        true_curve1 = calculate_observable(data1, param0_list[line1], param1_list[line1], param2_list[line1])

        plt.title('Graph of observables vs X')

        plt.errorbar(X[line1*7:(line1+1)*7], F[line1*7:(line1+1)*7], errF[line1*7:(line1+1)*7],  None, 'bo', label='t={0} x_b={1} Q={2}'.format(t[line1*7],x_b[line1*7], Q[line1*7])) # plot the raw data
        plt.plot(x_axis, true_curve1, 'b--', alpha=0.5, label='Curve fit') # plot the raw data
        plt.plot(x_axis, dnn_curve1, 'b-', label='Deep Network fit') # plot the raw data

        plt.xlabel('X value')
        plt.ylabel('Observables')

        plt.legend()
        plt.show()

        exit_check2 = input('Exit graphing? (Y/N)')
        if exit_check2=='Y' or exit_check2=='y':
            break





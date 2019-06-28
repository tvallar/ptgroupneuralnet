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

def rmse(estimated, actual):
    n = len(estimated)
    return (1/n)*np.sqrt(np.sum((estimated-actual)*(estimated-actual)))

def calculate_observable(data, par0, par1, par2):
    x, x_b, t, Q = data
    M_p = 0.938 #GeV
    #-1/(par[3]*par[3]*par[4]*(1+2*par[3]*M_p*2*par[3]*M_p/(par[5]*par[5]))*(1+2*par[3]*M_p*2*par[3]*M_p/(par[5]*par[5])))*(par[0]
    #+ par[1]*cos(x[0]) + par[2]*cos(x[0]*x[0]));
    return -1/(x_b*x_b*t*(1+2*x_b*M_p*2*x_b*M_p/(Q*Q))*(1+2*x_b*M_p*2*x_b*M_p/(Q*Q)))*(par0 + par1*np.cos(x) + par2*np.cos(x*x))

data = pd.read_csv('./Compton_FF_Code/data_ff.csv')
attributes =['X', 'X_b', 'Q', 't', 'F']
scatter_matrix(data[attributes])
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

fittedParameters, pcov = curve_fit(calculate_observable, h, y, initialParameters)#, bounds=constraints)

p0,p1,p2 = fittedParameters
output = calculate_observable(h, p0, p1, p2)



print()
print('Curve fit:')
print('Fitted ', fittedParameters)
print('Actual: ? ')
print('RMSE: ', rmse(y, output))


#-------------------------------------------------------------------------_#
#           Model               #

num_inputs = 3
num_outputs = 3

model_deep_network = CFN.CurveFittingNetwork([num_inputs, 30, num_outputs])

#model_gradient_boosting=GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, 
#    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=42, 
#    max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)


normalizer = 0.002
y_data = np.zeros((len(x_b), 3)) + [p0/normalizer, p1/normalizer, p2/normalizer]
X_data = []
for i in range(len(x_b)):
    X_data.append([x_b[i], t[i], Q[i]])
X_data = np.array(X_data)

X_train = X_data[:train_num]
y_train = y_data[:train_num]

X_test = X_data[train_num:]
y_test = y_data[train_num:]
#np.random.shuffle(X_data)

print(np.shape(y_train))
print(np.shape(X_train))

training_data = []
for i in range(len(X_train)):
    training_data.append((np.reshape(X_train[i],(num_inputs,1)), np.reshape(y_train[i],(num_outputs,1))))

test_eval_data = []
for i in range(len(X_test)):
    test_eval_data.append((np.reshape(X_test[i],(num_inputs,1)), np.reshape(y_test[i],(num_outputs,1))))


model_deep_network.SGD(training_data, 100, 30, 0.005, 0.0001, evaluation_data=test_eval_data,
                         monitor_evaluation_accuracy=True, monitor_evaluation_cost=True)

predicted_dnn =[]
actual_dnn = []
for (x,y) in test_eval_data: 
    predicted_dnn.append(model_deep_network.feedforward(x))
    actual_dnn.append(y)
predicted_dnn=np.array(predicted_dnn)
actual_dnn=np.array(actual_dnn)
#for xt, yt in zip(predicted[:10], y_test[:10]):
    #print (xt, ' | ', yt)


print('Model Scoring Results')
print('RMSE of Parameter Predictions for DNN: ', rmse(predicted_dnn, actual_dnn))
print('RMSE of Observable Values: ', )

def get_graph_arrays(line_value, x_axis, model):
    line1 = line_value
    #x_axis = np.linspace(0, 6, num=100)
    x_b1 = np.zeros((len(x_axis))) + x_b[line1*7]
    t_1 = np.zeros((len(x_axis))) + t[line1*7]
    Q_1 = np.zeros((len(x_axis))) + Q[line1*7]
    data1 = (x_axis, x_b1, t_1, Q_1)

    model_curve1 = []
    for i in range(len(x_axis)):
        params_tmp = model.feedforward(np.array([[x_b1[i]], [t_1[i]], [Q_1[i]]]))*normalizer
        data1_tmp = (x_axis[i],x_b1[i], t_1[i], Q_1[i])
        model_curve1.append(calculate_observable(data1_tmp, params_tmp[0][0], params_tmp[1][0], params_tmp[2][0]))

    return data1, model_curve1

x_axis = np.linspace(0, 6, num=100)
line1 = 0
data1, dnn_curve1 = get_graph_arrays(line1, x_axis, model_deep_network)
true_curve1 = calculate_observable(data1, p0, p1, p2)
#print('Chi Squared of First Curve for Deep Network: ', stats.chisquare(true_curve1, dnn_curve1) ) 

line2=60
data2, dnn_curve2 = get_graph_arrays(line2, x_axis, model_deep_network)
true_curve2 = calculate_observable(data1, p0, p1, p2)
#print('Chi Squared of Second Curve for Deep Network: ', stats.chisquare(true_curve2, dnn_curve2) ) 

line3 = 65
data3, dnn_curve3 = get_graph_arrays(line3, x_axis, model_deep_network)
true_curve3 = calculate_observable(data3, p0, p1, p2)
#print('Chi Squared of Third Curve for Deep Network: ', stats.chisquare(true_curve3, dnn_curve3) ) 



plt.title('Graph of observables vs X')

plt.errorbar(X[line1*7:(line1+1)*7], F[line1*7:(line1+1)*7], errF[line1*7:(line1+1)*7],  None, 'bo', label='t={0} x_b={1} Q={2}'.format(t[line1*7],x_b[line1*7], Q[line1*7])) # plot the raw data
plt.plot(x_axis, true_curve1, 'b--', alpha=0.5, label='Curve fit') # plot the raw data
plt.plot(x_axis, dnn_curve1, 'b-', label='Deep Network fit') # plot the raw data

plt.xlabel('X value')
plt.ylabel('Observables')

plt.legend()
plt.show()

plt.errorbar(X[line2*7:(line2+1)*7], F[line2*7:(line2+1)*7], errF[line2*7:(line2+1)*7], None,  'go', label='t={0} x_b={1} Q={2}'.format(t[line2*7],x_b[line2*7], Q[line2*7])) # plot the raw data
plt.plot(x_axis, calculate_observable(data2, p0, p1, p2), 'g--', alpha=0.5, label='Curve fit 2') # plot the raw data
plt.plot(x_axis, dnn_curve2, 'g-', label='Model fit') # plot the raw data

plt.xlabel('X value')
plt.ylabel('Observables')

plt.legend()
plt.show()

plt.errorbar(X[line3*7:(line3+1)*7], F[line3*7:(line3+1)*7], errF[line3*7:(line3+1)*7], None,  'ro', label='t={0} x_b={1} Q={2}'.format(t[line3*7],x_b[line3*7], Q[line3*7])) # plot the raw data
plt.plot(x_axis, calculate_observable(data3, p0, p1, p2), 'r--', alpha=0.5, label='Curve fit 3') # plot the raw data
plt.plot(x_axis, dnn_curve3, 'r-', label='Model fit') # plot the raw data

plt.xlabel('X value')
plt.ylabel('Observables')

plt.legend()
plt.show()

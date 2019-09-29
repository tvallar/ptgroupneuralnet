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
import Curve_Fitting_Network_Single as CFNS
import Regular_Network as RN 
import BHDVCS as BHDVCS

## Model Parameters
num_inputs = 4
num_outputs = 3
learning_rate = 0.0008
regularization_rate = 0.05
F_error_scaling = np.array([[0.003], [0.003], [0.002]])
iterations = 20
batch_size = 15
layers = [num_inputs, 30, num_outputs]


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

def generate_set(X_data, F, errF, angle_data, y_params, num=-1, shuffle=True,):
    data_set = []
    tot = 0
    x_tot = len(X_data)
    if num==-1:
        tot=len(X_data)
    else:
        tot=num
    for i in range(tot):
        F_tmp = np.random.normal(F[i%x_tot], errF[i%x_tot])
        data_set.append((np.reshape(X_data[i%x_tot],(num_inputs,1)), np.reshape(np.array([F_tmp]),(1)), angle_data[i%x_tot], np.reshape(y_params[i%x_tot], (num_outputs, 1))))
    if shuffle:
        np.random.shuffle(data_set)
    return data_set


networks = []

file_base = './networks/group-mini-const-t-v1'
for i in range(10):
        print('{0}-{1}.txt'.format(file_base, i))
        file_name = '{0}-{1}.txt'.format(file_base, i)
        networks.append(CFNS.load(file_name))

line_num = 0
file_data = './Compton_FF_Code/DVCS_cross_fixed_t.csv'

data = pd.read_csv(file_data)

k_vals = np.array(data['k'])
x_b = np.array(data['x_b'])
Q = np.array(data['QQ'])
t = np.array(data['t'])
X = np.array(data['phi_x'])#*math.pi/180.0

ReH = np.array(data['ReH'])
ReE = np.array(data['ReE'])
ReHT = np.array(data['ReHTilde'])

F = np.array(data['F'])
errF = np.array(data['errF'])

X_data = []
y_data = []
y_data_params = []
angle_data=[]
for i in range(len(x_b)):
    X_data.append([x_b[i], t[i], Q[i], k_vals[i]])
    angle_data.append([X[i]])
    y_data_params.append([ReH[i], ReE[i], ReHT[i]])

X_data = np.array(X_data)
angle_data = np.array(angle_data)

tot_num = len(X)
train_num = 2900
test_num = tot_num-train_num
num_points = 36

print('Graph model on lines (normal ones are 0, 60, 65)')
line1 = int(input('Enter Line Number (0-65): '))
print('Correct Parameters - ReH: ', ReH[line1*num_points], ' ReE: ', ReE[num_points*line1], ' ReHT: ', ReHT[num_points*line1])
param1_estimates = []
param2_estimates = []
param3_estimates = []
dex = line1*num_points
par0 = []
par1 = []
par2 = []
for j in range(10):
        
        out = networks[j].feedforward(np.reshape(X_data[dex], (num_inputs,1)))
        print(out)
        par0.append(out[0][0])
        par1.append(out[1][0])
        par2.append(out[2][0])

        # Set QQ, xB, t and k and calculate 4-vector products
 
p1_m, p1_std = get_mean_and_std(par0)
p2_m, p2_std = get_mean_and_std(par1)
p3_m, p3_std = get_mean_and_std(par2)

## Curve Fitting: 
initialParameters = np.array([1.0, 1.0, 1.0])
constraints = ((0.0, 0.0, 0.0), # Parameter Lower Bounds
        (1.5, 1.5, 1.5)) # Parameter upper bounds

constants = np.zeros((num_points)) + 0.014863
curve_fit_y = F[dex:dex+num_points]
data = (X[dex:dex+num_points], Q[dex:dex+num_points], x_b[dex:dex+num_points], t[dex:dex+num_points], k_vals[dex:dex+num_points], constants)

func = BHDVCS.BHDVCS()
p_fit, pcov = curve_fit(func.TotalUUXS_curve_fit, data, curve_fit_y, bounds=constraints, sigma=errF[dex:dex+num_points])
perr = np.sqrt(np.diag(pcov))

actual_points =[]
curve_points = []
network_points = []

for p in range(num_points):
        
        pars1 = [Q[dex+p], x_b[dex+p], t[dex+p], k_vals[dex+p], p1_m, p2_m, p1_m, p2_m, p3_m,  0.014863]
        pars2 = [Q[dex+p], x_b[dex+p], t[dex+p], k_vals[dex+p], p_fit[0], p_fit[1], p_fit[0], p_fit[1], p2_m,  0.014863]
        angle = [X[dex+p]]
        
        actual_points.append(F[dex+p])
        network_points.append(func.TotalUUXS(angle, pars1))
        curve_points.append(func.TotalUUXS(angle, pars2))
        
written = open('Fitted_points.csv', 'w')

written.write('Parameters: \n')
written.write('model,par1=ReH,err1,par2=ReE,err2,par3=ReHT,err3\n')
        #ReH[line1*num_points], ' ReE: ', ReE[num_points*line1], ' ReHT: ', ReHT[num_points*line1]
written.write('Correct,{0},{1},{2},{3},{4},{5}\n'.format(ReH[line1*num_points], '?', ReE[num_points*line1], '?', ReHT[num_points], '?'))
written.write('network,{0},{1},{2},{3},{4},{5}\n'.format(p1_m, p1_std, p2_m, p2_std, p3_m, p3_std))
written.write('curve fit,{0},{1},{2},{3},{4},{5}\n'.format(p_fit[0], perr[0], p_fit[1], perr[1], p_fit[2], perr[2]))

written.write('\n')
written.write('Q,x_b,t,k\n')
written.write('{0},{1},{2},{3}'.format(Q[dex+p], x_b[dex+p], t[dex+p], k_vals[dex+p]))
written.write('\n')
written.write('X_Value,Curve_F,network_F,actual_F,errF\n')
for i in range(num_points):
        #print('{0},{1},{2}'.format(curve_points[i], network_points[i], actual_points[i]))
        written.write('{0},{1},{2},{3},{4}\n'.format(X[dex+i], curve_points[i], network_points[i], actual_points[i], errF[dex+i]))

written.close()
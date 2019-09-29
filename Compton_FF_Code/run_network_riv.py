import numpy as np
import os
import pandas as pd
import math
#from functools import partial
import matplotlib
import matplotlib.pyplot as plt

import Curve_Fitting_Network as CFN
import Curve_Fitting_Network_2 as CFN2
import Regular_Network as RN
import Curve_Fitting_Network_Single as CFNS

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


###----------------------------------------------------------------------####

colors = ['b', 'g', 'r', 'c', 'm', 'y']
data = pd.read_csv('./Compton_FF_Code/DVCS_cross.csv')
attributes =['X', 'X_b', 'Q', 't', 'F']

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

## Model Parameters
num_inputs = 4
num_outputs = 3
learning_rate = 0.0008
regularization_rate = 0.05
F_error_scaling = np.array([[0.003], [0.003], [0.002]])
iterations = 30
batch_size = 10
layers = [num_inputs, 30, num_outputs]


#               Data stuff 
### ---------------------------------------------- ###

num = 20 #number networks
filen = 'group-const-t=-0.323-n{0}-mini-v1'.format(num)
total_examples = 2000
train_num = 1500

networks = []
train_sets = []
test_sets = []
network_results = []

line_num = -1

### Data generation
file_name = './Compton_FF_Code/DVCS_cross.csv'
#print('Generating ', num, ' datasets from ', file_name)
for j in range(num):
    if line_num!=-1:
        data = generate_set(X_data[line_num*num_points:(line_num+1)*num_points], F[line_num*num_points:(line_num+1)*num_points], errF[line_num*num_points:(line_num+1)*num_points], 
            angle_data[line_num*num_points:(line_num+1)*num_points], y_data_params[line_num*num_points:(line_num+1)*num_points], num=1000)
    else:
        data = generate_set(X_data, F, errF, angle_data, y_data_params, num=total_examples)
    training_data_tmp = data[:train_num]
    test_eval_data_tmp = data[train_num:]
    train_sets.append(training_data_tmp)
    test_sets.append(test_eval_data_tmp)

## Network training
#print('Data Generated.')
#print('Training Networks...')


for j in range(num):
    #print('Network ', j,)
    tmp_network = CFNS.CurveFittingNetwork(layers)

    eval_cost, eval_acc, train_cost, train_acc = tmp_network.SGD(train_sets[j], iterations, batch_size, learning_rate, 
                        lmbda=regularization_rate, evaluation_data=test_sets[j], param_ranges=[0.0, 1.5],
                        monitor_training_accuracy=True, monitor_training_cost=True, monitor_evaluation_accuracy=True, monitor_evaluation_cost=True)

    network_results.append([eval_cost, eval_acc, train_cost, train_acc])

    networks.append(tmp_network)

    tmp_network.save('./networks/{0}-{1}.txt'.format(filen, j))
    #print('---------')

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




# You might want to use the following packages
import numpy as np

print(np.__file__)
import os
import tensorflow as tf
import pandas as pd
tf.logging.set_verbosity(tf.logging.ERROR) #reduce annoying warning messages
from functools import partial
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt


# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

# Your code goes here for this section.


#data = pd.read_csv('./Compton_FF_Code/signal_background_1.csv', sep=' ')
#print(data.columns)

#####--------------- Analysis ------------------------------------######

data = pd.read_csv('./Compton_FF_Code/data_ff.csv')
attributes =['X', 'X_b', 'Q', 't', 'F']
#scatter_matrix(data[attributes])
#plt.show()

num_inputs = 3
num_outputs = 3

x_b = np.array(data['X_b'])
Q = np.array(data['Q'])
t = np.array(data['t'])
X = np.array(data['X'])

axis = np.arange(len(X))

F = np.array(data['F'])
errF = np.array(data['errF'])

h = (X, x_b, t, Q)

def get_value(val):
    return val

def rmse(estimated, actual):
    n = len(estimated)
    return (1/n)*np.sqrt(np.sum((estimated-actual)*(estimated-actual)))

def calculate_observable(par):
    xt, x_bt, tt, Qt = h
    M_p = 0.938 #GeV
    #-1/(par[3]*par[3]*par[4]*(1+2*par[3]*M_p*2*par[3]*M_p/(par[5]*par[5]))*(1+2*par[3]*M_p*2*par[3]*M_p/(par[5]*par[5])))*(par[0]
    #+ par[1]*cos(x[0]) + par[2]*cos(x[0]*x[0]));
    k = -1/(x_bt*x_bt*tt*(1+2*x_bt*M_p*2*x_bt*M_p/(Qt*Qt))*(1+2*x_bt*M_p*2*x_bt*M_p/(Qt*Qt)))
    out = (par*[[1], [np.cos(xt)], [np.cos(xt*xt)]])
    print(out)
    print(k)
    return k*out






def neural_net_model(X_data, input_dim, output_dim):

    layer_1_num = 10
    layer_2_num = 10
    
    W_1 = tf.Variable(tf.random_uniform([input_dim,layer_1_num]))
    b_1 = tf.Variable(tf.zeros([layer_1_num]))
    layer_1 = tf.add(tf.matmul(X_data,W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)
    # layer 1 multiplying and adding bias then activation function    
    

    W_2 = tf.Variable(tf.random_uniform([layer_1_num,layer_2_num]))
    b_2 = tf.Variable(tf.zeros([layer_2_num]))
    layer_2 = tf.add(tf.matmul(layer_1,W_2), b_2)
    layer_2 = tf.nn.relu(layer_2)    
    # layer 2 multiplying and adding bias then activation function    
    
    
    W_O = tf.Variable(tf.random_uniform([layer_2_num, output_dim]))
    b_O = tf.Variable(tf.zeros([output_dim]))
    output = tf.add(tf.matmul(layer_2, W_O), b_O)    
    # O/p layer multiplying and adding bias then activation function    
    # notice output layer has one node only since performing #regression     
    return output


xs = tf.placeholder("float")
ys = tf.placeholder("float")
k = tf.placeholder('float')
k2 = tf.placeholder('float')
index = 0


output = neural_net_model(xs, num_inputs, num_outputs)
print('Output ', output)
cost = tf.reduce_mean(tf.square(tf.matmul(k,tf.matmul(par,k2))-ys))# our mean squared error cost function

train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

##---------------------------------------------------------------------------###

tot_num = len(X)
train_num = 400
test_num = tot_num-train_num

h = (X, x_b, t, Q)
y_data = F
X_data = []
for i in range(len(x_b)):
    X_data.append([x_b[i], t[i], Q[i]])
X_data = np.array(X_data)
#np.random.shuffle(X_data)

X_train = X_data[:train_num]
y_train = y_data[:train_num]

X_test = X_data[train_num:]
y_test = y_data[train_num:]

c_t=[]
c_test=[]

with tf.Session() as sess:    # Initiate session and initialize all vaiables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(sess,'yahoo_dataset.ckpt')
    for i in range(100):
        #for j in range(X_train.shape[0]):
        sess.run([cost,train],feed_dict= {xs:X_train, ys:y_train})
            # Run cost and train with each sample        
        c_t.append(sess.run(cost, feed_dict={xs:X_train,ys:y_train}))
        c_test.append(sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
        print('Epoch :',i,'Cost :',c_t[i])

plt.plot(c_t, np.arange(len(c_t)))
plt.plot(c_test, np.arange(len(c_test)))
plt.show()
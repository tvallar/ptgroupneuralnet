
import numpy as np
import pandas as pd
import tensorflow as tf
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

tot_num = len(X)
train_num = 2900
test_num = tot_num-train_num
num_points = 36

## Model Parameters
num_inputs = 2
num_outputs = 3
learning_rate = 0.001
regularization_rate = 0.05
F_error_scaling = np.array([[0.003], [0.003], [0.002]])
iterations = 20
batch_size = 60
layers = [num_inputs, 20, num_outputs]

X_train = X_data[:train_num]
y_train = y_data_params[:train_num]

X_test = X_data[train_num:]
y_test = y_data_params[train_num:]


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
cost = tf.reduce_mean(tf.square(output-ys))# our mean squared error cost function

train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

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
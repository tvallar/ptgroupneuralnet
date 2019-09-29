import Neural_Net as nn
import model_functions as mf
import random
import numpy as np
#from sklearn import cross_validation                     


num_points = 20000
X_all, Y_all = mf.get_data(20000)
X_train = X_all[:18000]
y_train = Y_all[:18000]
X_test = X_all[18000:]
y_test = Y_all[18000:]
X_valid, X_train = X_train[16000:], X_train[:16000]
y_valid, y_train = y_train[16000:], y_train[:16000]


means = X_train.mean(axis=0, keepdims=True)
stds = X_train.std(axis=0, keepdims=True) + 1e-10


X_train = (X_train-means)/stds
X_valid = (X_valid-means)/stds
X_test = (X_test-means)/stds


# solver = 'adam', 'lbfgs'
model = MLPRegressor(hidden_layer_sizes=(80, ), activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto', learning_rate='constant', 
    learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, 
    momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)


model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print('score: ', score)


x1_graph = np.linspace(-1,1,num=50)
x2_graph = np.zeros((50))+-0.5

X_graph = (x1_graph, x2_graph)
Y1_graph = np.array(mf.mod1((x1[i], x2[i]), c_p[0], c_p[1], c_p[2], c_p[3], c_p[4], c_p[5], c_p[6], c_p[7]))*(1+0.1*(1 - 2*np.random.rand(len(x)))) +  2*(1 - 2*np.random.rand(len(x)))
Y2_graph = np.array(mf.mod2((x1[i], x2[i]), c_p[0], c_p[1], c_p[2], c_p[3], c_p[4], c_p[5], c_p[6], c_p[7]))*(1+0.1*(1 - 2*np.random.rand(len(x)))) +  2*(1 - 2*np.random.rand(len(x)))
Y3_graph = np.array(mf.mod3((x1[i], x2[i]), c_p[0], c_p[1], c_p[2], c_p[3], c_p[4], c_p[5], c_p[6], c_p[7]))*(1+0.1*(1 - 2*np.random.rand(len(x)))) +  2*(1 - 2*np.random.rand(len(x)))
Y4_graph = np.array(mf.mod4((x1[i], x2[i]), c_p[0], c_p[1], c_p[2], c_p[3], c_p[4], c_p[5], c_p[6], c_p[7]))*(1+0.1*(1 - 2*np.random.rand(len(x)))) +  2*(1 - 2*np.random.rand(len(x)))


#validation_set = []
#for i in range(len(X_valid)):
    #validation_set.append([X_valid[i], y_valid[i]])



#net = nn.NeuralNetwork(2, 6, 4)# hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
#iterations = 1
#for i in range(iterations*len(X_train)):
#    net.train(X_train[i%len(X_train)], y_train[i%len(X_train)])
#    if i%100==0:
#        print(i,': ', net.calculate_total_error(validation_set))
#pred = net.feed_forward(X_test[0])
#print(X_test[0], ' ', pred, ' ', y_test[0])
#rmse = mf.rmse(pred, y_test)
#print('RMSE: ', rmse)
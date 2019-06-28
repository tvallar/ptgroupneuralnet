import numpy as np
import math


class CurveFittingNetwork:

    def __init__(self, fit_function, layers, learn_rate, activation_func='sigmoid', verbose=False, random_state=None):
        self.layers_sizes = layers
        self.learn_rate = learn_rate
        self.activation_func = activation_func
        self.verbose = verbose
        self.layers = []
        self.fit_function = fit_function

        self.num_layers = len(layers)
        self.num_inputs = layers[0]
        self.num_outputs = layers[len(layers)-1]

        if random_state !=None:
            np.random.seed(random_state)

        self.set_up_network()


    def train(self, training_data, epochs, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        x, y = training_data
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.layers]
        for j in range(epochs):
            delta_nabla_b, delta_nabla_w = self.back_propagate(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.layers = [w-(eta)*nw for w, nw in zip(self.layers, nabla_w)]
            self.biases = [b-(eta)*nb for b, nb in zip(self.biases, nabla_b)]
            if self.verbose and j%20==0:
                print("Epoch {0} complete: {1}".format(j, self.feed_forward(x)))
                print(nabla_b)
                print(nabla_w)
    
    def set_up_network(self):
        for i in range(1, len(self.layers_sizes)):
            self.layers.append(np.random.random(size=(self.layers_sizes[i], self.layers_sizes[i-1])))

        self.biases = [np.random.randn(y, 1) for y in self.layers_sizes[1:]]

    def feed_forward(self, X):
        tmp = X
        for i in range(len(self.layers)):
            #tmp2 = np.append(tmp, [1])
            tmp = self.layers[i].dot(tmp) +self.biases[i]
        return self.activation(tmp)


    def back_propagate(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.layers]
        # feedforward
        activation = x
        activations = [x] # list to store all the alen(mini_batch)tivations, layer by layer
        zs = [] # list to store all the z lay, layelen(mini_batch) by layer
        for b, w in zip(self.biases, self.layers):
            z = np.dot(w, activation)+b
            print(z)
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward y_truepass
        #print(activations[-1])
        delta = self.cost_derivative(activations[-1], y) * \
        self.derivative_act(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = deriv_sigmoid(z)
            delta = np.dot(self.layers[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)
    
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)


    def activation(self, val):
        if self.activation_func=='sigmoid':
            return sigmoid(val)
        if self.activation_func=='ReLU':
            return ReLU(val)
        if self.activation_func=='tanh':
            return tanh(val)
        if self.activation_func=='ELU':
            return ELU(val)
    
    def derivative_act(self, val):
        if self.activation_func=='sigmoid':
            return deriv_sigmoid(val)
        if self.activation_func=='ReLU':
            return ReLU(val)
        if self.activation_func=='tanh':
            return tanh(val)
        if self.activation_func=='ELU':
            return ELU(val)

def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()

def sigmoid(val):
    return 1/(1+np.exp(-1*val))

def tanh(val):
    return (np.exp(val)-np.exp(-1*val))/(np.exp(val)+np.exp(-1*val))

def ReLU(val):
    return np.max([0,val])

def ELU(val):
    if val >= 0:
        return val
    else:
        return np.exp(val)-1.0


learning_rate=0.005
model = CurveFittingNetwork(None, [3,2], 0.001)

X=np.array([[1.0], [0.5], [0.2]])
y_pred = np.array([[0.5], [0.4]])
y = model.feed_forward(X)
print(y)
model.train((X,y_pred), 1000, learning_rate)
y = model.feed_forward(X)
print(y)


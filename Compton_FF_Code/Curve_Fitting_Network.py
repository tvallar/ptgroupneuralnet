"""network2.py
~~~~~~~~~~~~~~

An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.
Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.  Note
that I have focused on making the code simple, easily readable, and
easily modifiable.  It is not optimized, and omits many desirable
features.

"""

#### Libraries
# Standard library
import json
import random
import sys
import BHDVCS
import math



# Third-party libraries
import numpy as np


#### Define the quadratic and cross-entropy cost functions

class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.

        """
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer.  Note that the
        parameter ``z`` is not used by the method.  It is included in
        the method's parameters in order to make the interface
        consistent with the delta method for other cost classes.

        """
        return (a-y)

def calculate_observable(data, par0, par1, par2):
    x, x_b, t, Q = data
    M_p = 0.938 #GeV
    #-1/(par[3]*par[3]*par[4]*(1+2*par[3]*M_p*2*par[3]*M_p/(par[5]*par[5]))*(1+2*par[3]*M_p*2*par[3]*M_p/(par[5]*par[5])))*(par[0]
    #+ par[1]*cos(x[0]) + par[2]*cos(x[0]*x[0]));
    return -1/(x_b*x_b*t*(1+2*x_b*M_p*2*x_b*M_p/(Q*Q))*(1+2*x_b*M_p*2*x_b*M_p/(Q*Q)))*(par0 + par1*np.cos(x) + par2*np.cos(x*x))

def calculate_observable_delta(angle, pars, par_num, bhdvcs):
    temp1 = bhdvcs.TotalUUXS(angle, pars)
    #print(angle)
    #print('test')
    step = 0.001
    if par_num == 0:
        
        pars[4] = pars[4]+step
        pars[6] = pars[6]+step
        #print('test2')
        temp2 = bhdvcs.TotalUUXS(angle, pars)
        #print('test3')
        return (temp2-temp1)/step
    elif par_num ==1:
        #step = 0.1
        pars[5] = pars[5]+step
        pars[7] = pars[7]+step
        temp2 = bhdvcs.TotalUUXS(angle, pars)
        return (temp2-temp1)/step
    elif par_num==2:
        #step = 0.1
        pars[8] = pars[8]+step
        temp2 = bhdvcs.TotalUUXS(angle, pars)
        return (temp2-temp1)/step
    else:
        return 0.0

def calculate_observable_numerical_delta(data, par_num, par_values):
    x, x_b, t, Q = data
    M_p = 0.938 #GeV
    if par_num == 0:
        return -1/(x_b*x_b*t*(1+2*x_b*M_p*2*x_b*M_p/(Q*Q))*(1+2*x_b*M_p*2*x_b*M_p/(Q*Q)))
    elif par_num ==1:
        return -1/(x_b*x_b*t*(1+2*x_b*M_p*2*x_b*M_p/(Q*Q))*(1+2*x_b*M_p*2*x_b*M_p/(Q*Q)))*np.cos(x)
    elif par_num==2:
        return -1/(x_b*x_b*t*(1+2*x_b*M_p*2*x_b*M_p/(Q*Q))*(1+2*x_b*M_p*2*x_b*M_p/(Q*Q)))*(np.cos(x*x))
    else:
        return 0.0

def curve_length(xb, t, q, par0, par1, par2):
    x_axis = np.linspace(0, 6, num=20)
    total=0.0
    for i in range(1,len(x_axis)):
        point1 = calculate_observable((x_axis[i-1], xb, t, q), par0, par1, par2)
        point2 = calculate_observable((x_axis[i], xb, t, q), par0, par1, par2)

        total+= np.sqrt((x_axis[i]-x_axis[i-1])*(x_axis[i]-x_axis[i-1])+(point2-point1)*(point2-point1))
    return total

#### Main Network class
class CurveFittingNetwork(object):

    def __init__(self, sizes, cost=QuadraticCost, activation='sigmoid', Function=calculate_observable, parameter_scaling=1.0):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.default_weight_initializer`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.parameter_scale = parameter_scaling
        self.sizes = sizes
        self.default_weight_initializer(parameter_scaling)
        self.cost=cost
        self.bhdvcs = BHDVCS.BHDVCS()
        self.best_params = {}
        self.param_ranges = {}
        self.next_params = {}
        self.next_params_count = {}

    def default_weight_initializer(self, parameter_scaling):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        self.biases = [(np.random.randn(y, 1)*self.parameter_scale) for y in self.sizes[1:]]
        self.weights = [(np.random.randn(y, x)*self.parameter_scale)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """Initialize the weights using a Gaussian distribution with mean 0
        and standard deviation 1.  Initialize the biases using a
        Gaussian distribution with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        This weight and bias initializer uses the same approach as in
        Chapter 1, and is included for purposes of comparison.  It
        will usually be better to use the default weight initializer
        instead.

        """
        self.biases = np.abs([np.random.randn(y, 1) for y in self.sizes[1:]])
        self.weights = np.abs([np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])])

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        zs = []
        for b, w in zip(self.biases, self.weights):
            zs = np.dot(w, a)+b
            a = sigmoid(np.dot(w, a)+b)
            
        return zs

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            scaling_value=0.1,
            param_ranges=[[0.0,1.5],[0.0,1.5],[0.0,1.5]],
            shrinking_learn_rate=False,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The ``training_data`` is a list of tuples ``(x, y)``
        representing the training inputs and the desired outputs.  The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter ``lmbda``.  The method also accepts
        ``evaluation_data``, usually either the validation or test
        data.  We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.  The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data.  All values are
        evaluated at the end of each training epoch.  So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set.

        """
        ###Initialization###
        for x, y, ang, y_par in training_data:
            dict_string = '{0}-{1}-{2}-{3}'.format(x[2], x[0], x[1], x[3])
            if not(dict_string in self.param_ranges.keys()):
                self.param_ranges[dict_string] = [[param_ranges[0], param_ranges[1]],[param_ranges[0], param_ranges[1]],[param_ranges[0], param_ranges[1]]]

            #pars = [x[2], x[0], x[1], x[3], zs[-1][0], zs[-1][1], zs[-1][0], zs[-1][1], zs[-1][2],  0.014863]
            if not(dict_string in self.best_params.keys()):
                self.best_params[dict_string]=[0.75, 0.75, 0.75]
                self.next_params[dict_string]=[0.0, 0.0, 0.0]
                self.next_params_count[dict_string]=0


        self.mini_batch_size = mini_batch_size
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            num_batches = len(mini_batches)
            count = 0
            for mini_batch in mini_batches:
                if count%10==0:
                    print(count, ' / ', num_batches)
                if shrinking_learn_rate:
                    self.update_mini_batch(
                        mini_batch, eta, lmbda, len(training_data), param_ranges)
                else:
                    self.update_mini_batch(
                        mini_batch, eta, lmbda, len(training_data), param_ranges)
            if j%1==0:
                print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                if j%1==0:
                    print ("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                if j%1==0:
                    print("Accuracy on training data: {}".format(accuracy))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                if j%1==0:
                    print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                if j%1==0:
                    print("Accuracy on evaluation data: {}".format(accuracy))
            if j%1==0:
                for x, y, ang, y_par in training_data[:1]:
                    out = self.feedforward(x)
                    pars = [x[2], x[0], x[1], x[3], out[0], out[1], out[0], out[1], out[2],  0.014863]
                    estimated = self.bhdvcs.TotalUUXS(ang, pars)
                    print('Actual: ', y_par)
                    print('Estimated: ', out)
                    print('Observable Actual: ', y)
                    print('Observable Estimated: ', estimated)
                    dict_string = '{0}-{1}-{2}-{3}'.format(x[2], x[0], x[1], x[3])
                    print('Paramter Ranges: ', self.param_ranges[dict_string])
                    print('Best Parameters: ', self.best_params[dict_string])
                print()
                print()
            
            for dict_string in self.best_params.keys():
                #dict_string = '{0}-{1}-{2}-{3}'.format(x[2], x[0], x[1], x[3])
                self.best_params[dict_string][0] = self.next_params[dict_string][0]/self.next_params_count[dict_string]
                self.best_params[dict_string][1] = self.next_params[dict_string][1]/self.next_params_count[dict_string]
                self.best_params[dict_string][2] = self.next_params[dict_string][2]/self.next_params_count[dict_string]
            
                self.next_params[dict_string][0]= 0.0
                self.next_params[dict_string][1]= 0.0
                self.next_params[dict_string][2]= 0.0
                
                self.next_params_count[dict_string]=0

        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, eta, lmbda, n, param_ranges):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #curve_len = 0.0
        #mini_batch_size = len(mini_batch)
        for x, y, ang, y_par in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y, ang, y_par, param_ranges)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            #out = self.feedforward(x)
            #print(out)
            #curve_len+=curve_length(x[0], x[1], x[2], out[0], out[1], out[2])
            #print(curve_len)
        #print('average ',  curve_len/mini_batch_size)
        #avg_curve_len = curve_len/len(mini_batch)
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

        

    def backprop(self, x, y, angle, y_par, param_ranges):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        
        # This is where the deltas for the output node is being found
        ## -------------------------------------- ###

        dict_string = '{0}-{1}-{2}-{3}'.format(x[2], x[0], x[1], x[3])

        #estimated_val = self.bhdvcs.TotalUUXS(angle, pars)
        parameter_set = get_parameter_permutations(self.param_ranges[dict_string], [zs[-1][0], zs[-1][1], zs[-1][2]], 0, [])
        #print('----')
        estimated_set = []
        for k in range(len(parameter_set)):
            temp = parameter_set[k]
            pars = [x[2], x[0], x[1], x[3], temp[0], temp[1], temp[0], temp[1], temp[2],  0.014863]
            estimated_set.append(self.bhdvcs.TotalUUXS(angle, pars))
            #print('Parameters: ', temp,'\n Estimated Val: ', estimated_set[-1], '\n Actual: ', y)

        estimated_set = np.array(estimated_set)
        estimated_set_del = np.abs(estimated_set - y)

        min_index = np.argmin(estimated_set_del)
        pars2 = [x[2], x[0], x[1], x[3], self.best_params[dict_string][0], self.best_params[dict_string][1], self.best_params[dict_string][0], 
                                        self.best_params[dict_string][1], self.best_params[dict_string][2],  0.014863]
        best_estimated = self.bhdvcs.TotalUUXS(angle, pars2)
        del_best_est = np.abs(best_estimated - y)

        #if del_best_est > estimated_set_del[min_index]:
        #    tmp_pars = self.best_params[dict_string]
        #    self.best_params[dict_string] = [tmp_pars[0]*0.8+0.2*parameter_set[min_index][0], tmp_pars[1]*0.8+0.2*parameter_set[min_index][1], tmp_pars[2]*0.8+0.2*parameter_set[min_index][2]]

        
        tmp_pars = self.best_params[dict_string]

        if del_best_est < estimated_set_del[min_index]:
            self.next_params[dict_string][0] = self.best_params[dict_string][0]
            self.next_params[dict_string][1] = self.best_params[dict_string][1]
            self.next_params[dict_string][2] = self.best_params[dict_string][2]
        else:
            self.next_params[dict_string][0] += parameter_set[min_index][0]
            self.next_params[dict_string][1] += parameter_set[min_index][1]
            self.next_params[dict_string][2] += parameter_set[min_index][2]
        
        self.next_params_count[dict_string] += 1

        delta = (zs[-1] - np.reshape(tmp_pars, (len(zs[-1]),1)))

        for i in range(len(self.param_ranges[dict_string])):
            self.param_ranges[dict_string][i][0] = self.param_ranges[dict_string][i][0]*0.999+0.001*parameter_set[min_index][i]
            self.param_ranges[dict_string][i][1] = self.param_ranges[dict_string][i][1]*0.999+0.001*parameter_set[min_index][i]

        #random_element = np.abs(np.random.random((3,1)))
        #chi2 = (estimated_val-y)/y
        #delta = chi2*param_deltas2*random_element
        
        if zs[-1][0] < 0:
             delta[0] = -5.0
        if zs[-1][1] < 0:
             delta[1] = -5.0
        if zs[-1][2] < 0:
             delta[2] = -5.0

        #print(delta)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        #-----------------------------------------##

    
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def predict(self, x, angle):
        out = self.feedforward(x)
        pars = [x[2], x[0], x[1], x[3], out[0], out[1], out[0], out[1], out[2],  0.014863]
        y_val = self.bhdvcs.TotalUUXS(angle, pars)
        return out, y_val

    def observable_equation(self, angle, pars):
        y_val = self.bhdvcs.TotalUUXS(angle, pars)
        return y_val


    def accuracy(self, data, convert=False):
        """Return the number of inputs in ``data`` for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag ``convert`` should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results ``y`` are
        represented in the different data sets.  In particular, it
        flags whether we need to convert between the different
        representations.  It may seem strange to use different
        representations for the different data sets.  Why not use the
        same representation for all three data sets?  It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up.  More details on the
        representations can be found in
        mnist_loader.load_data_wrapper.

        """


        results = [(x, self.feedforward(x), y, a) for (x, y, a, yp) in data]
        mse_sum = 0.0
        count = 0
        for (x, out, y_true, angle) in results:
            pars = [x[2], x[0], x[1], x[3], out[0], out[1], out[0], out[1], out[2],  0.014863]
            y_est = self.bhdvcs.TotalUUXS([angle], pars)
            count+=1
            tmp = sum((y_est-y_true)*(y_est-y_true))
            mse_sum+= tmp
        out=(mse_sum / count)*1.0
        return out

    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set ``data``.  The flag
        ``convert`` should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the ``accuracy`` method, above.
        """
        cost = 0.0
        for x, y, ang, y_par in data:
            a = self.feedforward(x)
            pars = [x[2], x[0], x[1], x[3], a[0][0], a[1][0], a[0][0], a[1][0], a[2][0],  0.014863]
            y_est = self.bhdvcs.TotalUUXS([ang], pars)
            cost += self.cost.fn(y_est, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost

    def save(self, filename):
        """Save the neural network to the file ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

#### Loading a Network
def load(filename):
    """Load a neural network from the file ``filename``.  Returns an
    instance of Network.

    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = CurveFittingNetwork(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

#### Miscellaneous functions
def vectorized_result(j, num):
    """Return a 10-dimensional unit vector with a 1.0 in the j'th position
    and zeroes elsewhere.  This is used to convert a digit (0...9)
    into a corresponding desired output from the neural network.

    """
    e = np.zeros((num, 1))
    e[j] = 1.0
    return e

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()

def tanh(val):
    return (np.exp(val)-np.exp(-1*val))/(np.exp(val)+np.exp(-1*val))

def tanh_prime(z):
    return 1-(tanh(z)**2)

def ReLU(val):
    return np.max([0,val])

def ELU(val):
    if val >= 0:
        return val
    else:
        return np.exp(val)-1.0

def get_parameter_permutations(param_ranges, parameters, index, cur_list):
    par_tmp_1 = np.random.uniform(low=param_ranges[index][0], high=parameters[index])#(parameters[index]+param_ranges[index][0])/2
    par_tmp_2 = parameters[index]
    par_tmp_3 = np.random.uniform(low=parameters[index], high=param_ranges[index][1])#(parameters[index]+param_ranges[index][1])/2
    if index==(len(parameters)-1):
        return [cur_list+[par_tmp_1], cur_list+[par_tmp_2], cur_list+[par_tmp_3]]

    out = [get_parameter_permutations(param_ranges, parameters, index+1, cur_list+[par_tmp_1]),
            get_parameter_permutations(param_ranges, parameters, index+1, cur_list+[par_tmp_3]),
            get_parameter_permutations(param_ranges, parameters, index+1, cur_list+[par_tmp_2])]

    ret = []
    for i in range(len(out)):
        for j in range(len(out[i])):
            ret.append(out[i][j])
    return ret
#def output_activation(val):




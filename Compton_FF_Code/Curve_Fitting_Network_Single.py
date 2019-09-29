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



#### Main Network class
class CurveFittingNetwork(object):

    def __init__(self, sizes, cost=QuadraticCost, activation='sigmoid', parameter_scaling=1.0):
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
        self.data_library = {}
        self.kinematics_list = []

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


    # This is the main training function, call it to begin the Stochastic gradient Descent (SGD)
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
        for x, y, oth, y_par in training_data:
            dict_string = '{0}-{1}'.format(x[0], x[1])
            if [x[0], x[1]] not in self.kinematics_list:
                self.kinematics_list.append([x[0], x[1]])
            
            ## initializing the parameter ranges
            if not(dict_string in self.param_ranges.keys()):
                self.param_ranges[dict_string] = [[param_ranges[0], param_ranges[1]],[param_ranges[0], param_ranges[1]],[param_ranges[0], param_ranges[1]]]
            
            # creating a data library for the curve points for each set of parameters
            # this was used to find the best parameters for each unique set of parameters in a mini batch
            # since the mini batch itself couldn't be used to find the best parameters
            # when multiple different curves were represented
            # No longer used in the Single Curve version of this code but useful to leave in and doesn't use too much memory
            if not(dict_string in self.data_library.keys()):
                self.data_library[dict_string] = [[oth[0], oth[1], oth[2], oth[3], oth[4], y]]
            else:
                self.data_library[dict_string].append([oth[0], oth[1], oth[2], oth[3], oth[4], y])
            #pars = [x[2], x[0], x[1], x[3], zs[-1][0], zs[-1][1], zs[-1][0], zs[-1][1], zs[-1][2],  0.014863]
            #initializing the best parameters to some middle ground
            if not(dict_string in self.best_params.keys()):
                self.best_params[dict_string]=[0.75, 0.75, 0.75]
        print('Num unique params: ', len(self.kinematics_list))


        self.mini_batch_size = mini_batch_size
        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        first = True
        # the epochs is how many times to run through the full training set
        # each epoch, the full training data is split into mini batches
        # these mini batches are then used to train the network
        # once each minibatch (total number len(data)/mini_batch_size) has been run through the update_mini_batch
        # training function, the cost for the network post training will be printed
        # the cost should shrink each iteration
        for j in range(epochs):

            random.shuffle(training_data)
            print()
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)] # creating the mini batches
            for mini_batch in mini_batches:
                if shrinking_learn_rate:
                    self.update_mini_batch(
                        mini_batch, eta, lmbda, len(training_data), param_ranges) # updating each mini batch
                else:
                    self.update_mini_batch(
                        mini_batch, eta, lmbda, len(training_data), param_ranges) # updating each mini batch
            # printing results
            if j%1==0:
                print("Epoch %s training complete" % j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                if j%1==0:
                    print ("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                training_accuracy.append(accuracy)
                if j%1==0:
                    print("Accuracy on training data: {}".format(accuracy))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda)
                evaluation_cost.append(cost)
                if j%1==0:
                    print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                if j%1==0:
                    print("Accuracy on evaluation data: {}".format(accuracy))
            if j%1==0:
                for x, y, oth, y_par in training_data[:1]:
                    out = self.feedforward(x)
                    pars = [oth[1], x[0], x[1], oth[2], oth[3], oth[4], out[0], out[1], out[2],  0.014863]
                    estimated = self.bhdvcs.TotalUUXS([oth[0]], pars)
                    print('Actual: ', y_par)
                    print('Estimated: ', out)
                    print('Observable Actual: ', y)
                    print('Observable Estimated: ', estimated)
                    dict_string = '{0}-{1}'.format(x[0], x[1])
                    print('Paramter Ranges: ', self.param_ranges[dict_string])
                    print('Best Parameters: ', self.best_params[dict_string])
                print()
                

        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy


    ## The update_parameters function
    # This was used when the network was training on multiple sets of parameters
    # This imlementation in Curve_Fitting_Network_Single only trains on a single
    # set of parameters, so this function is not used
    # However, it is a good way to understand how the parameter ranges function
    # works
    def update_parameters(self):
        print('Updating params...')
        num_kins = len(self.kinematics_list)
        for ii in range(num_kins):
            if ii%2==0:
                print(ii, ' \ ', num_kins)
            x = self.kinematics_list[ii]
            dict_string = '{0}-{1}'.format(x[0], x[1])
            zs = np.abs(self.feedforward(np.array(x)))
            best_pars = self.best_params[dict_string]
            parameter_set = get_parameter_permutations(self.param_ranges[dict_string], [best_pars[0], best_pars[1], best_pars[2]], 0, [])
            point_set = self.data_library[dict_string][:50]
            best_error = 0.0
            

            estimated_error = np.zeros((len(parameter_set)))
            for p in point_set:
                pars_b = [p[1], x[0], x[1], p[2], p[3], p[4], best_pars[0], best_pars[1], best_pars[2], 0.014863]    
                best_error += np.abs(self.bhdvcs.TotalUUXS([p[0]], pars_b)-p[5])

            for jj in range(len(parameter_set)):
                for p in point_set:
                    pars = [p[1], x[0], x[1], p[2], p[3], p[4], parameter_set[jj][0], parameter_set[jj][1], parameter_set[jj][2], 0.014863]    
                    estimated_error[jj] += np.abs(self.bhdvcs.TotalUUXS([p[0]], pars)-p[5])

            min_index = np.argmin(estimated_error)

            for k2 in range(len(estimated_error)):
                print(k2, ': ', parameter_set[k2], ' ', estimated_error[k2])
            print('Mini Batch Best: ', parameter_set[min_index], ' ', estimated_error[min_index])

            if best_error > estimated_error[min_index]:
                self.best_params[dict_string][0] = parameter_set[min_index][0]#*0.2 + self.best_params[dict_string][0]*0.8
                self.best_params[dict_string][1] = parameter_set[min_index][1]#*0.2 + self.best_params[dict_string][1]*0.8
                self.best_params[dict_string][2] = parameter_set[min_index][2]#*0.2 + self.best_params[dict_string][2]*0.8

            ### UPDATE PARAMETER RANGES: 
            for i in range(len(self.param_ranges[dict_string])):
                self.param_ranges[dict_string][i][0] = self.param_ranges[dict_string][i][0]*0.95+0.05*self.best_params[dict_string][i]
                self.param_ranges[dict_string][i][1] = self.param_ranges[dict_string][i][1]*0.95+0.05*self.best_params[dict_string][i]

    def update_mini_batch(self, mini_batch, eta, lmbda, n, param_ranges):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        first = True
        
        parameter_set = []
        #print('----')
        best_error = 0.0
        actual_params_error = 0.0

        #### Finding the best set of parameters
        zs = None
        for x, y, oth, y_par in mini_batch:
            dict_string = '{0}-{1}'.format(x[0], x[1]) #this is the string that would be used when there were multiple
            # sets of parameters that were trained on. At this point its left in just so that major changes to the parameter
            # would not have to be made, but rather the same system that worked for multiple parameters also works
            # for training on a single set
            temp = self.best_params[dict_string] #previous best parameters
            pars_b = [oth[1], x[0], x[1], oth[2], oth[3], oth[4], temp[0], temp[1], temp[2],  0.014863] #full parameters for 
            # cross section equation using the best parameters
            best_error += np.abs(self.bhdvcs.TotalUUXS([oth[0]], pars_b)-y) # adding the error from the current point onto the best error
            pars_a = [oth[1], x[0], x[1], oth[2], oth[3], oth[4], y_par[0], y_par[1], y_par[2],  0.014863] #this is using the correct parameters
            # supplied in the csv file as a way to compare errors and make sure the system is working as hoped
            actual_params_error += np.abs(self.bhdvcs.TotalUUXS([oth[0]], pars_a)-y) #adding to the error using the correct params
            if first:
                zs = np.abs(self.feedforward(x))
                first = False
                #generating the parameters set, returns 125 permutations of parameters randomly generated based on the parameter ranges passed in
                # since this is in first, it only runs on the first iteration, making sure the same sets of parameters are used throughout
                # the whole mini batch
                parameter_set = get_parameter_permutations(self.param_ranges[dict_string], [ self.best_params[dict_string][0], 
                                                                                            self.best_params[dict_string][1], 
                                                                                            self.best_params[dict_string][2]], 0, [])
                estimated_error = np.zeros((len(parameter_set))) #error for each of the sets of parameters
            # Now we are cycling through all 125 sets of parameters for the current x, y, oth, y_par in the minibatch and adding the errors
            # to the correct index
            for k in range(len(parameter_set)):
                temp = parameter_set[k] #current parameter set being tested
                pars = [oth[1], x[0], x[1], oth[2], oth[3], oth[4], temp[0], temp[1], temp[2],  0.014863] #full parameter set for the cross section equation
                estimated_error[k] += np.abs(self.bhdvcs.TotalUUXS([oth[0]], pars)-y)

        min_index = np.argmin(estimated_error) #finding the minimum error index after all points in the mini batch have been used
        #if the minimum error is lower than the error using the best parameters found so far, the best parameters b
        # will be replaced by the new ones
        if best_error > estimated_error[min_index]:
                self.best_params[dict_string][0] = parameter_set[min_index][0]#*0.2 + self.best_params[dict_string][0]*0.8
                self.best_params[dict_string][1] = parameter_set[min_index][1]#*0.2 + self.best_params[dict_string][1]*0.8
                self.best_params[dict_string][2] = parameter_set[min_index][2]#*0.2 + self.best_params[dict_string][2]*0.8

        ### UPDATE PARAMETER RANGES: 
        # once the parameters are updated, the parameter ranges (the range in which randomly generated parameters can be gotten)
        # will shrink around the parameters chosen
        # since this is done in the mini batch, the rate at which the windows shrink is a function of the mini
        # batch size
        for i in range(len(self.param_ranges[dict_string])):
                self.param_ranges[dict_string][i][0] = self.param_ranges[dict_string][i][0]*0.99+0.01*self.best_params[dict_string][i]
                self.param_ranges[dict_string][i][1] = self.param_ranges[dict_string][i][1]*0.99+0.01*self.best_params[dict_string][i]

        #now we begin backpropigating the found best parameters for each data point in the minibatch
        #the error for each will be aggregated in nabla_b and nabla_w, which at the end of the cycle
        #will be used to train the weights and biases
        for x, y, oth, y_par in mini_batch:

            dict_string = '{0}-{1}'.format(x[0], x[1])
            best_pars_mini = self.best_params[dict_string]
            delta_nabla_b, delta_nabla_w = self.backprop(x, y, oth, y_par, param_ranges, back_prop_pars=best_pars_mini)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #which is done here
        #in this equation, the (1-eta*(lmbda/n)) is the regularization, eta is a parameter passed in at the
        # beginning of training and the larger eta is, the smaller the weights will be forced to be
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

        

    def backprop(self, x, y, oth, y_par, param_ranges, back_prop_pars = None):
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
        
        tmp_pars = back_prop_pars # the parameters being back prop-ed are the ones
        # found during the minibatch analysis and parameter permutations

        delta = (zs[-1] - np.reshape(tmp_pars, (len(zs[-1]),1)))
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

    def predict(self, x, oth):
        out = self.feedforward(x)
        pars = [oth[1], x[0], x[1], oth[2], out[0], out[1], out[0], out[1], out[2],  0.014863]
        y_val = self.bhdvcs.TotalUUXS([oth[0]], pars)
        return out, y_val

    def observable_equation(self, angle, pars):
        y_val = self.bhdvcs.TotalUUXS(angle, pars)
        return y_val


    def accuracy(self, data):
        """Return the mean squared error for the the output of 
        the TotalUUXS equation using the parameters outputed
        by the network. The error is gotten using the correct
        measurements of the cross section as provided in the
        data.

        """


        results = [(x, self.feedforward(x), y, a) for (x, y, a, yp) in data]
        mse_sum = 0.0
        count = 0
        for (x, out, y_true, oth) in results:
            pars = [oth[1], x[0], x[1], oth[2], oth[3], oth[4], out[0], out[1], out[2],  0.014863]
            y_est = self.bhdvcs.TotalUUXS([oth[0]], pars)
            count+=1
            tmp = sum((y_est-y_true)*(y_est-y_true))
            mse_sum+= tmp
        out=(mse_sum / count)*1.0
        return out

    def total_cost(self, data, lmbda):
        """Return the total cost for the data set ``data``.
        """
        cost = 0.0
        for x, y, oth, y_par in data:
            a = self.feedforward(x)
            pars = [oth[1], x[0], x[1], oth[2], oth[3], oth[4], a[0][0], a[1][0], a[2][0],  0.014863]
            y_est = self.bhdvcs.TotalUUXS([oth[0]], pars)
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
    par_tmp_in = parameters[index]
    if par_tmp_in > param_ranges[index][1]:
        par_tmp_in = (param_ranges[index][1] + param_ranges[index][0])/2
    par_tmp_1 = np.random.uniform(low=param_ranges[index][0], high=param_ranges[index][1])#(parameters[index]+param_ranges[index][0])/2
    par_tmp_2 = np.random.uniform(low=param_ranges[index][0], high=param_ranges[index][1])#(parameters[index]+param_ranges[index][0])/2
    par_tmp_3 = parameters[index]
    par_tmp_4 = np.random.uniform(low=param_ranges[index][0], high=param_ranges[index][1])#(parameters[index]+param_ranges[index][1])/2
    par_tmp_5 = np.random.uniform(low=param_ranges[index][0], high=param_ranges[index][1])#(parameters[index]+param_ranges[index][1])/2
    
    if index==(len(parameters)-1):
        return [cur_list+[par_tmp_1], cur_list+[par_tmp_2], cur_list+[par_tmp_3], cur_list+[par_tmp_4], cur_list+[par_tmp_5]]
    out = [ get_parameter_permutations(param_ranges, parameters, index+1, cur_list+[par_tmp_1]),
            get_parameter_permutations(param_ranges, parameters, index+1, cur_list+[par_tmp_3]),
            get_parameter_permutations(param_ranges, parameters, index+1, cur_list+[par_tmp_2]),
            get_parameter_permutations(param_ranges, parameters, index+1, cur_list+[par_tmp_4]),
            get_parameter_permutations(param_ranges, parameters, index+1, cur_list+[par_tmp_5])]

    ret = []
    for i in range(len(out)):
        for j in range(len(out[i])):
            ret.append(out[i][j])
    return ret
#def output_activation(val):

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




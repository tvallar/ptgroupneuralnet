import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.neural_network import MLPRegressor  

# Setup
#--------------------------------------------------------------------------------------#


### MODIFY EQUATIONS AS NECESSARY
# add or remove both equations and parameters
# !!!! NOTE: You must pass all parameters to all equations regardless of whether
# they are used by that equation, parameter p0 for mod1 will be the exact same as p0 for mod2

def mod1(data, p0, p1, p2, p3, p4, p5, p6, p7): # not all parameters are used here
    x, y = data
    return (p0*p0 + p1+p1 + p2*p2 +p3*p3) + (p4*p4+p5*p5+4*p6*p6+4*p7*p7)*x + (p0*p4+p1*p5+p2*p6+p3*p7)*y# + error

def mod2(data, p0, p1, p2, p3, p4, p5, p6, p7): # not all parameters are used here
    x, y = data
    return (p0*p2+p1*p3)+(2*p4*p6+p5*2*p7)*x + (p0*p6+p1*p7+p2*p4+p3*p5)*y*y# + error

def mod3(data, p0, p1, p2, p3, p4, p5, p6, p7):
    #error = 1 - 2*np.random.rand(len(data))
    x, y = data
    return (p1*p4-p0*p5+p2*p7-p3*p6)*x*y# + error
    #return (p3*p3 + p0*p0)*np.cos(3*data) + (p2*p3 + p4*p5)*np.sin(data)  + (p3*p3 - p5*p5)

def mod4(data, p0, p1, p2, p3, p4, p5, p6, p7):
    #error = 1 - 2*np.random.rand(len(data))
    x, y = data
    return (-1*p2*p4-p3*p5+p0*p6+p1*p7+(p6*p4+p7*p5)*x)*y #+ error
    #return (p3*p3 + p0*p0)*np.cos(3*data) + (p2*p3 + p4*p5)ata[0]*np.sin(data) + p1



def rmse(estimated, actual):
    n = len(estimated)
    return (1/n)*np.sqrt(np.sum((estimated-actual)*(estimated-actual)))



def comboFunc(comboData, p0, p1, p2, p3, p4, p5, p6, p7):
    # single data set passed in, extract separate data
    #num_dp = len(h)
    #extract1 = comboData[:len(y1)] # first data
    #extract2 = comboData[len(y2):(len(y2)+len(y3))] # second data
    #extract3 = comboData[(len(y2)+len(y3)):(3*num_dp)]
    #extract4 = comboData[(3*num_dp):]


    result1 = mod1(comboData, p0, p1, p2, p3, p4, p5, p6, p7)
    result2 = mod2(comboData, p0, p1, p2, p3, p4, p5, p6, p7)
    result3 = mod3(comboData, p0, p1, p2, p3, p4, p5, p6, p7)
    result4 = mod4(comboData, p0, p1, p2, p3, p4, p5, p6, p7)

    out =np.append(result1, result2)
    out = np.append(out, result3)
    return np.append(out, result4)




#### SET YOUR PARAMETERS:
num_equations = 4
num_parameters = 8
num_datapoints = 8

param_max = 2
x_max=2

# some initial parameter values, can be set to specific value or sign and constrained or set to 1.0 as default
initialParameters = np.array([3.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
constraints = ((-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf), # Parameter Lower Bounds
               (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)) # Parameter upper bounds

sigma=[0.0]*num_datapoints

## the data points
# Var : value for data point 0, value for data point 1, ....
x = -x_max + 2*x_max*np.random.rand(num_datapoints)
y = -x_max + 2*x_max*np.random.rand(num_datapoints)

#combine all variables
h = (x,y)

## The correct parameters for creating the y data 
#(need observables if correct parameters are not known)
c_p = -1*param_max + 2*param_max*np.random.rand(num_parameters)
### GENERATE TARGET Y VALUES USING CORRECT PARAMETERS (ERROR INCLUDED)
y1 = np.array(mod1(h, c_p[0], c_p[1], c_p[2], c_p[3], c_p[4], c_p[5], c_p[6], c_p[7]))#*(1+0.1*(1 - 2*np.random.rand(len(x)))) +  2*(1 - 2*np.random.rand(len(x)))
y2 = np.array(mod2(h, c_p[0], c_p[1], c_p[2], c_p[3], c_p[4], c_p[5], c_p[6], c_p[7]))#*(1+0.1*(1 - 2*np.random.rand(len(x)))) +  2*(1 - 2*np.random.rand(len(x)))
y3 = np.array(mod3(h, c_p[0], c_p[1], c_p[2], c_p[3], c_p[4], c_p[5], c_p[6], c_p[7]))#*(1+0.1*(1 - 2*np.random.rand(len(x)))) +  2*(1 - 2*np.random.rand(len(x)))
y4 = np.array(mod4(h, c_p[0], c_p[1], c_p[2], c_p[3], c_p[4], c_p[5], c_p[6], c_p[7]))#*(1+0.1*(1 - 2*np.random.rand(len(x)))) +  2*(1 - 2*np.random.rand(len(x)))



def get_data(num_dp, x_max=1, param_max=2):
    X_out = []
    Y_out = []
    x1 = -1*x_max + 2*x_max*np.random.rand(num_dp)
    x2 = -1*x_max + 2*x_max*np.random.rand(num_dp)
    #c_p = -1*param_max + 2*param_max*np.random.rand(num_parameters)

    for i in range(len(x1)):
        X_out.append([x1[i], x2[i]])
        y1 = mod1((x1[i], x2[i]), c_p[0], c_p[1], c_p[2], c_p[3], c_p[4], c_p[5], c_p[6], c_p[7])#*(1+0.1*(1 - 2*np.random.rand(len(x)))) +  2*(1 - 2*np.random.rand(len(x)))
        y2 = mod2((x1[i], x2[i]), c_p[0], c_p[1], c_p[2], c_p[3], c_p[4], c_p[5], c_p[6], c_p[7])#*(1+0.1*(1 - 2*np.random.rand(len(x)))) +  2*(1 - 2*np.random.rand(len(x)))
        y3 = mod3((x1[i], x2[i]), c_p[0], c_p[1], c_p[2], c_p[3], c_p[4], c_p[5], c_p[6], c_p[7])#*(1+0.1*(1 - 2*np.random.rand(len(x)))) +  2*(1 - 2*np.random.rand(len(x)))
        y4 = mod4((x1[i], x2[i]), c_p[0], c_p[1], c_p[2], c_p[3], c_p[4], c_p[5], c_p[6], c_p[7])#*(1+0.1*(1 - 2*np.random.rand(len(x)))) +  2*(1 - 2*np.random.rand(len(x)))
        #print(y1)
        Y_out.append([y1, y2, y3, y4])
    #print(X_out)
    #print(Y_out)
    return np.array(X_out), np.array(Y_out)



#------------------------------------------------------------------------------------------#

# END SETUP, 

if (num_equations*len(h) < num_parameters):
    print('ERROR: The number of equations multiplied by the number of datapoints must be greater than or equal to the number of parameters')



comboY = np.append(y1, y2)
comboY = np.append(comboY, y3)
comboY = np.append(comboY, y4)



# curve fit the combined data to the combined function
fittedParameters, pcov = curve_fit(comboFunc, h, comboY, initialParameters, bounds=constraints)
print('Fitted:')
print(fittedParameters)

# values for display of fitted function
p0, p1, p2, p3, p4, p5, p6, p7 = fittedParameters
x_axis = np.linspace(-x_max,x_max,50)
y_axis = np.zeros((len(x_axis)))+0.5
combined = (x_axis, y_axis)

y_fit_1 = mod1(combined, p0, p1, p2, p3, p4, p5, p6, p7) # first data set, first equation
y_fit_2 = mod2(combined, p0, p1, p2, p3, p4, p5, p6, p7) # second data set, second equation
y_fit_3 = mod3(combined, p0, p1, p2, p3, p4, p5, p6, p7) # third data set, third equation
y_fit_4 = mod4(combined, p0, p1, p2, p3, p4, p5, p6, p7) # third data set, third equation

y_true_1 = mod1(combined, c_p[0], c_p[1], c_p[2], c_p[3], c_p[4], c_p[5], c_p[6], c_p[7]) # first data set, first equation
y_true_2 = mod2(combined, c_p[0], c_p[1], c_p[2], c_p[3], c_p[4], c_p[5], c_p[6], c_p[7]) # second data set, second equation
y_true_3 = mod3(combined, c_p[0], c_p[1], c_p[2], c_p[3], c_p[4], c_p[5], c_p[6], c_p[7]) # second data set, second equation
y_true_4 = mod4(combined, c_p[0], c_p[1], c_p[2], c_p[3], c_p[4], c_p[5], c_p[6], c_p[7]) # second data set, second equation

num_dp = num_datapoints


##ERROR and other statistics
print()
print('Curve fit on all 3 equations:')
print('Fitted ', fittedParameters)
print('Actual ', c_p)
print('RMSE for y1: ', rmse(y_fit_1, y_true_1))
print('RMSE for y2: ', rmse(y_fit_2, y_true_2))
print('RMSE for y3: ', rmse(y_fit_3, y_true_3))
print('RMSE for y4: ', rmse(y_fit_4, y_true_4))

y1_dp_fit = np.array(mod1(h, p0, p1, p2, p3, p4, p5, p6, p7))
y2_dp_fit = np.array(mod2(h, p0, p1, p2, p3, p4, p5, p6, p7))
y3_dp_fit = np.array(mod3(h, p0, p1, p2, p3, p4, p5, p6, p7))
y4_dp_fit = np.array(mod4(h, p0, p1, p2, p3, p4, p5, p6, p7))
comboY_fit = np.append(y1_dp_fit, y2_dp_fit)
comboY_fit = np.append(comboY_fit, y3_dp_fit)
comboY_fit = np.append(comboY_fit, y4_dp_fit)

ChiSqr = stats.chisquare(comboY, comboY_fit)
print('Chi Squared Statistic On Data Points: ', ChiSqr)
ChiSqr2 = stats.chisquare(c_p, fittedParameters)
print('Chi Squared Statistic On Parameter Values: ', ChiSqr2)


### Parameters using only one function to fit
y1_fit_p, y1_pcov = curve_fit(mod1, h, y1, initialParameters)
y2_fit_p, y2_pcov = curve_fit(mod2, h, y2, initialParameters)
y3_fit_p, y3_pcov = curve_fit(mod3, h, y3, initialParameters)
y4_fit_p, y4_pcov = curve_fit(mod4, h, y4, initialParameters)


###Curve fit on just y1 data and curve
print()
print('Curve fit on just Y1:')
print('Fitted ', y1_fit_p)
print('Actual ', c_p)
print('RMSE for y1: ', rmse(mod1(combined, y1_fit_p[0], y1_fit_p[1], y1_fit_p[2], y1_fit_p[3], y1_fit_p[4], y1_fit_p[5], y1_fit_p[6], y1_fit_p[7]), y_true_1))
print('RMSE for y2: ', rmse(mod2(combined, y1_fit_p[0], y1_fit_p[1], y1_fit_p[2], y1_fit_p[3], y1_fit_p[4], y1_fit_p[5], y1_fit_p[6], y1_fit_p[7]), y_true_2))
print('RMSE for y3: ', rmse(mod3(combined, y1_fit_p[0], y1_fit_p[1], y1_fit_p[2], y1_fit_p[3], y1_fit_p[4], y1_fit_p[5], y1_fit_p[6], y1_fit_p[7]), y_true_3))
print('RMSE for y4: ', rmse(mod4(combined, y1_fit_p[0], y1_fit_p[1], y1_fit_p[2], y1_fit_p[3], y1_fit_p[4], y1_fit_p[5], y1_fit_p[6], y1_fit_p[7]), y_true_4))


###Curve fit on just Y2
print()
print('Curve fit on just Y2:')
print('Fitted ', y2_fit_p)
print('Actual ', c_p)
print('RMSE for y1: ', rmse(mod1(combined, y2_fit_p[0], y2_fit_p[1], y2_fit_p[2], y2_fit_p[3], y2_fit_p[4], y2_fit_p[5], y2_fit_p[6], y2_fit_p[7]), y_true_1))
print('RMSE for y2: ', rmse(mod2(combined, y2_fit_p[0], y2_fit_p[1], y2_fit_p[2], y2_fit_p[3], y2_fit_p[4], y2_fit_p[5], y2_fit_p[6], y2_fit_p[7]), y_true_2))
print('RMSE for y3: ', rmse(mod3(combined, y2_fit_p[0], y2_fit_p[1], y2_fit_p[2], y2_fit_p[3], y2_fit_p[4], y2_fit_p[5], y2_fit_p[6], y2_fit_p[7]), y_true_3))
print('RMSE for y4: ', rmse(mod4(combined, y2_fit_p[0], y2_fit_p[1], y2_fit_p[2], y2_fit_p[3], y2_fit_p[4], y2_fit_p[5], y2_fit_p[6], y2_fit_p[7]), y_true_4))



## Curve fit on just Y3
print()
print('Curve fit on just Y3:')
print('Fitted ', y3_fit_p)
print('Actual ', c_p)
print('RMSE for y1: ', rmse(mod1(combined, y3_fit_p[0], y3_fit_p[1], y3_fit_p[2], y3_fit_p[3], y3_fit_p[4], y3_fit_p[5], y3_fit_p[6], y3_fit_p[7]), y_true_1))
print('RMSE for y2: ', rmse(mod2(combined, y3_fit_p[0], y3_fit_p[1], y3_fit_p[2], y3_fit_p[3], y3_fit_p[4], y3_fit_p[5], y3_fit_p[6], y3_fit_p[7]), y_true_2))
print('RMSE for y3: ', rmse(mod3(combined, y3_fit_p[0], y3_fit_p[1], y3_fit_p[2], y3_fit_p[3], y3_fit_p[4], y3_fit_p[5], y3_fit_p[6], y3_fit_p[7]), y_true_3))
print('RMSE for y4: ', rmse(mod4(combined, y3_fit_p[0], y3_fit_p[1], y3_fit_p[2], y3_fit_p[3], y3_fit_p[4], y3_fit_p[5], y3_fit_p[6], y3_fit_p[7]), y_true_4))


## Curve fit on just Y3
print()
print('Curve fit on just Y4:')
print('Fitted ', y4_fit_p)
print('Actual ', c_p)
print('RMSE for y1: ', rmse(mod1(combined, y4_fit_p[0], y4_fit_p[1], y4_fit_p[2], y4_fit_p[3], y4_fit_p[4], y4_fit_p[5], y4_fit_p[6], y4_fit_p[7]), y_true_1))
print('RMSE for y2: ', rmse(mod2(combined, y4_fit_p[0], y4_fit_p[1], y4_fit_p[2], y4_fit_p[3], y4_fit_p[4], y4_fit_p[5], y4_fit_p[6], y4_fit_p[7]), y_true_2))
print('RMSE for y3: ', rmse(mod3(combined, y4_fit_p[0], y4_fit_p[1], y4_fit_p[2], y4_fit_p[3], y4_fit_p[4], y4_fit_p[5], y4_fit_p[6], y4_fit_p[7]), y_true_3))
print('RMSE for y4: ', rmse(mod4(combined, y4_fit_p[0], y4_fit_p[1], y4_fit_p[2], y4_fit_p[3], y4_fit_p[4], y4_fit_p[5], y4_fit_p[6], y4_fit_p[7]), y_true_4))


### Model Fitting
num_points = 20000
X_all, Y_all = get_data(60000)
X_train = X_all[:55000]
y_train = Y_all[:55000]
X_test = X_all[55000:]
y_test = Y_all[55000:]
X_valid, X_train = X_train[50000:], X_train[:50000]
y_valid, y_train = y_train[50000:], y_train[:50000]


means = X_train.mean(axis=0, keepdims=True)
stds = X_train.std(axis=0, keepdims=True) + 1e-10


X_train = (X_train-means)/stds
X_valid = (X_valid-means)/stds
X_test = (X_test-means)/stds


print()
print('Model Accuracy;')
# solver = 'adam', 'lbfgs' 'sgd'
# activation = 'tanh', 'relu', logistic, 
model = MLPRegressor(hidden_layer_sizes=(300, ), activation='relu', solver='adam', alpha=0.001, batch_size='auto', learning_rate='constant', 
    learning_rate_init=0.00025, power_t=0.5, max_iter=400, shuffle=True, random_state=42, tol=0.0001, verbose=False, warm_start=False, 
    momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)


model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print('score: ', score)

predictions = model.predict(X_test)
for i in range(200):
    print('Predicted: ', predictions[i], ' | Actual: ', y_test[i], ' | X1, X2: ', X_test[i])


X_graph = []
for i in range(len(x_axis)):
    X_graph.append([x_axis[i], y_axis[i]])
X_graph = np.array(X_graph)
Y_model = model.predict(X_graph)
y1_model = []
y2_model = []
y3_model = []
y4_model = []
for i in range(len(x_axis)):
    y1_model.append(Y_model[i][0])
    y2_model.append(Y_model[i][1])
    y3_model.append(Y_model[i][2])
    y4_model.append(Y_model[i][3])

print()
print('MODEL results in comparison to real equation')
print('RMSE for y1: ', rmse(y1_model, y_true_1))
print('RMSE for y2: ', rmse(y2_model, y_true_2))
print('RMSE for y3: ', rmse(y3_model, y_true_3))
print('RMSE for y4: ', rmse(y4_model, y_true_4))



### Y1 Plot
plt.subplot(2,2,1)
plt.title('Y1')
plt.plot(x, comboY[:num_dp], 'bo', label='y1 datapoints') # plot the raw data
plt.plot(x_axis, y_fit_1, 'b-', label='y1 curve fit') # plot the equation using the fitted parameters
plt.plot(x_axis, y_true_1, 'r--', label='y1 true', alpha=0.5) # plot the equation using the fitted parameters
plt.plot(x_axis, y1_model, 'g', label='model fit') # plot the equation using the fitted parameters
plt.legend()

### Y2 Plot
plt.subplot(2,2,2)
plt.title('Y2')
plt.plot(x, comboY[num_dp:2*num_dp], 'ro', label='y2 datapoints') # plot the raw data
plt.plot(x_axis, y_fit_2, 'b-', label='y2 fit') # plot the equation using the fitted parameters
plt.plot(x_axis, y_true_2, 'r--', label='y2 true', alpha=0.5) # plot the equation using the fitted parameters
plt.plot(x_axis, y2_model, 'g', label='model fit') # plot the equation using the fitted parameters
plt.legend()

### Y3 Plot
plt.subplot(2,2,3)
plt.title('Y3')
plt.plot(x, comboY[2*num_dp:3*num_dp], 'go', label='y3 datapoints') # plot the raw data
plt.plot(x_axis, y_fit_3, 'b-', label='y3 fit') # plot the equation using the fitted parameters
plt.plot(x_axis, y_true_3, 'r--', label='y3 true', alpha=0.5) # plot the equation using the fitted parameters
plt.plot(x_axis, y3_model, 'g', label='model fit') # plot the equation using model
plt.legend()

## Y4 plot
plt.subplot(2,2,4)
plt.title('Y4')
plt.plot(x, comboY[3*num_dp:], 'go', label='y4 datapoints') # plot the raw data
plt.plot(x_axis, y_fit_4, 'b-', label='y4 fit') # plot the equation using the fitted parameters
plt.plot(x_axis, y_true_4, 'r--', label='y4 true', alpha=0.5) # plot the equation using the fitted parameters
plt.plot(x_axis, y4_model, 'g', label='model fit') # plot the equation using themodel
plt.legend()



plt.show()


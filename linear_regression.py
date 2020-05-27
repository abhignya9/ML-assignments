"""Name - Abhignya Goje
   UCM ID - 700703549
   I certify that the codes/answers of this assignment are entirely my own work. """

import sys
import numpy as np
import pandas as pd
import scipy.optimize as opt
  
if len(sys.argv)==4:
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    degree = int(sys.argv[3])
    train_data = pd.read_table(train_file, delim_whitespace=True, header=None, dtype=float) 
    test_data = pd.read_table(test_file, delim_whitespace=True, header=None, dtype=float) 
else:
    print("Usage: logistic_regression.py <training_file> <test file> ")

#normalize the data
def normalize(x):               
    mean = x.mean(axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / std

#setting c value
def intercept_term(x):     
    X = np.ones(shape=(x.shape[0], x.shape[1] + 1))
    X[:, 1:] = x                                   
    return X   

#finding the squared error
def rmse(num1, num2):        
    return np.square(num1 - num2)

#cost to be minimised
def cost_function(theta, X, y):   
    length = len(y)
    Y_pred = X @ theta
    squared_errors = np.sum(np.square(Y_pred - y))
    return (squared_errors) / (2 * length)

#defining gradient descent
def gradient_descent(theta, X, y): 
    Y_pred = X @ theta
    gradient = X.transpose() @ (Y_pred - y)
    return (gradient) / (len(y))

#converting the data to poynomial LR based on degree passed
def polynomial_lr(train, degree):
    poly_lr_var = []
    for x in train:
        poly_lr = np.zeros(shape=(len(x), degree))
        for i in range(0, degree):
            poly_lr[:, i] = x.squeeze() ** (i + 1);
        poly_lr_var.append(poly_lr.flatten())
    return poly_lr_var
def linear_regression(X, y):
    theta = np.zeros(X.shape[1])
    return opt.fmin_cg(cost_function, theta, gradient_descent, (X, y), disp=False)

#**********************************************driver code**********************************************
train = []
train_y = train_data[train_data.shape[1]-1]
train_data = normalize(train_data)
train = np.array(train_data)[:, :-1]
poly_lr = polynomial_lr(train,degree)
poly_lr = intercept_term(np.array(poly_lr))
theta = linear_regression(poly_lr, train_y)
hypothesis = poly_lr @ theta
i = 0
for x in theta:
    print("θ%d=%f" % (i, x))
    i += 1

test = []
test_y = test_data[test_data.shape[1]-1]
test_x = normalize(test_data)
test = np.array(test_x)[:, :-1]
poly_lr = polynomial_lr(test,degree)
poly_lr = intercept_term(np.array(poly_lr))

hypothesis = poly_lr @ theta

i = 0
index = 0
for test_point in poly_lr:
    i += 1
    print("Object_ID=%3d, output=%8.4f, target value = %8.4f, squared error = %3f" % (i, hypothesis[index], test_y[index], rmse( hypothesis[index], test_y[index])))
    index = index+1


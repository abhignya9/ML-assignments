"""Name - Abhignya Goje
   UCM ID - 700703549
   I certify that the codes/answers of this assignment are entirely my own work. """

import sys
import numpy as np
    
#normalize the data
def normalize(x):               
    mean = x.mean(axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / std

#hypothesis function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#finding J(θ) or cost
def cost(theta, train_X, train_y):
    pred = sigmoid(train_X @ theta)
    pred[pred == 1] = 0.999 # log(1)=0 causes division error during optimization
    error = -train_y * np.log(pred) - (1 - train_y) * np.log(1 - pred)
    return sum(error) / len(train_y)

#the differentiation(gradient) of the cost function
def cost_gradient(theta, train_X, train_y):
    pred = sigmoid(train_X @ theta)
    pred[pred == 1] = 0.999 # log(1)=0 causes division error during optimization
    return  (train_X.transpose() @ (pred - train_y)) / len(train_y)

#minimizing the gradient function
def gradient_descent(train_X, train_y, theta, alpha, num_iters):
    for i in range(num_iters):    
        gradient = cost_gradient(theta,train_X,train_y)
        theta = theta - (alpha * gradient) 
    return theta

##########################################################################################

#start of the execution of the program
#receving the command lines arguments into the program
if len(sys.argv)==3:
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    
else:
    print("Usage: logistic_regression.py <training_file> <test_file> ")
    
#load the files using the numpy loadtxt() function
train_data = np.loadtxt(train_file)
test_data = np.loadtxt(test_file)

#separate the labels from the obervations/features
train_x = train_data[:,0:(train_data.shape[1]-1)]
train_x = normalize(train_x)
train_y = train_data[:,-1]

#add the weight(which is 1) of theta0 values to the train data    
train_X = np.ones(shape=(train_x.shape[0], train_x.shape[1] + 1))
train_X[:, 1:] = train_x

#set the training parameters
alpha = 0.01
num_iters = 1000
uniq_label = np.unique(train_y)
numFeatures = train_x.shape[1]

#array to store the model parameters for each classifier
classifiers = np.zeros(shape=(len(uniq_label), numFeatures + 1))

#set values to 1 where the corresponding values in y are equal to the current class, 
#and we set the rest of the values to 0
k=0
for l in uniq_label:
    train_Y = np.array([1 if l == i else 0 for i in train_y])
    theta = np.zeros(train_X.shape[1]) #initializing the theta values to all 0's
    print("initial cost for class %d with theta values initialized to 0s : %f"%(l,cost(theta,train_X,train_Y)))
    classifiers[k, :] = gradient_descent(train_X, train_Y, theta, alpha, num_iters)
    k+=1

#print the theta vaues for each class 
for i in range(classifiers.shape[0]):
    print("Theta values for class - ",int(uniq_label[i]))
    print("Improved cost after performing the gradient descent : ",cost(classifiers[i,:], train_X,train_Y))
    for j in range(classifiers.shape[1]):
        #just for formatting the output prints(spaces)
        if (classifiers[i,j] < 0):
            print("θ%d  = %f"%(j,classifiers[i,j]))
        else:
            print("θ%d  =  %f"%(j,classifiers[i,j]))

#separate the observations and the target values in the test file
test_x = test_data[:,0:(np.shape(test_data)[1]-1)]
test_x = normalize(test_x)
test_y = test_data[:,-1]

#add the weight(which is 1) of theta0 values to the test data
test_X = np.ones(shape=(test_x.shape[0], test_x.shape[1] + 1))
test_X[:, 1:] = test_x

#Predict the probabilities of each class for each test data
classProbabilities = sigmoid(test_X @ classifiers.transpose())

#pick the class with the max probability value
predicted_class = classProbabilities.argmax(axis=1)  

for index in range(len(test_y)):
    accuracy = 0
    if test_y[index]==predicted_class[index]:
        accuracy = 1
    print("Object ID = %3d, output = %d, target value = %d, Accuracy = %d" % (index+1, predicted_class[index], test_y[index], accuracy))
#print("Test accuracy:", str(100 * np.mean(pred_class == test_y)) + " %")

"""Name - Abhignya Goje
   UCM ID - 700703549
   I certify that the codes/answers of this assignment are entirely my own work. """


import sys
import numpy as np
from collections import Counter

def load_data(file):
    return np.loadtxt(file)

class KNN(object):
    classification_accuracy = 0
    def __init__(self, train_data, test_data, k):
        self.train_data = train_data[:, :-1]
        self.train_label = train_data[:, -1]
        self.test_data = test_data[:, :-1]
        self.test_label = test_data[:, -1]
        self.k = k
        
    def normalize(self):
        mean = np.mean(self.train_data, axis=0)
        std_dev = np.std(self.train_data, axis=0)
        
        # distributing/normalizing the test data over the mean and standard deviation of the training data
        # for consistency
        for i in range(0, self.train_data.shape[1]):
            self.train_data[:, i] = self.train_data[:, i] - mean[i]
            self.train_data[:, i] = self.train_data[:, i] / std_dev[i]
            self.test_data[:, i] = self.test_data[:, i] - mean[i]
            self.test_data[:, i] = self.test_data[:, i] / std_dev[i]
            
    def classify(self):
        #make predictions
        for i in range(0, len(self.test_data)):
            distance = euclidian_distance(self.test_data[i, :], self.train_data)
            distance = (np.concatenate([[distance, self.train_label]])).T
            distance = np.array(sorted(distance, key=lambda x:x[0]))
            self.show_results(distance, i)
            
    def show_results(self, distance, row_number):
        print("Object ID       : ", row_number)
        
        # pick the true classification label from the test data 
        true = self.test_label[row_number]
        print("True Class      : ", true)
        
        # when k=1, no ties no complications!
        if self.k == 1:
            k_neighbours = distance[self.k, :]
            predicted = k_neighbours[1]
            if(true==predicted):
                accuracy=1
                self.classification_accuracy = self.classification_accuracy+accuracy
            else:
                accuracy=0
            print("Predicted Class : ", predicted) 
            print("Accuracy        : ",accuracy)
            print("-----------------------------")
            
        #when k>1
        elif self.k > 1:
            k_neighbours = distance[0: self.k, :]
            
            # when the k-rows have the same training label(no-tie), select that training label
            if len(np.unique(k_neighbours[:,1])) == 1:
                #predicted=np.unique(k_neighbours[:,1])
                predicted = k_neighbours[0,1]
                #print("Predicted Class : ", predicted)
                if(true==predicted):
                    accuracy=1
                    self.classification_accuracy = self.classification_accuracy+accuracy
                else:
                    accuracy=0
                print("Predicted Class : ", predicted) 
                print("Accuracy        : ",accuracy)
                print("-----------------------------")
                
            # when the k rows have k different training labels (no-tie),select the first training label since 
            # it belongs to the training data nearest to the test data point
            elif len(np.unique(k_neighbours[:,1])) == self.k:
                predicted=k_neighbours[0,1]
                if(true==predicted):
                    accuracy=1
                    self.classification_accuracy = self.classification_accuracy+accuracy
                else:
                    accuracy=0
                print("Predicted Class : ", predicted) 
                print("Accuracy        : ",accuracy)
                print("-----------------------------")

            # when chances of tie
            else:
                # Counter outputs a dictionary of format {"key":"no.of times value appeared in the list/array"}
                counts=dict(Counter(k_neighbours[:,1]))
                counts_s = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1])}
                #print(counts_s)
                k = list(counts_s.keys())
                v = list(counts_s.values())
                
                #selecting the predicted value to be the class that appeared most in k-radius of the test point
                predicted = k[v.index(np.amax(v))]
                print("Predicted Class : ", predicted)
                
                #accuracy calculation based on question
                j=0
                for i in range(len(v)-1):
                    if v[i]==v[i+1]:
                        j+=1
                    else:
                        break

                if ((true in list(counts_s.keys())[0:j]) and (j!=0)):
                    accuracy = float(1/(j+1))
                elif j==0 and true==predicted:
                    accuracy=1
                else:
                    accuracy=0
                    
                self.classification_accuracy = self.classification_accuracy+accuracy 
                print("Accuracy        : ",accuracy)
                print("-----------------------------")
        
        
    def overall_acc(self):
        # overall accuracy of the knn-classifier   
        print("Overall accuracy for k = ", self.k, " : ", self.classification_accuracy/self.test_label.shape[0])

def euclidian_distance(test, train):
    distance = np.square(test - train)
    distance = np.sum(distance, axis = 1)
    distance = np.sqrt(distance)
    return distance

def main():

    if len(sys.argv)==4:
        train_file = sys.argv[1]
        test_file = sys.argv[2]
        k_value = int(sys.argv[3])
        train = load_data(train_file)
        test = load_data(test_file)
        knn = KNN(train, test, k_value)
        knn.normalize()
        knn.classify()
        knn.overall_acc()
    else:
        print("Usage: knn_classify.py <training_file> <test file> <k_value> ")

main()

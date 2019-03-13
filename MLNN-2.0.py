import math
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import time

#2.0version
#final version
#train and test iris dataset, wine dataset
#the best parameter :
#==========================================================================
#iris   input    hidden   learning-rate    training-times    highest-score
#         4        4           0.005            10000             1
#--------------------------------------------------------------------------
#wine     4        8/10        0.0005           100000           0.85
#==========================================================================
#
#algorithm reference "Introduction to Machine Learning" p164.
#


#depend on present time create a true random seed.
now_time = time.time() * 100000
time_1 = now_time // 100
time_seed = round(now_time % time_1)
random.seed(time_seed)
#end

def rand_num():
    if(random.random()>0.5):             #random select negative or positive numbers
        return random.random()*0.0001    #if do not take 0.0001, sometimes will have Overflow.
    else:
        return -0.0001*random.random()

def create_matrix(m, n, fill=0.0):  # m*n zero matrix
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat

def conver_label(data_label):      # Conversion label
    label = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
    output_label = []
    for i in range(len(data_label)):
        output_label.append(label[int(data_label[i])])
    return np.array(output_label)


def re_conver_label(output_label):  # reverse conversion label
    pre_label = []
    data_label=[]
    for i in range(output_label.shape[0]):
        for j in range(3):
            if (output_label[i][j] == 1):
                data_label = j
        pre_label.append(data_label)
    return np.array(pre_label)


def sigmoid(x):  # Activation function
    #print(x)
    #x=round(x,2)
    #print("again:",x)
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_derivative(x):  #H*(1-H)
    return x * (1 - x)


class ML_NeuralNetwork:    #Multi-layered network
    def __init__(self):    #Initialization
        self.input_num = 0
        self.hidden_num = 0
        self.output_num = 0
        self.input_units = []
        self.hidden_units = []
        self.output_units = []
        self.input_weights = []
        self.output_weights = []
        self.input_bias = []
        self.hidden_bias = []
        self.delta_hidden_weights=[]
        self.delta_input_weights=[]

    def set_para(self, ni, nh, no):
        # set the number of input, hidden, output units
        self.input_num = ni
        self.hidden_num = nh
        self.output_num = no
        # set the units
        self.input_units = [1.0] * self.input_num    #神经元的个数，float类型
        self.hidden_units = [1.0] * self.hidden_num
        self.output_units = [1.0] * self.output_num
        # set the weights
        self.input_weights = create_matrix(self.input_num, self.hidden_num)
        self.hidden_weights = create_matrix(self.hidden_num, self.output_num)
        self.delta_input_weights = create_matrix(self.input_num, self.hidden_num)
        self.delta_hidden_weights = create_matrix(self.hidden_num,self.output_num)

        # give random numbers to weights
        for i in range(self.input_num):    #there are (input_num* hidden_num) input weights
            for h in range(self.hidden_num):
                self.input_weights[i][h] = rand_num()
        for h in range(self.hidden_num):   #there are (hidden_num* output_num) hidden weights
            for o in range(self.output_num):
                self.hidden_weights[h][o] = rand_num()
        # give random numbers to bias
        self.input_bias=[rand_num()]*self.hidden_num  #the number of hidden units
        self.hidden_bias=[rand_num()]*self.output_num #the number of output units

    def predict(self, inputs):
        # input layer
        for i in range(self.input_num):
            self.input_units[i] = inputs[i]    #assign value to each input units
            #print("predict :inputs[i] ",inputs[i])
        # hidden layer
        for j in range(self.hidden_num):
            total = 0.0
            for i in range(self.input_num):
                total += self.input_units[i] * self.input_weights[i][j]  #wx+wx+wx+......
            total += self.input_bias[j]                 #WX+input_bias
            self.hidden_units[j] = sigmoid(total)       #active
        # output layer
        for k in range(self.output_num):
            total = 0.0
            for j in range(self.hidden_num):
                total += self.hidden_units[j] * self.hidden_weights[j][k]  #vh+vh+vh+......
            total += self.hidden_bias[k]                     #VH+hidden_bias
            self.output_units[k] = sigmoid(total)       #active

        return self.output_units[:]   #return prediction

    def back_propagate(self, sample, label, learn):
        # Backpropagation algorithm

        self.predict(sample)

        # calculate updated hidden weights
        output_Error = [0.0] *self.output_num       #clear
        for h in range(self.hidden_num):
            for o in range(self.output_num):        #calculate output ERROR
                output_Error[o] = (label[o] - self.output_units[o]) #save each error of output unit
                self.delta_hidden_weights[h][o]=learn*output_Error[o]*self.hidden_units[h]

        # calculate  updated input weights
        hidden_Error = [0.0] * self.hidden_num      #clear
        for h in range(self.hidden_num):            #calculate hidden ERROR
            output_sumError=0.0                     #clear
            for o in range(self.output_num):
                output_sumError +=output_Error[o]*self.hidden_weights[h][o]
            hidden_Error[h] = output_sumError*sigmoid_derivative(self.hidden_units[h])
            for i in range(self.input_num):
                self.delta_input_weights[i][h] = learn*hidden_Error[h]*self.input_units[i]

        # update hidden weights and bias
        for h in range(self.hidden_num):
            for o in range(self.output_num):
                self.hidden_weights[h][o] += self.delta_hidden_weights[h][o]
                if h==0:
                    self.hidden_bias[o] += learn * output_Error[o]

        # update input weights and bias
        for i in range(self.input_num):
            for h in range(self.hidden_num):
                #change_v=hidden_Error[h]*self.input_units[i]
                self.input_weights[i][h] += self.delta_input_weights[i][h]
                if i==0:
                    self.input_bias[h] += learn * hidden_Error[h]
        # calculate global error
        global_Error = 0.0
        for o in range(len(label)):
            global_Error += 0.5*(label[o] - self.output_units[o])**2 #global error
        return global_Error

    def train(self, samples, labels, iteration=1000, learn=0.01):
        #train(x_train, y_train, 1000, 0.2, 0.05)
        #
        for j in range(iteration):
            error = 0.0
            for i in range(len(samples)):
                label = labels[i]
                sample = samples[i]
                error += self.back_propagate(sample, label, learn)


    def fit(self, x_test):
        predicition = []
        for sample in x_test:
            pred_label = self.predict(sample)          #predict this sample
            for i in range(len(pred_label)):
                if (pred_label[i] == max(pred_label)):   #set predicted label of max value to true label
                    pred_label[i] = 1
                else:
                    pred_label[i] = 0
            predicition.append(pred_label)
        return re_conver_label(np.array(predicition))


if __name__ == '__main__':  # main function
    test_nn = ML_NeuralNetwork()

    iris = datasets.load_iris()  # 1.0
    wine=datasets.load_wine()    #0.85

    rs_iris=random.randint(1,100)   #random state
    rs_wine=random.randint(1,100)

    X_iris = iris.data   # output 3
    y_iris = iris.target
    y_iris = conver_label(y_iris)
    x_train, x_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state=rs_iris, test_size=0.5,train_size=0.5)

    X_wine = wine.data[:,:4]  # output 3
    y_wine = wine.target
    y_wine = conver_label(y_wine)
    x_train1, x_test1, y_train1, y_test1 = train_test_split(X_wine, y_wine, random_state=rs_wine, test_size=0.5, train_size=0.5)

    print("start training iris dataset")
    test_nn.set_para(4, 4, 3)            # set parameter  iris: 4 4 3 0.005 1000
    test_nn.train(x_train, y_train, 100, 0.005) # training
    pred_label = test_nn.fit(x_test)     # test
    con_label = re_conver_label(y_test)
    iris_s = accuracy_score(pred_label, con_label)
    print(iris_s)
    print("finish dataset iris!")

    print("start training wine dataset")
    test_nn.set_para(4, 8, 3)  # set parameter  wine:4 8 3 0.0005 100000
    test_nn.train(x_train1, y_train1, 100, 0.0005)  # training
    pred_label1 = test_nn.fit(x_test1)  # test
    con_label1 = re_conver_label(y_test1)
    wine_s = accuracy_score(pred_label1, con_label1)
    print(wine_s)
    print("finish dataset wine!")




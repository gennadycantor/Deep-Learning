# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:06:06 2019

@author: Miru Park
"""

import numpy as np
from csv import reader
from math import exp
import matplotlib.pyplot as plt
import os
""" image preprocessing """
import pandas as pd



""" read csv """
# TODO
def get_data(filename):
    data = list()
    #open file for reading
    with open(filename, 'r') as file:
        read_csv = reader(file)
        #for each row in csv
        for row in read_csv:
            #append it to our "list"
            data.append(row)
    return data

""" train test split """
def train_test_split(cleanedData):
    # my test portion is 81a -> 200b
    # which is 161th to 201st index wise
    # cleanedData is a 2d array
    test = cleanedData[161:201,:]
    train1 = cleanedData[0:161,:]
    train2 = cleanedData[201:,:]
    # concat train 1 and train 2
    train = np.concatenate((train1, train2), axis=0)
    return train, test

def get_final(filename):
    data = list()
    #open file for reading
    with open(filename, 'r') as file:
        read_csv = reader(file)
        #for each row in csv
        for row in read_csv:
            #append it to our "list"
            data.append(row)
    return data
            
    

""" preprocess and clean data """
def clean_data(data):
    # first turn our list into a matirx
    X = np.array(data)
    # change to float 
    X = X.astype(np.float)
    # now change the label
    #X = X[:,1:] / float(255)
    #for x in range(0, X.shape[0]):        
    #for x in range(0, 400):
    #    if X[x,0] == 4.0:
            #counter4 += 1
    #        X[x,0] = 0
    #    elif X[x,0] == 6.0:
            #counter6  += 1
    #        X[x,0] = 1
   
    #return X[:400,:]             
    return X
    
""" compute cost """
def compute_cost(a,Y):
    cost = np.zeros(len(a))
    for i in range(len(a)):
        if Y[i] == 0:
            if (1-a[i]) < 0.0001:
                cost[i] = 1*100
            else:
                cost[i] = -1*np.log(1-a[i])
        elif Y[i] == 1:
            if (a[i]) < 0.0001:
                cost[i] = 1*100
            else:
                cost[i] = -1*np.log(a[i])
    cost = 1*np.sum(cost)
    return cost

# RELU, SOFTMAX and TANH SHOULD BE USED #

""" apply sigmoid activation function to inner product: w.Tx """
def activation(theta):
    theta = 1 / (1 + np.exp(-theta))
    return theta

def genWeightMatrixL1(X):
    X = X.T
    numFeatures = X.shape[0] # 1
    numRowsW = X.shape[0]
    W1 = np.random.randn(numFeatures, numRowsW) * 0.01
    return W1
    
def forwardPropL1(W,X,b):
    # note: X is a transposed of X and so is b
    return np.matmul(W,X) + b 

def forwardPropL2(W,A,b):
    return np.matmul(W,A) + b

def backPropL2(A2, Y, AL1):
    dZ2 = A2 - Y
    dW2 = np.matmul(dZ2,AL1.T) # Al1.T
    dB2 = np.sum(dZ2, axis = 0, keepdims = True) # why axis = 1?
    return dW2, dB2, dZ2

def computeSigmoidDer(Z):
    # compute 1 - g(Z)
    allOnes = np.ones((Z.shape[0], Z.shape[1]))
    M1 = activation(Z)
    M2 = activation(Z)
    M2 = allOnes - M2
    derivativeM = np.multiply(M1, M2)
    return derivativeM

def backPropL1(W2, dZ2, deriv, X):
    dZ1 = np.matmul(W2.T, dZ2)
    dZ1 = np.multiply(dZ1, deriv)
    dW1 = np.matmul(dZ1, X)
    dB1 = np.sum(dZ1, axis = 0, keepdims = True) # why axis = 1?
    return dW1, dB1

""" for mini batch implementatino, TODO """
def shuffleMini(X,Y, i, numitr):
    #11760 training examples
    # divide the number of row dimensions by num itr
    rowDim = X.shape[0]
    numBatches = rowDim / numitr
    lastBatch = rowDim % numitr 
    if (i+2)*numBatches < rowDim:
        return X[i:(i+1)*numBatches, :], Y[i:(i+1)*numBatches]
    else:
        return X[(i+1)*numBatches:,:], Y[(i+1)*numBatches:]
    
""" gradient descend and minimize our cost function """
def train2NN(X,Y, alpha, numitr):
    print(59)
    W1 = genWeightMatrixL1(X)
    print(49)
    #b1 = np.zeros((1, X.shape[1]))
    b1 = np.zeros(1)
    #b1 = b1.T
    numW2 = (X.T).shape[0]
    W2 = np.random.randn(1, numW2) * 0.01
    b2 = np.zeros(1)
    #b2 = np.zeros(1)
    # we are going to append cost/epoch in the list below
    cost = []
    for i in range(0,numitr):

        
        # split the training set  in to batches 
        # split the label in the the same number of batches
        # shuffle them
        #miniX, miniY = shuffleMini(X, Y, i, numitr)
        
        
        # we need to forward propagate twice.
        ZL1 = forwardPropL1(W1,X.T,b1) # originally b1.T
        AL1 = activation(ZL1)
        # ZL2 = (1xW.shape[0])X (W.shape[0]xZl1.shape[1])
        ZL2 = forwardPropL2(W2, AL1, b2) #AL1 originally b2.T
        #ZL2 = ZL2.T
        A2 = activation(ZL2)
        
        # compute the cost
        costPer = compute_cost(A2.T, Y)        
        if i % 100 == 0:
            cost.append(costPer)
            print("@ Epoch " + str(i/100) + " cost is: " + str(cost[i/100]))
       
        # backpropagate
        # minibatch... or SGD...
        dW2, dB2, dZ2 = backPropL2(A2, Y, AL1)
        sigmoidDer = computeSigmoidDer(ZL1)
        dW1, dB1 = backPropL1(W2, dZ2, sigmoidDer, X)
        
        # update weights
        W2 -= alpha*dW2
        # IMPORTANT: broadcasting is not supported by a += b
        #b2 -= alpha * dB2
        b2 = b2 - (alpha*dB2)
        W1 -= alpha * dW1
        #b1 -= alpha * dB1
        b1 = b1 - (alpha*dB1)
        
    return W1, b1, W2, b2, cost

""" fit our model after training on train set and evaluate accuracy """
def evaluate(W1, W2, b1, b2, testX, testY):
    ZL1 = forwardPropL1(W1,testX.T,b1)
    AL1 = activation(ZL1)
     # ZL2 = (1xW.shape[0])X (W.shape[0]xZl1.shape[1])
    ZL2 = forwardPropL2(W2, AL1, b2)
     #ZL2 = ZL2.T
    A2 = activation(ZL2)
    A2 = A2.tolist()
    output = []
    for i in range(len(A2[0])):
        if A2[0][i] > 0.5:
            A2[0][i] = 1
            output.append(1)
        elif A2[0][i] < 0.5:
            A2[0][i] = 0
            output.append(0)
    # now evaluate
    counter = 0
    numInstances = len(testY)
    for i in range(len(A2[0])):
        if A2[0][i] == testY[i]:
            counter += 1
    score = counter / float(numInstances)
    return score, output

""" final eval """
def evaluate_final(W1, W2, b1, b2 ,X):
    ZL1 = forwardPropL1(W1,X.T,b1)
    AL1 = activation(ZL1)
    hidden = AL1[:,0]
     # ZL2 = (1xW.shape[0])X (W.shape[0]xZl1.shape[1])
    ZL2 = forwardPropL2(W2, AL1, b2)
     #ZL2 = ZL2.T
    A2 = activation(ZL2)
    A2 = A2.tolist()
    outputs = np.zeros(len(A2[0]))
    #print("dimension of a:", a.shape)
    for i in range(len(A2[0])):
        if A2[0][i] > 0.5:
            A2[0][i] = 1
            outputs[i] = 1
        else:
            A2[0][i] = 0
            outputs[i] = 0
    return outputs, hidden
    
""" normalize our train and test set """
def normalize(X):
    #X = X[:, 1:] / float(255)
    X = X / float(255)
    return X

""" split our train and test into trainLabel/trainFeauture & testLabel/testFeature """
def partition(X):
    Feature = X[:, 1:]
    Label = X[:,0]
    return Feature,Label

""" main function """
def main():
    
    rawData = get_data('faces.csv')
    #train = get_data('mnist_train.csv') 
    #test = get_data('mnist_test.csv') 
    # clean data
    #train = clean_data(train)
    #test = clean_data(test)
    cleanedData = clean_data(rawData)
    # now we have an 2d matrix of data 
    trainData, testData = train_test_split(cleanedData)
    print("dimension of trainData: ", trainData.shape)
    print("dimension of testData: ", testData.shape)
    
    trainX, trainY = partition(trainData)
    testX, testY = partition(testData)
    trainX = normalize(trainX)
    testX = normalize(testX)
    print("we should have this many weights: ", trainX.shape[1])
     
    
    ''' Hyperparameter Tuning Block'''
    #alphaList = np.zeros(2)
    #alphaList[0] = 0.1
    #alphaList[0] = 0.01
    #alphaList[1] = 0.001
    # alpha = 0.01 or 0.001 seems good
    #alpha = 0.01
    alpha = 0.001
    #for elem in alphaList:
    #    weights,bias = gradient_descent(trainX, trainY, elem, 1000, coeff)
    #    print(weights, bias)
    # try numitr = 2000 with alpha = 0.01
    W1, b1, W2, b2, cost = train2NN(trainX, trainY, alpha, 1000)
    print("check: ", b1.shape)
    print("check b2 : ", b2.shape )
    M = np.matmul(W1,testX.T)
    print("plz: ", M.shape)
    ''' display and plot cost '''
    plt.plot(cost)
    plt.xlabel(' number of iterations (100s) ')
    plt.ylabel('cost')
    plt.show
    
    ''' evaluate and print accuracy of our model '''
    print("comeon...: ", len(testY))
    newDim = len(testY)
    b1 = b1.T
    b2 = b2.T
    b1 = b1[:newDim]
    print(len(b1))
    b2 = b2[:newDim]
    b1 = b1.T
    b2 = b2.T
    accuracy, output = evaluate(W1, W2, b1, b2, testX, testY)
    accuracy = accuracy * 100
    print("accuracy of the model is: " + str(accuracy) + " %")
    
    ''' final '''
    #test4 = get_final('test_4.csv')
    #test6 = get_final('test_6.csv')
    # convert each to matrix and concetenate
    #X_4 = np.array(test4)
    #X_6 = np.array(test6)
    
    #final = np.concatenate((X_4, X_6), axis=0)
    # convert to float
    #final = final.astype(np.float)
    # normalize
    #final = normalize(final)
    #final = final / float(255)
    sample1 = open("hidden.txt", "w+")
    finalOutputs, hidden = evaluate_final(W1, W2, b1, b2, testX)
    print(hidden[0])
    print(type(hidden))
    for i in range(len(hidden)):
        sample1.write(str(hidden[i]) + ",")
    sample1.close()
    # modify so that Z's first column (1st sample)'s weight is stored in a file    
    #counters
    counter1 = 0
    counter0 = 0
    ''' see number of 4's and 6's '''
    for i in range(len(finalOutputs)):
        if finalOutputs[i] == 1.0:
            counter1 += 1
        else:
            counter0 += 1
    print("number of 1's: ", counter1)
    print("number of 0's: ", counter0)
   
    
    foFinal = open("output.txt", "w+")
    for i in range(len(output)):
        foFinal.write(str(int(output[i])) + "\n")
    foFinal.close()
    
    
main()    

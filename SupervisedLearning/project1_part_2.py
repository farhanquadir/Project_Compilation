#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:13:02 2019

@author: farhan
"""
import numpy as np
import csv
import matplotlib.pyplot as plt
from PIL import Image
import random
from sklearn.utils import shuffle
import os, sys

#workdir = "/home/farhan/Downloads/NN Class/Data/"
workdir = "/home/farhan/Downloads/NN_Class/Data/Part2/"
onefile_test = "Part2_1_Test.csv"
onefile_train = "Part2_1_Train.csv"
threefile_test = "Part2_3_Test.csv"
threefile_train = "Part2_3_Train.csv"


# Build a simple 1D convolution with shared weights
# sigmoid
def sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))
###################################################################################################################
# tanh loss function 
def tanh(x, derive=False):
    if derive:
        return (1 - x*x)
    return (np.exp(2*x)-1) / (1 + np.exp(2*x))
###################################################################################################################
# binary cross entropy
def crossEntropy (yHat,y, derive=False): #yHat = predicted, y = actual
    if derive:
        if y ==1:
            return (-1/yHat)
        else: 
             return (1/(1-yHat))
    if (y==1):
        return -np.log(yHat)
    else:
        return -np.log(1 - yHat)    
###################################################################################################################
#Mean Squared Error
def msqe(d,y,derive=False):
    if derive == True:
        return -1*(d-y)
    return 0.5 * (d-y)*(d-y)
###################################################################################################################
def relu(x, derive=False):
    if derive:
        y = np.ones(x.shape)
        fi = (x <= 0)
        y[fi] = 0
        return y
    return np.maximum(x,0)
#######################################################################################################################
def lrelu(x, alpha, derive=False):
    if derive:
        y = np.ones(x.shape)
        fi = (x <= 0)
        y[fi] = alpha
        return y
    else:
        y = np.copy(x)
        fi = (y <= 0)
        y[fi] = alpha * y[fi]
        return y
#######################################################################################################################
def loadData(filename):
    data = np.loadtxt(filename, delimiter=",")
    return data
########################################################################################################################
def activate(node, func):
    if func == "relu":
        return relu(node)
    if func == "sigmoid":
        return sigmoid(node)
    if func == "lrelu":
        return lrelu(node)
    if func == "tanh":
        return tanh(node)
        
    return 
########################################################################################################################
def normalize(v1,v2):
    s = v1+v2
    nv1 = v1/s
    nv2 = v2/s
    return (nv1,nv2)
########################################################################################################################
def predict(data, w1,w2):
    n1_value = np.dot(data,w1)
    n2_value = np.dot(data,w2)
    fi_n1 = activate(n1_value, "sigmoid")
    fi_n2 = activate(n2_value, "sigmoid")
    #fi_n1,fi_n2 = normalize(fi_n1,fi_n2) #not sure to normalize here
    
    Y_pred = np.zeros((2))
    Y_pred[0] = fi_n1
    Y_pred[1] = fi_n2
    return Y_pred
########################################################################################################################
########################################################################################################################
def loadWeights(folder,weights_file):
    file1 = folder+weights_file+"w1.txt"
    file2 = folder+weights_file+"w2.txt"
    if not (os.path.exists(file1)):
        print("Weights file not found!")
        print("Reinitializing weights...")
        return
    if not (os.path.exists(file2)):
        print("Weights file not found!")
        print("Reinitializing weights...")
        return
    print("Previous weights found...loading them...")
    w1=np.loadtxt(file1)
    w2=np.loadtxt(file2)
    #with open(workdir+folder+weights_file, "w") as f:
        
    return w1, w2
########################################################################################################################
def saveBestWeights(folder,weights_file,w1,w2):
    if not (os.path.exists(folder)):
        
        os.mkdir(folder) #problem in space in "NN Class"
    #    print("Here")
    
    file1 = folder+weights_file+"w1.txt"
    file2 = folder+weights_file+"w2.txt"
    if (os.path.exists(file1)):
        print("Removing previously saved weights!")
        os.system("rm -f "+file1)
    if (os.path.exists(file2)):
        os.system("rm -f "+file2)
    print("Saving new weights...")
    np.savetxt(file1, w1)
    np.savetxt(file2, w2)
    #with open(workdir+folder+weights_file, "w") as f:
       
    return
########################################################################################################################
def learn(X, Y, w1, w2, X_test, Y_test, lr, epochs, activation="relu"):
    
    error = np.zeros((epochs))
    val_error = np.zeros((epochs))
    #print(error)
    
    index = np.array(range(len(X)))
    indextest = np.array(range(len(X_test)))
    index = shuffle(index)
    indextest = shuffle(indextest)
    #print(error)
    for epoch in range(epochs):
        #Forward pass
        error[epoch] = 0
        for i in range(len(X)): #go through the training data
            indx = index[i]
            data = X[indx]
            #Forward Pass
            n1_value = np.dot(data,w1)
            n2_value = np.dot(data,w2)
            fi_n1 = activate(n1_value, "sigmoid")
            fi_n2 = activate(n2_value, "sigmoid")
            #print(indx, fi_n1, fi_n2)
            y0 = int(Y[indx][0])
            y1 = int(Y[indx][1])
            #print(y0)
            #print(y1)
            #fi_n1, fi_n2 = normalize(fi_n1, fi_n2)
            #Back prop
            error_n1 = crossEntropy(fi_n1, y0, derive= False)
            error_n2 = crossEntropy(fi_n2, y1, derive= False)
            #print(i,error_n1,error_n2)
            error[epoch] += (error_n1 + error_n2)
            #print(epoch, error_n1,error_n2, error[epoch])
            delta_Ey_n1 = crossEntropy(fi_n1, y0, True)
            delta_Ey_n2 = crossEntropy(fi_n2, y1, True)
            
            delta_yv_n1 = sigmoid(fi_n1,True)
            delta_yv_n2 = sigmoid(fi_n2,True)
            
            del_w1= np.ones((784))
            del_w2= np.ones((784))
            #Calculate delta_w for both neurons
            for j in range(len(del_w1)):
                del_w1[j] = delta_Ey_n1 * delta_yv_n1 * data[j]
                del_w2[j] = delta_Ey_n2 * delta_yv_n2 * data[j]
                w1[j]-= lr * del_w1[j]
                w2[j]-= lr * del_w2[j]
        
        #Test the network
        TP = 0
        for i in range(len(X_test)):
            indx= indextest[i]
            Y_pred = predict(X_test[indx], w1,w2)
            Y_actual = Y_test[indx]
            err1 = msqe(Y_actual[0],Y_pred[0],False)
            err2 = msqe(Y_actual[1],Y_pred[1],False)
            val_error[epoch] += err1 + err2
            if (Y_pred[0]>Y_pred[1]):
                Y_pred[0]=1
                Y_pred[1]=0
            else:
                Y_pred[0]=0
                Y_pred[1]=1
            #print(np.shape(Y_pred),np.shape(Y_actual))
            if np.array_equal(Y_pred, Y_actual):
                TP+=1
            #if epoch == epochs-1:
            #    print(indx, Y_pred, Y_actual)
    """
    for epoch in range(epochs):
        print(epoch, error[epoch])
    print("TP=",TP)
    plt.plot(error, label = "Training Loss only")
    plt.legend()
    plt.close()
    plt.plot(val_error, label = "Validation Loss only")
    plt.legend()
    plt.close()
    plt.plot(val_error, label = "Validation loss")
    plt.plot(error, label = "Training Loss")
    #plt.show()
    plt.legend()
    """
    return TP, error, val_error, w1, w2
########################3 Main Program #################################################################################

#Load the train and test data

one_train = loadData(workdir+onefile_train)
one_test = loadData(workdir+onefile_test)
three_train = loadData(workdir+threefile_train)
three_test = loadData(workdir+threefile_test)

#create X dataset for all datas
X_train = np.concatenate((one_train,three_train))
X_test = np.concatenate((one_test,three_test))

#set labelsprint_detailed_accuracy_on_this_data(pdb_list,diffinvmap,epoch
#l1 = np.array([1,0]) #[1,0] if one
#l3 = np.array([0,1]) #[1,0] if three

l1 = np.zeros((2))
l1[0] =1 #[1,0] if one

l3 = np.zeros((2))
l3[1] =1 #[0,1] if three

#Y = np.array([])
Y = []
for i in range(17+17):
    if i < len(one_train):
        Y.append(l1)
    else:
        Y.append(l3)
Y_train = np.asarray(Y)
#Y_train = Y
Y = []

for i in range(100):
    if i < len(one_test):
        Y.append(l1)
    else:
        Y.append(l3)
Y_test = np.asarray(Y)

#set epochs
epochs = 1000

#set learning rate
lr = 0.01

#initialize random weights

#w1 = np.random.uniform(-1,1,(784)) # range [-2,2]
#w2 = np.random.uniform(-1,1,(784)) # range [+2,2]

folder = "best_weights/"
weights_file = "best_weights_bncr"+str(epochs)+"_"+str(lr)+"_"

TP_old = 0
error_best = np.zeros((epochs))
val_error_best = np.zeros((epochs))

for out_epochs in range (10):
    #initialize random weights
    w1 = np.random.uniform(-1,1,(784)) # range [-2,2]
    w2 = np.random.uniform(-1,1,(784)) # range [+2,2]
    TP_new, error, val_error, w1_new, w2_new = learn(X_train, Y_train, w1, w2, X_test, Y_test, lr, epochs=1000)
    

    if (TP_new > TP_old):
        TP_old = TP_new
        saveBestWeights(workdir+folder,weights_file,w1_new,w2_new)
        error_best = error
        val_error_best = val_error
        if (os.path.exists(workdir+"train_error_bncr.txt")):
            print("Removing...\n"+workdir+"train_error_bncr.txt")
            os.system("rm -f "+workdir+"train_error_bncr.txt")
        if (os.path.exists(workdir+"val_error_bncr.txt")):
            print("Removing...\n"+workdir+"val_error_bncr.txt")
            os.system("rm -f "+workdir+"val_error_bncr.txt")
        np.savetxt(workdir+"train_error_bncr.txt",error_best)
        np.savetxt(workdir+"val_error_bncr.txt",val_error_best)

print(TP_old)
plt.plot(val_error_best, label = "Validation loss")
plt.plot(error_best, label = "Training Loss")

plt.legend()
#plt.close()

#plt.imshow(w1_new.reshape(28,28))
#plt.imshow(w2_new.reshape(28,28))
#plt.show()

#print(error)

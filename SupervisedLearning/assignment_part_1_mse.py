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
workdir = "/data/farhan/SupervisedLearning/"
x_data = "X.txt"
y_data = "Y.txt"


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
def mse(d,y,derive=False):
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
    data = np.loadtxt(filename)
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
def predict(data, w1):
    n1_value = np.dot(data,w1)
    #n2_value = np.dot(data,w2)
    fi_n1 = activate(n1_value, "sigmoid")
    #fi_n2 = activate(n2_value, "sigmoid")
    #fi_n1,fi_n2 = normalize(fi_n1,fi_n2) #not sure to normalize here
    Y_pred = np.zeros((1))
    Y_pred[0] = fi_n1
    #Y_pred[1] = fi_n2
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
def getTestTrain(data):
    L=len(data)
    l_test=int(0.2*L)
    l_tr=L-l_test
    index = np.array(range(L))
    index = shuffle(index)
    train=[]
    test=[]
    for i in range(L):
        if (i<=l_tr):
            train.append(data[index[i]])
        else:
            test.append(data[index[i]])

    return train,test
########################################################################################################################
def learn(X, Y, w1, w2, X_test, Y_test, lr, epochs, activation="relu"):
    error = np.zeros((epochs))
    val_error = np.zeros((epochs))
    acc=np.zeros((epochs))
    index = np.array(range(len(X)))
    indextest = np.array(range(len(X_test)))
    index = shuffle(index)
    indextest = shuffle(indextest)

    for epoch in range(epochs):
        for i in range(len(X)): #go through the training data
            indx = index[i]
            data = X[indx]
            
            #Forward Pass
            n1_value = np.dot(data,w1)
            #n2_value = np.dot(data,w2)
            fi_n1 = activate(n1_value, "sigmoid")
            #fi_n2 = activate(n2_value, "sigmoid")
            y0 = int(Y[indx][0])
            #y1 = int(Y[indx][1])
            #Back prop
            error_n1 = mse(y0, fi_n1, derive= False)
            #error_n2 = mse(y1, fi_n2, derive= False)
            error[epoch] += (error_n1)# + error_n2)
            delta_Ey_n1 = mse(y0, fi_n1, True)
            #delta_Ey_n2 = mse(y1, fi_n2, True)
            delta_yv_n1 = sigmoid(fi_n1,True)
            #delta_yv_n2 = sigmoid(fi_n2,True)
            del_w1= np.ones((5))
            #del_w2= np.ones((784))
            #Calculate delta_w for both neurons
            for j in range(len(del_w1)):
                del_w1[j] = delta_Ey_n1 * delta_yv_n1 * data[j]
                #del_w2[j] = delta_Ey_n2 * delta_yv_n2 * data[j]
                w1[j]-= lr * del_w1[j]
                #w2[j]-= lr * del_w2[j]
        TP = 0
        for i in range(len(X_test)):
            indx= indextest[i]
            Y_pred = predict(X_test[indx], w1)
            Y_actual = Y_test[indx]
            err1 = mse(Y_actual[0],Y_pred[0],False)
            #err2 = mse(Y_actual[1],Y_pred[1],False)
            val_error[epoch] += err1 #+ err2
            if (Y_pred[0]>0.5):
                Y_pred[0]=1
                #Y_pred[1]=0
            else:
                Y_pred[0]=0
                #Y_pred[1]=1
            if (Y_pred[0]== Y_actual[0]):
                TP+=1
        acc[epoch]=100*TP/8
    return TP, error, val_error, w1, acc
########################3 Main Program #################################################################################

#Load the train and test data

x_data=loadData(workdir+x_data)
y_data=loadData(workdir+y_data)
ones=np.ones((len(x_data),1))
sqr=np.ones((len(x_data),1))
for i in range(len(x_data)):
    sqr[i]=x_data[i]*x_data[i]

x_data=np.append(ones,x_data,axis=1)
x_data=np.append(x_data,sqr,axis=1)

#print(s_data)
#print(len(X_train),len(Y_train))
#print(len(X_test),len(Y_test))
#print(Y_test)
#set learning rate

lr = 0.01
epochs = 1000
folder = "best_weights/"
weights_file = "best_weights_mse"+str(epochs)+"_"+str(lr)+"_"

TP_old = 0
error_best = np.zeros((epochs))
val_error_best = np.zeros((epochs))
acc=np.zeros((epochs))
for out_epochs in range (1):
    #initialize random weights
    w1 = np.random.uniform(-1,1,(3)) # range [-1,1]
    w2 = np.random.uniform(0,1,(784))*0.001 # range [0,1]
    TP_new, error, val_error, w1_new, acc_new = learn(X_train, Y_train, w1, w2, X_test, Y_test, lr, epochs=1000)

    if (TP_new > TP_old):
        TP_old = TP_new
        #saveBestWeights(workdir+folder,weights_file,w1_new)
        error_best = error
        val_error_best = val_error
        acc=acc_new
        if (os.path.exists(workdir+"train_error_mse.txt")):
            print("Removing...\n"+workdir+"train_error_mse.txt")
            os.system("rm -f "+workdir+"train_error_mse.txt")
        if (os.path.exists(workdir+"val_error_mse.txt")):
            print("Removing...\n"+workdir+"val_error_mse.txt")
            os.system("rm -f "+workdir+"val_error_mse.txt")
        np.savetxt(workdir+"train_error_mse.txt",error_best)
        np.savetxt(workdir+"val_error_mse.txt",val_error_best)

print(TP_old)
plt.plot(val_error_best, label = "Validation loss")
plt.plot(error_best, label = "Training Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.title("Validation and Test Error Plots")

#plt.plot(acc, label = "Training Loss")
plt.show()
#plt.legend()
#plt.imshow(w1_new.reshape(28,28))
#plt.imshow(w2_new.reshape(28,28))
#plt.show()


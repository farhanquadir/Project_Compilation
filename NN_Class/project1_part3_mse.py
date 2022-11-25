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
import pandas as pd

#workdir = "/home/farhan/Downloads/NN Class/Data/"
workdir = "/home/farhan/Downloads/NN_Class/Data/Part3/"
fileprefix = "Part3_"
tail_test ="_Test.csv"
#file_test = "Part3_
tail_train = "_Train.csv"
number = np.array(range(10))

onefile_test = "Part3_1_Test.csv"
onefile_train = "Part3_1_Train.csv"
twofile_test = "Part3_2_Test.csv"
twofile_train = "Part3_2_Train.csv"
threefile_test = "Part3_3_Test.csv"
threefile_train = "Part3_3_Train.csv"
fourfile_test = "Part3_4_Test.csv"
fourfile_train = "Part3_4_Train.csv"
fivefile_test = "Part3_5_Test.csv"
fivefile_train = "Part3_5_Train.csv"
sixfile_test = "Part3_6_Test.csv"
sixfile_train = "Part3_6_Train.csv"
sevenfile_test = "Part3_7_Test.csv"
sevenfile_train = "Part3_7_Train.csv"
eightfile_test = "Part3_8_Test.csv"
eightfile_train = "Part3_8_Train.csv"
ninefile_test = "Part3_9_Test.csv"
ninefile_train = "Part3_9_Train.csv"
zerofile_test = "Part3_0_Test.csv"
zerofile_train = "Part3_0_Train.csv"


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
    """
    if derive:
        y = np.ones(x.shape)
        fi = (x <= 0)
        y[fi] = 0
        return y
    """
    if derive:
        if x<=0:
            return 0
        else:
            return 1
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
def activate(node, func, derive=False):
    if func == "relu":
        return relu(node,derive)
    if func == "sigmoid":
        return sigmoid(node,derive)
    if func == "lrelu":
        return lrelu(node,derive)
    if func == "tanh":
        return tanh(node,derive)
        
    return 
########################################################################################################################
def normalize2(v1,v2):
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
    fi_n1,fi_n2 = normalize(fi_n1,fi_n2) #not sure to normalize here
    Y_pred = np.zeros((2))
    Y_pred[0] = fi_n1
    Y_pred[1] = fi_n2
    return Y_pred
########################################################################################################################
def normalize(A):
    l = A.shape[0]
    total = A.sum()
    for i in range(l):
        A[i] = A[i]/total
    return A
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
        os.system("rm "+file1)
    if (os.path.exists(file2)):
        os.system("rm "+file2)
    print("Saving new weights...")
    np.savetxt(file1, w1)
    np.savetxt(file2, w2)
    #with open(workdir+folder+weights_file, "w") as f:
       
    return
########################################################################################################################
def slide(data, w):
    #out = np.zeros((22*22))
    stride = 1
    nl = int(((data.shape[0]-w.shape[0])/stride)+1)
    #print(nl)
    out = np.zeros((nl,nl))
    l = len(data)
    w_flat = w.flatten()
    
    io = 0
    jo = 0
    for i in range(0,l,stride):
        if (i+len(w)>l):
            break
        for j in range(0,l,stride):
            if (j+len(w)>l):
                break
            
            portion = data[i:i+len(w),j:j+len(w)]
            portion = portion.flatten()
                       
            portion_val = np.dot(portion,w_flat)
            
            out[io,jo] = portion_val
            jo+=1
            if (jo==nl):
                io+=1
                jo=0
            if (io==nl):
                break
        #io+=1
    return out #out.reshape(22,22)
########################################################################################################################
def softMax(a):
    #from scipy.special import softmax
    l = a.shape[0]
    print(a)
    sum_exp =0
    for i in range (l):
        sum_exp += np.exp(a[i])
    for i in range(l):
        a[i]= np.exp(a[i])/sum_exp
    return a
########################################################################################################################
########################################################################################################################
def softMaxDer(a):
    #from scipy.special import softmax
    #s = softmax(a)
    v = np.zeros((a.shape[0]))
    sum_exp = 0
    for i in range(10):
        sum_exp += np.exp(a[i])
    for i in range(10):
        v[i] = (np.exp(a[i])*(sum_exp-np.exp(a[i])))/((sum_exp*sum_exp))
    return v
########################################################################################################################
#def learn(X, Y, w1, w2, X_test, Y_test, lr, epochs, activation="relu"):
def learn(X, Y, w1, w2, X_test, Y_test, lr, epochs, activation="relu"):
    filternum = len(w1)
    l1_size = 22
    l1_out = np.zeros((filternum,l1_size,l1_size))
    
    #for i in range(filternum):
    #    l1
    error = np.zeros((epochs))
    val_error = np.zeros((epochs))
    #print(error)
    
    index = np.array(range(len(X)))
    indextest = np.array(range(len(X_test)))
    index = shuffle(index)
    indextest = shuffle(indextest)
    TP = 0
    #print(error)
    for epoch in range(epochs):
        #Forward pass
        error[epoch] = 0
        err = 0
        for i in range(len(X)): #go through the training data
            indx = index[i]
            #indx = 0
            data = X[indx]
            #print(data.shape)
            #sys.exit()
            #Forward Pass
            #Layer1
            for nw1 in range(len(w1)):
                l1_out[nw1] = slide(data,w1[nw1])
                print("l1_out=,", l1_out[nw1])
            
            #Flatten to 16*22*22=7744
            #print(l1_out.shape)
            l1_V = l1_out.flatten()
            
            #print(l1_V)
            l1_V_relu = np.zeros((16*22*22))
            l1_V_relu_der = np.zeros((16*22*22))
            #Activation
            for rel in range(len(l1_V)):
                l1_V_relu[rel] = activate(l1_V[rel],"relu")
                #l1_V_relu[rel] = activate(l1_V[rel],"sigmoid")
                #l1_V_relu[rel] = activate(l1_V[rel],"tanh")
                print(l1_V_relu[rel])
                #l1_V_relu_der[rel] = relu(l1_V_relu[rel], True)
                #print(l1_V_relu_der[rel])
                l1_V_relu_der[rel] = relu(l1_V[rel], True) #derivative
                #l1_V_relu_der[rel] = sigmoid(l1_V[rel], True) #derivative
                #l1_V_relu_der[rel] = tanh(l1_V[rel], True) #derivative
                #print(l1_V_relu_der[rel])
            
            #l1_V_relu = normalize(l1_V_relu) #Haywire
            #for rel in range(len(l1_V)):
            #    l1_V_relu_der[rel] = sigmoid(l1_V[rel], True)
            
            #Mapping 16*22*22 nodes to 10 output nodes:
            Y_out = np.zeros((10))
            for nw2 in range(len(w2)):
                vall = np.dot(l1_V_relu,w2[nw2])
                #print(vall)
                
                #print(w2[nw2])
                Y_out[nw2] = vall
                #print("Vall=",vall)
                if (vall == "nan"):
                    sys.exit()
                #print(vall)
                #Y_out[nw2] = activate(np.dot(l1_V_relu,w2[nw2]),"sigmoid") #creating problems here vall is very large
                #Y_out[nw2] = vall
                #print(Y_out)
                #print((np.dot(l1_V_relu,w2[nw2])).shape)
            #print("Before: ", Y_out)
            Y_out = softMax(Y_out)
            sm_der = softMaxDer(Y_out)
            #print("After: ", Y_out)
            for oo in range(10):
                #Y_out[oo] = activate(Y_out[00],"sigmoid")
                #print("Loop Y:",Y_out[oo])
                print("")
            #Calculate error for 10
            mse_out = np.zeros((10))
            delta_Ey = np.zeros((10))
            delta_yv = np.zeros((10))
            #print(Y_out)
            for n in range(len(Y_out)):
                mse_out[n] = mse(Y[indx][n],Y_out[n])
                #print(mse_out[n])# = mse(Y[indx][n],Y_out[n]))
                delta_Ey[n] = mse(Y[indx][n],Y_out[n], True)
                #delta_yv[n] = sigmoid(Y_out[n], True)
                delta_yv[n] = sm_der[n]
            
            #Total error
            error[epoch] += mse_out.sum()
            #print("Epoch =",epoch,"Error=",error[epoch])
            
            #backprop
            #Layer = H
            del_w2 = np.ones((10,16*22*22))
            for j in range(len(del_w2)):
                #del_w2[j] = delta
                for bg in range(16*22*22):
                    del_w2[j][bg] = delta_Ey[j] * delta_yv[j] * l1_V_relu[bg]
            #Layer = H-1
            s_delta = np.zeros((16*22*22))
            for m in range(16*22*22):
                for n in range(10):
                    s_delta[m] += delta_Ey[n] * delta_yv[n] * w2[n][m]
                    
            del_w1 = np.zeros((filternum,7,7))
            for filtnum in range(filternum):
               
                for x in range(7):
                    for y in range(7):
                        #del_w1[0][x][y]+= inp * l1_V_relu_der[0] * delta_Ey[0] * delta_yv[0] * w2[0][0] #x 10
                        for outnumx in range(22):
                            for outnumy in range(22):
                                inp = data[x+outnumx][y+outnumy]             #22 for each filter
                                
                                #del_w1[filtnum][x][y]+= inp * l1_V_relu_der[filtnum*22+outnumy] * delta_Ey[outnum] * delta_yv[outnum] * w2[filtnum][y+outnumy] #x 10
                                del_w1[filtnum][x][y]+= inp * l1_V_relu_der[filtnum*22+outnumy] * s_delta[filtnum*22+outnumy]
            
            
            #Update weights
            #w2
            
            for n in range(len(w2)):
                w2[n]-= lr * del_w2[n]
            #w1
            for n in range(len(w1)):
                w1[n]-= lr * del_w1[n]
        #print("Epoch =",epoch,"Error=",error[epoch])
        with open(workdir+"error.log","w+") as f:
            f.write(str(error[epoch])+"\n")
        #Validation
        TP = 0
        for i in range(len(X_test)): #go through the training data
            indx = index[i]
            #indx = 0
            data = X_test[indx]
            #print(data)
            #Forward Pass
            #Layer1
            for nw1 in range(len(w1)):
                l1_out[nw1] = slide(data,w1[nw1])
                #print(l1_out[nw1].shape)
            #Flatten to 16*22*22=7744
            
            l1_V = l1_out.flatten()
            
            l1_V_relu = np.zeros((16*22*22))
            l1_V_relu_der = np.zeros((16*22*22))
            for rel in range(len(l1_V)):
                l1_V_relu[rel] = activate(l1_V[rel],"relu")
                #l1_V_relu_der[rel] = relu(l1_V_relu[rel], True)
                l1_V_relu_der[rel] = relu(l1_V[rel], True)
            
            #Mapping 16*22*22 nodes to 10 output nodes:
            Y_out_test = np.zeros((10))
            for nw2 in range(len(w2)):
                Y_out_test[nw2] = activate(np.dot(l1_V_relu,w2[nw2]),"sigmoid")
                #print(Y_out[nw2])
                #print((np.dot(l1_V_relu,w2[nw2])).shape)
            #Calculate error for 10
            mse_out_val = np.zeros((10))
            #delta_Ey = np.zeros((10))
            #delta_yv = np.zeros((10))
            for n in range(len(Y_out)):
                mse_out_val[n] = mse(Y_test[indx][n],Y_out_test[n])
                #delta_Ey[n] = mse(Y_test[indx][n],Y_out[n], True)
                #delta_yv[n] = sigmoid(Y_out[n], True)
            Y_soft = softMax(Y_out_test)
            Y_pred = np.zeros((10))
            maxindx = Y_soft.argmax()
            Y_pred[maxindx] = 1
            if (np.array_equal(Y_pred, Y_test[indx])):
                TP+=1
            #Total error
            val_error[epoch] += mse_out_val.sum()
        
        #print(epoch, TP)
        
        with open(workdir+"val_error.log","w+") as f:
            f.write(str(val_error[epoch])+"\n")
        
        #print("Epoch =",epoch,"Error=",error[epoch], "Val_error=", val_error[epoch])
            
            
            
    
    return error, val_error, TP, w1, w2        
            
########################3 Main Program #################################################################################

#Load the train and test data
data_train = []
data_test = []

for i in range(10):
    num = str(i)
    data_train.append(loadData(workdir+fileprefix+num+tail_train))
    data_test.append(loadData(workdir+fileprefix+num+tail_test))

#print(len(data_train), len(data_test))

X_train = np.concatenate((data_train[0],data_train[1]))
X_test = np.concatenate((data_test[0],data_test[1]))

#create X dataset for all datas

for i in range(2,len(data_train)):
    X_train = np.concatenate((X_train,data_train[i]))
    X_test = np.concatenate((X_test,data_test[i]))



X_train_T = np.zeros((len(X_train), 28,28))


for im in range (len(X_train)):
    X_train_T[im] = X_train[im].reshape(28,28).T

X_test_T = np.zeros((len(X_test), 28,28))

for im in range (len(X_test)):
    X_test_T[im] = X_test[im].reshape(28,28).T

#Labels
Y_train =np.zeros((5000,10))
Y_test =np.zeros((5000,10))

pos = 0
for i in range(0,len(Y_train),500):
    for j in range(500):
        Y_train[i+j][pos] = 1
        Y_test[i+j][pos] = 1
    pos+=1

#Random weights 16 7x7 filters
w1 = np.zeros((16,7,7)) #Filters for layer 1
for i in range (16):
    w_init = np.random.uniform(0,1,(49)) # range [0,1]
    w1[i] = w_init.reshape(7,7)

#For mapping 16*22*22 to 10 outputs
w2 = np.zeros((10,16*22*22))
for i in range (10):
    w2[i] = np.random.uniform(-1,1,(16*22*22)) # range [0,1]
    
#print(w2.shape)
ii = np.random.randint(0,4999,20)
X_small_train = np.zeros((20,28,28))
Y_small_train = np.zeros((20,10))
X_small_test = np.zeros((20,28,28))
Y_small_test = np.zeros((20,10))

#print(ii.shape)

for s in range(20):
    
    X_small_train[s] = X_train_T[ii[s]]
    Y_small_train[s] = Y_train[ii[s]]
    X_small_test[s] = X_test_T[ii[s]]
    Y_small_test[s] = Y_test[ii[s]]

#error, val_error, TP, w1_new, w2_new = learn(X_train_T, Y_train, w1, w2, X_test_T, Y_test, lr=0.01, epochs=100, activation="relu")

error, val_error, TP, w1_new, w2_new = learn(X_small_train, Y_small_train, w1, w2, X_small_test, Y_small_test, lr=0.01, epochs=10, activation="relu")

plt.plot(error, label = "Training Loss")
plt.plot(val_error, label = "Validation Loss")
plt.legend()

#set learning rate
lr = 0.01
epochs = 0

folder = "best_weights/"
weights_file = "best_weights_mse"+str(epochs)+"_"+str(lr)+"_"

np.savetxt(workdir+"error_p3.txt", error)
np.savetxt(workdir+"val_error_p3.txt", val_error)
np.savetxt(workdir+"w1_p3.txt", w1.flatten())
np.savetxt(workdir+"w2_p3.txt", w2.flatten())


#TP_old = 0
#error_best = np.zeros((epochs))
#val_error_best = np.zeros((epochs))
"""
for out_epochs in range (10):
    #initialize random weights
    w1 = np.random.uniform(-1,1,(784)) # range [-1,1]
    w2 = np.random.uniform(-1,1,(784)) # range [+1,1]
    TP_new, error, val_error, w1_new, w2_new = learn(X_train, Y_train, w1, w2, X_test, Y_test, lr, epochs=1000)
    

    if (TP_new > TP_old):
        TP_old = TP_new
        saveBestWeights(workdir+folder,weights_file,w1_new,w2_new)
        error_best = error
        val_error_best = val_error
        if (os.path.exists(workdir+"test_error.txt")):
            print("Removing...\n"+workdir+"test_error.txt")
            os.system("rm "+workdir+"test_error.txt")
        if (os.path.exists(workdir+"val_error.txt")):
            print("Removing...\n"+workdir+"val_error.txt")
            os.system("rm "+workdir+"val_error.txt")
        np.savetxt(workdir+"test_error.txt",error_best)
        np.savetxt(workdir+"val_error.txt",val_error_best)

print(TP_old)

"""
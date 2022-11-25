#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:25:13 2019

@author: farhan
"""

import numpy as np
import sys
import random

# our nonlinear function (and its derivative); lam = 1 (so fixed)
def sigmoid(x, derive=False):
    if derive:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))
###################################################################################################################3
def tanh(x, derive=False):
    if derive:
        return (1 - x*x)
    return (np.exp(2*x)-1) / (1 + np.exp(2*x))
###################################################################################################################
#Mean Squared Error
def msqe(d,y,derive=False):
    if derive == True:
        return -1*(d-y)
    return 0.5 * (d-y)*(d-y)
###################################################################################################################


# define the XOR data set
X =np.array([
        [0.0, 0.0, 0.8, 0.4, 0.4, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.3, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.8, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.4, 0.4, 0.1],
        [0.8, 0.4, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.8, 0.3]])

    # its labels
y = np.array([
        [-1],
        [1],
        [1],
        [-1],
        [-1],
        [1]])

# learning rate
eta = 0.2

# initialize weights with random numbers
#n1_w = np.random.normal(0,1,(3, 1))
#n2_w = np.random.normal(0,1,(3, 1))
#n3_w = np.random.normal(0,1,(3, 1))

n1_w = green_ = np.array([1.73673761, 1.89791391, -2.10677342, -0.14891209, 0.58306155])

n2_w = orange_ = np.array([-2.25923303, 0.13723954, -0.70121322, -0.62078008, -0.47961976])

n3_w = wout_ = np.array([1.20973877, -1.07518386, 0.80691921, -0.29078347, -0.22094764, -0.16915604, 1.10083444, 0.08251052, 
                 -0.00437558, -1.72255825, 1.05755642, -2.51791281, -1.91064012])

###############################################
# Epochs
###############################################
epoch = 1 #1000 # how many epochs? (each epoch will pass all 4 data points through)
err = np.zeros((epoch,1)) # lets record error to plot (get a convergence plot)
inds = np.asarray([0,1,2,3,4,5]) # array of our 4 indices (data point index references)
for k in range(1): 
    
    # init error
    err[k] = 0    
    
    # random shuffle of data each epoch
    
    #inds = np.random.permutation(inds)
    
    for i in range(4): 
        
        # what index?
        inx = inds[i]
        
        # forward pass
        v = np.ones((12, 1))
        
        for i in range (6):
            v[i] = np.dot(X[i:i+4],n1_w[0:4])+n1_w[4]
            v[i]   = tanh[v[i]]
        for i in range (6,12):
            v[i] = np.dot(X[i-6:i-6+4],n2_w[0:4])+n2_w[4]
            v[i]   = tanh[v[i]]
        
        
        #v[0] = np.dot(X[inx,:], n1_w) # neuron 1 fires (x as input)
        #v[0] = tanh(v[0])        # neuron 1 sigmoid
        #v[1] = np.dot(X[inx,:], n2_w) # neuron 2 fires (x as input)
        #v[1] = sigmoid(v[1])    
        
        oo = np.dot(np.transpose(v), n3_w) # neuron 3 fires, taking neuron 1 and 2 as input
        o = tanh(oo) # hey, result of our net!!!
        
        # error
        err[k] = err[k] + ((1.0/2.0) * np.power((o - y[inx]), 2.0))
        
        # backprop time!!!
        
        # output layer
        delta_1 = o - y[inx]
        delta_2 = tanh(o,derive=True)
        # now, lets prop it back to the weights
        delta_ow = np.ones((13, 1))
        # format is
        # delta_index = (input to final neuron) * (Err derivative * Sigmoid derivative)
        
        for l in range(12):
            delta_ow[l] = v[l]  *  (delta_1*delta_2)
        
        delta_ow[12] = 1  *  (delta_1*delta_2)
        
        #delta_ow[0] = v[0]  *  (delta_1*delta_2)
        #delta_ow[1] = v[1]  *  (delta_1*delta_2)
        #delta_ow[2] = v[2]  *  (delta_1*delta_2)
        #print(v[2])
        
        
        # neuron n1
        
        #delta_3 = tanh(v[0],derive=True)
        delta_n = np.ones((12,1))
        delta_n = delta_n.squeeze()
        
        for l in range(12): #neuron n1 to n12
            delta_n[l] = tanh(v[l],derive=True)
        
        # same, need to prop back to weights
        delta_hw1 = np.zeros((5, 1))
        # format
                #              input     this Tanh der     error from output   weight to output neuron
        for l in range(4):
            for m in range(6):
                delta_hw1[l] = X[inx,l+m]  *  delta_n[m]  *  ((delta_1*delta_2)   *n3_w[m])
        
        #for j in range(4):
        #        for i in range (6):
        #            delta_green[j] += X[j+i] * tanh(V_tanh[i], True) * delta_Ey * delta_yv * wout[i]
        #            delta_orange[j] += X[j+i] * tanh(V_tanh[i+6], True) * delta_Ey * delta_yv * wout[i+6]
        
        #delta_hw1[0] = X[inx,0]  *  delta_n[]  *  ((delta_1*delta_2)   *n3_w[0])
        #delta_hw1[1] = X[inx,1]  *  delta_n[]  *  ((delta_1*delta_2)   *n3_w[0])
        #delta_hw1[2] = X[inx,2]  *  delta_n[]  *  ((delta_1*delta_2)   *n3_w[0])     
        
        # neuron n2
        delta_4 = sigmoid(v[1],derive=True)
        # same, need to prop back to weights        
        delta_hw2 = np.ones((5, 1))
        delta_hw2[0] = X[inx,0]  *  delta_4  *   ((delta_1*delta_2)   *n3_w[1])
        delta_hw2[1] = X[inx,1]  *  delta_4  *   ((delta_1*delta_2)   *n3_w[1])
        delta_hw2[2] = X[inx,2]  *  delta_4  *   ((delta_1*delta_2)   *n3_w[1])
        
        # update rule, so old value + eta weighted version of delta's above!!!
        n1_w = n1_w - eta * delta_hw1
        n2_w = n2_w - eta * delta_hw2
        n3_w = n3_w - eta * delta_ow
      
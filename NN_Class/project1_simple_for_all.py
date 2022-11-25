#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 21:44:20 2019

@author: farhan
"""

import numpy as np


# Declare global variables
#learning rate 

lr = 0.2

#inputs X
X_inp =np.array([
        [0.0, 0.0, 0.8, 0.4, 0.4, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.3, 0.8, 0.3, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.8, 0.3, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.4, 0.4, 0.1],
        [0.8, 0.4, 0.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.8, 0.3]])

# labels y
Y_inp = np.array([
        [-1],
        [1],
        [1],
        [-1],
        [-1],
        [1]])

# shared weights. Last element is the bias

green_ = np.array([1.73673761, 1.89791391, -2.10677342, -0.14891209, 0.58306155])

orange_ = np.array([-2.25923303, 0.13723954, -0.70121322, -0.62078008, -0.47961976])

wout_ = np.array([1.20973877, -1.07518386, 0.80691921, -0.29078347, -0.22094764, -0.16915604, 1.10083444, 0.08251052, 
                 -0.00437558, -1.72255825, 1.05755642, -2.51791281, -1.91064012])



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
#Mean Squared Error
def msqe(d,y,derive=False):
    if derive == True:
        return -1*(d-y)
    return 0.5 * (d-y)*(d-y)
###################################################################################################################
def learn(X, Y, green, orange, wout, lr, epochs=1):
    for _ in range (epochs):
            
        #initialize V and psi(V_tanh) function 
        V = np.ones((12,1))
        V_tanh = np.ones((13,1))
     
        #forward pass first layer
        
        for i in range (6):
            V[i] = np.dot(X[i:i+4],green[0:4])+green[4]
            V_tanh[i] = tanh(V[i])
        
        
        for i in range (6,12):
            V[i] = np.dot(X[i-6:i-6+4],orange[0:4])+orange[4]
            V_tanh[i] = tanh(V[i])
       
        #forward  pass second layer
        
        
        #OV = np.dot(V_tanh.transpose()[:],wout[:])
        OV = np.dot(V_tanh.squeeze()[:],wout[:])
        OV_tanh = tanh(OV)
        #end of forward pass
        #print (msqe(OV_tanh, Y[0]))
        
        delta_Ey = msqe(OV_tanh, Y[0], True)
        delta_yv = tanh(OV_tanh, True)
        
        #Output layer weights wout
        delta_wout = np.ones((13,1))
        for i in range(13):
            delta_wout[i]= delta_Ey * delta_yv * V_tanh[i]
        
        #initialize  the changes in weights for shared weights
            
        delta_green = np.zeros((5,1))
        delta_green= delta_green.squeeze()
        
        delta_orange = np.zeros((5,1))
        delta_orange= delta_orange.squeeze()
        """
        for j in range(4):
            for i in range(6) :
                #delta          =  input * delta_yh *  
                delta_green[j] += X[j+i] * tanh(V_tanh[i],True) * delta_Ey * delta_yv * wout[j+i]
                delta_orange[j] += X[j+i] * tanh(V_tanh[i+6],True) * delta_Ey * delta_yv * wout[j+i+6]
        """
        """
        for i in range (6):
            delta_green[0] += X[i] * tanh(V_tanh[i], True) * delta_Ey * delta_yv * wout[i]
            delta_green[1] += X[i+1] * tanh(V_tanh[i], True) * delta_Ey * delta_yv * wout[i]
            delta_green[2] += X[i+2] * tanh(V_tanh[i], True) * delta_Ey * delta_yv * wout[i]
            delta_green[3] += X[i+3] * tanh(V_tanh[i], True) * delta_Ey * delta_yv * wout[i]
            delta_green[4] += 1 * tanh(V_tanh[i], True) * delta_Ey * delta_yv * wout[i]
            
        for i in range (6):
            delta_orange[0] += X[i] * tanh(V_tanh[i+6], True) * delta_Ey * delta_yv * wout[i+6]
            delta_orange[1] += X[i+1] * tanh(V_tanh[i+6], True) * delta_Ey * delta_yv * wout[i+6]
            delta_orange[2] += X[i+2] * tanh(V_tanh[i+6], True) * delta_Ey * delta_yv * wout[i+6]
            delta_orange[3] += X[i+3] * tanh(V_tanh[i+6], True) * delta_Ey * delta_yv * wout[i+6]
            delta_orange[4] += 1 * tanh(V_tanh[i+6], True) * delta_Ey * delta_yv * wout[i+6]
        """
        
        # calculate the delta_w for the shared weights except the bias
        for j in range(4):
            for i in range (6):
                delta_green[j] += X[j+i] * tanh(V_tanh[i], True) * delta_Ey * delta_yv * wout[i]
                delta_orange[j] += X[j+i] * tanh(V_tanh[i+6], True) * delta_Ey * delta_yv * wout[i+6]
        
        
        # calculate the delta_w for the bias term of the shared weights
        for i in range(6):
            delta_green[4] += 1 * tanh(V_tanh[i], True) * delta_Ey * delta_yv * wout[i]
            delta_orange[4] += 1 * tanh(V_tanh[i+6], True) * delta_Ey * delta_yv * wout[i+6]
        
        # perform the updates to the weights of the output layer
        for i in range (13):
            wout[i] -= lr * delta_wout[i]
        
        # perform the changes for the shared weights
        for i in range (5):
            green[i] -= lr * delta_green[i]
            orange[i] -= lr * delta_orange[i]
        
    #print(green)
    #print(orange)
    #print(wout)
    return green, orange, wout
    
    
##############################################################################################################
############## Main Program #####################################################
for i in range (6):
    green_,orange_, wout_ = learn(X_inp[i], Y_inp[i], green_, orange_, wout_, lr, 1)
    print(green_)
    print(orange_)
    print(wout_)
#learn(X_inp[1], Y_inp[1], 1)


#print(green_)
#print(orange_)
#print(wout_)
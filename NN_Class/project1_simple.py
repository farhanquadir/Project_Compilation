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
X =np.array(
        [0.0, 0.0, 0.8, 0.4, 0.4, 0.1, 0.0, 0.0, 0.0],
        )

# labels y
Y = np.array(
        [-1]
        )

# shared weights. Last element is the bias

green = np.array([1.73673761, 1.89791391, -2.10677342, -0.14891209, 0.58306155])

orange = np.array([-2.25923303, 0.13723954, -0.70121322, -0.62078008, -0.47961976])

wout = np.array([1.20973877, -1.07518386, 0.80691921, -0.29078347, -0.22094764, -0.16915604, 1.10083444, 0.08251052, 
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
# binary cross entropy
def crossEntropy (yHat,y, derive=False):
    if derive:
        if y ==1:
            return (-1/yHat)
        else: 
             return (1/(1-yHat))
    if y==1:
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
    
V = np.ones((12,1))
V_tanh = np.ones((13,1))
#print(X[0:4])
#print(green[0:4])

#forward pass first layer

for i in range (6):
    V[i] = np.dot(X[i:i+4],green[0:4])+green[4]
    V_tanh[i] = tanh(V[i])


for i in range (6,12):
    V[i] = np.dot(X[i-6:i-6+4],orange[0:4])+orange[4]
    V_tanh[i] = tanh(V[i])
#print (V_tanh)
#forward  pass second layer
#OV = np.ones((1,1))

#OV = np.dot(V_tanh.transpose()[:],wout[:])
OV = np.dot(V_tanh.squeeze()[:],wout[:])
OV_tanh = tanh(OV)
#end of forward pass

delta_Ey = msqe(OV_tanh, Y[0], True)
delta_yv = tanh(OV_tanh, True)

#Output layer weights wout
delta_wout = np.ones((13,1))
for i in range(13):
    delta_wout[i]= delta_Ey * delta_yv * V_tanh[i]

#initialize  the changes in weights
    
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
for j in range(4):
    for i in range (6):
        delta_green[j] += X[j+i] * tanh(V_tanh[i], True) * delta_Ey * delta_yv * wout[i]
        delta_orange[j] += X[j+i] * tanh(V_tanh[i+6], True) * delta_Ey * delta_yv * wout[i+6]

# change for the bias terms in the shared weights
for i in range(6):
    delta_green[4] += 1 * tanh(V_tanh[i], True) * delta_Ey * delta_yv * wout[i]
    delta_orange[4] += 1 * tanh(V_tanh[i+6], True) * delta_Ey * delta_yv * wout[i+6]
#print(V_tanh)

#print(delta_wout)

for i in range (13):
    wout[i] -= lr * delta_wout[i]

for i in range (5):
    green[i] -= lr * delta_green[i]
    orange[i] -= lr * delta_orange[i]

print(green)
print(orange)
print(wout)


#print(V)
#print(V_tanh)

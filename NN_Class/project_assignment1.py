#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:14:00 2019

@author: farhan
"""
import numpy as np


# Declare global variables
#learning rate 

lr = 0.2

#inputs X
x =np.array(
        [0.0, 0.0, 0.8, 0.4, 0.4, 0.1, 0.0, 0.0, 0.0])

# labels y
y = np.array(
        [-1])

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
"""
def calculateError(error, output, y, W):
    print (x)
    #For the output layer always needed to be done 
    H = len(error)-1
    error_update = error
    outlayer_nodes = len(error_update[H])
    #print(no_outputs)
    for current_node in range(outlayer_nodes):
        error_update[H][current_node] += msqe(output, y)
        #print(out_nodes)
        
    #error_update []
    for h in reversed(range(H)):
        current_layer_nodes = len(error_update[h])
        for current_node in range(current_layer_nodes):
            out_layer_nodes = len(error_update[h+1])
            for front_nodes in range(out_layer_nodes):
                error_update[h][current_node] += error_update[h+1][front_nodes]*
"""
###################################################################################################################
def calculateError(error, output_tanh, y, W, V, V_tanh, X):
    H = len(error)-1 #H=3-1 = 2
    #Only one MSE in output node
    error [H][0] = msqe(output_tanh, y, True) * tanh(V[H-1],True)#Error from output node i.e. delta(j)[h==H]
    
    for hlnode in range(13): #Error backpropagating from output node to H-1 layer
        #error[H-1][hlnode] = error [H][0] * W[H-1][hlnode] #delta_W(j)[h==H] = delta(j)[H]*W(j)[H]
        error[H-1][hlnode] = error [H][0] * V_tanh[H-2][hlnode] #delta_W(j)[h==H] = delta(j)[H]*W(j)[H]
        
    #Error for H-2 layer. This is the shared weights layer
    for wnode in range (5): #Green weights
        for hdnode in range (6):# sum of error; 
            #error[H-2][wnode]+= error[H-1][hdnode] * W[H-2][wnode] * tanh(V[H-2][hdnode],True)
            error[H-2][wnode] = error[H][0] * W[H-1][hdnode] * tanh(V_tanh[H-1][hdnode],True) * X[hdnode]
        
    for wnode in range (5,10): #Orange weights
        #print("Next")
        for hdnode in range (6,12): 
            #print(tanh(V[H-2][hdnode],True))
            error[H-2][wnode]+= error[H-1][hdnode] * W[H-2][wnode] * tanh(V[H-2][hdnode],True)
            #print(error[H-2][wnode])
    #print(error[H-2][:])
    return error
############################################################################################
def changeWeights(W, error):
    for layer in range(len(W)):
        for i in range(len(W[layer])):
            W[layer][i] -= lr*error[layer][i]
    return W
############################################################################################
#Main program

v1 = np.zeros((6,1))
v1_tanh = np.zeros((6,1))
#print(green[0:4])
#print((x[0:len(green)]).shape)

sharing_range = len(green)-2

for i in range(len(v1)):
    v1[i] = np.dot(x[i:(i+len(green)-1)],green[0:4])+green[4]
    #print(v1[i])
    v1_tanh[i] = tanh(v1[i])

v2 = np.zeros((6,1))
v2_tanh = np.ones((7,1))
for i in range(len(v2)):
    v2[i] = np.dot(x[i:(i+len(orange)-1)],orange[0:4])+orange[4]
    #print(v2[i])
    v2_tanh[i] = tanh(v2[i])

v_all = np.concatenate((v1,v2))
v_all = v_all.squeeze()
#print(v_all)
v_all_tanh = np.concatenate((v1_tanh,v2_tanh))
v_all_tanh = v_all_tanh.squeeze()


output = np.dot(v_all_tanh[:],wout[:])
V = np.array([v_all, output])

#print (v_all)
output_tanh = tanh(output) # output node

V_tanh = np.array([v_all_tanh, output_tanh])

# some constant values
H = 3 #Layers

# errors at each layer
# Create a W x H matrix. Each layer has the corresponding weights W(ji)
e_h1 =  np.zeros((10))
e_h2 =  np.zeros((13))
e_hH =  np.zeros((1)) 
error=np.array([e_h1, e_h2, e_hH]) #maps errors to each node layerwise
W = np.array([np.concatenate((green[0:5], orange[0:5])), wout[0:13]]) #weights mapped to each node in the entire NN layerwise


#print(V[0])
print(V_tanh[0])


#print(V_tanh[1])

#print (W[0])

#error = calculateError(error, output_tanh, y, W, V, V_tanh, x) #computes the error flowing back i.e. delta_w
#print(error[1])

#W = changeWeights(W,error) #adjust weights according to the error
#print(error[0])
#print(W[0])
#print(W[1])

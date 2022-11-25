#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 04:32:48 2020

@author: farhan
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
dloss=np.loadtxt("d_loss.txt",dtype=float)
gloss=np.loadtxt("g_loss.txt",dtype=float)

g=0
d=0
small_d=[]
small_g=[]
#small_d_avg=[]
for i in range(0,len(dloss),6):
    d=np.sum(dloss[i:i+6])
    small_d.append(d)
    g=np.sum(gloss[i:i+6])
    small_g.append(g)
    #print (i,l)
    #print(dloss[i:i+6])

#print (small_d)
#print (small_d[0])
#print (small_d[1])
#print(dloss[0:6])
    
    

#print (gloss.shape)
losses=np.column_stack((dloss,gloss)) #np.concatenate((dloss,gloss),axis=1)
#losses=np.column_stack((np.asarray(small_d)/6,np.asarray(small_g)/6)) #np.concatenate((dloss,gloss),axis=1)
print (losses.shape)

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()
plt.show()

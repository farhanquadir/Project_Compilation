#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:48:12 2019

@author: farhan
"""

#plotting results

import numpy as np
import csv
import matplotlib.pyplot as plt
from PIL import Image
import random
from sklearn.utils import shuffle
import os, sys

workdir = "/home/farhan/Downloads/NN_Class/Data/Part2/"
errorfile = "test_error.txt"
valfile = "val_error.txt"

#error = np.loadtxt(workdir+errorfile)
#val_error = np.loadtxt(workdir+valfile)

p3err="/home/farhan/Downloads/NN_Class/Data/Part3/error_p3.txt"
p3val= "/home/farhan/Downloads/NN_Class/Data/Part3/val_error_p3.txt"

error = np.loadtxt(p3err)
val_error = np.loadtxt(p3val)

#plt.plot(error, label = "Training Loss")
#plt.plot(val_error, label = "Validation loss")

#w1file = "/home/farhan/Downloads/NN_Class/Data/Part3/w1_p3.txt"
w1file = "/home/farhan/Downloads/NN_Class/Data/Part3/w2_p3.txt"
w1 = np.loadtxt(w1file)
w1 = w1.reshape(10,16,22,22)
#print(w1.shape)
plt.imshow(w1[0][9])
plt.show()
#plt.legend()


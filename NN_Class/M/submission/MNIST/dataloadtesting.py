#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 23:57:26 2020

@author: farhan
"""

import torch.nn as nn                                                                      
from torchvision import models
import os,sys
import torch
import numpy as np
from torchvision import datasets
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as tech 

train = datasets.MNIST( root = './', # where to download data set to
                       train = True, # If True, creates dataset from training.pt, otherwise from test.pt
                       transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()]), # convert data into tensor vs PIL image 
                       download = True)
c1 = 1
c2 = 3

#idx=train.train_labels==c1
#dataset.train_labels=train.train_labels[idx]
#print (len(train))
#print (len(idx))
dataset = datasets.MNIST(root='./')
idx = dataset.targets==c1
idx+=dataset.targets==c2
dataset.targets= dataset.targets[idx]
dataset.data = dataset.data[idx]
print (len(idx))
print (len(dataset.targets))
print (len(dataset.data))
batch_size=1
train_ld = tech.DataLoader(dataset = train, shuffle = True, batch_size = batch_size)
     
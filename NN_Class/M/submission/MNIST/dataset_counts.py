#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 02:56:36 2020

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
from tqdm import tqdm
import seaborn as sn  
import pandas as pd
from torchsummary import summary

train = datasets.FashionMNIST(root='./',train = True, 
                       transform=torchvision.transforms.Compose([
                       torchvision.transforms.ToTensor()]),download = True)
c1 = 1
c2 = 3

trainidx = torch.LongTensor(train.targets) == c1
trainidx += torch.LongTensor(train.targets) == c2
train.targets= train.targets[trainidx]
train.data = train.data[trainidx]

test = datasets.FashionMNIST(root = './',train = False, 
                       transform=torchvision.transforms.Compose([
                       torchvision.transforms.ToTensor()]),download = True)

testidx = torch.LongTensor(test.targets) == c1
testidx += torch.LongTensor(test.targets) == c2
test.targets= test.targets[testidx]
test.data = test.data[testidx]

train_ld = tech.DataLoader(dataset = train, shuffle = True, batch_size = 1)       
test = tech.DataLoader(dataset = test, shuffle = False, batch_size = 1) 

#print(train_ld.dataset.targets[0])
print("Max target value is = ",max(train_ld.dataset.targets))
print("Min target value is = ",min(train_ld.dataset.targets))


count_dict={}
for i in range(10):
    count_dict[i]=0
    
for sample, label in train_ld:
    # what is its label?
    #labs=label.copy()
    labels=label
    label = int(label.numpy())
    count_dict[label]+=1
    
for key,val in count_dict.items():
    print (key,":",val)



count_dict={}
for i in range(10):
    count_dict[i]=0
    
for sample, label in test:
    # what is its label?
    #labs=label.copy()
    labels=label
    label = int(label.numpy())
    count_dict[label]+=1
    
for key,val in count_dict.items():
    print (key,":",val)



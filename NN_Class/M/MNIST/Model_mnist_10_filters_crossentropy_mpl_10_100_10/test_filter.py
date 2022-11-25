# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 01:42:16 2020

@author: Farhan
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 21:53:08 2020

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

weight_file=sys.argv[1]
#test_value=int(sys.argv[2])
test_value=9
test = datasets.MNIST( root = '../', 
                       train = False, 
                       transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()]), # convert data into tensor vs PIL image                       
                       download = True)
testidx = torch.LongTensor(test.targets) == test_value
#testidx += torch.LongTensor(test.targets) == c2
test.targets= test.targets[testidx]
test.data = test.data[testidx]

test = tech.DataLoader(dataset = test, shuffle = False, batch_size = 1)

#if (".png" in weight_file):
#    from PIL import Image
#    im_frame = Image.open(weight_file)
#    weight= np.array(im_frame.getdata())
#else:
weight=np.loadtxt(weight_file)


print(len(test.dataset.data))
print(test.dataset.data[0].shape)
data=test.dataset.data[0].numpy()
print (data.shape)
#print (data.numpy())
#plt.figure()
#plt.imshow(data)
#plt.imsave(data,"test.png")
#plt.savefig("test.png")
#plt.close()

mult=data.dot(weight)
print(mult.shape)
df=pd.DataFrame(mult)
np.savetxt("mult.txt",mult)
plt.figure()
plt.imshow(mult)
plt.savefig("mult.png")
plt.close()

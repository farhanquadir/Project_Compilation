#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 08:14:56 2019

@author: farhan
"""
import numpy as np
import matplotlib.pyplot as plt

#file = "history_cifar10_CNN.txt"
#file = "history_resnet_mnist.txt"
#file = "history_resnet_cifar10_5block.txt"
file = "history_mnist_CNN.txt"
lines=[]
with open(file,"r") as f:
    for line in f:
        if line.startswith("E"):
            continue
        lines.append(line.strip())

train_loss=[]
val_loss=[]
train_acc=[]
val_acc=[]

#split=lines[0].split()
#print(len(split))
#print(split[0])
#print(split[1])
#print(split[2])
#print(split[3])
#print(split[4])
for i in range(len(lines)):
    split=lines[i].split()
    train_loss.append(float(split[1]))
    val_loss.append(float(split[2]))
    train_acc.append(float(split[3]))
    val_acc.append(float(split[4]))


fig = plt.figure()
plt.subplot(1,2,1)
plt.plot(train_acc)
plt.plot(val_acc)
#plt.xlim(0,100)
plt.title('Plot of Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.subplot(1,2,2)
#plt.xlim(0,100)
plt.plot(train_loss)
plt.plot(val_loss)
plt.title('Plot of Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.tight_layout()
plt.show()
#fig

#plt.title("Model Loss")

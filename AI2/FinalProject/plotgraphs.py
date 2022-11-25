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
#from PIL import Image
import random
#from sklearn.utils import shuffle
import os, sys

#error = np.loadtxt(workdir+errorfile)
#val_error = np.loadtxt(workdir+valfile)

p3err= sys.argv[1]#"/home/farhan/Downloads/NN_Class/M/multicom_save/Model_mnist_10_filters_mse_train_error.txt"
p3val= p3err.replace("train_loss.txt","val_loss.txt")#"/home/farhan/Downloads/NN_Class/M/multicom_save/Model_mnist_10_filters_mse_val_error.txt"
name=p3err.replace("_train_loss.txt","")

error = np.loadtxt(p3err)
val_error = np.loadtxt(p3val)
plt.title("Training and Validation Loss Curves")
plt.plot(error, label = "Training Loss")
plt.plot(val_error, label = "Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
#w1file = "/home/farhan/Downloads/NN_Class/Data/Part3/w1_p3.txt"
#w1file = "/home/farhan/Downloads/NN_Class/Data/Part3/w2_p3.txt"
#w1 = np.loadtxt(w1file)
#w1 = w1.reshape(10,16,22,22)
#print(w1.shape)
#plt.imshow(w1[0][9])
#plt.show()
plt.savefig("./"+name+"_train_val_error.png")
plt.close()


train_acc = np.loadtxt(name+"_train_acc.txt")
val_acc = np.loadtxt(name+"_val_acc.txt")
plt.title("Training and Validation Accuracy Curves")
plt.plot(train_acc, label = "Training accuracy")
plt.plot(val_acc, label = "Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
#w1file = "/home/farhan/Downloads/NN_Class/Data/Part3/w1_p3.txt"
#w1file = "/home/farhan/Downloads/NN_Class/Data/Part3/w2_p3.txt"
#w1 = np.loadtxt(w1file)
#w1 = w1.reshape(10,16,22,22)
#print(w1.shape)
#plt.imshow(w1[0][9])
#plt.show()
plt.savefig("./"+name+"_train_val_accuracy.png")
plt.close()

epochs=np.arange(0,len(train_acc),1)
#Loss vs Acc vs Epochs
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss', color=color)
ax1.plot(error, color=color)
ax1.tick_params(axis='y', labelcolor=color)
color = 'tab:blue'
ax2 = ax1.twinx()
ax2.set_ylabel('Training Accuracy', color=color)
#ax2.set_ylim([0,1])
ax2.plot(train_acc, color=color)
ax2.tick_params(axis='y', labelcolor=color)
#plt.show()
plt.title("Training Loss and Accuracy Comparision")
fig.tight_layout()
plt.savefig("./"+name+"_train_error_accuracy_epoch.png")
plt.close()

#Val Loss vs Acc vs Epochs
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Validation Loss', color=color)
ax1.plot(val_error, color=color)
ax1.tick_params(axis='y', labelcolor=color)
color = 'tab:blue'
ax2 = ax1.twinx()
ax2.set_ylabel('Validation Accuracy', color=color)
#ax2.set_ylim([0,1])
ax2.plot(val_acc, color=color)
ax2.tick_params(axis='y', labelcolor=color)
#plt.show()
plt.title("Validation Loss and Accuracy Comparision")
fig.tight_layout()
plt.savefig("./"+name+"_val_error_accuracy_epoch.png")
plt.close()

#Train Acc vs Loss
plt.title("Training Accuracy vs Loss Curves")
#plt.plot(np.flipud(error),np.flipud(train_acc))
plt.plot(error,train_acc)
#plt.plot(, label = "Validation accuracy")
plt.xlabel("Loss")
plt.ylabel("Accuracy")
#plt.legend()
plt.savefig("./"+name+"_train_error_accuracy.png")
plt.close()

#Validation Acc vs Loss
plt.title("Validation Accuracy vs Loss Curves")
plt.plot(val_error,val_acc)
#plt.plot(, label = "Validation accuracy")
plt.xlabel("Loss")
plt.ylabel("Accuracy")
#plt.legend()
plt.savefig("./"+name+"_val_error_accuracy.png")
plt.close()



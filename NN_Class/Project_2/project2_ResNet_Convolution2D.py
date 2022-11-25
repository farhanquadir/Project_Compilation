#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 21:27:57 2019

@author: farhan
"""


from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Activation, Flatten, Dense, Conv2D, Convolution2D, Input
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta, SGD, RMSprop, Adagrad, Adamax, Nadam
from keras.backend import epsilon
from resnet import Residual
import tensorflow as tf
import matplotlib.pyplot as plt
import os,sys


def getModel():
    input_shape=(28,28,1)
    input_var=Input(shape=input_shape)
    
    #model = Sequential()
    conv = Convolution2D(64, 3, 3, border_mode="same", activation = "relu") (input_var)
    conv2 =Convolution2D(8, 3, 3, activation="relu") (conv)
    resnet = conv2
    # Add as many layers
    for _ in range (5):
        resnet = Residual(Convolution2D(8, 3, 3, border_mode="same"))(resnet)
        resnet = Activation("relu")(resnet)
        
    flat = Flatten()(resnet)
    softmax = Dense(10,activation="softmax")(flat)
    model = Model(input=[input_var],output=[softmax])
    
    return model

def train(model, epoch=1):
    
    optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])
    history=model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs= epoch)
    return history


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train=X_train.reshape(len(X_train),28,28,1)
Y_train= to_categorical(Y_train)
X_test=X_test.reshape(len(X_test),28,28,1)
Y_test= to_categorical(Y_test)
epoch = 100

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement = True)
tf.GPUOptions(per_process_gpu_memory_fraction = 0.99)
config.gpu_options.allow_growth = True
sess= tf.Session(config = config)

path=os.getcwd()
modelsave=path+"\out_dir\model_ResNet.h5"
model = getModel()
model.summary()
"""
history = train(model, epoch)
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
train_acc= history.history["acc"]
val_acc=history.history["val_acc"]

with open ("history_resnet.txt","w") as f:
    f.write("Epoch\t\t Train_Loss\t\t\t Val_loss\t\t\t Train_acc\t\t\t Val_acc\n")
    for ep in range(len(train_loss)):
        f.write(str(ep)+"\t\t "+str(train_loss[ep])+"\t\t "+str(val_loss[ep])+"\t\t "+str(train_acc[ep])+"\t\t "+str(val_acc[ep])+"\n")
"""
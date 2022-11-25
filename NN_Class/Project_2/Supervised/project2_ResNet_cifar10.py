#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 21:27:57 2019

@author: farhan
"""

from __future__ import print_function

from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Activation, Flatten, Dense, Conv2D, Convolution2D, Input, Add, add
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta, SGD, RMSprop, Adagrad, Adamax, Nadam
from keras.backend import epsilon
import keras.backend as K
#from resnet import Residual
import tensorflow as tf
import matplotlib.pyplot as plt
import os,sys

def residualBlock(input_var):
    def f(input):
        CHANNEL_AXIS=3
        input_shape = K.int_shape(input)
        x = input
        input_var=Input(shape=input_shape)
        
        l1 = Conv2D(64, kernel_size=3, padding="same", activation = "relu") (input)
        #l2 =Conv2D(32, kernel_size=3, padding="same",activation="relu") (l1)
        l2 =Conv2D(32, kernel_size=3, padding="same") (l1)
        residual_shape = K.int_shape(l2)
        stride_width = 1
        stride_height = 1
        equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]
        
        if (not equal_channels):
            x = Conv2D(filters=residual_shape[CHANNEL_AXIS], kernel_size=(1,1), strides = (stride_width, stride_height), padding="valid")(x)
        y = add([x, l2])
        return y
    return f

def getModel():
    CHANNEL_AXIS=3
    input_shape=(32,32,3)
    input_var=Input(shape=input_shape)
    
    resnet = residualBlock(input_var)(input_var)
    # Add as many layers
    for _ in range (5):
        resnet = residualBlock(resnet)(resnet)
        resnet = Activation("relu")(resnet)
        
    flat = Flatten()(resnet)
    dense2=Dense(128,activation="relu")(flat)
    dense3=Dense(64,activation="relu")(dense2)
    softmax = Dense(10,activation="softmax")(dense3)
    model = Model(input=[input_var],output=[softmax])
    
    return model

def saveModel(model):
    model.save("ResNet_cifar10_5block.h5")
    model.save_weights("ResNet_cifar10_weights_5block.h5")
    return


def train(model, epoch=1):
    
    optim = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])
    history=model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs= epoch, batch_size=batch_size)
    saveModel(model)
    return history



(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
#X_train=X_train.reshape(len(X_train),28,28,1)
Y_train= to_categorical(Y_train)
#X_test=X_test.reshape(len(X_test),28,28,1)
Y_test= to_categorical(Y_test)
epochs = 100
batch_size=128
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#print(X_train, X_test)
X_train /= 255
X_test /= 255

#print(X_train, X_test)

#X_train=X_train[0:1]
#Y_train=Y_train[0:1]
#X_test=X_test[0:1]
#Y_test=Y_test[0:1]


#print('X_train shape:', X_train.shape)
#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement = True)
tf.GPUOptions(per_process_gpu_memory_fraction = 0.99)
config.gpu_options.allow_growth = True
sess= tf.Session(config = config)

path=os.getcwd()
modelsave=path+"\out_dir\model_ResNet.h5"
model = getModel()
model.summary()

history = train(model, epoch=epochs)
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
train_acc= history.history["acc"]
val_acc=history.history["val_acc"]

with open ("history_resnet_cifar10_5block.txt","w") as f:
    f.write("Epoch\t\t Train_Loss\t\t\t Val_loss\t\t\t Train_acc\t\t\t Val_acc\n")
    for ep in range(len(train_loss)):
        f.write(str(ep)+"\t\t "+str(train_loss[ep])+"\t\t "+str(val_loss[ep])+"\t\t "+str(train_acc[ep])+"\t\t "+str(val_acc[ep])+"\n")
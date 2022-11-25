import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.utils import np_utils,to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LSTM
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam, Adadelta, SGD, RMSprop, Adagrad, Adamax, Nadam
from keras.regularizers import l2
import tensorflow as tf
import numpy as np
import os,sys
import pandas as pd
import csv
import time
 
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0001
    return lrate

def readListFile(list_file):
    file_list=[]
    with open (list_file,"r") as f:
        for line in f:
            file_list.append(line.strip())
    return file_list

def loadData(datafile):
    data = []
    label = []
    with open (datafile) as f:
        csv_line=csv.reader(f,delimiter=",")
        for parts in csv_line:
            if parts[0]=="Text": continue
            data.append(parts[0])
            label.append(int(parts[1]))
    return data,label

def normalize(arr):
    mini=np.min(arr[0])
    maxi=np.max(arr[0])
    arr[0]=(arr[0]-mini)/(maxi-mini)
    mini=np.min(arr[1])
    maxi=np.max(arr[1])
    arr[1]=(arr[1]-mini)/(maxi-mini)
    mini=np.min(arr[2])
    maxi=np.max(arr[2])
    arr[2]=(arr[2]-mini)/(maxi-mini)
    return arr

def normalize_2(arr):
    mini=np.min(arr)
    maxi=np.max(arr)
    arr=arr/maxi
    return arr

def createModel(x_train):
    model=Sequential()
    #model.add(LSTM(2000, input_shape=(x_train[0].shape),activation="tanh"))
    model.add(LSTM(1024, input_shape=(x_train[0].shape),activation="tanh"))
    model.add(Dropout(0.1))
    #model.add(Dense(1024, activation="tanh"))
    #model.add(Dropout(0.1))
    model.add(Dense(512, activation="tanh"))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation="tanh"))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation="tanh"))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation="tanh"))
    model.add(Dropout(0.1))
    model.add(Dense(10, activation="tanh"))
    model.add(Dense(1,activation="sigmoid"))
   
    
    return model

def saveModel(model):
    model.save(sys.argv[0].replace(".py","")+"HW4.h5")
    model.save_weights(sys.argv[0].replace(".py","")+"HW4_weights.h5")
    return

def trainModel(model,X_train,Y_train,X_test,Y_test,epoch=1,batch_size=1):
    optim = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=["accuracy"])
    model.compile(loss="binary_crossentropy", optimizer=optim, metrics=["accuracy"])
    #history=model.fit(X_train, Y_train, validation_data=(X_test,Y_test), epochs= epoch, batch_size=batch_size)
    history=model.fit(X_train, Y_train, validation_split=0.2, epochs= epoch, batch_size=batch_size)
    saveModel(model)
    return history

def testModel(X_test,Y_test,model=None):
    if (model==None):
        model=load_model(sys.argv[0].replace(".py","")+"HW4.h5")
    result=model.predict(X_test)
    """
    for i in range(len(X_test)):
        val=model.predict(X_test[i])
        if (val<0.5): result.append([val,0,test_y[i]])
        if (val>=0.5): result.append([val,1,test_y[i]])
    """
    return model,result


def trainIt(model,train_x,train_y,test_x,test_y,epoch):
    history=trainModel(model,train_x,train_y,test_x,test_y,epoch=50)
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    train_acc= history.history["acc"]
    val_acc=history.history["val_acc"]
    
    with open (sys.argv[0].replace(".py","")+"_train_loss.txt","w") as f:
        for loss in train_loss:
            f.write(str(loss)+"\n")
    
    with open (sys.argv[0].replace(".py","")+"_val_loss.txt","w") as f:
        for loss in val_loss:
            f.write(str(loss)+"\n")
    
    with open (sys.argv[0].replace(".py","")+"_train_acc.txt","w") as f:
        for acc in train_acc:
            f.write(str(acc)+"\n")
    
    with open (sys.argv[0].replace(".py","")+"_val_acc.txt","w") as f:
        for acc in val_acc:
            f.write(str(acc)+"\n")
    

#Read the list of training and test data
#The following loads the training data and labels into a list. Labels are one-hot encoded
training_list_file="HW5_training.csv"
test_list_file="HW5_test.csv"
train_x,train_y=loadData(training_list_file)
test_x,test_y=loadData(test_list_file)
n_train=len(train_x)
n_test=len(test_x)
total=train_x+test_x
l=[]
for x in total:
    l.append(len(x))
print (max(l))
max_len=max(l)
token_train_x=Tokenizer(num_words=1000)#.fit_on_texts(train_x)
token_train_x.fit_on_texts(train_x)
print (train_x[0])
#Tokenize training text
train_x=token_train_x.texts_to_matrix(train_x,  mode="tfidf")
train_x=train_x.reshape(len(train_x),1,train_x.shape[1])
print(train_x.shape)
#Tokenize test text
token_test_x=Tokenizer(num_words=1000)#.fit_on_texts(train_x)
token_test_x.fit_on_texts(test_x)
test_x=token_test_x.texts_to_matrix(test_x,  mode="tfidf")
test_x=test_x.reshape(len(test_x),1,test_x.shape[1])
train_y=np.asarray(train_y)
test_y=np.asarray(test_y)

#train_x=normalize(train_x)
#test_x=normalize(test_x)

#Activate cuda environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement = True)
tf.GPUOptions(per_process_gpu_memory_fraction = 0.99)
config.gpu_options.allow_growth = True
sess= tf.Session(config = config)

model = createModel(train_x)
model.summary()
trainIt(model,train_x,train_y,test_x,test_y,epoch=50)

model,predictions=testModel(test_x,test_y)
print(predictions)
print(model.evaluate(test_x,test_y,batch_size=20))
tp_00=0
tp_11=0
fp_01=0
fp_10=0
print("Test:")
for i in range(20):
    if predictions[i]<0.5:
        print ([0,test_y[i]])
        if (test_y[i]==0): tp_00+=1
        if (test_y[i]==1): fp_10+=1
    else:
        print ([1,test_y[i]])
        if (test_y[i]==1): tp_11+=1
        if (test_y[i]==0): fp_01+=1
        
print ("00",tp_00)
print ("01",fp_01)
print ("10",fp_10)
print ("11",tp_11)

print ("Training:")

model,predictions=testModel(train_x,train_y)
print(predictions)
print(model.evaluate(train_x,train_y,batch_size=1))
tp_00=0
tp_11=0
fp_01=0
fp_10=0

for i in range(100):
    if predictions[i]<0.5:
        print ([0,train_y[i]])
        if (train_y[i]==0): tp_00+=1
        if (train_y[i]==1): fp_10+=1
    else:
        print ([1,train_y[i]])
        if (train_y[i]==1): tp_11+=1
        if (train_y[i]==0): fp_01+=1
        
print ("00",tp_00)
print ("01",fp_01)
print ("10",fp_10)
print ("11",tp_11)


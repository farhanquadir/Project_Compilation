import os, sys
import re
import string
import nltk
import nltk.corpus
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sn
from keras import Sequential
from keras.initializers import Constant
from keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional, GRU, Concatenate
from keras.regularizers import l2
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adadelta

def loadEnsembleData(datafile):
    x=[]
    labels=[]
    with open (datafile, "r") as f:
        for line in f:
            x.append([float(line.split(",")[0]),float(line.split(",")[1]),float(line.split(",")[2])])
            labels.append(int(line.split(",")[3]))
    return np.asarray(x), np.asarray(labels)

def train_ensemble(x, labels):
    model=Sequential()
#    model.add(Dense(3, activation="relu"))
    model.add(Dense(1, activation="relu"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    


#previous results datafile
datafile="all_results.txt"
testdatafile="test_results.txt"
X_train, Y_train = loadData(datafile)
x_test, y_test = loadData(testdatafile)
model=train_ensemble(X_train,Y_train)
start = time.time()
history = model.fit(X_train, y_train, epochs=100, verbose=2,validation_data=(X_test, y_test), batch_size=20)
stop = time.time()
print(f"Overall training time: {stop - start} s")

# evaluate the model
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Test Accuracy: {:.4f}".format(accuracy))


########## plot accuracy and loss
plt.style.use('ggplot')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
x = range(1, len(acc) + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x, acc, 'b', label='Training acc')
plt.plot(x, val_acc, 'r', label='Test acc')
plt.ylabel('Accuracy', fontsize = 14)
plt.xlabel('Epoch', fontsize = 14)
plt.title('Training and test accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(x, loss, 'b', label='Training loss')
plt.plot(x, val_loss, 'r', label='Test loss')
plt.ylabel('Loss', fontsize = 14)
plt.xlabel('Epoch', fontsize = 14)
plt.title('Training and test loss')
plt.legend()


# In[4]:


# training confusion matrix
y_pred = (np.round(model.predict(X_train))).astype(int)
test_cm = confusion_matrix(y_train, y_pred)
sn.heatmap(test_cm, annot=True, annot_kws={"size": 12}, cmap=plt.cm.Blues)
plt.title('Training confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# test confusion matrix
y_pred = (np.round(model.predict(X_test))).astype(int)
test_cm = confusion_matrix(y_test, y_pred)
sn.heatmap(test_cm, annot=True, annot_kws={"size": 12}, cmap=plt.cm.Blues)
plt.title('Test confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


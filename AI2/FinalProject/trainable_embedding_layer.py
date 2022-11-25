#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os,sys
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
from keras.layers import Embedding, Dense, GRU
from keras.regularizers import l2
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adadelta, SGD
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
#ntlk.download("")

#########Load data
train = pd.read_csv("training.csv")
test  = pd.read_csv("test.csv")

X_train = train['Text'].values.tolist()
X_test = test['Text'].values.tolist()

y_train = train['Label'].values
y_test = test['Label'].values


#########text preprocessing
all_text = X_train+X_test
text_lines = list()
for line in all_text:
    #tokenize 
    tokens = word_tokenize(line) 
    #convert to lower case
    tokens = [w.lower() for w in tokens]
    #remove punctuation
    table = str.maketrans('','', string.punctuation)
    stripped = [w.translate(table) for w in tokens] 
    #remove non alphabetic tokens
    tokens = [w for w in stripped if w.isalpha()]
    #filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    #keep meaningful words
    #english_words = set(words.words())
    #tokens = [w for w in tokens if w in english_words ]         
    text_lines.append(tokens)

######### Vectorize the text
tk = Tokenizer()
tk.fit_on_texts(text_lines)
line_lengths = [len(line) for line in text_lines]
max_len = max(line_lengths)
vocab_size = len (tk.word_index) + 1

print('Found', vocab_size, 'unique tokens.')
print('Maximum sequence length is', max_len)

######### Pade sequences
sequences = tk.texts_to_sequences(text_lines)
text_pad = pad_sequences(sequences, maxlen=max_len)

X_train = text_pad[:-200]
X_test = text_pad[-200:]

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print (vocab_size)
sys.exit()
########## Create the model
EMBEDDING_DIM = 100
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIM,input_length=max_len, 
                    embeddings_initializer='uniform', embeddings_regularizer=l2(0.01),
                    activity_regularizer=l2(0.01))) 
model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

######### fit the model

start = time.time()

history = model.fit(X_train, y_train,
                    epochs=10,
                    verbose=2,
                    validation_data=(X_test, y_test),
                    batch_size=20)
stop = time.time()
print(f"Overall training time: {stop - start} s")

######### evaluate the model
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Test Accuracy: {:.4f}".format(accuracy))


# In[12]:


# plot accuracy and loss
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


# In[6]:


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


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
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
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import words
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
from gensim.models import Word2Vec
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)



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

######### Train word2vec embedding 
EMBEDDING_DIM = 100
model = Word2Vec (sentences=text_lines, size=EMBEDDING_DIM, window=5, workers=3, min_count=2)
words = list(model.wv.vocab)

# save embedding model
filename = 'text_embedding_word2vec.txt'
model.wv.save_word2vec_format(filename, binary=False)


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


embeddings_index = {}
f = open(os.path.join('','text_embedding_word2vec.txt'), encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word] = coefs
f.close

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in tk.word_index.items():
    if i > vocab_size:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        #words not found in embedding index will be all-zeros
        embedding_matrix[i] = embedding_vector



########## Create the model
model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIM, 
                    embeddings_initializer=Constant(embedding_matrix),
                    input_length = max_len,
                    trainable=False))
model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
#model.add(Bidirectional(LSTM(64, activation='relu')))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

####### fit the model

start = time.time()
history = model.fit(X_train, y_train,
                    epochs=10,
                    verbose=2,
                    validation_data=(X_test, y_test),
                    batch_size=10)
stop = time.time()
print(f"Overall training time: {stop - start} s")

# evaluate the model
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Test Accuracy: {:.4f}".format(accuracy))


# In[ ]:


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


# In[ ]:


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


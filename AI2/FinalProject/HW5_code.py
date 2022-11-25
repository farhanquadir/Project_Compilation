#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Mariam Alshehri
import re,sys
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
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
#from keras import optimizers.RMSprop
from keras.regularizers import l2
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Load data
train = pd.read_csv("training.csv")
test  = pd.read_csv("test.csv")

X_train = train['Text']
Y_train = train['Label'].values
Y_train = Y_train[:,np.newaxis]
X_test = test['Text']
Y_test = test['Label'].values
Y_test = Y_test[:,np.newaxis]
#1. data cleaning
def clean(text):
    text = text.lower() # convert to lower case
    text = re.sub(r'#\S+ ', r'', text) # remove hashtag words
    #text = re.sub(r"(\.\d+)_\d+", text) # remove underscore and the characters before it
    text = re.sub('['+string.punctuation+']', '', text) # remove punctuation
    text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text) # remove redundant words
    return text
X_train = X_train.apply(lambda x: clean(x))
X_test = X_test.apply(lambda x: clean(x))

#2. data tokenizing
X_train = [word_tokenize(sen) for sen in X_train]
X_test = [word_tokenize(sen) for sen in X_test]

#3. removing stopwords
nltk.download("stopwords")
stoplist = stopwords.words('english')
def removeStopWords(sen): 
    return [word for word in sen if word not in stoplist]
X_train = [removeStopWords(sen) for sen in X_train]
X_test = [removeStopWords(sen) for sen in X_test]

# build vocabulary and get maximum text length and total number of words
# for training data
all_training_words = [word for tokens in X_train for word in tokens]
training_sentence_lengths = [len(tokens) for tokens in X_train]
TRAINING_VOCAB = sorted(list(set(all_training_words)))
VOCAB_SIZE = len(TRAINING_VOCAB)
MAX_SEQUENCE_LENGTH = max(training_sentence_lengths)
# for test data
all_test_words = [word for tokens in X_test for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in X_test]
TEST_VOCAB = sorted(list(set(all_test_words)))
MAX_SEQUENCE_LENGTH2 = max(test_sentence_lengths)

# Tokenize and Pad sequences
    # for train data
tokenizer = Tokenizer(num_words=len(TRAINING_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(X_train)
training_sequences = tokenizer.texts_to_sequences(X_train)
train_word_index = tokenizer.word_index
X_train = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

print("Training data:")
print("%s words total, with a vocabulary size of %s" % (len(all_training_words), VOCAB_SIZE))
print("Max text length is %s" % MAX_SEQUENCE_LENGTH)
print("Found %s unique words." % len(train_word_index))
print("Training input size = ", X_train.shape)

    # for test data
tokenizer = Tokenizer(num_words=len(TEST_VOCAB), lower=True, char_level=False)
tokenizer.fit_on_texts(X_test)
test_sequences = tokenizer.texts_to_sequences(X_test)
test_word_index = tokenizer.word_index
X_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

print("\nTest data:")
print("%s words total, with a vocabulary size of %s" % (len(all_test_words), len(TEST_VOCAB)))
print("Max text length is %s" % MAX_SEQUENCE_LENGTH2)
print("Found %s unique words." % len(test_word_index))
print("Test input size = ", X_test.shape)


# In[ ]:

print (VOCAB_SIZE)
sys.exit()
# create the model
model = Sequential()
model.add(Embedding(VOCAB_SIZE, 64, embeddings_initializer='uniform',
                   embeddings_regularizer=l2(0.01), 
                    activity_regularizer=l2(0.01),mask_zero=True))
model.add(Bidirectional(LSTM(32, activation='relu',
                        activity_regularizer=l2(0.001),
                        recurrent_regularizer=l2(0.01), 
                        bias_regularizer=l2(0.01))))
model.add(Dense(64, activation='relu', activity_regularizer=l2(0.001)))
model.add(Dense(1, activation='sigmoid'))

#rms = optimizers.RMSprop(learning_rate=0.001, rho=0.9)
rms = RMSprop(lr=0.001, rho=0.9)
model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])
print(model.summary())

# fit the model
start = time.time()
print (X_train.shape)
print (Y_train.shape)
#sys.exit()
history = model.fit(X_train, Y_train,
                    epochs=30,
                    verbose=2,
                    validation_data=(X_test, Y_test),
                    batch_size=10)
stop = time.time()
print(f"Overall training time: {stop - start} s")

# evaluate the model
loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
print("Test Accuracy: {:.4f}".format(accuracy))

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


# In[4]:


# training confusion matrix
Y_pred = (np.round(model.predict(X_train))).astype(int)
test_cm = confusion_matrix(Y_train, Y_pred)
sn.heatmap(test_cm, annot=True, annot_kws={"size": 12}, cmap=plt.cm.Blues)
plt.title('Training confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# test confusion matrix
Y_pred = (np.round(model.predict(X_test))).astype(int)
test_cm = confusion_matrix(Y_test, Y_pred)
sn.heatmap(test_cm, annot=True, annot_kws={"size": 12}, cmap=plt.cm.Blues)
plt.title('Test confusion matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[ ]:





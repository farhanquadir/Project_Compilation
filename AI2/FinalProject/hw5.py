import csv
import time
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.layers import Dense, LSTM
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
from keras.layers import Dense, Conv2D, Flatten, Dropout

import matplotlib.pyplot as plt


def remove_unwanted_character(_input):
    _input.replace("Text", "")
    unwanted_char = ["#", "-", "'", "_"]
    new_input = _input
    for val in unwanted_char:
        new_input = new_input.replace(val, "")
    return new_input.strip().lower()


def confusion_matix(_prediction, _test_y):
    # rounded__pred = np.argmax(_prediction, axis=1)
    _prediction[_prediction > 0.5] = 1
    _prediction[_prediction < 0.5] = 0
    rounded__pred = _prediction

    rounded__test = _test_y
    counter = 0
    t_p = 0
    t_n = 0
    f_n = 0
    f_p = 0
    for val in _test_y:
        if rounded__test[counter] == 1 and rounded__pred[counter] == 1:
            t_p = t_p + 1
        if rounded__test[counter] == 0 and rounded__pred[counter] == 0:
            t_n = t_n + 1
        if rounded__test[counter] == 0 and rounded__pred[counter] == 1:
            f_p = f_p + 1
        if rounded__test[counter] == 1 and rounded__pred[counter] == 0:
            f_n = f_n + 1
        counter = counter + 1


    print(t_p)
    print(t_n)
    print(f_p)
    print(f_n)


def training_details(_history):
    # Details of training
    os.chdir('/home/rajroy/PycharmProjects/hw_4')
    train_loss = _history.history["loss"]
    val_loss = _history.history["val_loss"]
    train_acc = _history.history["accuracy"]
    val_acc = _history.history["val_accuracy"]

    # saving the results
    with open('./hw5' + '_' + str('100') + '_' + ".txt", "w") as f:
        f.write("Epoch\t\t Train_Loss\t\t\t Val_loss\t\t\t Train_acc\t\t\t Val_acc\n")
        for ep in range(len(train_loss)):
            f.write(str(ep) + "\t\t " + str(train_loss[ep]) + "\t\t " + str(val_loss[ep]) + "\t\t " + str(
                train_acc[ep]) + "\t\t " + str(val_acc[ep]) + "\n")


def train(_model, _epoch=1):  # training function
    # ADAM optimizer
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    _model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    history = _model.fit(x=train_X_tokenized, y=train_Y, validation_split=0.2, epochs=_epoch, verbose=2, shuffle=True)
    training_details(history)
    return _model


def getModel():
    model = Sequential()
    model.add(LSTM(1024, input_shape=(1, 1000), activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(512, activation='tanh', input_dim=1000))
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='tanh', input_dim=512))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    return model


def read_data(_input):
    with open(_input) as csvfile:
        data = []
        label = []
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            data.append(remove_unwanted_character(row[0]))
            label.append(int(row[1]))
    return data, label


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
tf.GPUOptions(per_process_gpu_memory_fraction=0.99)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
train_X, train_Y = read_data('HW5_training.csv')
test_X, test_Y = read_data('HW5_test.csv')

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(train_X)

train_X_tokenized = tokenizer.texts_to_matrix(train_X, mode="tfidf")
train_X_tokenized = np.reshape(train_X_tokenized,
                               (train_X_tokenized.shape[0], 1, train_X_tokenized.shape[1]))

print(np.shape(train_X_tokenized))
print(np.shape(test_Y))
test_X_tokenized = tokenizer.texts_to_matrix(test_X, mode="tfidf")
test_X_tokenized = np.reshape(test_X_tokenized,
                              (test_X_tokenized.shape[0], 1, test_X_tokenized.shape[1]))
model = getModel()
model.summary()
start = time.time()
trained_model = train(model, 50)
done = time.time()
elapsed = done - start
print(' time elapsed ' + str(elapsed / 50) + '\n')
# model = Sequential()
# model.add(LSTM(512, input_shape=(1, 1000)))
# model.add(Dense(100, activation='relu', input_dim=1000))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
print(train_Y)
# history = model.fit(train_X_tokenized, train_Y, epochs=10,validation_split=0.2, verbose=2,batch_size=10)

prediction = model.predict(test_X_tokenized)
confusion_matix(prediction, test_X)
prediction_score = model.evaluate(test_X_tokenized, test_Y)

print(100 * prediction_score[1])

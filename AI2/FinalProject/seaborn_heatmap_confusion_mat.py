import csv
import time
import pandas as pd
import numpy as np
import seaborn as sns
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


def plot_confusion_matrix():
    # Reference for Confusion Matrix
    # https://stackoverflow.com/questions/54589669/confusion-matrix-error-classification-metrics-cant-handle-a-mix-of-multilabel
    # https://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_digits_active_learning.html#sphx-glr-auto-examples-semi-supervised-plot-label-propagation-digits-active-learning-py

    # converting the data from 1 hot to

    df_cm = [[78, 22], [18, 82]]
    fig,ax=plt.subplots(1)
    #plt.figure(figsize=(10, 10))
#    ax.set_xlabel("Predicted labels")
#    ax.set_ylabel("True labels")
    #ax.set_title("Confuzion Matrix for Small Dataset (Test)")
    sns_plot = sns.heatmap(df_cm, annot=True, fmt='d', cmap=plt.cm.Blues, annot_kws={'size': 16}, ax=ax)
    fig = sns_plot.get_figure()
    plt.title("Confuzion Matrix for Large Dataset (Test)")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")

    plt.show()


plot_confusion_matrix()

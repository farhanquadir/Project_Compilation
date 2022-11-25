#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 05:11:04 2020

@author: farhan
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pickle as pkl

# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
    i=0 #"img"
    fig, axes = plt.subplots(figsize=(3,3), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        print (img.shape)
        img = ((img +1)*255 / (2)).astype(np.uint8) # rescale to pixel range (0-255)
#        plt.imsave("img_"+str(i)+".png",img)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        print ("IMage display")
        im = ax.imshow(img.reshape((32,32,3)))
#        plt.imsave("img_"+str(i)+".png",img.reshape((32,32,3)))
        i+=1
    fig.savefig("full_figure_5b5.png")


print ("START")

with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)
_ = view_samples(-1, samples)
print (len(samples[-1]))

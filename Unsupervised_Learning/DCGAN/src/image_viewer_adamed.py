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
import os, sys
import cv2 as cv
from glob import glob
import time

def zeroPad(image,pad):
    padded_image=np.zeros((image.shape[0]+2*pad,image.shape[1]+2*pad))
    padded_image[pad:-pad,pad:-pad]=image[:,:]
    return padded_image

def StageB(window):
    h,w = window.shape
    Zmin = np.min(window)
    Zmed = np.median(window)
    Zmax = np.max(window)
    Zxy = window[h//2,w//2]
    B1 = Zxy - Zmin
    B2 = Zxy - Zmax
    if B1 > 0 and B2 < 0 :
        return Zxy
    else:
        return Zmed

def StageA(mat,x,y,s,sMax,st):
    window = mat[x-(s//2):x+(s//2)+1,y-(s//2):y+(s//2)+1]
    Zmin = np.min(window)
    Zmed = np.median(window)
    Zmax = np.max(window)
    A1 = Zmed - Zmin
    A2 = Zmed - Zmax
    if A1 > 0 and A2 < 0:
        return StageB(window)
    else:
        #print (s,time.time()-st)
        s += 2
    if s <= sMax:
        return StageA(mat,x,y,s,sMax,st)
    else:
        return Zmed

def runAdaMed(_img,sMax_val):

    sMax_range=[sMax_val]

    for sMax in sMax_range:
        #sMax=21
        s=3

        image_list=glob("./img_*.png") #["./testimages/dc_gan_images4b4.png"] #glob("./testimages/*")
        print (image_list)

        #sys.exit()
#        image_list=[image_list[-1]]
    
        for k in range(1):
            img=cv.imread("temp.png",1) #_img #cv.imread(image_list[k],1)
            print (img.shape)
    #        sys.exit()
            H,W,C = img.shape
            final_image=[]
            for channel in range (C):
            
                a = sMax//2
                padded_img = zeroPad(img[:,:,channel],a)            
                f_img = np.zeros(padded_img.shape)
                start_time=time.time()
                for i in range(a,H+a+1):
                    for j in range(a,W+a+1):
                        #print("(",i,",",j,"")
                        value = StageA(padded_img,i,j,s,sMax,st=start_time)
                        f_img[i,j] = value
                print (sMax, time.time()-start_time)
                final_image.append(f_img[a:-a,a:-a])
            final_color_image=np.stack(final_image,axis=-1)
        cv.imwrite("temp_cv.png",final_color_image)
        return final_color_image
#            cv.imwrite(os.path.basename(image_list[k]).split(".")[0]+"_smax_"+str(sMax)+"_out.png",final_color_image)



# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples, smax=0):
    i=0 #"img"
    fig, axes = plt.subplots(figsize=(4,4), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        print (img.shape)
        img = ((img +1)*255 / (2)).astype(np.uint8) # rescale to pixel range (0-255)
        plt.imsave("temp.png",img)
        if smax!=0: 
            img = runAdaMed(img,3)
            img=plt.imread("temp_cv.png")
            img = np.transpose(img, (1, 2, 0))
        #plt.imsave("img_adamed_"+str(i)+".png",img)
#        img = np.transpose(img, (1, 2, 0))
#        cv.imwrite("img_adamed_"+str(i)+".png",img)
        img = ((img +1)*255 / (2)).astype(np.uint8) 
        plt.imsave("img_adamed_"+str(i)+".png",img)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        print ("IMage display")
        im = ax.imshow(img.reshape((32,32,3)))
#        plt.imsave("img_"+str(i)+".png",img.reshape((32,32,3)))
        i+=1
    fig.savefig("full_figure_unstretched_"+str(smax)+"_.png")


print ("START")

with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)
_ = view_samples(-1, samples,3)

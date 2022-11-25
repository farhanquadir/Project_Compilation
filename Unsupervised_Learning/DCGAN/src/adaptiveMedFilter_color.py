# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 21:36:56 2021

@author: MaamMaam
"""
import numpy as np
import cv2 as cv
import os
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

sMax_range=list(range(3,12,2))

for sMax in sMax_range:
    #sMax=21
    s=3
    from glob import glob
    image_list=glob("./img_*.png") #["./testimages/dc_gan_images4b4.png"] #glob("./testimages/*")
    print (image_list)
    import sys
    #sys.exit()
    image_list=[image_list[-1]]
    
    for k in range(len(image_list)):
        img=cv.imread(image_list[k],1)
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
        cv.imwrite(os.path.basename(image_list[k]).split(".")[0]+"_smax_"+str(sMax)+"_out.png",final_color_image)

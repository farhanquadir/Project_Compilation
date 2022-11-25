#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 18:55:07 2019

@author: farhan
"""
import os, sys
import numpy as np

def slide(data, w):
    #out = np.zeros((22*22))
    stride = 1
    nl = int(((data.shape[0]-w.shape[0])/stride)+1)
    #print(nl)
    out = np.zeros((nl,nl))
    l = len(data)
    w_flat = w.flatten()
    #print("w.shape=",w.shape)
    o = 0
    print(data)
    io = 0
    jo = 0
    for i in range(0,l,stride):
        if (i+len(w)>l):
            break
        for j in range(0,l,stride):
            if (j+len(w)>l):
                break
            #print("i=",i,"j=",j)
            portion = data[i:i+len(w),j:j+len(w)]
            portion = portion.flatten()
            #print("Data seg= ",data[i:len(w),j:len(w)])
            #print(portion)
            #print(portion.shape)
            #print(np.dot(portion,w_flat))
            portion_val = np.dot(portion,w_flat)
            #print(io,jo)
            out[io,jo] = portion_val
            jo+=1
            if (jo==nl):
                io+=1
                jo=0
            if (io==nl):
                break
        #io+=1
    return out #out.reshape(22,22)


a = np.array(range(16))
a = a.reshape(4,4)

w = np.ones((2,2))

print(slide(a,w))

#print(a[0:2,2:4])
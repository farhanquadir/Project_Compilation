#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 01:36:02 2019

@author: farhan
"""

import numpy as np

def getDist(vA,vB):
    x1,y1,z1=getXYZ(vA)
    x2,y2,z2=getXYZ(vB)
    (np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2))
    return np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)

def getXYZ(txt):
    split=txt.split()
    return float(split[0]),float(split[1]),float(split[2])
valB_28="18.158   1.377 -18.794"
valB_29="20.780   5.271 -19.351 "
valB_33="22.490  11.271 -22.684"
valB_56="16.567  -3.044 -29.998"
valB_57="17.131  -2.050 -34.204 "
valB_58="14.236   1.856 -34.462 "
valB_59="17.730   2.953 -30.918"
valB_60="21.175   0.159 -33.584 "
valB_61="18.482   1.558 -37.904"
valB_62="18.678   6.520 -35.939 "
valB_63="23.690   5.729 -34.587 "
valB_64="24.396   3.741 -39.827 "

valB=["18.158   1.377 -18.794",
      "20.780   5.271 -19.351 ",
      "22.490  11.271 -22.684",
      "16.567  -3.044 -29.998",
      "17.131  -2.050 -34.204 ",
      "14.236   1.856 -34.462 ",
      "17.730   2.953 -30.918",
      "21.175   0.159 -33.584 ",
      "18.482   1.558 -37.904",
      "18.678   6.520 -35.939 ",
      "23.690   5.729 -34.587 ",
      "24.396   3.741 -39.827 "]


valA_49="-0.835  -4.369 -27.501 "
valA_50="-3.272  -9.083 -25.198  "
valA_51="-8.310  -7.479 -24.105"
valA_52="-6.428 -11.240 -20.981 "
valA_53="-9.105 -11.639 -25.783 "
valA_54="-4.979 -10.445 -28.691"
valA_55="-1.978 -12.437 -25.157 "
valA_56="-6.026 -15.793 -25.122"
valA_57="-5.211 -16.463 -29.295"
valA_58=" -0.824 -14.344 -29.390 "

valA=["-0.835  -4.369 -27.501 ",
      "-3.272  -9.083 -25.198  ",
      "-8.310  -7.479 -24.105",
      "-6.428 -11.240 -20.981 ",
      "-9.105 -11.639 -25.783 ",
      "-4.979 -10.445 -28.691",
      "-1.978 -12.437 -25.157 ",
      "-6.026 -15.793 -25.122",
      "-5.211 -16.463 -29.295",
      " -0.824 -14.344 -29.390 "]
"""
for vA in valA:
    for vB in valB:
        x1,y1,z1=getXYZ(vA)
        x2,y2,z2=getXYZ(vB)
        print (np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2))
"""
valA_376_45="5.552  -8.445 -29.265"
valA_367_44="8.674  -8.573 -25.232"
valA_343_41="11.319 -10.683 -29.320"

valB_423_51="8.755  -6.839 -29.113"
valB_435_53="13.044  -6.929 -30.762"
valB_429_52="12.170  -4.157 -25.998"

print (getDist(valA_376_45,valB_423_51))
print (getDist(valA_367_44,valB_423_51))
print (getDist(valA_343_41,valB_435_53))
print (getDist(valA_343_41,valB_423_51))
print (getDist(valA_367_44,valB_429_52))
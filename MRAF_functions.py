# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:18:35 2017

@author: Stella
"""

#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
import numpy as np
import cmath as c
from PIL import Image
import cv2
refPt=[]
#-----------------------------------------------------------------------------
# Functions & Definitions
#-----------------------------------------------------------------------------
roi=[]
def target_image(path, roi, sig ,N, M, blackIsBright):
    img = cv2.imread(path)
    #crop to ROI
    pts1 = np.float32([[roi[2],roi[0]],[roi[3],roi[0]], [roi[3],roi[1]], [roi[2],roi[1]]])
    pts2 = np.float32([[0,0],[M,0],[M,N],[0,N]])
    trans = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,trans,(1920,1920)) 
    target = rgb2gray(dst)
    #just take SR
    img = np.zeros((M, N))
    if blackIsBright == 'n':
        for i in range(sig[0],sig[1]):
            for j in range(sig[2],sig[3]):
                img[i,j] = target[i,j]
    #if black in target image means bright, flip black and white
    elif blackIsBright == 'y': 
        for i in range(sig[0],sig[1]):
            for j in range(sig[2],sig[3]):
                img[i,j] = abs(target[i,j] - 255.0)
    #Normalise target image
    T_power=0
    for i in range(M):
        for j in range(N):
            T_power+=img[i,j]*img[i,j]
    target = img / np.sqrt(T_power)
    return img


#Source Wave
def Ein_0(source, N,M):
    E = source
    S_power = 0
    for i in range(M):
        for j in range(N):
            S_power += E[i,j] * E[i,j]
    E = E / np.sqrt(S_power)
    return E

#Output Wave
def Eout(Ein, sig, N,M,m):
    out = np.fft.fft2(Ein)
    O_power = 0
    for i in range(M):
        for j in range(N):
            if sig[0] < i < sig[1] and sig[2] < j < sig[3]:
                O_power += (m * abs(out[i,j])) * (m * abs(out[i,j]))   
            else:
                O_power += ((1-m) * abs(out[i,j])) * ((1-m) * abs(out[i,j]))
    out = out / np.sqrt(O_power)
    return out     

#Generated New Field Output
def newFieldOutput(target, Eout,m,sig):
    phase = np.angle(Eout) #phase of output wave
    
    g = (m-1)*Eout #noise region amplitude
    
    for i in range(sig[0],sig[1]):
        for j in range(sig[2],sig[3]):
            g[i,j] = m * target[i,j] #signal region amplitude
    
    g = g * np.exp(1J * phase) #include phase
    return g

#New Field Input
def newFieldInput(source,G):
    fft_g = np.fft.ifft2((G)) #FFT generated output
    phase=np.angle(fft_g)
    Ein = source * np.exp(1J * phase)
    return Ein

#Figure of Merit
def merit(target, Eout):
    dif = abs(target) - abs(Eout)
    av_dif = np.mean(dif)
    return av_dif/np.max(abs(target))

#24-bit to 8-bit Convertion
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


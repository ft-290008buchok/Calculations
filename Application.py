import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pydicom
from pydicom.uid import RLELossless
import pydicom.data
from pydicom.fileset import FileSet

def loadFiles():
    arterial = np.load('arterial.npy')
    maskZip = np.load('segmentationseg.npz')
    mask = maskZip['arr_0']
    images = glob.glob('./28_02_2021_14_35_06/*.dcm')
    ds = pydicom.dcmread(images[0])
    return [arterial, mask, ds]

def calcDensityParams(arterial, mask):
    arterial -= 1000
    #apply a mask
    arterial = arterial * mask

    mean = np.mean(arterial[arterial > 0])
    median = np.median(arterial[arterial > 0])
    std = np.std(arterial[arterial > 0])

    return [mean, median, std]

def calcMaxValues(mask, ds):
    z = 0
    for i in range(len(mask)):
        if len(np.flatnonzero(mask[i])) > 0:
            z += 1
    z = z * ds.SliceThickness

    x = 0
    mx = np.moveaxis(mask, 2, 0)
    for i in range(len(mx)):
        if len(np.flatnonzero(mx[i])) > 0:
            x += 1
    x = x * ds.PixelSpacing[0]

    y = 0
    my = np.moveaxis(mask, 0, 1)
    for i in range(len(my)):
        if len(np.flatnonzero(my[i])) > 0:
            y += 1
    y = y * ds.PixelSpacing[0]
    
    return [x, y, z]

def calcVolume(mask, ds):
    deltaS = ds.PixelSpacing[0] * ds.PixelSpacing[0]
    V = 0
    for i in range(len(mask)):
        ones = len(np.flatnonzero(mask[i]))
        if ones > 0:
            V += ones * deltaS
    V = V * ds.SliceThickness
    V = V / 1000
    return V
#-------------------------------------
arterial, mask, ds = loadFiles()
#mean, median, std = calcDensityParams(arterial, mask)

#print('mean density value = ', mean)
#print('median density value = ', median)
#print('std density value = ', std)
#print('----------------------------------')

x, y, z = calcMaxValues(mask, ds)

print('x-max = ', x, ' mm')
print('y-max = ', y, ' mm')
print('z-max = ', z, ' mm')
print('----------------------------------')

V = calcVolume(mask, ds)

print('Volume = ', V, ' sm^3')

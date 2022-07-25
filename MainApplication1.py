import numpy as np
import math
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

#       -------------------
#     /|           *      /|
#    / |  **       *     / |
#   /  |                /  |
#  /   |               /   |
#  --------------------  * |
# | *  |              |  * |
# | *  |    **        |    |
# |     --------------|----
# |   /               |   /
# |  /       **       |  /
# | /                 | /
# |/                  |/
#  -------------------
def paralUpperPoint(mask, ds):
    #This function calculates one of possible ellipsoid points
    #It's not sertainly elipsoid point, but real point should be nearby
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            points = np.flatnonzero(mask[i][j])
            if len(points) > 0:
                x = points[len(points) // 2] * ds.PixelSpacing[0]
                y = j * ds.PixelSpacing[0]
                z = i * ds.SliceThickness
                return [x, y, z]

def paralLowerPoint(mask, ds):
    maskSize = len(mask)
    for i in range(maskSize):
        for j in range(len(mask[maskSize - 1 - i])):
            points = np.flatnonzero(mask[maskSize - 1 - i][j])
            if len(points) > 0:
                x = points[len(points) // 2] * ds.PixelSpacing[0]
                y = j * ds.PixelSpacing[0]
                z = (maskSize - 1 - i) * ds.SliceThickness
                return [x, y, z]

def paralFurtherPoint(mask, ds):
    my = np.moveaxis(mask, 0, 1)
    for i in range(len(my)):
        for j in range(len(my[i])):
            points = np.flatnonzero(my[i][j])
            if len(points) > 0:
                x = points[len(points) // 2] * ds.PixelSpacing[0]
                y = i * ds.PixelSpacing[0]
                z = j * ds.SliceThickness
                return [x, y, z]

def calcMajorAndMinor(mask, ds):
    x1, y1, z1 = paralUpperPoint(mask, ds)
    x2, y2, z2 = paralLowerPoint(mask, ds)
    x3, y3, z3 = paralFurtherPoint(mask, ds)

    A = [[x1*x1, y1*y1, z1*z1],
         [x2*x2, y2*y2, z2*z2],
         [x3*x3, y3*y3, z3*z3]]
    B = [1, 1, 1]

    X = np.linalg.solve(A, B)
    X = 1 / X
    X = np.abs(X)
    X = np.sqrt(X)
    major = np.amax(X)
    minor = np.amin(X)
    return [major, minor]
#-------------------------------------
arterial, mask, ds = loadFiles()
mean, median, std = calcDensityParams(arterial, mask)

print('mean density value = ', mean)
print('median density value = ', median)
print('std density value = ', std)
print('----------------------------------')

x, y, z = calcMaxValues(mask, ds)

print('x-max = ', x, ' mm')
print('y-max = ', y, ' mm')
print('z-max = ', z, ' mm')
print('----------------------------------')

V = calcVolume(mask, ds)

print('Volume = ', V, ' sm^3')
print('----------------------------------')

major, minor = calcMajorAndMinor(mask, ds)
print('major = ', major, ' mm')
print('minor = ', minor, ' mm')

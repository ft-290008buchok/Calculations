import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pydicom
from pydicom.uid import RLELossless
import pydicom.data
from pydicom.fileset import FileSet

images = glob.glob('./28_02_2021_14_35_06/*.dcm')
IMAGES = []
for photo in images:
    ds = pydicom.dcmread(photo)
    IMAGES.append(ds)

arr = np.zeros((len(IMAGES), IMAGES[0].Rows, IMAGES[0].Columns), )
a = len(arr)
for i in range(a):
    arr[i] = IMAGES[i].pixel_array

arterial = np.zeros((a, 512, 512))
for i in range(a - 1, -1, -1):
    arterial[a - 1 - i] = arr[i]
np.save('arterial', arterial)
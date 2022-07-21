import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pydicom
from pydicom.uid import RLELossless
import pydicom.data
from pydicom.fileset import FileSet

base = r"C:\python-progects"
pass_dicom = "IMG-0008-00035.dcm"
filename = pydicom.data.data_manager.get_files(base, pass_dicom)[0]

ds = pydicom.dcmread(filename)
plt.imshow(ds.pixel_array, cmap=plt.cm.bone)  # set the color map to bone
plt.show()
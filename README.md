# DICOM-data processor
This feature set is designed for some basic DICOM medical image processing. It uses PyDicom and NumPy libraries.
(This code is not a commercial product and is written for educational purposes.)   
# Input parameters   
This feature set accepts a DICOM snapshot and a binary mask in the ".npz" format of segmented shape as input.  Use the "load to numpy.py" to upload a DICOM snapshot.  load to numpy.py converts a DICOM snapshot into a three-dimensional NumPy array that you can open in MainApplication.py.   
# Output parameters   
At the output, it gives:   
the average and median values of the formation density,   
the standard deviation,   
the maximum values of the formation lengths along the three coordinate axes,    
the volume of the formation, the major and minor lengths,    
the radius of the sphere described around the formation,    
the voxel coordinates of the points of the formation surface,   
the coordinates of the most distant points of the surface and the distance between them   

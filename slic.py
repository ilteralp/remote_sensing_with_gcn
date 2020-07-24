# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:20:53 2020

@author: melike
"""
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
import Constants

image = img_as_float(io.imread(Constants.IMG_PATH))                  # Load image and convert it to a floating point data type
# image = img_as_float(io.imread(Constants.LENNA_IMG_PATH))

dpi = 100
height, width, nbands = image.shape
figsize = width / float(dpi), height / float(dpi)                    # DPI is for making output image size same with the input.

# for numSegments in (100, 200):
numSegments = 2000
segments = slic(image, n_segments = numSegments, sigma = 5)          # Apply SLIC & extract (appr.) given number of segments
segment_ids = np.unique(segments)
print('There are %d segments.' % len(segment_ids))
float_segments = segments.astype(np.single)                          # Convert to float
# int_segments = segments.astype(np.uint64)

cv2.imwrite(os.path.join(Constants.SLIC_FOLDER_PATH, 
                         'segments=%d.tif' % numSegments), float_segments) # Only TIF works! PNG ranges in [0,255].
                                                                     # There might be more than 255 segments!

fig = plt.figure("Superpixels -- %d segments" % (numSegments), 
                 figsize=figsize, frameon=False)
fig.set_size_inches(width / float(dpi), height / float(dpi))
ax = plt.Axes(fig, [0., 0., 1., 1.])                                # Set size same with input
ax.axis('off')
fig.add_axes(ax)
ax.imshow(mark_boundaries(image, segments, mode='thick'), aspect='auto')
# plt.savefig(os.path.join(Constants.SLIC_FOLDER_PATH, str(numSegments) + '.png'), dpi=dpi)
plt.savefig(os.path.join(Constants.SLIC_FOLDER_PATH, '%d.png' % numSegments), dpi=dpi)
    
# # show the plots
# plt.show()
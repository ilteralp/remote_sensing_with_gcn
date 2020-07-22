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
import argparse
import os


IMG_PATH = 'C:\\Users\\melike\\RS\\vaihingen\\image\\png\\top_mosaic_09cm_area1.png'
OUT_FOLDER = 'C:\\Users\\melike\\RS\\vaihingen\\segmented'
# # load the image and convert it to a floating point data type
image = img_as_float(io.imread(IMG_PATH))

dpi = 100
height, width, nbands = image.shape
figsize = width / float(dpi), height / float(dpi)
COLOR_CONTOUR = (1., 0., 0.)


# loop over the number of segments
for numSegments in (100, 200):
    # apply SLIC and extract (approximately) the supplied number
    # of segments
    segments = slic(image, n_segments = numSegments, sigma = 5)
    print('segments:', type(segments), segments.shape)
    
    # show the output of SLIC
    fig = plt.figure("Superpixels -- %d segments" % (numSegments), figsize=figsize, frameon=False)
    fig.set_size_inches(width / float(dpi), height / float(dpi))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    
    ax.imshow(mark_boundaries(image, segments, mode='thick'), aspect='auto')
    plt.savefig(os.path.join(OUT_FOLDER, str(numSegments) + '.png'), dpi=dpi)
    
# # show the plots
# plt.show()
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 09:17:57 2022

@author: turnerp
"""

import numpy as np
from skimage import exposure
import cv2
import tifffile
import os
from glob2 import glob
import pandas as pd
import mat4py
import datetime
import json
import matplotlib.pyplot as plt
import hashlib
from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
import scipy



def get_histogram(image, bins):
    """calculates and returns histogram"""

    # array with size of bins, set to zeros
    histogram = np.zeros(bins)

    # loop through pixels and sum up counts of pixels

    for pixel in image:
        try:
            histogram[int(pixel)] += 1
        except:
            pass

    return histogram


def cumsum(a):
    """cumulative sum function"""

    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)


def autocontrast_values(image, clip_hist_percent=1):

    # calculate histogram
    hist, bin_edges = np.histogram(image, bins =  (2 ** 16) - 1)
    hist_size = len(hist)
    
    # plt.plot(hist)
    # plt.show()

    # calculate cumulative distribution from the histogram
    accumulator = cumsum(hist)
    
    # plt.plot(accumulator)
    # plt.show()

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    try:
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1
    except:
        pass

    # Locate right cut
    maximum_gray = hist_size - 1
    try:
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1
    except:
        pass
    
    
    if maximum_gray > minimum_gray:
        contrast_limit = [minimum_gray, maximum_gray]
    else:
        contrast_limit = [np.min(img),np.max(img)]
        

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    
    # calculate gamma value
    img = cv2.convertScaleAbs(image.copy(), alpha=alpha, beta=beta)
    mid = 0.5
    mean = np.mean(img)
    gamma = np.log(mid * 255) / np.log(mean)
    
    if gamma > 2:
        gamma = 2
    if gamma < 0:
        gamma = 0

    return contrast_limit, alpha, beta, gamma





path = r"\\CMDAQ4.physics.ox.ac.uk\AKGroup\Piers\AKSEG\Images\AF\images\control\SUM_multiabx_control_MG1655_ctrl1_posXY10_channels_t0_posZ0_colour1.tif"
# path = r"\\CMDAQ4.physics.ox.ac.uk\AKGroup\Piers\AKSEG\Images\AF\images\chloramphenicol\SUM_multipleabx_cam2_posXY0_channels_t0_posZ0_colour1.tif"

image = tifffile.imread(path)


def rescale_image(image, precision = "int16"):
    
    precision_dict = {"int8":np.uint8, "int16":np.uint16, "int32":np.uint32, "native":image.dtype}
    
    dtype = precision_dict[precision]
    
    if "int" in str(dtype):
        max_value = np.iinfo(dtype).max
    else:
        max_value = np.finfo(dtype).max
        
    if precision != "native":
        image = ((image - np.min(image))/np.max(image)) * max_value
        image = image.astype(dtype)

    return image
    


image = rescale_image(image, precision = "int16")


# plt.imshow(image[500:,200:])



# image = ((image - np.min(image))/np.max(image)) * ((2**16) - 1)

# # hist, bin_edges = np.histogram(image, bins =  (2 ** 16) - 1)



# # plt.plot(hist)

# # percentiles = np.percentile(image, (0.5, 99.5))
# # array([ 1., 28.])
# # scaled = exposure.rescale_intensity(image, in_range=tuple(percentiles))

# # plt.imshow(scaled)

contrast_limit, alpha, beta, gamma = autocontrast_values(image, clip_hist_percent=0.01)

manual_result = cv2.convertScaleAbs(image.astype(int), alpha=alpha, beta=beta)

plt.imshow(image[500:,200:])
plt.show()

plt.imshow(manual_result[500:,200:])
plt.show()




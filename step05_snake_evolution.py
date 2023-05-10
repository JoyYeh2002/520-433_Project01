'''
EN.520.433 Medical Image Analysis
Spring 2023

Step04: Get curvature and area information from the gt marking
- Load the images, initial contours, and 

Updated 05.09.2023

Joy Yeh
'''

import pickle
import SimpleITK as sitk
import matplotlib.pylab as plt
import cv2
import numpy as np
from helper_functions import pre_process
import time

# 0. Load the variables from the pickle file
patient_idx = 1
channel_number = 2
out_file_name = "outputs\pickles\patient{0:04d}_{1}_CH_evolution_variables.pkl".format(patient_idx, channel_number)

with open(out_file_name, "rb") as f:
    patient_idx = pickle.load(f)
    channel_number = pickle.load(f)
    lines = pickle.load(f)

    I = pickle.load(f)
    I = I[0]
    I_gt = pickle.load(f)
    r2 = pickle.load(f) # the binary red contour (the most important one)
    
    I_out = pickle.load(f)
    I_seq = pickle.load(f)

    c1 = pickle.load(f)
    c2 = pickle.load(f)
    c3 = pickle.load(f)

''' Method 1: Brute-force segmentation with cartoons '''

# Grab an image without markings 
for idx in range(1): #(int(len(I_seq)/2)):
    img = I_seq[idx]
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply thresholding to create a binary image
    # _, binary = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)

        # Apply median filtering to remove small details
    filtered = cv2.medianBlur(img, 25)

    f = filtered.copy()
    # Identify pixels with values between 30 and 100 and set them to 50

    thresh_low = 30
    thresh_high = 140
    f[(f >= thresh_low) & (f <= thresh_high)] = 50

    # Apply adaptive thresholding to create a binary image with gray, black, and white regions
    block_size = 21
    c = 2
    binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                    cv2.THRESH_BINARY, block_size, c)

    fig, axs = plt.subplots(1, 4)

    # Plot the images on the subplots
    axs[0].imshow(I, cmap = 'gray')
    axs[1].imshow(binary, cmap = 'gray')
    axs[2].imshow(filtered, cmap = 'gray')
    axs[3].imshow(f, cmap = 'gray')

    axs[0].set_title('P{0} Img'.format(patient_idx))
    axs[1].set_title('block = {0}, c = {1}'.format(block_size, c))
    axs[2].set_title('med blur = 25')
    axs[3].set_title('thresh = {0}/{1}'.format(thresh_low, thresh_high))

    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


''' Not Currently Using '''
plot_seq_samples = False
if plot_seq_samples == True:
    fig, axs = plt.subplots(1, int(len(I_seq)/2), figsize=(25, 25))
    axs[0].imshow(I_out, cmap = 'gray')

    for i in range(1, int(len(I_seq)/2)):
        axs[i].imshow(I_seq[i], cmap='gray')
        axs[i].axis('off')
    
    plt.show()
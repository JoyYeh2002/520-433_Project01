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
patient_idx = 15
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
for idx in range(1): #int(len(I_seq)/2)):
    img = I_out# I_seq[idx]
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply median filtering to remove small details
    filtered = cv2.medianBlur(img, 25)

    # Identify pixels with values between 30 and 100 and set them to 50
    f = filtered.copy()
    thresh_low = 30
    thresh_high = 140
    f[(f >= thresh_low) & (f <= thresh_high)] = 50

    ''' Box Drawing '''
    img = f

    # Define the size of the white rectangle
    white_rect_width = 140
    white_rect_height = 36

    # Calculate the coordinates of the top left and bottom right corners of the white rectangle
    white_rect_x1 = int(img.shape[1]/2 - white_rect_width/2)  # add 140 to center the rectangle horizontally
    white_rect_y1 = int(img.shape[0]/2 - white_rect_height/2)  # add 36 to center the rectangle vertically
    white_rect_x2 = white_rect_x1 + white_rect_width
    white_rect_y2 = white_rect_y1 + white_rect_height

    # Create a white rectangle
    white_rect = np.zeros_like(img)
    white_rect[white_rect_y1:white_rect_y2, white_rect_x1:white_rect_x2] = 255

    # Define the size of the black rectangles
    black_rect_width = white_rect_width
    black_rect_height = 30

    # Calculate the coordinates of the top left and bottom right corners of the top black rectangle
    top_black_rect_x1 = white_rect_x1
    top_black_rect_y1 = white_rect_y1 - black_rect_height
    top_black_rect_x2 = top_black_rect_x1 + black_rect_width
    top_black_rect_y2 = top_black_rect_y1 + black_rect_height

    # Create the top black rectangle
    top_black_rect = np.zeros_like(img)
    img[top_black_rect_y1:top_black_rect_y2, top_black_rect_x1:top_black_rect_x2] = 0

    # Calculate the coordinates of the top left and bottom right corners of the bottom black rectangle
    bottom_black_rect_x1 = white_rect_x1
    bottom_black_rect_y1 = white_rect_y2
    bottom_black_rect_x2 = bottom_black_rect_x1 + black_rect_width
    bottom_black_rect_y2 = bottom_black_rect_y1 + black_rect_height

    # Create the bottom black rectangle
    bottom_black_rect = np.zeros_like(img)
    img[bottom_black_rect_y1:bottom_black_rect_y2, bottom_black_rect_x1:bottom_black_rect_x2] = 0

    # Add the black rectangles to the white rectangle to create the sandwich shape
    sandwich = cv2.add(top_black_rect, white_rect)
    sandwich = cv2.add(sandwich, bottom_black_rect)

    # Add the sandwich shape to the original image
    img_with_rect = cv2.add(img, sandwich)

    # Display the image
    cv2.imshow("Image with Rectangle", img_with_rect)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plot_comparisons = False
    if plot_comparisons == True:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(I, cmap = 'gray')
        axs[1].imshow(f, cmap = 'gray')

        axs[0].set_title('P{0} Img'.format(patient_idx))
        axs[1].set_title('thresh = {0}/{1}'.format(thresh_low, thresh_high))

        for ax in axs.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()


''' Not Currently Using '''
 # Apply adaptive thresholding to create a binary image with gray, black, and white regions
    # block_size = 21
    # c = 2
    # binary = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
    #                                 cv2.THRESH_BINARY, block_size, c)
    
    # Perform Canny edge detection
    # cthresh_low = 40
    # cthresh_high = 70
    # edges = cv2.Canny(filtered, cthresh_low, cthresh_high)

plot_seq_samples = False
if plot_seq_samples == True:
    fig, axs = plt.subplots(1, int(len(I_seq)/2), figsize=(25, 25))
    axs[0].imshow(I_out, cmap = 'gray')

    for i in range(1, int(len(I_seq)/2)):
        axs[i].imshow(I_seq[i], cmap='gray')
        axs[i].axis('off')
    
    plt.show()
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

# 0. Load the variables from the pickle file
patient_idx = 1
channel_number = 4
out_file_name = "outputs\pickles\patient{0:04d}_{1}_CH_evolution_variables.pkl".format(patient_idx, channel_number)

with open(out_file_name, "rb") as f:
    patient_idx = pickle.load(f)
    channel_number = pickle.load(f)
    lines = pickle.load(f)

    I = pickle.load(f)
    I_gt = pickle.load(f)
    r2 = pickle.load(f) # the binary red contour (the most important one)
    
    I_out = pickle.load(f)
    I_seq = pickle.load(f)

    c1 = pickle.load(f)
    c2 = pickle.load(f)
    c3 = pickle.load(f)

# 1. Plots for sanity check
plot_on_img = False
if plot_on_img == True:
    I_out = cv2.cvtColor(I_out, cv2.COLOR_GRAY2RGB)

    # Set the color of the contour pixels to red in the RGB image
    linewidth = 4
    cv2.drawContours(I_out, c2, -1, (0, 255, 0), linewidth)
    cv2.drawContours(I_out, c1, -1, (0, 0, 255), linewidth)
    cv2.drawContours(I_out, c3, -1, (255, 0, 0), linewidth)

    # Save the canvas with the contours in red
    cv2.imwrite('outputs\snakes\patinet{0:04d}_{1}CH_contour_label.jpg'.format(patient_idx, channel_number), I_out)

    # Display the canvas with the contour in red color
    cv2.imshow('Contour Image', I_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

plot_seq_samples = True
if plot_seq_samples == True:
    fig, axs = plt.subplots(1, int(len(I_seq)/2), figsize=(25, 25))
    axs[0].imshow(I_out, cmap = 'gray')

    for i in range(1, int(len(I_seq)/2)):
        axs[i].imshow(I_seq[i], cmap='gray')
        axs[i].axis('off')
    
    plt.show()
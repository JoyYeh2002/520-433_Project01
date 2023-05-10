'''
EN.520.433 Medical Image Analysis
Spring 2023

Step06: Given the markings (r2) for channel 2 and 4, assume they are orthogonal, apply 
a formula to calculate the bean sag shape volume

Updated 05.09.2023

Hannah Qu
'''

import pickle
import SimpleITK as sitk
import matplotlib.pylab as plt
import cv2
import numpy as np
from helper_functions import pre_process
from itertools import combinations

# 0. Load the variables from the pickle file
patient_idx = 20
channel_number = 4
out_file_name_2 = "outputs\pickles\patient{0:04d}_{1}_CH_evolution_variables.pkl".format(patient_idx, 2)

out_file_name_4 = "outputs\pickles\patient{0:04d}_{1}_CH_evolution_variables.pkl".format(patient_idx, 4)

with open(out_file_name_2, "rb") as f:
    patient_idx_2 = pickle.load(f)
    channel_number_2 = pickle.load(f)
    lines_2 = pickle.load(f)

    I_2 = pickle.load(f)
    I_gt_2 = pickle.load(f)
    r2_2 = pickle.load(f) # the binary red contour (the most important one)
    
    I_out_2 = pickle.load(f)
    I_seq_2 = pickle.load(f)

    c1_2 = pickle.load(f)
    c2_2 = pickle.load(f)
    c3_2 = pickle.load(f)

with open(out_file_name_4, "rb") as f:
    patient_idx_4 = pickle.load(f)
    channel_number_4 = pickle.load(f)
    lines_4 = pickle.load(f)

    I_4 = pickle.load(f)
    I_gt_4 = pickle.load(f)
    r2_4 = pickle.load(f) # the binary red contour (the most important one)
    
    I_out_4 = pickle.load(f)
    I_seq_4 = pickle.load(f)

    c1_4 = pickle.load(f)
    c2_4 = pickle.load(f)
    c3_4 = pickle.load(f)    

# 1. Plots for sanity check
plot_on_img = True
if plot_on_img == True:
    I_out_2 = cv2.cvtColor(I_out_2, cv2.COLOR_GRAY2RGB)
    I_out_4 = cv2.cvtColor(I_out_4, cv2.COLOR_GRAY2RGB)

    # Set the color of the contour pixels to red in the RGB image
    linewidth = 4
    # cv2.drawContours(I_out, c2, -1, (0, 255, 0), linewidth) # green
    cv2.drawContours(I_out_2, c1_2, -1, (0, 0, 255), linewidth) # red
    # cv2.drawContours(I_out, c3, -1, (255, 0, 0), linewidth) # blue
    cv2.drawContours(I_out_4, c1_4, -1, (0, 0, 255), linewidth) # red

    # Save the canvas with the contours in red
    cv2.imwrite('outputs\snakes\patinet{0:04d}_{1}CH_contour_label.jpg'.format(patient_idx, channel_number), I_out_2)
    cv2.imwrite('outputs\snakes\patinet{0:04d}_{1}CH_contour_label.jpg'.format(patient_idx, channel_number), I_out_4)

    # Display the canvas with the contour in red color
    cv2.imshow('Contour Image', I_out_2)
    cv2.imshow('Contour Image', I_out_4)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#2 Create binary mask and find number of pixels in LV
plot_mask_find_area = True
if plot_mask_find_area == True:
    contour_2 = np.array(c1_2, dtype = np.int32)
    contour_4 = np.array(c1_4, dtype = np.int32)
    mask_2 = np.zeros_like(I_out_2)
    mask_4 = np.zeros_like(I_out_4)
    cv2.drawContours(mask_2, contour_2, -1, 255, -1)
    cv2.drawContours(mask_4, contour_4, -1, 255, -1)
    cv2.imshow('Binary Mask', mask_2)
    cv2.imshow('Binary Mask', mask_4)

    pixels_2 = np.count_nonzero(mask_2) # get num pixels
    pixels_4 = np.count_nonzero(mask_4) # get num pixels
    print("# Pixels in mask:", pixels_2, "pixels")
    print("# Pixels in mask:", pixels_4, "pixels")

    conversion_factor = 1; #FIGURE THIS OUT
    area_2 = pixels_2*conversion_factor # get area of LV
    area_4 = pixels_4*conversion_factor # get area of LV
    print("Area of mask:", area_2, "cm^2")
    print("Area of mask:", area_4, "cm^2")

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


current_max_2 = 0
max_a_2=  0
max_b_2=  0
max_a_4=  0
max_b_4=  0

for tup in c1_2: # iterate through tuple
    #print("type", type(tup))
    c1_s_2 = np.squeeze(c1_2)
    #print(c1_s)
    for i in c1_s_2:
        a = i[0]
        b = i[1]
        current_distance = np.linalg.norm(a-b)
        if current_distance > current_max_2:
            current_max_2 = current_distance
            max_a_2 = a
            max_b_2 = b
print(current_max_2)

current_max_4 = 0

for tup in c1_4: # iterate through tuple
    #print("type", type(tup))
    c1_s_4 = np.squeeze(c1_4)
    #print(c1_s)
    for i in c1_s_4:
        a = i[0]
        b = i[1]
        current_distance = np.linalg.norm(a-b)
        if current_distance > current_max_4:
            current_max_4 = current_distance
            max_a_4 = a
            max_b_4 = b
print(current_max_4)

# Dodge Estimate: V = (math.pi*L/6) * (pi*area_2/pi*current_max_2)*(2*area_4/pi*current_max_4)

plot_seq_samples = False
if plot_seq_samples == True:
    fig, axs = plt.subplots(1, int(len(I_seq_2)/2), figsize=(25, 25))
    axs[0].imshow(I_out_2, cmap = 'gray')

    for i in range(1, int(len(I_seq_2)/2)):
        axs[i].imshow(I_seq_2[i], cmap='gray')
        axs[i].axis('off')
    
    plt.show()
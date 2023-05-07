'''
EN.520.433 Medical Image Analysis
Spring 2023

Step02: Save the marking data and compare with the ultrasound images
[LEGACY CODE] Failed to apply otsu's method with satisfactory results. However, the code is good for future reference.
Updated 05.05.2023

Joy Yeh
'''

import SimpleITK as sitk
import numpy as np
import matplotlib.pylab as plt
from skimage import exposure
from skimage.filters import threshold_otsu
import scipy.ndimage as ndi
import cv2

# 0. Basic Control Panel
patient_idx01 = 1
patient_idx02 = 5

heartbeat_state = 'ES'  # set to 'ED' or 'ES' to load the corresponding files

folder_name01 = "data/patient" + str(patient_idx01).zfill(4) + "/"
folder_name02 = "data/patient" + str(patient_idx02).zfill(4) + "/"

display_markings = True
display_sequence = False
channel_number = 2

loop_through = True
loop_idx = range(1, 36, 4)

# 1. Open the .cfg file
cfg_name = 'Info_{0}CH.cfg'.format(channel_number)

with open(folder_name01 + cfg_name, 'r') as f:
    # Read the lines of the file into a list
    lines = f.readlines()

with open(folder_name02 + cfg_name, 'r') as f:
    # Read the lines of the file into a list
    lines = f.readlines()

if display_markings == True:
    # construct the filenames based on the heartbeat_state
    mhp01 = 'patient{0:04d}_2CH_{1}.mhd'.format(patient_idx01, heartbeat_state)
    mhp02 = 'patient{0:04d}_2CH_{1}.mhd'.format(patient_idx02, heartbeat_state)

    # load the images and ground truth files
    I1 = sitk.GetArrayFromImage(sitk.ReadImage(folder_name01 + mhp01, sitk.sitkFloat32))
    I2 = sitk.GetArrayFromImage(sitk.ReadImage(folder_name02 + mhp02, sitk.sitkFloat32))
    
    # Draw the histogram, excluding 0's
    # Compute the histogram of the non-zero pixel values
    hist, bins = np.histogram(I2[I2 != 0], bins=256, range=[0, 256])
    avg_intensity = np.average(bins[:-1], weights=hist)

    show_histogram = False
    if show_histogram == True:
        plt.bar(bins[:-1], hist, width=1)
        plt.axvline(x=avg_intensity, color='red')

        plt.suptitle('Patient {0}, {1} sequence Raw Histogram'.format(patient_idx02, heartbeat_state))
        plt.ylim([0, 7000])
        plt.show()
    
    # Add 100 intensity
    I = I2
    I2[I2 != 0] += 100

    # Clip the pixel values to the range [0, 255]
    img = np.clip(I2, 0, 255)

    show_4_subplots = True
    if show_4_subplots == True:
        # create a subplot with three images
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].imshow(I[0], cmap='gray')
        axs[1].imshow(img[0], cmap='gray')
        axs[2].imshow(I1[0], cmap = 'gray')
        
        axs[0].set_title('Original Image')
        axs[1].set_title('Brightened')
        axs[2].set_title('Patient 1')
       
        # remove the ticks from the subplots
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        print('HELLO')
        plt.show()
    
    # [THESE ARE AFTER THE BREAKPOINT]
    cont = False
    if cont== True:
        # apply a Gaussian filter with a standard deviation of 2 in the x and y dimensions, and 1 in the z dimension
        sigma = (5, 5, 1)
        gaussian_filtered = ndi.gaussian_filter(I, sigma)

        # Median blurring for denoising
        denoised = cv2.medianBlur(gaussian_filtered[0, :, :], 5) # 5 is the size of the filter window
        denoised = np.expand_dims(denoised, axis=0)

        # apply Otsu's method again
        thresh = threshold_otsu(denoised)
        binary = denoised > thresh

        # Contrast strecthing
        p2, p98 = np.percentile(I, (20, 80))
        contrasted = exposure.rescale_intensity(denoised, in_range=(p2, p98))

        # create a subplot with three images
        fig, axs = plt.subplots(1, 4, figsize=(10, 5))

        # display the original image in the first subplot
        axs[0].imshow(I_2ch[0], cmap='gray')
        axs[0].set_title('Original')

        # display the filtered image in the second subplot
        axs[1].imshow(denoised[0], cmap='gray')
        axs[1].set_title('Filt + Denoise')

        axs[2].imshow(binary[0], cmap='gray')
        axs[2].set_title('Contrast enhanced')

        # display the gt
        axs[3].imshow(I_2ch_gt[0], cmap='gray')
        axs[3].set_title('Labels')

        # set the title of the figure
        plt.suptitle('Patient {0}, {1}CH Sequence'.format(patient_idx, channel_number))

        # remove the ticks from the subplots
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        # display the figure
        plt.show()
    
    display = False
    if display == True:
        fig, axs = plt.subplots(2, 4, figsize=(10, 5))
        plt.gray()
        plt.subplots_adjust(0, 0, 1, 1, 0.01, 0.01)

        # display each slice of the images side by side in the same subplot
        for i in range(min(I_2ch.shape[0], 2*4)):
            axs[i//4, i%4].imshow(I_2ch[i])
            axs[i//4, i%4].axis('off')
            axs[i//4+1, i%4].imshow(I_2ch_gt[i])
            axs[i//4+1, i%4].axis('off')

        # display the figure
        plt.setp(axs, xticks=[], yticks=[])
        plt.suptitle('Patient {0}, {1} sequence'.format(patient_idx, heartbeat_state))
        plt.show()

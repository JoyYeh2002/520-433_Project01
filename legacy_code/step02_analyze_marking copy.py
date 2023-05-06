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
patient_idx = 10
heartbeat_state = 'ES'  # set to 'ED' or 'ES' to load the corresponding files
folder_name = "data/patient" + str(patient_idx).zfill(4) + "/"
display_markings = True
display_sequence = False
channel_number = 2

loop_through = True
loop_idx = range(1, 36, 4)

# 1. Open the .cfg file
cfg_name = 'Info_{0}CH.cfg'.format(channel_number)
with open(folder_name + cfg_name, 'r') as f:
    # Read the lines of the file into a list
    lines = f.readlines()

if display_markings == True:
    # construct the filenames based on the heartbeat_state
    mhp_2ch = 'patient{0:04d}_2CH_{1}.mhd'.format(patient_idx, heartbeat_state)
    mhp_2ch_gt = 'patient{0:04d}_2CH_{1}_gt.mhd'.format(patient_idx, heartbeat_state)

    # load the images and ground truth files
    I_2ch = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + mhp_2ch, sitk.sitkFloat32))
    I_2ch_gt = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + mhp_2ch_gt, sitk.sitkFloat32))
    
    I = I_2ch

    # create a binary mask where 0 indicates pixels to ignore
    mask = np.where(I == 0, 0, 1)

    # apply the mask to the image
    masked_img = I * mask
    masked_img = masked_img.astype(np.uint16)
    I_2ch = I_2ch.astype(np.uint16)

    # apply Otsu's method to the masked image
    otsu_thresh, otsu_img = cv2.threshold(masked_img[0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(otsu_img.shape)

    # create a subplot with three images
    fig, axs = plt.subplots(1, 4, figsize=(10, 5))
    axs[0].imshow(I_2ch[0], cmap='gray')
    axs[1].imshow(mask[0], cmap='gray')
    axs[2].imshow(masked_img[0], cmap='gray')
    axs[3].imshow(otsu_img, cmap='gray')

    axs[0].set_title('Original Image')
    axs[1].set_title('Binary Mask')
    axs[2].set_title('Masked Image')
    axs[3].set_title('Otsu on the masked image')

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

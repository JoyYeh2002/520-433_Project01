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

display_markings = True
if display_markings == True:
    # construct the filenames based on the heartbeat_state
    mhp01 = 'patient{0:04d}_2CH_{1}.mhd'.format(patient_idx01, heartbeat_state)
    mhp02 = 'patient{0:04d}_2CH_{1}.mhd'.format(patient_idx02, heartbeat_state)

    # load the images and ground truth files
    I1 = sitk.GetArrayFromImage(sitk.ReadImage(folder_name01 + mhp01, sitk.sitkFloat32))
    I2 = sitk.GetArrayFromImage(sitk.ReadImage(folder_name02 + mhp02, sitk.sitkFloat32))
    
    # Linearly scale the pixel values to the range [0, 255]
    contrasted01 = exposure.equalize_hist(I1[0], nbins=256)
    contrasted02 = exposure.equalize_hist(I2[0], nbins=256)

    con03 = exposure.equalize_hist(I1[0], nbins=128)
    con04 = exposure.equalize_hist(I1[0], nbins=64)
    con05 = exposure.equalize_hist(I1[0], nbins=16)

    show_4_subplots = True
    if show_4_subplots == True:
        # create a subplot with three images
        fig, axs = plt.subplots(1, 4, figsize=(10, 5))

        axs[0].imshow(contrasted01, cmap = 'gray')
        axs[1].imshow(con03, cmap = 'gray')
        axs[2].imshow(con04, cmap='gray')
        axs[3].imshow(con05, cmap = 'gray')
        
        axs[0].set_title('256')
        axs[1].set_title('128')
        axs[2].set_title('64')
        axs[3].set_title('16')

        # axs[0].imshow(I1[0], cmap = 'gray')
        # axs[1].imshow(contrasted01, cmap = 'gray')
        # axs[2].imshow(I2[0], cmap='gray')
        # axs[3].imshow(contrasted02, cmap = 'gray')
        
        # axs[0].set_title('P#{0}Orig'.format(patient_idx01))
        # axs[1].set_title('P1 Enhanced')
        # axs[2].set_title('P#{0}Orig'.format(patient_idx02))
        # axs[3].set_title('P2 Enhanced')
      
        # remove the ticks from the subplots
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.show()
    
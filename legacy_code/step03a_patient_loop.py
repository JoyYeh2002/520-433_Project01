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
patient_idx = [1, 5, 10, 20, 25, 30, 35, 154]
patient_idx01 = 1
patient_idx02 = 5

heartbeat_state = 'ES'  # set to 'ED' or 'ES' to load the corresponding files
folder_name = []
for i in range(len(patient_idx)):
    folder_name.append("data/patient" + str(patient_idx[i]).zfill(4) + "/")

display_markings = True
display_sequence = False
channel_number = 2

loop_through = True
loop_idx = range(1, 36, 4)

# 1. Open the .cfg file
cfg_name = 'Info_{0}CH.cfg'.format(channel_number)

for i in range(len(folder_name)):
    with open(folder_name[i] + cfg_name, 'r') as f:
        # Read the lines of the file into a list
        lines = f.readlines()

display_markings = True
if display_markings == True:
    # construct the filenames based on the heartbeat_state
    mhp_arr = []
    for i in range(len(patient_idx)):
        mhp_arr.append('patient{0:04d}_2CH_{1}.mhd'.format(patient_idx[i], heartbeat_state))
   
    # load the images and ground truth files
    im_arr = []
    contrast_arr = []
    for i in range(len(mhp_arr)):
        im_arr.append(sitk.GetArrayFromImage(sitk.ReadImage(folder_name[i] + mhp_arr[i], sitk.sitkFloat32)))
        # Linearly scale the pixel values to the range [0, 255]
        contrast_arr.append(exposure.equalize_hist(im_arr[i][0], nbins=256))
  
    show_4_subplots = True
    if show_4_subplots == True:
        # create a subplot with three images
        fig, axs = plt.subplots(2, len(im_arr), figsize=(10, 5))
        print(len(axs))
        count = 0
        for i in range(0, 2*len(im_arr)-1, 2):
            axs[i//len(im_arr), i%len(im_arr)].imshow(im_arr[count][0], cmap = 'gray')
            axs[i//len(im_arr), i%len(im_arr)+1].imshow(contrast_arr[count], cmap = 'gray')
            print(i)
           
            axs[i//len(im_arr), i%len(im_arr)].set_title('P#{0}Orig'.format(patient_idx[count]))
            axs[i//len(im_arr), i%len(im_arr)+1].set_title('P{0} Enhanced'.format(patient_idx[count]))
            count+=1
        
        # remove the ticks from the subplots
        for arow in axs:
            for i in range(2):
                for acol in axs[i]:
                    acol.set_xticks([])
                    acol.set_yticks([])

        plt.show()
    

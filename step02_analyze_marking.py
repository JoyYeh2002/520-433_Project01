'''
EN.520.433 Medical Image Analysis
Spring 2023

Step02: Save the marking data and compare with the ultrasound images

Updated 05.05.2023

Joy Yeh
'''

import SimpleITK as sitk
import numpy as np
import matplotlib.pylab as plt
from skimage import filters, restoration
from PIL import Image
import scipy.ndimage as ndi
import cv2

# 0. Basic Control Panel
patient_idx = 1
heartbeat_state = 'ES'  # set to 'ED' or 'ES' to load the corresponding files
folder_name = "data/patient" + str(patient_idx).zfill(4) + "/"
display_markings = True
display_sequence = False
channel_number = 2


# 1. Open the .cfg file
cfg_name = 'Info_2CH.cfg'
with open(folder_name + cfg_name, 'r') as f:
    # Read the lines of the file into a list
    lines = f.readlines()

# Extract the relevant information
age = float(lines[0].split(':')[1].strip())
weight = float(lines[1].split(':')[1].strip())
gender = lines[2].split(':')[1].strip()
edv = lines[3].split(':')[1].strip()
esv = lines[4].split(':')[1].strip()
ef = lines[5].split(':')[1].strip()

# Print the extracted information
print("Information of Patient #" + str(patient_idx).zfill(4))
print("Age:", age)
print("Weight:", weight)
print("Gender:", gender)
print("EDV:", edv)
print("ESV:", esv)
print("EF:", ef)

if display_markings == True:
    # construct the filenames based on the heartbeat_state
    mhp_2ch = 'patient{0:04d}_2CH_{1}.mhd'.format(patient_idx, heartbeat_state)
    mhp_2ch_gt = 'patient{0:04d}_2CH_{1}_gt.mhd'.format(patient_idx, heartbeat_state)

    # load the images and ground truth files
    I_2ch = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + mhp_2ch, sitk.sitkFloat32))
    I_2ch_gt = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + mhp_2ch_gt, sitk.sitkFloat32))
    
    # print(np.max(I_2ch))
    # create a figure with subplots arranged in a 2x4 grid
    # read in the image
    I = I_2ch
   
    # apply a Gaussian filter with a standard deviation of 2 in the x and y dimensions, and 1 in the z dimension
    sigma = (5, 5, 1)
    gaussian_filtered = ndi.gaussian_filter(I, sigma)

    denoised = cv2.medianBlur(gaussian_filtered[0, :, :], 9) # 5 is the size of the filter window
    denoised = np.expand_dims(denoised, axis=0)


    # create a subplot with three images
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    # display the original image in the first subplot
    axs[0].imshow(I[0], cmap='gray')
    axs[0].set_title('Original')

    # display the filtered image in the second subplot
    axs[1].imshow(gaussian_filtered[0], cmap='gray')
    axs[1].set_title('Gaussian filtered')

    # display the denoised image in the third subplot
    axs[2].imshow(denoised[0], cmap='gray')
    axs[2].set_title('Denoised')

    # set the title of the figure
    fig.suptitle('Image processing example')

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

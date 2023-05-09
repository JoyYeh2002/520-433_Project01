'''
EN.520.433 Medical Image Analysis
Spring 2023

Step02: Do image processing on the raw data, so that it looks more like the markings
Updated 05.06.2023

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
patient_idx01 = 5
patient_idx02 = 154

heartbeat_state = 'ED'  # set to 'ED' or 'ES' to load the corresponding files

folder_name01 = "data/patient" + str(patient_idx01).zfill(4) + "/"
folder_name02 = "data/patient" + str(patient_idx02).zfill(4) + "/"

display_markings = True
display_sequence = False
channel_number = 2

# 1. Open the .cfg file
cfg_name = 'Info_{0}CH.cfg'.format(channel_number)

with open(folder_name01 + cfg_name, 'r') as f:
    # Read the lines of the file into a list
    lines = f.readlines()

with open(folder_name02 + cfg_name, 'r') as f:
    # Read the lines of the file into a list
    lines = f.readlines()

# construct the filenames based on the heartbeat_state
mhp01 = 'patient{0:04d}_2CH_{1}.mhd'.format(patient_idx01, heartbeat_state)
mhp02 = 'patient{0:04d}_2CH_{1}.mhd'.format(patient_idx02, heartbeat_state)
mhp01_gt = 'patient{0:04d}_2CH_{1}_gt.mhd'.format(patient_idx01, heartbeat_state)
mhp02_gt = 'patient{0:04d}_2CH_{1}_gt.mhd'.format(patient_idx02, heartbeat_state)

# load the images and ground truth files
I1 = sitk.GetArrayFromImage(sitk.ReadImage(folder_name01 + mhp01, sitk.sitkFloat32))
I2 = sitk.GetArrayFromImage(sitk.ReadImage(folder_name02 + mhp02, sitk.sitkFloat32))
I1_gt = sitk.GetArrayFromImage(sitk.ReadImage(folder_name01 + mhp01_gt, sitk.sitkFloat32))
I2_gt = sitk.GetArrayFromImage(sitk.ReadImage(folder_name02 + mhp02_gt, sitk.sitkFloat32))

''' Image processing starts here'''
# 1. Contrast enhancement
I1_c = exposure.equalize_hist(I1[0], nbins=256)
I2_c = exposure.equalize_hist(I2[0], nbins=256)

# 2. Denoising
# apply a Gaussian filter with a standard deviation of 2 in the x and y dimensions, and 1 in the z dimension
sigma = (1, 1)
gaussian_filtered1 = ndi.gaussian_filter(I1_c, sigma)
gaussian_filtered2 = ndi.gaussian_filter(I2_c, sigma)

# Median blurring for denoising
denoised1 = cv2.medianBlur(gaussian_filtered1[:, :], 5) # 5 is the size of the filter window
denoised1 = np.expand_dims(denoised1, axis=0)

denoised2 = cv2.medianBlur(gaussian_filtered2[:, :], 5) # 5 is the size of the filter window
denoised2 = np.expand_dims(denoised2, axis=0)

# create a subplot with three images
fig, axs = plt.subplots(2, 2, figsize=(10, 5))

# Display the images
axs[0, 0].imshow(I1_c, cmap='gray')
axs[0, 1].imshow(denoised1[0], cmap='gray')
#axs[0, 2].imshow(I1_gt[0], cmap='gray')

axs[1, 0].imshow(I2_c, cmap='gray')
axs[1, 1].imshow(denoised2[0], cmap='gray')
#axs[1, 2].imshow(I2_gt[0], cmap='gray')

# Set the titles
axs[0, 0].set_title('P#{0} Raw'.format(patient_idx01))
axs[0, 1].set_title('P#{0} Denoised'.format(patient_idx01))
#axs[0, 2].set_title('P#{0} Labels'.format(patient_idx01))

axs[1, 0].set_title('P#{0} Raw'.format(patient_idx02))
axs[1, 1].set_title('P#{0} Denoised'.format(patient_idx02))
#axs[1, 2].set_title('P#{0} Labels'.format(patient_idx02))


# # Display the images
# axs[0, 0].imshow(I1[0], cmap='gray')
# axs[0, 1].imshow(I1_c, cmap='gray')
# axs[0, 2].imshow(I1_gt[0], cmap='gray')

# axs[1, 0].imshow(I2[0], cmap='gray')
# axs[1, 1].imshow(I2_c, cmap='gray')
# axs[1, 2].imshow(I2_gt[0], cmap='gray')

# # Set the titles
# axs[0, 0].set_title('P#{0} Orig'.format(patient_idx01))
# axs[0, 1].set_title('P#{0} Enhanced'.format(patient_idx01))
# axs[0, 2].set_title('P#{0} Labels'.format(patient_idx01))

# axs[1, 0].set_title('P#{0} Orig'.format(patient_idx02))
# axs[1, 1].set_title('P#{0} Enhanced'.format(patient_idx02))
# axs[1, 2].set_title('P#{0} Labels'.format(patient_idx02))

# Remove the ticks from the subplots
plt.setp(axs, xticks=[], yticks=[])
plt.show()

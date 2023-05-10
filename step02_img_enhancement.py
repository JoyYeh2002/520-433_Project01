'''
EN.520.433 Medical Image Analysis
Spring 2023

Step02: Do image processing on the raw data, so that it looks more like the markings
- exposure.equalizehist()
- gaussian blurring
- weighted average
- contrast strecthing again

Updated 05.09.2023

Joy Yeh
'''

import SimpleITK as sitk
import numpy as np
import matplotlib.pylab as plt
from skimage import exposure
import scipy.ndimage as ndi
import cv2

# 0. Basic Control Panel
patient_idx = 25
heartbeat_state = 'ED'  # set to 'ED' or 'ES' to load the corresponding files
folder_name = "data/patient" + str(patient_idx).zfill(4) + "/"
channel_number = 2

# 1. Open the .cfg file
cfg_name = 'Info_{0}CH.cfg'.format(channel_number)

with open(folder_name + cfg_name, 'r') as f:
    # Read the lines of the file into a list
    lines = f.readlines()

# construct the filenames based on the heartbeat_state
mhp01 = 'patient{0:04d}_{1}CH_{2}.mhd'.format(patient_idx, channel_number, heartbeat_state)
mhp01_gt = 'patient{0:04d}_{1}CH_{2}_gt.mhd'.format(patient_idx, channel_number, heartbeat_state)

# load the images and ground truth files
I1 = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + mhp01, sitk.sitkFloat32))
I1_gt = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + mhp01_gt, sitk.sitkFloat32))

''' Image processing starts here'''
# 1. Contrast enhancement
I1_c = exposure.equalize_hist(I1[0], nbins=256)

# 2. Gaussion blurring
sigma = (20, 20)
g_filt01 = ndi.gaussian_filter(I1_c, sigma)

# 3. Weighted avg
# Compute the noise levels and add weighted
noise_I = cv2.meanStdDev(I1_c)[1][0][0]
noise_gf = cv2.meanStdDev(g_filt01)[1][0][0]
w_I = noise_gf / (noise_I + noise_gf)
w_gf = noise_I / (noise_I + noise_gf)
I_comb = cv2.addWeighted(I1_c, w_I, g_filt01, w_gf, 0)

# 4. Contrast stretching
stretch_range = (10, 95)
p2, p98 = np.percentile(I_comb, stretch_range)
I_out = exposure.rescale_intensity(I_comb, in_range=(p2, p98))

''' Same-patient technique comparison '''
# Create a figure with a single row and four columns
fig, axs = plt.subplots(1, 5)

# Plot the images on the subplots
axs[0].imshow(I1[0], cmap = 'gray')
axs[1].imshow(I1_c, cmap = 'gray')
axs[2].imshow(I_comb, cmap='gray')
axs[3].imshow(I_out, cmap = 'gray')
axs[4].imshow(I1_gt[0], cmap = 'gray')

# Set the title for each subplot
axs[0].set_title('P{0} Orig'.format(patient_idx))
axs[1].set_title('Contrast')
axs[2].set_title('Combined')
axs[3].set_title('EQ = {0}'.format(stretch_range))
axs[4].set_title('Labels')

# Hide the x and y ticks for all subplots
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.1)

# Display the subplots
plt.show()


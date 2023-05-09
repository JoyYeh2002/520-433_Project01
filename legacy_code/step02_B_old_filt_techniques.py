'''
EN.520.433 Medical Image Analysis
Spring 2023

Step02: Do image processing on the raw data, so that it looks more like the markings
Updated 05.09.2023

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

''' Control Panel '''
sigma = (10, 10)
window_size = 5
window_size2 = (1, 1)


# Compute the noise levels of the two images
noise_I = cv2.meanStdDev(I)[1][0][0]
noise_gf = cv2.meanStdDev(gf)[1][0][0]

# Compute the weights for each image
w_I = noise_gf / (noise_I + noise_gf)
w_gf = noise_I / (noise_I + noise_gf)

# Combine the two images using a weighted average
I_comb = cv2.addWeighted(I, w_I, gf, w_gf, 0)


# 2. Denoising
# apply a Gaussian filter with a standard deviation of 2 in the x and y dimensions, and 1 in the z dimension
g_filt01 = ndi.gaussian_filter(I1_c, sigma)
g_filt02 = ndi.gaussian_filter(I2_c, sigma)

# Median blurring for denoising
d1 = cv2.medianBlur(g_filt01[:, :], window_size) # 5 is the size of the filter window
d1 = np.expand_dims(d1, axis=0)
print(d1.shape)

d2 = cv2.medianBlur(g_filt02[:, :], window_size) # 5 is the size of the filter window
d2 = np.expand_dims(d2, axis=0)

# blur #2 on image 1
d1b = np.squeeze(d1)
d1b = np.expand_dims(d1b, axis=2)
blur1 = cv2.blur(d1b, window_size2)

''' Same-patient technique comparison '''
# Create a figure with a single row and four columns
fig, axs = plt.subplots(1, 5)

# Plot the images on the subplots
axs[0].imshow(I1_c, cmap = 'gray')
axs[1].imshow(g_filt01, cmap = 'gray')
axs[2].imshow(d1[0], cmap='gray')
axs[3].imshow(blur1, cmap = 'gray')
axs[4].imshow(I1_gt[0], cmap = 'gray')

# Set the title for each subplot
axs[0].set_title('P1 Orig')
axs[1].set_title('Sigma = {0}'.format(sigma))
axs[2].set_title('Blur = {0}'.format(window_size))
axs[3].set_title('Blur = {0}'.format(window_size2))
axs[4].set_title('Labels')


# Hide the x and y ticks for all subplots
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.1)

# Display the subplots
plt.show()

''' Cross-patient plotting comparison '''
compare_two_patients = False
if compare_two_patients == True:
    # create a subplot with three images
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))

    # Display the images
    axs[0, 0].imshow(I1_c, cmap='gray')
    axs[0, 1].imshow(d1[0], cmap='gray')
    #axs[0, 2].imshow(I1_gt[0], cmap='gray')

    axs[1, 0].imshow(I2_c, cmap='gray')
    axs[1, 1].imshow(d2[0], cmap='gray')
    #axs[1, 2].imshow(I2_gt[0], cmap='gray')

    # Set the titles
    axs[0, 0].set_title('P#{0} Raw'.format(patient_idx01))
    axs[0, 1].set_title('P#{0} Denoised'.format(patient_idx01))
    #axs[0, 2].set_title('P#{0} Labels'.format(patient_idx01))

    axs[1, 0].set_title('P#{0} Raw'.format(patient_idx02))
    axs[1, 1].set_title('P#{0} Denoised'.format(patient_idx02))
    #axs[1, 2].set_title('P#{0} Labels'.format(patient_idx02))

    # Remove the ticks from the subplots
    plt.setp(axs, xticks=[], yticks=[])
    plt.show()

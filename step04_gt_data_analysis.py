'''
EN.520.433 Medical Image Analysis
Spring 2023

Step04: Get curvature and area information from the gt marking
- Plot the contour overlayed on the image
- Save the contour arrays
- Evolve the snake???

Updated 05.09.2023

Hannah Qu, Joy Yeh
'''

import SimpleITK as sitk
import matplotlib.pylab as plt
import cv2
import numpy as np

# 0. Basic Control Panel
patient_idx = 25
heartbeat_state = 'ES'  # set to 'ED' or 'ES' to load the corresponding files
folder_name = "data/patient" + str(patient_idx).zfill(4) + "/"
display_markings = True
display_sequence = False
channel_number = 2

# 1. Open the .cfg file
cfg_name = 'Info_{0}CH.cfg'.format(channel_number)
with open(folder_name + cfg_name, 'r') as f:
    # Read the lines of the file into a list
    lines = f.readlines()

# construct the filenames based on the heartbeat_state
mhp = 'patient{0:04d}_{1}CH_{2}.mhd'.format(patient_idx, channel_number, heartbeat_state)
mhp_gt = 'patient{0:04d}_{1}CH_{2}_gt.mhd'.format(patient_idx, channel_number, heartbeat_state)

# load the images and ground truth files
I = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + mhp, sitk.sitkFloat32))
I_gt = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + mhp_gt, sitk.sitkFloat32))

# Threshold the image to isolate pixels below a certain intensity
regions = []
for i in range (4):
    regions.append(I_gt.copy())
    regions[i][I_gt != i] = 0
regions = regions[1:]

''' Work with the region'''
r1 = regions[0][0]

# plt.imshow(r1, cmap='gray')
# plt.show()

img = r1

# Get the contours of the shape
contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank canvas with the same dimensions as the input image
# canvas = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

# Convert the grayscale image into RGB
# Scale the intensity range of the grayscale image to 0-255
img_scaled = cv2.normalize(I[0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

# Convert the grayscale image into RGB
img_rgb = cv2.cvtColor(img_scaled, cv2.COLOR_GRAY2RGB)

# Set the color of the contour pixels to red in the RGB image
cv2.drawContours(img_rgb, contours, -1, (0, 0, 255), 2)

# Display the canvas with the contour in red color
cv2.imshow('Contour Image', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

''' Plot the marking regions '''
examine_segments = False
if examine_segments == True:
    fig, axs = plt.subplots(1, 4, figsize=(10, 5))
    for i in range(3):
        axs[i+1].imshow(regions[i][0], cmap='gray')
    axs[0].imshow(I_gt[0], cmap='gray')

    axs[0].set_title('P#{0} Original'.format(patient_idx))
    axs[1].set_title('P#{0} region 1'.format(patient_idx))
    axs[2].set_title('P#{0} region 2'.format(patient_idx))
    axs[3].set_title('P#{0} region 3'.format(patient_idx))

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
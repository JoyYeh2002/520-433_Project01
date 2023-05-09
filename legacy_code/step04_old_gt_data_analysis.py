'''
EN.520.433 Medical Image Analysis
Spring 2023

Step04: Segment gt marking (regions 1, 2, 3, 4)
Legacy code 
Hannah Qu
'''

import SimpleITK as sitk
import matplotlib.pylab as plt
import cv2
import numpy as np
from helper_functions import pre_process

# 0. Basic Control Panel
patient_idx = 5
heartbeat_state = 'ES'  # set to 'ED' or 'ES' to load the corresponding files
folder_name = "data/patient" + str(patient_idx).zfill(4) + "/"
display_markings = True
display_sequence = False
channel_number = 4

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

# Create a memory copy of original image I
img = I.copy()

# Threshold the image to isolate pixels below a certain intensity
regions = []
for i in range (4):
    regions.append(I_gt.copy())
    regions[i][I_gt != i] = 0
regions = regions[1:]

''' Work with the region'''
r1 = regions[0][0]
r2 = regions[1][0]
r3 = regions[2][0]

# Get the 3 contours of markings
c1, _ = cv2.findContours(r1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c2, _ = cv2.findContours(r2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c3, _ = cv2.findContours(r3.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

''' [Legacy code] Plot the marking regions '''
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
'''
EN.520.433 Medical Image Analysis
Spring 2023

Step04: Get curvature and area information from the gt marking

Updated 05.05.2023

Hannah Qu
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
cfg_name = 'Info_2CH.cfg'
with open(folder_name + cfg_name, 'r') as f:
    # Read the lines of the file into a list
    lines = f.readlines()

# construct the filenames based on the heartbeat_state
mhp_2ch = 'patient{0:04d}_2CH_{1}.mhd'.format(patient_idx, heartbeat_state)
mhp_2ch_gt = 'patient{0:04d}_2CH_{1}_gt.mhd'.format(patient_idx, heartbeat_state)

# load the images and ground truth files
I = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + mhp_2ch, sitk.sitkFloat32))
I_gt = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + mhp_2ch_gt, sitk.sitkFloat32))


# Threshold the image to isolate pixels below a certain intensity
threshold = 0
low = 0
high = 255
binary = []
for i in range (4):
    binary.append(I_gt.copy())
    binary[i][I_gt != i] = 0
#binary[binary != ] = 255
#print(np.unique(binary))


fig, axs = plt.subplots(1, 4, figsize=(10, 5))

for i in range(4):
    axs[i].imshow(binary[i][0], cmap='gray')
# axs[0].imshow(I[0], cmap = 'gray')
# axs[1].imshow(I_gt[0], cmap = 'gray')
# axs[2].imshow(binary[0], cmap='gray')
# axs[3].imshow(I_gt[0], cmap = 'gray')

axs[0].set_title('P#{0} Binary1'.format(patient_idx))
axs[1].set_title('P#{0} Binary2'.format(patient_idx))
axs[2].set_title('P#{0} Binary3'.format(patient_idx))
axs[3].set_title('P#{0} Binary4'.format(patient_idx))

# remove the ticks from the subplots
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
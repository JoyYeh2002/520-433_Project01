'''
EN.520.433 Medical Image Analysis
Spring 2023

Step04: Get curvature and area information from the gt marking
- Plot the contour overlayed on the image
- Save the contour arrays and processed images to pickle files

Updated 05.09.2023

Joy Yeh, Hannah Qu
'''
import pickle
import SimpleITK as sitk
import matplotlib.pylab as plt
import cv2
import numpy as np
from helper_functions import pre_process

# 0. Basic Control Panel
patient_idx = 10
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
c1, _ = cv2.findContours(r1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # RED
c2, _ = cv2.findContours(r2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # GREEN
c3, _ = cv2.findContours(r3.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # BLUE

''' Plot on top of original image '''
plot_on_orig_img = False
if plot_on_orig_img == True:
    # Scale the intensity range of the grayscale image to 0-255
    I = cv2.normalize(I[0], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    I = cv2.cvtColor(I, cv2.COLOR_GRAY2RGB)

    # Set the color of the contour pixels to red in the RGB image
    linewidth = 4
    cv2.drawContours(I, c2, -1, (0, 255, 0), linewidth)
    cv2.drawContours(I, c1, -1, (0, 0, 255), linewidth)
    cv2.drawContours(I, c3, -1, (255, 0, 0), linewidth)

    # Save the canvas with the contours in red
    cv2.imwrite('outputs\images\patinet_{0}_{1}CH_contours_orig.jpg'.format(patient_idx, channel_number), I)

''' Plot on top of processed/enhanced image '''
I_out = pre_process(np.squeeze(img))
I_out = cv2.normalize(I_out, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

plot_on_img = False
if plot_on_img == True:
    I_out = cv2.cvtColor(I_out, cv2.COLOR_GRAY2RGB)

    # Set the color of the contour pixels to red in the RGB image
    linewidth = 4
    cv2.drawContours(I_out, c2, -1, (0, 255, 0), linewidth)
    cv2.drawContours(I_out, c1, -1, (0, 0, 255), linewidth)
    cv2.drawContours(I_out, c3, -1, (255, 0, 0), linewidth)

    # Save the canvas with the contours in red
    cv2.imwrite('outputs\images\patinet{0:04d}_{1}CH_contours_processed.jpg'.format(patient_idx, channel_number), I_out)

    # Display the canvas with the contour in red color
    cv2.imshow('Contour Image', I_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

''' SNAKE data preparation strats here'''
# Load and pre-process the sequence
sequence_file = 'patient{0:04d}_{1}CH_sequence.mhd'.format(patient_idx, channel_number)
I_seq = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + sequence_file, sitk.sitkFloat32))

seq_out = []
for i in range(I_seq.shape[0]):
    this_I = I_seq[i]
    seq_out.append(pre_process(I_seq[i]))

# Save these to a pickle file
out_file_name = "outputs\pickles\patient{0:04d}_{1}_CH_evolution_variables.pkl".format(patient_idx, channel_number)
with open(out_file_name, "wb") as f:
    pickle.dump(patient_idx, f)
    pickle.dump(channel_number, f)
    pickle.dump(lines, f)

    pickle.dump(I, f) # Original Image
    pickle.dump(I_gt, f) # the label
    pickle.dump(r2, f) # the binary red contour (the most important one)
    pickle.dump(I_out, f) # Processed img
    pickle.dump(seq_out, f) # Processed seq

    pickle.dump(c1, f) # Green contour
    pickle.dump(c2, f) # Red contour
    pickle.dump(c3, f) # Blue contour

print("Pickle File: patient{0:04d}_{1}_CH_evolution_variables.pkl is saved".format(patient_idx, channel_number))


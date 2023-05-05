'''
EN.520.433 Medical Image Analysis
Spring 2023

Step01: Image loading and display

Updated 05.03.2023

Joy Yeh
'''

import SimpleITK as sitk
import numpy as np
import matplotlib.pylab as plt

# 0. Basic Control Panel
patient_idx = 1
folder_name = "data/patient" + str(patient_idx).zfill(4) + "/"

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

# 2. Load the .mhd file
mhp_name = 'patient' + str(patient_idx).zfill(4) + '_2CH_ED.mhd'
I = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + mhp_name, sitk.sitkFloat32))

# plt.figure(figsize=(20,16))
plt.figure()
plt.gray()
plt.subplots_adjust(0,0,1,1,0.01,0.01)
for i in range(I.shape[0]):
    plt.subplot(5,6,i+1), plt.imshow(I[i]), plt.axis('off')
    # use plt.savefig(...) here if you want to save the images as .jpg, e.g.,
plt.show()


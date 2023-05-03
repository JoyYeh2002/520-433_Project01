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

# Load the .cfg file
patient_idx = 1
folder_name = "data/patient" + str(patient_idx).zfill(4) + "/"
# image = sitk.ReadImage(folder_name + 'info_4CH.cfg')

# Load the .mhd file
I = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + 'patient' + str(patient_idx).zfill(4) + '_2CH_ED.mhd', sitk.sitkFloat32))

# print(ct_scans.shape())
plt.figure(figsize=(20,16))
plt.gray()
plt.subplots_adjust(0,0,1,1,0.01,0.01)
for i in range(I.shape[0]):
    plt.subplot(5,6,i+1), plt.imshow(I[i]), plt.axis('off')
    # use plt.savefig(...) here if you want to save the images as .jpg, e.g.,
plt.show()
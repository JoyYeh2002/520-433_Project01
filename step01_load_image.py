'''
EN.520.433 Medical Image Analysis
Spring 2023

Step01: Image loading and display

Updated 05.03.2023

Joy Yeh
'''

import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot as plt

# Load the .cfg file
patient_idx = 1
folder_name = "data/patient" + str(patient_idx).zfill(4) + "/"
image = sitk.ReadImage(folder_name + 'info_2CH.cfg')

# Convert the image to a NumPy array
image_array = sitk.GetArrayFromImage(image)

# Display the image using matplotlib
plt.imshow(image_array, cmap='gray')
plt.show()

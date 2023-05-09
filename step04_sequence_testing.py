'''
EN.520.433 Medical Image Analysis
Spring 2023

Step02: Package the pre-processing into a helper function, then test on a sequence

Updated 05.09.2023

Joy Yeh
'''

import SimpleITK as sitk
import numpy as np
import matplotlib.pylab as plt
from skimage import exposure
import scipy.ndimage as ndi
import cv2

def pre_process(I_in):
    # 1. Contrast enhancement
    I1_c = exposure.equalize_hist(I_in, nbins=256)

    # 2. Gaussian blurring
    sigma = (20, 20)
    g_filt01 = ndi.gaussian_filter(I1_c, sigma)

    # 3. Weighted average
    # Compute the noise levels and add weighted
    noise_I = cv2.meanStdDev(I1_c)[1][0][0]
    noise_gf = cv2.meanStdDev(g_filt01)[1][0][0]
    w_I = noise_gf / (noise_I + noise_gf)
    w_gf = noise_I / (noise_I + noise_gf)
    I_comb = cv2.addWeighted(I1_c, w_I, g_filt01, w_gf, 0)

    # 4. Contrast stretching
    stretch_range = (5, 95)
    p2, p98 = np.percentile(I_comb, stretch_range)
    I_out = exposure.rescale_intensity(I_comb, in_range=(p2, p98))

    return I_out

''' Main starts here '''
# Basic Control Panel
patient_idx = 30
heartbeat_state = 'ED'  # set to 'ED' or 'ES' to load the corresponding files
folder_name = "data/patient" + str(patient_idx).zfill(4) + "/"
channel_number = 2

display_sequence = True
if display_sequence == True:
    # construct the filename based on the channel_number
    sequence_file = 'patient{0:04d}_{1}CH_sequence.mhd'.format(patient_idx, channel_number)

    # load the sequence of images
    I_seq = sitk.GetArrayFromImage(sitk.ReadImage(folder_name + sequence_file, sitk.sitkFloat32))

    # Put them into the helper function
    out_img_arr = []
    for i in range(I_seq.shape[0]):
        this_I = I_seq[i]
        out_img_arr.append(pre_process(I_seq[i]))

    # display the images using matplotlib
    fig, axs = plt.subplots(1, I_seq.shape[0], figsize=(25, 25))
    for i in range(I_seq.shape[0]):
        axs[i].imshow(out_img_arr[i], cmap='gray')
        axs[i].imshow(I_seq[i], cmap='gray')
        axs[i].axis('off')

    plt.suptitle('Patient {0}, {1}CH Sequence'.format(patient_idx, channel_number))
    plt.show()

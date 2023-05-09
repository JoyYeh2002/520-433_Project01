# helper_functions.py

import SimpleITK as sitk
import numpy as np
import matplotlib.pylab as plt
from skimage import exposure
import scipy.ndimage as ndi
import cv2

def pre_process(img_in):
    # 1. Ccontrast enhancement
    I1_c = exposure.equalize_hist(img_in, nbins=256)

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

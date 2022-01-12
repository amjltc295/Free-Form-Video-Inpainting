from glob import glob
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.function_base import average

from skimage.measure import compare_psnr, compare_ssim, compare_mse

resdir = './saved/C2C_model_clouds_train/0111_170820/test_outputs/epoch_140/test_cloud/'
gtdir = '../dataset/clouds/inpaint/test/image/'

mse_vals = []
psnr_vals = []
ssim_vals = []
for cnt in range(len(glob(resdir + '/*'))):
    resimf = resdir + f'/result_{cnt:04d}/frame_0004.png'
    gtimf = gtdir + f'{cnt:03d}/005.png'
    resim = np.asarray(Image.open(resimf))
    resgt = np.asarray(Image.open(gtimf))

    if np.all(resim == resgt):
        continue
    mse_vals += [compare_mse(resgt, resim)]
    psnr_vals += [compare_psnr(resgt, resim, data_range=255)]
    ssim_vals += [compare_ssim(resgt, resim, data_range=255, multichannel=True)]
    # print(imfile)
    # plt.imshow(im)
    # plt.show()
# mse_vals = np.array(mse_vals)
# psnr_vals = np.array(psnr_vals)
# ssim_vals = np.array(ssim_vals)
# mask = psnr_vals != np.inf
print(f"MSE: {np.average(mse_vals):.4f}, PSNR {np.average(psnr_vals):.4f}, SSIM {np.average(ssim_vals):.4f}")
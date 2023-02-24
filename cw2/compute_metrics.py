# Create function to compute similarity metrics
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compute_metrics(original_slice, re_sliced_image):
    # Compute the MSE between the two slices
    mse = np.mean((original_slice - re_sliced_image) ** 2)

    # Compute the SSIM between the two slices
    ssim_value = ssim(original_slice, re_sliced_image, gaussian_weights=False, channel_axis=None)
   
    # Compute the PSNR between the two slices
    # Compute the maximum pixel value of the images
    max_pixel = np.iinfo(original_slice.dtype).max

    # Compute the peak signal-to-noise ratio
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))



    return mse, ssim_value, psnr
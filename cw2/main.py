# This script compares the performance of two approaches:
# 3D filtering before reslicing and 2D filtering after reslicing.

# 1. # Implement the 3D before filtering approach
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Import the functions from the other scripts
# reslice:function that takes a 3D image and returns a desired 2D slice of the image in an oblique view.
from reslice import reslice 
# compute_metrics: function to compute similarity metrics
from compute_metrics import compute_metrics
# bilateral_filters: functions to apply a 2D/3D bilateral filter to an image
from bilateral_filters import bilateral_filter_3d, bilateral_filter_2d
# plot_oblique_view: function to plot the oblique view of a slice
from plot_oblique_view import plot_oblique_view
# compute_metrics: function to compute similarity metrics
from compute_metrics import compute_metrics

# Load image
image = np.load("test_trus.npy").T


# Create helper function to apply 3D bilateral filter before reslicing
# Create function for 3D filtering before reslicing
def filter_3d_before_reslice(image, slice_idx, diameter, sigma_intensity, sigma_spatial, theta, phi, pre_filtered_image = False, filtered_image_3d = None, exp_num = None):
    """
    Function that applies a 3D bilateral filter to an image,
    creates an oblique view of the image, computes the similarity metrics, 
    plots the oblique view of the image and saves the results in a text file.

    Parameters
    ----------
    image: 3D numpy array
        Input image.
    slice_idx: int 
        Index of the slice to be extracted.
    diameter: int
        Diameter of the filter.
    sigma_intensity: float
        Standard deviation of the intensity Gaussian.
    sigma_spatial: float   
        Standard deviation of the spatial Gaussian.
    theta: float
        Angle of rotation about the z-axis.
    phi: float
        Angle of rotation about the x-axis.
    pre_filtered_image: bool
        Boolean indicating whether the a pre-filtered image will be provided hence the 3D filter will not be applied.
    filtered_image_3d: 3D numpy array
        Pre-filtered image.
    exp_num: int
        Experiment number.

    Returns
    -------
    Prints the similarity metrics and saves the results in a text file.
    """


    # Apply the 3D bilateral filter
    if pre_filtered_image == False:
        filtered_image_3d = bilateral_filter_3d(image[:, :, :3], diameter, sigma_intensity, sigma_spatial)
    else:
        filtered_image_3d = filtered_image_3d
    

    # Reslice the image
    #theta = 0 # angle of rotation about the z-axis
    #phi = -25 # angle of rotation about the x-axis
    xx, yy, zz, slice_values = reslice(filtered_image_3d, theta, phi, slice_idx)

    # Plot the oblique view of the filtered image
    plot_oblique_view(xx, yy, zz, slice_values, slice_idx, filtered_image_3d, f"Experiment {exp_num} 3d_filtered")

    # Compute the similarity metrics
    re_sliced_image = slice_values.reshape(xx.shape) # reshape the slice to the original shape
    mse, ssim_value, psnr = compute_metrics(image[:, :, slice_idx], re_sliced_image)

    # Print the results
    print(f"Results for 3D filtering before reslicing for slice {slice_idx}, experiment {exp_num}: \n")
    print(f"MSE: {mse} \n")
    print(f"SSIM: {ssim_value} \n")
    print(f"PSNR: {psnr} \n")

    # Save the results in a text file
    with open(f"results_3d_{slice_idx}.txt", "a") as f:
        f.write(f"\nResults for 3D filtering before reslicing for slice {slice_idx}, experiment {exp_num}: \n")
        f.write(f"MSE: {mse} \n")
        f.write(f"SSIM: {ssim_value} \n")
        f.write(f"PSNR: {psnr} \n")

# Create helper function to apply 2D bilateral filter after reslicing
# Create function for 2D filtering after reslicing
def filter_2d_after_reslice(image, slice_idx, diameter, sigma_intensity, sigma_spatial, theta, phi, pre_filtered_image = False, filtered_image_2d = None, exp_num = None):
    """
    Function that creates an oblique view of the image, applies a 2D bilateral filter to the slice,
    computes the similarity metrics, plots the oblique view of the image and saves the results in a text file.
    
    Parameters
    ----------
    image: 3D numpy array
        Input image.
    slice_idx: int 
        Index of the slice to be extracted.
    diameter: int
        Diameter of the filter.
    sigma_intensity: float
        Standard deviation of the intensity Gaussian.
    sigma_spatial: float   
        Standard deviation of the spatial Gaussian.
    theta: float
        Angle of rotation about the z-axis.
    phi: float
        Angle of rotation about the x-axis.
    pre_filtered_image: bool
        Boolean indicating whether the a pre-filtered image will be provided hence the 3D filter will not be applied.
    filtered_image_3d: 3D numpy array
        Pre-filtered image.
    exp_num: int
        Experiment number.

    Returns
    -------
    Prints the similarity metrics and saves the results in a text file.

    """

    # a. Apply reslicing
    #theta = 0 # angle of rotation about the z-axis
    #phi = -25 # angle of rotation about the x-axis
    xx, yy, zz, slice_values = reslice(image, theta, phi, slice_idx)

    # b. Apply the 2D bilateral filter
    if pre_filtered_image == False:
        re_sliced_image = slice_values.reshape(xx.shape)
        filtered_image_2d = bilateral_filter_2d(re_sliced_image, diameter, sigma_intensity, sigma_spatial)
    else:
        filtered_image_2d = filtered_image_2d
    # re_sliced_image = slice_values.reshape(xx.shape) # reshape the slice to the original shape
    # filtered_image_2d = bilateral_filter_2d(re_sliced_image, diameter, sigma_intensity, sigma_spatial)

    # Plot the oblique view of the filtered image
    plot_oblique_view(xx, yy, zz, filtered_image_2d, slice_idx, image, f"Experiment {exp_num} 2d_filtered")

    # Compute the similarity metrics
    mse, ssim_value, psnr = compute_metrics(image[:, :, slice_idx], filtered_image_2d)

    # Print the results
    print(f"Results for 2D filtering after reslicing for slice {slice_idx}, experiment {exp_num}: \n")
    print(f"MSE: {mse} \n")
    print(f"SSIM: {ssim_value} \n")
    print(f"PSNR: {psnr} \n")

    # Save the results in a text file that does not overwrite the previous results
    with open(f"results_2d_{slice_idx}.txt", "a") as f:
        f.write(f"\nResults for 2D filtering after reslicing for slice {slice_idx}, experiment {exp_num}: \n")
        f.write(f"\nMSE: {mse} \n")
        f.write(f"SSIM: {ssim_value} \n")
        f.write(f"PSNR: {psnr} \n")



#-------------Investigate the effect of changing the diameter of the filter---------------------#
# Define the slice to be extracted
slice_idx = 0

# Define the parameters for the bilateral filters
sigma_spatial = 5
sigma_intensity = 5
diameter = 3

# Define the angles of rotation
theta = 0 # angle of rotation about the z-axis
phi = -25 # angle of rotation about the x-axis

# Apply the 3D bilateral filter before reslicing
filtered_image_3d = np.load("Full_filtered_image_3d.npy")
filter_3d_before_reslice(image, slice_idx, diameter, sigma_intensity, sigma_spatial, theta, phi,
                         pre_filtered_image = True, filtered_image_3d = filtered_image_3d, exp_num=1)

# Apply the 2D bilateral filter after reslicing
filter_2d_after_reslice(image, slice_idx, diameter, sigma_intensity, sigma_spatial, theta, phi,
                        pre_filtered_image = False, filtered_image_2d = None, exp_num=1)

# Change the diameter of the filter
diameter = 5
filtered_image_3d = np.load("Full_filtered_image_3d_d.npy")
filter_3d_before_reslice(image, slice_idx, diameter, sigma_intensity, sigma_spatial, theta, phi,
                         pre_filtered_image = True, filtered_image_3d = filtered_image_3d, exp_num=2)

# Apply the 2D bilateral filter after reslicing
filter_2d_after_reslice(image, slice_idx, diameter, sigma_intensity, sigma_spatial, theta, phi,
                        pre_filtered_image = False, filtered_image_2d = None, exp_num=2)


#-----------------Investigate the effect of changing the sigma_spatial of the filter---------------------#
# Define the slice to be extracted
slice_idx = 0

# Define the parameters for the bilateral filters
sigma_spatial = 5
sigma_intensity = 5
diameter = 3

# Define the angles of rotation
theta = 0 # angle of rotation about the z-axis
phi = -25 # angle of rotation about the x-axis

# Apply the 3D bilateral filter before reslicing
filtered_image_3d = np.load("Full_filtered_image_3d.npy")
filter_3d_before_reslice(image, slice_idx, diameter, sigma_intensity, sigma_spatial, theta, phi,
                         pre_filtered_image = True, filtered_image_3d = filtered_image_3d, exp_num=3)

# Apply the 2D bilateral filter after reslicing
filter_2d_after_reslice(image, slice_idx, diameter, sigma_intensity, sigma_spatial, theta, phi,
                        pre_filtered_image = False, filtered_image_2d = None, exp_num=3)

# Change the sigma_spatial of the filter
sigma_spatial = 80
filtered_image_3d = np.load("Full_filtered_image_3d_ssp.npy")
filter_3d_before_reslice(image, slice_idx, diameter, sigma_intensity, sigma_spatial, theta, phi,
                            pre_filtered_image = True, filtered_image_3d = filtered_image_3d, exp_num=4)

# Apply the 2D bilateral filter after reslicing
filter_2d_after_reslice(image, slice_idx, diameter, sigma_intensity, sigma_spatial, theta, phi,
                        pre_filtered_image = False, filtered_image_2d = None, exp_num=4)

#--------------------Investigate the effect of changing the sigma_intensity of the filter-------------------#
# Define the slice to be extracted
slice_idx = 0

# Define the parameters for the bilateral filters
sigma_spatial = 5
sigma_intensity = 5
diameter = 3

# Define the angles of rotation
theta = 0 # angle of rotation about the z-axis
phi = -25 # angle of rotation about the x-axis

# Apply the 3D bilateral filter before reslicing
filtered_image_3d = np.load("Full_filtered_image_3d.npy")
filter_3d_before_reslice(image, slice_idx, diameter, sigma_intensity, sigma_spatial, theta, phi,
                            pre_filtered_image = True, filtered_image_3d = filtered_image_3d, exp_num=5)

# Apply the 2D bilateral filter after reslicing
filter_2d_after_reslice(image, slice_idx, diameter, sigma_intensity, sigma_spatial, theta, phi,
                        pre_filtered_image = False, filtered_image_2d = None, exp_num=5)

# Change the sigma_intensity of the filter
sigma_intensity = 80
filtered_image_3d = np.load("Full_filtered_image_3d_scol_80.npy")
filter_3d_before_reslice(image, slice_idx, diameter, sigma_intensity, sigma_spatial, theta, phi,
                            pre_filtered_image = True, filtered_image_3d = filtered_image_3d, exp_num=6)

# Apply the 2D bilateral filter after reslicing
filter_2d_after_reslice(image, slice_idx, diameter, sigma_intensity, sigma_spatial, theta, phi,
                        pre_filtered_image = False, filtered_image_2d = None, exp_num=6)





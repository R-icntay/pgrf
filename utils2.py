"""
utility functions for use in the image registration exercises 2 for module MPHY0025 (IPMI)

Jamie McClelland
UCL
"""

import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('default')
import scipy.interpolate as scii

def dispImage(img, int_lims = []):
  """
 function to display a grey-scale image that is stored in 'standard
 orientation' with y-axis on the 2nd dimension and 0 at the bottom

 SYNTAX:
   dispImage(img)
   dispImage(img, int_lims)

 INPUTS:
   img - image to be displayed
   int_lims - the intensity limits to use when displaying the image
       int_lims(1) = min intensity to display
       int_lims(2) = max intensity to display
       default = [np.nanmin(img), np.nanmax(img)]
  """

  #check if intensity limits have been provided, and if not set to min and
  #max of image
  if not int_lims:
    int_lims = [np.nanmin(img), np.nanmax(img)]
    #check if min and max are same (i.e. all values in img are equal)
    if int_lims[0] == int_lims[1]:
      #add one to int_lims(2) and subtract one from int_lims(1), so that
      #int_lims(2) is larger than int_lims(1) as required by imagesc
      #function
      int_lims[0] -= 1
      int_lims[1] += 1
  
  # take transpose of image to switch x and y dimensions and display with
  # first pixel having coordinates 0,0
  img = img.T
  plt.gca().clear()
  plt.imshow(img, cmap = 'gray', vmin = int_lims[0], vmax = int_lims[1], origin='lower')
  
  #set axis to be scaled equally (assumes isotropic pixel dimensions), tight
  #around the image
  plt.axis('image')
  plt.tight_layout()

def defFieldFromAffineMatrix(aff_mat, num_pix_x, num_pix_y):
  """
 function to create a 2D deformation field from an affine matrix

 SYNTAX:
   def_field = defFieldFromAffineMatrix(aff_mat, num_pix_x, num_pix_y)

 INPUTS:
   aff_mat - a 3 x 3 numpy array representing the 2D affine transformation
       in homogeneous coordinates
   num_pix_x - number of pixels in the deformation field along the x
       dimension
   num_pix_y - number of pixels in the deformation field along the y
       dimension

 OUTPUTS:
   def_field - the 2D deformation field as a 3D numpy array

 NOTES:
   the function calculates the pixel coordinates of the deformation field
   as:
       x = 0:num_pix_x - 1
       y = 0:num_pix_y - 1
  """
  # form 2D arrays containing all the pixel coordinates
  [X,Y] = np.mgrid[0:num_pix_x, 0:num_pix_y]

  # reshape and combine coordinate arrays into a 3 x N array, where N is
  # the total number of pixels (num_pix_x x num_pix_y)
  # the 1st row contains the x coordinates, the 2nd the y coordinates, and the
  # 3rd row is all set to 1 (i.e. using homogenous coordinates)
  total_pix = num_pix_x * num_pix_y
  pix_coords = np.array([np.reshape(X, -1), np.reshape(Y, -1), np.ones(total_pix)])
  
  # apply transformation to pixel coordinates
  trans_coords = aff_mat @ pix_coords
  
  #reshape into deformation field by first creating an empty deformation field
  def_field = np.zeros((num_pix_x, num_pix_y, 2))
  def_field[:,:,0] = np.reshape(trans_coords[0,:], (num_pix_x, num_pix_y))
  def_field[:,:,1] = np.reshape(trans_coords[1,:], (num_pix_x, num_pix_y))
  return def_field

def resampImageWithDefField(source_img, def_field, interp_method = 'linear', pad_value=np.NaN):
  """
 function to resample a 2D image with a 2D deformation field

 SYNTAX:
   resamp_img = resampImageWithDefField(source_img, def_field)
   resamp_img = resampImageWithDefField(source_img, def_field, interp_method)
   resamp_img = resampImageWithDefField(source_img, def_field, interp_method, pad_value)

 INPUTS:
   source_img - the source image to be resampled, as a 2D array
   def_field - the deformation field, as a 3D array
   inter_method - any of the interpolation methods accepted by interpn
       function ('linear', 'nearest' and 'splinef2d')
       default = 'linear'
   pad_value - the value to assign to pixels that are outside the source
       image
       default = NaN

 OUTPUTS:
   resamp_img - the resampled image

 NOTES:
   the deformation field should be a 3D array, where the size of the
   first two dimensions is the size of the resampled image, and the size
   of the 3rd dimension is 2. def_field(:,:,1) contains the x coordinates
   of the transformed pixels, def_field(:,:,2) contains the y coordinates
   of the transformed pixels.
   the origin of the source image is assumed to be the bottom left pixel
  """
  x_coords = np.arange(source_img.shape[0], dtype = 'float')
  y_coords = np.arange(source_img.shape[1], dtype = 'float')
  
  # resample image using interpn function
  return scii.interpn((x_coords, y_coords), source_img, def_field, bounds_error=False, fill_value=pad_value, method=interp_method)

def affineMatrixForRotationAboutPoint(theta, p_coords):
  """
  function to calculate the affine matrix corresponding to an anticlockwise
  rotation about a point

  SYNTAX:
    aff_mat = affineMatrixForRotationAboutPoint(theta, p_coords)
  
  INPUTS:
    theta - the angle of the rotation, specified in degrees
    p_coords - the 2D coordinates of the point that is the centre of rotation.
        p_coords[0] is the x coordinate,
        p_coords[1] is the y coordinate
  
  OUTPUTS:
    aff_mat - a numpy array representing the 3 x 3 affine matrix
  """ 
  #convert theta to radians
  theta = np.pi * theta / 180

  #form matrices for translation and rotation
  T1 = np.array([[1, 0, -p_coords[0]],
                  [0, 1, -p_coords[1]],
                  [0,0,1]])
  T2 = np.array([[1, 0, p_coords[0]],
                  [0, 1, p_coords[1]],
                  [0,0,1]])
  R = np.array([[np.cos(theta), -np.sin(theta), 0],
                 [np.sin(theta), np.cos(theta), 0],
                 [0, 0, 1]])
  
  #compose matrices using matrix multiplication
  aff_mat = T2 @ R @ T1
  # ***************
  return aff_mat

def calcSSD(A,B):
  """
 function to calculate the sum of squared differences between two images

 SYNTAX:
   SSD = calcSSD(A, B)

 INPUTS:
   A - an image stored as a 2D array
   B - an image stored as a 2D array. B must the the same size as A

 OUTPUTS:
   SSD - the value of the sum of squared differences

 NOTE:
   if either of the images contain NaN values these pixels should be 
   ignored when calculating the SSD.
  """
  # use nansum function to find sum of squared differences ignoring NaNs
  return np.nansum((A-B)*(A-B))

def calcMSD(A,B):
  """
 function to calculate the mean of squared differences between two images

 SYNTAX:
   MSD = calcMSD(A, B)

 INPUTS:
   A - an image stored as a 2D array
   B - an image stored as a 2D array. B must the the same size as A

 OUTPUTS:
   MSD - the value of the mean of squared differences

 NOTE:
   if either of the images contain NaN values these pixels should be 
   ignored when calculating the SSD.
  """
  # use nansum function to find sum of squared differences ignoring NaNs
  return np.nanmean((A-B)*(A-B))

def calcEntropies(A, B, num_bins = [32,32]):
  """
 function to calculate the joint and marginal entropies for two images

 SYNTAX:
   [H_AB, H_A, H_B] = calcEntropies(A, B)
   [H_AB, H_A, H_B] = calcEntropies(A, B, num_bins)

 INPUTS:
   A - an image stored as a 2D array
   B - an image stored as a 2D array. B must the the same size as A
   num_bins - a 2 element vector specifying the number of bins to in the
       joint histogram for each image
       default = 32, 32

 OUTPUTS:
   H_AB - the joint entropy between A and B
   H_A - the marginal entropy in A
   H_B - the marginal entropy in B

 NOTE:
   if either of the images contain NaN values these pixels should be
   ignored when calculating the entropies.
  """
  #first remove NaNs and flatten
  nan_inds = np.logical_or(np.isnan(A), np.isnan(B))
  A = A[np.logical_not(nan_inds)]
  B = B[np.logical_not(nan_inds)]
  
  #use histogram2d function to generate joint histogram, an convert to
  #probabilities
  joint_hist,_,_ = np.histogram2d(A, B, bins = num_bins)
  probs_AB = joint_hist / np.sum(joint_hist)
  
  #calculate marginal probability distributions for A and B
  probs_A = np.sum(probs_AB, axis=1)
  probs_B = np.sum(probs_AB, axis=0)
    
  #calculate joint entropy and marginal entropies
  #note, when taking sums must check for nan values as
  #0 * log(0) = nan
  H_AB = -np.nansum(probs_AB * np.log(probs_AB))
  H_A = -np.nansum(probs_A * np.log(probs_A))
  H_B = -np.nansum(probs_B * np.log(probs_B))
  return H_AB, H_A, H_B
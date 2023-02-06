# Import libraries
import numpy as np
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter

# Create a class Image3D
class Image3D:

    """
    A class which takes in an image and voxel dimensions and provides methods for resizing the image,
    with or without pre-filtering with a Gaussian filter.

    Attributes
    ----------
    image : numpy.ndarray
        A 3D array of image data in the form of a numpy array with shape (x, y, z).
    voxel_dimension : tuple
        A tuple of three elements, each of which is a float, representing the voxel dimensions in the x, y, and z directions (mm).





    """




    # Initialise the class which takes in an image and voxel dimensions
    def __init__(self, image, voxel_dimension = ()):
        self.image = image
        self.voxel_dimension = voxel_dimension

         # Define a local image coordinate system
        x_coords, y_coords, z_coords = np.mgrid[:self.image.shape[0], :self.image.shape[1], :self.image.shape[2]]

        # Create a 3D array of coordinates
        # The voxel coordinates in this coordinate system correspond to the indices of the voxels in the image data array.
        # The rows are numbered from top to bottom, the columns from left to right, and the slices from front to back.
        # The origin of the coordinate system is at the top left corner of the image.
        self.voxel_coords = np.vstack((x_coords.flatten(), y_coords.flatten(), z_coords.flatten())).T

        # The voxel coordinates are stored as integers, so if we want to convert them
        # to physical coordinates (i.e., coordinates in millimeters or some other unit),
        # we will need to multiply them by the voxel size.
        self.voxel_coords = self.voxel_coords * self.voxel_dimension

    # Implement a class member function that takes a resize ratio and resizes original image
    def volume_resize(self, resize_ratio):

        # Compute the new voxel dimensions by dividing the original voxel dimensions by the ratio
        new_voxel_dimension = tuple(np.array(self.voxel_dimension) / resize_ratio)

        # Compute the new shape of the resized image by multiplying the original shape by the ratio
        new_shape  = np.round(np.array(self.image.shape) * resize_ratio).astype(int)

        # Create a 3D array of coordinates in the new coordinate system;
        # We multiply the new coordinates by the new voxel dimensions
        # to convert them to physical coordinates (i.e., coordinates in millimeters or some other unit)
        new_coords = np.stack(np.mgrid[:new_shape[0], :new_shape[1], :new_shape[2]], axis = -1) * new_voxel_dimension

        # Interpolate the data array at the new coordinates
        new_data = interpn((np.arange(self.image.shape[0])*self.voxel_dimension[0], np.arange(self.image.shape[1])*self.voxel_dimension[1], np.arange(self.image.shape[2])*self.voxel_dimension[2]),
         self.image, 
         new_coords, 
         method = 'linear', 
         bounds_error = False,
         fill_value = None)

        return Image3D(new_data, new_voxel_dimension)

    # Implement volume_resize_antialias which applies a Gaussian filter to the original image before interpolation
    def volume_resize_antialias(self, resize_ratio, sigma):
        # Apply a Gaussian filter to the original image;
        # filtered image is the same size as the original image
        # When finding the sigma value for each dimension,
        # we divide the voxel dimension by the sigma value as a rule of thumb
        filtered_image = gaussian_filter(self.image, sigma = np.array(self.voxel_dimension)/sigma)

        # Compute the new voxel dimensions by dividing the original voxel dimensions by the ratio
        new_voxel_dimension = tuple(np.array(self.voxel_dimension) / resize_ratio)

        # Compute the new shape of the resized image by multiplying the original shape by the ratio
        new_shape  = np.round(np.array(self.image.shape) * resize_ratio).astype(int)

        # Create a 3D array of coordinates in the new coordinate system;
        # We multiply the new coordinates by the new voxel dimensions 
        # to convert them to physical coordinates (i.e., coordinates in millimeters or some other unit)
        new_coords = np.stack(np.mgrid[:new_shape[0], :new_shape[1], :new_shape[2]], axis = -1) * new_voxel_dimension

        # Interpolate the data array from the filtered image
        new_data = interpn((np.arange(self.image.shape[0])*self.voxel_dimension[0], np.arange(self.image.shape[1])*self.voxel_dimension[1],np.arange(self.image.shape[2])*self.voxel_dimension[2]),
        filtered_image,
        new_coords,
        method = 'linear',
        bounds_error = False,
        fill_value = None)

        return Image3D(new_data, new_voxel_dimension)



        


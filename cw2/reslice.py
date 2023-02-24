# Create re-slicing function/algorithm
import numpy as np
from scipy.interpolate import interpn

def reslice(image, theta, phi, slice_index):
    """
    Reslice is a function that takes a 3D image and returns a desired 2D slice of the image
    in an oblique view. The oblique view is defined by the angles theta and phi.
    The resulting image is obtained by applying a transformation matrix to the
    coordinates of the image and then interpolating the image values to the
    transformed coordinates.
    Parameters
    ----------
    image : 3D array
        Input image.
    theta : float
        Angle of rotation about the z-axis.
    phi : float
        Angle of rotation about the x-axis.
    slice_index : int
        Index of the slice to be extracted.
    Returns
    -------
    xx : 2D array
        x-coordinates of the slice.
    yy : 2D array
        y-coordinates of the slice.
    zz : 2D array   
        z-coordinates of the slice.
    slice_values : 2D array
        Intensity values of the slice.




    """




    # Define the transformation matrix for the oblique view
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    rotation_matrix_theta = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
    
    rotation_matrix_phi = np.array([[1, 0, 0],
                                    [0, np.cos(phi), -np.sin(phi)],
                                    [0, np.sin(phi), np.cos(phi)]])
    # Define the coordinates of the slice
    x_len, y_len, z_len = image.shape

    # Create a composed affine matrix
    rotation_matrix = np.matmul(rotation_matrix_phi, rotation_matrix_theta)


    # Define the coordinates of the oblique slice
    xx, yy = np.meshgrid(np.arange(x_len), np.arange(y_len), indexing='ij')
    zz = np.ones_like(xx) * slice_index


    # Apply the transformation matrix to the coordinates
    xyz = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
    xyz = np.dot(xyz, rotation_matrix)

    # Reshape the transformed coordinates and the slice values
    xx, yy, zz = xyz[:, 0].reshape(xx.shape), xyz[:, 1].reshape(yy.shape), xyz[:, 2].reshape(zz.shape)



    # Extract the slice values
    slice_values = image[:, :, slice_index]


    # Interpolate the slice values to the transformed coordinates
    slice_values = interpn((np.arange(x_len), np.arange(y_len)), slice_values, np.vstack((xx.flatten(), yy.flatten())).T, method='nearest', bounds_error=False, fill_value=None)


    return xx, yy, zz, slice_values
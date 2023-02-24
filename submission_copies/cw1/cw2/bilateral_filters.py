# # A bilateral filter is a non-linear, edge preserving and noise reducing smoothing filter for images.
# **It replaces the intensity of each pixel with a weighted average of intensity values from nearby pixels.**

# - This weight can be based on a Gaussian distribution.
# - The weight does not only depend on the Euclidean distance of pixels, but also on the radiometric differences e.g color intensity 
# - The Gaussian distribution is used to give more weight to pixels that are closer to the central pixel and also have similar intensity values.

# Therefore, To implement a bilateral filter from scratch that filters a 3D image without having to take individual 2D slices,
# we can use a nested loop to iterate through each pixel in the 3D image,
# and then use the intensity and spatial distance of the pixel to compute the weight of the pixel.

import numpy as np

def bilateral_filter_3d(image, diameter, sigma_intensity, sigma_spatial):
    """Apply a bilateral filter to an image.
    Parameters
    ----------
    image : 3D array
        Input image.
    diameter : int
        Diameter of each pixel neighborhood that is used during filtering.
    sigma_intensity : float
        Filter sigma in the color/intensity range space.
    sigma_spatial : float
        Filter sigma in the coordinate space.
    Returns
    -------
    filtered_image : ndarray
        Filtered output image.
    """
    # Create an empty image to store the output
    filtered_image = np.zeros_like(image)
    
    # Loop through all the pixels in the image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            for z in range(image.shape[2]):
                # Get the pixel value at (x, y, z)
                center_pixel = image[x, y, z]

                # Initialize the weight sum
                neighborhood_sum = 0
                neighborhood_weight_sum = 0
                weight_sum = 0
                
                # Loop through the local neighborhood
                for x_n in range(x-diameter, x+diameter + 1):
                    for y_n in range(y-diameter, y+diameter + 1):
                        for z_n in range(z-diameter, z+diameter + 1):
                            # Get the coordinates of the neighbour
                            neighbor_x = x_n
                            neighbor_y = y_n
                            neighbor_z = z_n

                            # Make sure the neighbor is inside the image
                            if 0 <= neighbor_x < image.shape[0] and \
                                0 <= neighbor_y < image.shape[1] and \
                                0 <= neighbor_z < image.shape[2]:

                                # Get the pixel value of the neighbor
                                neighbor_pixel = image[neighbor_x, neighbor_y, neighbor_z]

                                # Calculate the spatial distance
                                spatial_distance = ((x-x_n) ** 2 + (y-y_n) ** 2 + (z-z_n) ** 2) ** 0.5
                                
                                # Calculate the intensity difference
                                intensity_distance = (center_pixel - neighbor_pixel)

                                # Calculate the intensity weight i.e the weight of the pixel based on the intensity difference
                                intensity_weight = (1.0/(2*np.pi*(sigma_intensity**2)))*np.exp(-(intensity_distance**2) / (2*sigma_intensity**2))

                                # Calculate the spatial weight i.e the weight of the pixel based on the spatial distance from the center pixel
                                spatial_weight = (1.0/(2*np.pi*(sigma_spatial**2)))*np.exp(-(spatial_distance**2) / (2*sigma_spatial**2))

                                # Calculate the overall weight by multiplying: spatial weight * intensity weight
                                # Compute the overall weight
                                weight = spatial_weight * intensity_weight

                                # Accumulate the weights 
                                weight_sum += weight

                                # Accumulate the weighted neighborhood sum ie the weighted sum of all the pixels in the neighborhood
                                neighborhood_sum += weight * image[x_n, y_n, z_n]

                                # Accumulate the neighborhood weight sum ie the sum of all the weights
                                neighborhood_weight_sum += weight
        
                # Compute the filtered pixel value
                filtered_image[x, y, z] = neighborhood_sum / neighborhood_weight_sum
    # Return the filtered image
    return filtered_image


# Implement a 2D version of the bilateral filter
# Modify code for 2D image
def bilateral_filter_2d(image, diameter, sigma_intensity, sigma_spatial):
    """Apply a bilateral filter to an image.
    Parameters
    ----------
    image : 2D array
        Input image.
    diameter : int
        Diameter of each pixel neighborhood that is used during filtering.
    sigma_intensity : float
        Filter sigma in the color/intensity range space.
    sigma_spatial : float
        Filter sigma in the coordinate space.
    Returns
    -------
    filtered_image : ndarray
        Filtered output image.
    """
    # Create an empty image to store the output
    filtered_image = np.zeros_like(image)
    
    # Loop through all the pixels in the image
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
        #for z in range(image.shape[2]):
            # Get the pixel value at (x, y, z)
            center_pixel = image[x, y]

            # Initialize the weight sum
            neighborhood_sum = 0
            neighborhood_weight_sum = 0
            weight_sum = 0
            
            # Loop through the local neighborhood
            for x_n in range(x-diameter, x+diameter + 1):
                for y_n in range(y-diameter, y+diameter + 1):
                    #for z_n in range(-diameter, diameter + 1):
                    # Get the coordinates of the neighbour
                    neighbor_x = x_n
                    neighbor_y = y_n
                    #neighbor_z = z + z_n

                    # Make sure the neighbor is inside the image
                    if 0 <= neighbor_x < image.shape[0] and 0 <= neighbor_y < image.shape[1]:

                        # Get the pixel value of the neighbor
                        neighbor_pixel = image[neighbor_x, neighbor_y]

                        # Calculate the spatial distance
                        spatial_distance = ((x-x_n) ** 2 + (y-y_n) ** 2) ** 0.5
                        
                        # Calculate the intensity difference
                        intensity_distance = (center_pixel - neighbor_pixel)

                        # Calculate the intensity weight i.e the weight of the pixel based on the intensity difference
                        intensity_weight = (1.0/(2*np.pi*(sigma_intensity**2)))*np.exp(-(intensity_distance**2) / (2*sigma_intensity**2))

                        # Calculate the spatial weight i.e the weight of the pixel based on the spatial distance from the center pixel
                        spatial_weight = (1.0/(2*np.pi*(sigma_spatial**2)))*np.exp(-(spatial_distance**2) / (2*sigma_spatial**2))

                        # Calculate the overall weight by multiplying: spatial weight * intensity weight
                        # Compute the overall weight
                        weight = spatial_weight * intensity_weight

                        # Accumulate the weights 
                        weight_sum += weight

                        # Accumulate the weighted neighborhood sum ie the weighted sum of all the pixels in the neighborhood
                        neighborhood_sum += weight * image[x_n, y_n]

                        # Accumulate the neighborhood weight sum ie the sum of all the weights
                        neighborhood_weight_sum += weight
        
            # Compute the filtered pixel value
            filtered_image[x, y] = neighborhood_sum / neighborhood_weight_sum

    return filtered_image
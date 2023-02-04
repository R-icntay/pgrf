# Import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure


# Load the segmentation file label_train00.npy and transpose it to x,y,z
label_train00 = np.load('label_train00.npy').T


# Use marching cubes to compute vertex coordinates in mm and
# triangles for representing the segmenation boundary
# The pixel spacing is usually specified as the distance between adjacent voxels in millimeters,
# and it tells you the size of the voxels in the image.

# Use marching cubes to obtain the vertex coordinates in mm and triangles
vertices, triangles, _, _ = measure.marching_cubes(label_train00, 0, spacing = (0.5, 0.5, 2))

# Plot the triangulated surface
fig = plt.figure(figsize = (10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the triangle faces
ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=triangles, alpha = 0.7, cmap='inferno', edgecolor='k')

# Set the axis labels
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_zlabel('z (mm)')

# Set the title
ax.set_title('Triangulated surface of the segmentation boundary')

# Save the figure
plt.savefig('original triangulated_surface.png')

# Close the figure
plt.close(fig)


# Create a triangulated surface
triangulated_surface = [vertices, triangles]

# Test three saggital planes to divide this triangulated surface into two surfaces, left and right
# Load the surface dividing function
from surface_dividing import surface_dividing
# Load the plotting function
from surface_dividing import plot_left_right_surfaces

# The first plane is at the median of the x values
sagittal_plane_1 = np.median(vertices[:, 0].flatten())

# Call the surface_dividing function and divide the surface into two surfaces, left and right
left_surfaces, right_surfaces = surface_dividing(triangulated_surface, sagittal_plane = sagittal_plane_1)

# Plot the left and right surfaces at the 50th percentile of the x values
plot_left_right_surfaces(left_surfaces, right_surfaces, sagittal_plane_1, '1_median_x_values')

# The second plane is at the 25th percentile of the x values
sagittal_plane_2 = np.percentile(vertices[:, 0].flatten(), 25)

# Call the surface_dividing function and divide the surface into two surfaces, left and right
left_surfaces, right_surfaces = surface_dividing(triangulated_surface, sagittal_plane = sagittal_plane_2)

# Plot the left and right surfaces at the 25th percentile of the x values
plot_left_right_surfaces(left_surfaces, right_surfaces, sagittal_plane_2, '2_25th_percentile_x_values')


# The third plane is at the 75th percentile of the x values
sagittal_plane_3 = np.percentile(vertices[:, 0].flatten(), 75)

# Call the surface_dividing function and divide the surface into two surfaces, left and right
left_surfaces, right_surfaces = surface_dividing(triangulated_surface, sagittal_plane = sagittal_plane_3)

# Plot the left and right surfaces at the 75th percentile of the x values
plot_left_right_surfaces(left_surfaces, right_surfaces, sagittal_plane_3, '3_75th_percentile_x_values')






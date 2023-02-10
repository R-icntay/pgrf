import numpy as np
from RigidTransform import RigidTransform
import random
from matplotlib import pyplot as plt

## Load the image file
volume = np.load('image_train00.npy').T

## ----- Manually define the ranges of the rotations and translations ------##
x_translation_range = np.arange(-16, 16, 2)
y_translation_range = np.arange(-16, 16, 2)
z_translation_range = np.arange(-4, 4, 2)
# Rotation parameters
x_rotation_range = np.arange(-15, 15, 2) # In degrees
y_rotation_range = np.arange(-15, 15, 2) # In degrees
z_rotation_range = np.arange(0, 2, 1) # In degrees


# Rationale for the ranges:
# The translation range range was set to a range of 1/8th of the image size in each dimension.
# For the translation, we want to translate the volume by a fraction of the volume size so that the volume is still in the image
# and doesn't introduce large gaps (which are likely to be filled with nans) in the transformed volume, which would make it unrecognizable. 
# For the rotation, the range was set to a range of -15 to 15 degrees for each dimension which is a reasonable range
# where the volume is still recognizable after rotation.    

# Experiment 1: test the implemented warping and transformation composing
# Experiment 1.1: Randomly sample 3 sets of rigid transformations from the ranges defined above
T1 = ((random.choice(x_rotation_range), random.choice(y_rotation_range), random.choice(z_rotation_range)), (random.choice(x_translation_range), random.choice(y_translation_range), random.choice(z_translation_range)))
T2 = ((random.choice(x_rotation_range), random.choice(y_rotation_range), random.choice(z_rotation_range)), (random.choice(x_translation_range), random.choice(y_translation_range), random.choice(z_translation_range)))
T3 = ((random.choice(x_rotation_range), random.choice(y_rotation_range), random.choice(z_rotation_range)), (random.choice(x_translation_range), random.choice(y_translation_range), random.choice(z_translation_range)))


# Experiment 1.2:Instantiate 3 objects such that they represent three rigid transformations, T1, T1T2, and T1T2T3
# where T1T2 is the composition of T1 and T2, and T1T2T3 is the composition of T1, T2, and T3

# Initialize RigidTransform objects for T1, T2, and T3
rigid_transform_t1 = RigidTransform(rotations = np.deg2rad(T1[0]), translations = T1[1], image_size = volume.shape, flag_composing_ddf = False)
rigid_transform_t2 = RigidTransform(rotations = np.deg2rad(T2[0]), translations = T2[1], image_size = volume.shape, flag_composing_ddf = False)
rigid_transform_t3 = RigidTransform(rotations = np.deg2rad(T3[0]), translations = T3[1], image_size = volume.shape, flag_composing_ddf = False)

# Composition of T1 and T2 ie applying T1 followed by T2: T1T2 = T2*T1
rigid_transform_t2t1 = rigid_transform_t1.compose(rotations2 = np.deg2rad(T2[0]), translations2 = T2[1], image_size = volume.shape)

# # Composition of T1 and T2 and T3 ie applying T1 followed by T2 then T3: T1T2T3 = T3*T1T2 = T3*T2*T1
rigid_transform_t3t2t1 = rigid_transform_t2t1.compose(rotations2 = np.deg2rad(T3[0]), translations2 = T3[1], image_size = volume.shape)



# Experiment 1.3: Compare the two warped images
# Experiment 1.3.1: Warp the image using the composed transformation T1T2T3
warped_volume_t3t2t1 = rigid_transform_t3t2t1.warp(volume)

# Experiment 1.3.2: Warp the image using the individual transformations sequentially
warped_volume_1 = rigid_transform_t1.warp(volume)

warped_volume_2 = rigid_transform_t2.warp(warped_volume_1)

warped_volume_3 = rigid_transform_t3.warp(warped_volume_2)

# Experiment 1.3.3: Compare the two warped images
print('The mean difference between the two warped images is: ', np.mean(np.abs(warped_volume_t3t2t1 - warped_volume_3)))

# Save slices from the two images
# For each warped image, save 5 slices from each image
# Create function to save axial slices of the resized images
def save_axial_slices(warped_volume, slice_number_origin, filename, title):
    """
    This is a helper function to save axial slices of the resized images

    Parameters

    ----------
    warped_volume : numpy array
        A volume that has been transformed by composition of rigid transformations or sequentially applied rigid transformations
    slice_number_origin : int
        The slice number in the original image to be saved
    filename : str
        The filename of the saved image
    title : str
        The title of the saved image

    Returns

    -------
    None

    """


    fig, (ax) = plt.subplots(1, figsize=(6, 6))
    ax.imshow(warped_volume[:, :, slice_number_origin], cmap = "bone")
    ax.set_title(f"Axial slice of {title}")
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("y (voxels)")
    plt.savefig(f"{filename}_slice_{slice_number_origin}.png")
    plt.close(fig)

# Save the slices
# save 5 axial slices of composed transformed image as png images
for slice in [1, 10, 15, 20, 29]:
#[15, 20, 22, 26, 31]:
    save_axial_slices(warped_volume = warped_volume_t3t2t1, slice_number_origin = slice,
    filename = "exp3_composed_warped", title= " a composed-transformation warped image")

# save 5 axial slices of sequentially transformed image as png images
for slice in [1, 10, 15, 20, 29]:
#[15, 20, 22, 26, 31]:
    save_axial_slices(warped_volume = warped_volume_3, slice_number_origin = slice,
    filename = "exp3_sequential_warped", title= " a sequentially warped image")

# Comparing the slices from the composed transformation and the sequential transformation, one can observe that some 
# slices are similar while others are more warped. This differences can be attributed to accumulation of errors, the transformations applied, and interpolation. 
# Sequential transformations result in the accumulation of rounding errors and floating point inaccuracies, leading to differences in the final output.
# Also, sequential transformations suffer more from the interpolation artifacts as transformations increase since some 
# transformations such as rotation can lead to cropping. 
# A composed transformation minimizes the error by doing all the transformations as a single step.


# Print the above comment
print('Comparing the slices from the composed transformation and the sequential transformation, one can observe that some slices are similar while others are more warped \n')
print('This differences can be attributed to accumulation of errors, the transformations applied, and interpolation. \n')
print('Sequential transformations result in the accumulation of rounding errors and floating point inaccuracies, leading to differences in the final output. \n')
print('Also, sequential transformations suffer more from the interpolation artifacts as transformations increase since some transformations such as rotation can lead to cropping. \n')
print('A composed transformation minimizes the error by doing all the transformations as a single step. \n')


# Experiment 2: Repeat Experiment 1 after enabling flag_composing_ddf = True
# Experiment 2.2:Instantiate 3 objects such that they represent three rigid transformations, T1, T1T2, and T1T2T3
# where T1T2 is the composition of T1 and T2, and T1T2T3 is the composition of T1, T2, and T3

# Initialize RigidTransform objects for T1, T2, and T3
rigid_transform_t1 = RigidTransform(rotations = np.deg2rad(T1[0]), translations = T1[1], image_size = volume.shape, flag_composing_ddf = True)
rigid_transform_t2 = RigidTransform(rotations = np.deg2rad(T2[0]), translations = T2[1], image_size = volume.shape, flag_composing_ddf = True)
rigid_transform_t3 = RigidTransform(rotations = np.deg2rad(T3[0]), translations = T3[1], image_size = volume.shape, flag_composing_ddf = True)



# Composition of T1 and T2 ie applying T1 followed by T2: T1T2 = T2*T1
rigid_transform_t2t1 = rigid_transform_t1.compose(rotations2 = np.deg2rad(T2[0]), translations2 = T2[1], image_size = volume.shape)


# # Composition of T1 and T2 and T3 ie applying T1 followed by T2 then T3: T1T2T3 = T3*T1T2 = T3*T2*T1
rigid_transform_t3t2t1 = rigid_transform_t2t1.compose(rotations2 = np.deg2rad(T3[0]), translations2 = T3[1], image_size = volume.shape)

# Experiment 2.3: Compare the two warped images
# Experiment 2.3.1: Warp the image using the composed transformation T1T2T3
warped_volume_t3t2t1_flag = rigid_transform_t3t2t1.warp(volume)

# Experiment 1.3.2: Warp the image using the individual transformations sequentially
warped_volume_1 = rigid_transform_t1.warp(volume)

warped_volume_2 = rigid_transform_t2.warp(warped_volume_1)

warped_volume_3_flag = rigid_transform_t3.warp(warped_volume_2)

# Compute the voxel-level difference between the two warped images from the two algorithms
composed_voxel_level_difference = np.abs(warped_volume_t3t2t1_flag - warped_volume_t3t2t1)
sequential_voxel_level_difference = np.abs(warped_volume_3_flag - warped_volume_3)
# Report the mean and standard deviation of the voxel-level difference
print('The mean and standard deviation of the voxel-level difference between the two warped images from the two algorithms are: \n')
print(f'Mean voxel level difference from composed transformation algorithm : {np.mean(composed_voxel_level_difference)} and \n Standard deviation of voxel level differences from composed transformation algorithm {np.std(composed_voxel_level_difference)} \n')
print(f'Mean voxel level difference from sequential transformation algorithm: {np.mean(sequential_voxel_level_difference)} and \n Standard deviation of voxel level differences from sequential transformation algorithm {np.std(sequential_voxel_level_difference)} \n')


# Save the slices
# save 5 axial slices of composed transformed image as png images
for slice in [1, 10, 15, 20, 29]:
#[15, 20, 22, 26, 31]:
    save_axial_slices(warped_volume = warped_volume_t3t2t1_flag, slice_number_origin = slice,
    filename = "exp3-2_flag_composed_warped", title= " a composed-transformation warped image")

# save 5 axial slices of sequentially transformed image as png images
for slice in [1, 10, 15, 20, 29]:
#[15, 20, 22, 26, 31]:
    save_axial_slices(warped_volume = warped_volume_3_flag, slice_number_origin = slice,
    filename = "exp3-2_flag_sequential_warped", title= " a sequentially warped image")


## Comment on the visual comparison to those obtained in Experiment 1
# From the visual comparison, the slices from the sequential transformation algorithm are similar to those obtained in Experiment 1.
# This makes sense since the sequential transformation algorithm is the same as the one used in Experiment 1.
# The slices from the composed transformation algorithm are subtly different from those obtained in Experiment 1.
# This observation can be supported by the small differences in the mean and standard deviation of the voxel-level difference between the two warped images from the two algorithms.
# This difference can be attributed to the accumulation of errors, during the calculation of the two separate ddfs.

# Print the above comment
print('From the visual comparison, the slices from the sequential transformation algorithm are similar to those obtained in Experiment 1. \n')
print('This makes sense since the sequential transformation algorithm is the same as the one used in Experiment 1. \n')
print('The slices from the composed transformation algorithm are subtly different from those obtained in Experiment 1. \n')
print('This observation can be supported by the small differences in the mean and standard deviation of the voxel-level difference between the two warped images from the two algorithms. \n')
print('This difference can be attributed to the accumulation of errors, during the calculation of the two separate ddfs. \n')
print("But overall, the two algorithms produce similar results as expected. \n")
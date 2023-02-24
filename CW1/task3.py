import numpy as np
from scipy.interpolate import interpn
from scipy.ndimage import map_coordinates
class RigidTransform:

    # Initialize class
    def __init__(self, rotations, translations, warped_image_size):
        self.rx = rotations[0]
        self.ry = rotations[1]
        self.rz = rotations[2]
        self.tx = translations[0]
        self.ty = translations[1]
        self.tz = translations[2]
        self.warped_image_size_x = warped_image_size[0]
        self.warped_image_size_y = warped_image_size[1]
        self.warped_image_size_z = warped_image_size[2]

        # Precompute rotation matrices around the X, Y, and Z axis
        rot_x = np.array([[1, 0, 0],
                                 [0, np.cos(self.rx), -np.sin(self.rx)],
                                 [0, np.sin(self.rx), np.cos(self.rx)]])
                                 
        rot_y = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                                [0, 1, 0],
                                [-np.sin(self.ry), 0, np.cos(self.ry)]])

        rot_z = np.array([[np.cos(self.rz), -np.sin(self.rz), 0],
                                [np.sin(self.rz), np.cos(self.rz), 0],
                                [0, 0, 1]])

        # Compose rotation matrix from yaw, pitch, and roll 
        # Correct order is Z, Y, X
        self.rot_vec = np.matmul(rot_z, np.matmul(rot_y, rot_x))
        #self.rot_vec = rot_z

        # Compose translation vector
        self.trans_vec = np.array([self.tx, self.ty, self.tz])

        # Precompute the dense displacement field DDF
        #self.ddf = self.compute_ddf((self.warped_image_size_x, self.warped_image_size_y, self.warped_image_size_z))
        self.compute_ddf((self.warped_image_size_x, self.warped_image_size_y, self.warped_image_size_z))

    # Implement a class member function compute_ddf that computes displacement vector
    # this code takes a point in the warped image,
    # undo the translation by subtracting the translation vector from the point,
    # then undo the rotation by applying the inverse rotation matrix, 
    # and the result is the displacement vector from the warped image to the original image
    #  at that voxel location.
    def compute_ddf(self, warped_image_size):
        # Compute displacement vector
        self.warped_image_size_x, self.warped_image_size_y, self.warped_image_size_z = warped_image_size
            # Pre-allocate displacement vector
        self.ddf = np.zeros((self.warped_image_size_x, self.warped_image_size_y, self.warped_image_size_z, 3))

        for x in range(self.warped_image_size_x):
            for y in range(self.warped_image_size_y):
                for z in range(self.warped_image_size_z):
                    # Compute displacement vector at each voxel
                    original = np.matmul(np.linalg.inv(self.rot_vec), np.array([x, y, z]) - self.trans_vec)
                    ddd = np.array([x, y, z]) - original
                    self.ddf[x, y, z] = np.array([x, y, z]) - original
                    #self.ddf[x, y, z, :] = np.matmul(np.linalg.inv(self.rot_vec), np.array([x, y, z]) - self.trans_vec)

# np.natmul(rot_vec, np.array([x, y, z]) +  self.
    # Implement a class member function warp that returns a warped image volume in a Numpy Array
    def warp(self, volume):
        # Pre-allocate warped volume
        warped_image_coords = []
        #warped_volume = np.zeros_like(volume)
        for x in range(volume.shape[0]):
            for y in range(volume.shape[1]):
                for z in range(volume.shape[2]):
                    # Compute new voxel coordinates in the warped image
                    warped_image_coords.append(np.matmul(self.rot_vec, np.array([x, y, z])) + self.trans_vec)
        warped_image_coords = np.array(warped_image_coords)

        # Resample intensity values at the new coordinates
        #warped_volume[x, y, z] = interpn(np.array([x, y, z], ), volume, np.array((new_x, new_y, new_z),), method = 'linear', bounds_error = False, fill_value = None)
        
        warped_volume = map_coordinates(volume, [warped_image_coords.T[0], warped_image_coords.T[1], warped_image_coords.T[2]], order=1, mode='constant', cval=np.NaN, prefilter=False)
        warped_volume = warped_volume.reshape(volume.shape)
        return warped_volume

    # Implement a class member function compose which represents a combination of two transforms
    def compose(self, rotations2, translations2):

        composed_transform = RigidTransform(rotations2, translations2,  warped_image_size = (128, 128, 32))

        # Update rotation matrix: Check with Ad
        composed_transform.rot_vec = np.matmul(composed_transform.rot_vec, self.rot_vec)

        # Update translation vector: Check with Ad
        composed_transform.trans_vec = np.matmul(composed_transform.rot_vec, self.trans_vec) + np.array(translations2)
        #composed_transform.trans_vec = np.matmul(self.rot_vec, np.array(translations2)) + self.trans_vec

        # Update the displacement field
        composed_transform.compute_ddf((self.warped_image_size_x, self.warped_image_size_y, self.warped_image_size_z))

    

        # # Compute the combined rotation matrix
        # rot_x2 = np.array([[1, 0, 0],
        #                     [0, np.cos(rotations2[0]), -np.sin(rotations2[0])],
        #                     [0, np.sin(rotations2[0]), np.cos(rotations2[0])]])

        # rot_y2 = np.array([[np.cos(rotations2[1]), 0, np.sin(rotations2[1])],
        #                     [0, 1, 0],
        #                     [-np.sin(rotations2[1]), 0, np.cos(rotations2[1])]])

        # rot_z2 = np.array([[np.cos(rotations2[2]), -np.sin(rotations2[2]), 0],
        #                     [np.sin(rotations2[2]), np.cos(rotations2[2]), 0],
        #                     [0, 0, 1]])

        # # Compose rotation matrix from yaw, pitch, and roll 
        # # Correct order is Z, Y, X
        # rot_vec2 = np.matmul(rot_z2, np.matmul(rot_y2, rot_x2))
        # R_composed = np.matmul(self.rot_vec, rot_vec2)

        
        # # Compute the combined translation vector
        # #T_composed = np.matmul(self.rot_vec, np.array(translations2)) + self.trans_vec
        # T_composed = np.matmul(self.rot_vec, self.trans_vec) + np.array(translations2)

        # # Create a new RigidTransform object with the composed transformations
        # composed_transform = RigidTransform(R_composed, T_composed)

        # # Update the displacement field
        # composed_transform.compute_ddf((self.warped_image_size_x, self.warped_image_size_y, self.warped_image_size_z))

        return composed_transform
    
    def compose2(self, rotations2, translations2):

        composed_transform = RigidTransform(rotations2, translations2,  warped_image_size = (128, 128, 32))
        return composed_transform


    # Implement a class member function that to take two ddfs and return the composition of the two ddfs without using the rotation and translation matrices

    # def composing_ddfs(self, ddf1, ddf2, grid):
    #     # Compute the composition of the two ddfs






    # def compose_ddf(self, ddf1, ddf2):
    #     # Compute the composition of the two ddfs
    #     # ddf_composed = np.zeros_like(ddf1)
    #     # for x in range(ddf1.shape[0]):
    #     #     for y in range(ddf1.shape[1]):
    #     #         for z in range(ddf1.shape[2]):
    #     #             ddf_composed[x, y, z, :] = np.matmul(self.rot_vec, ddf1[x, y, z, :]) + ddf2[x, y, z, :]

    #     return ddf_composed
        

# Load image train 00
volume = np.load('CW1/image_train00.npy').T
orig_coords = np.stack(np.mgrid[:volume.shape[0], :volume.shape[1], :volume.shape[2]], axis = -1)
# Define translation and rotation parameters
rotations = np.deg2rad(np.array([5, 0, 0]))
translations = np.array([0, 0, 0])


rigid_transform = RigidTransform(rotations = rotations, translations = translations, warped_image_size = (128, 128, 32))

# Extract ddf
ddf1 = rigid_transform.ddf
# Implement a class RigidTransform which should specify a 3Drigid transformation
# which can warp 3D image volumes.
import numpy as np
from scipy.interpolate import interpn
from scipy.ndimage import map_coordinates
from scipy.interpolate import griddata

class RigidTransform:

    """
    This class implements a 3D rigid transformation which can compute the displacement vector field,
    warp an image volume and combine two rigid transformations. The class is initialized with a rotation vector and a translation vector.

    compute_ddf: computes the displacement vector field from the warped image to the original image
    warp: warps an image volume
    compose: combines two rigid transformations

    """



    # Initialize class with rotations and translations
    def __init__(self, rotations, translations, image_size, flag_composing_ddf = False):
        self.rx = rotations[0]
        self.ry = rotations[1]
        self.rz = rotations[2]
        self.tx = translations[0]
        self.ty = translations[1]
        self.tz = translations[2]
        self.flag_composing_ddf = flag_composing_ddf
       

        # Precompute a rotation matrix and a translation vector stored in the class object
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

        # Compose rotation matrix by multiplying matrices in the correct order
        # Correct order is Z, Y, X
        self.rot_vec = np.matmul(rot_z, np.matmul(rot_y, rot_x))

        # Compose translation vector
        self.trans_vec = np.array([self.tx, self.ty, self.tz])

        # Compute displacement vector field
        self.compute_ddf(image_size)
    
    # Implement a class function compute_ddf which returns a 3D displacement vector
    # from warped image to original image at each warped image voxel location
    def compute_ddf(self, warped_image_size):
        # Docstring describing the image coordinate system used in this function
        """
        The image coordinate system is a Cartesian coordinate system defined as follows:
        The origin of the coordinate system is at the top left corner of the image.
        The x-axis points to the right, the y-axis points down, and the z-axis points into the image.
        The unit used is voxel, with the distance between two adjacent voxels being 1 in all three dimensions (x, y, z).
        This coordinate system is consistent for both the warped image and the original image,
        meaning that the coordinates of the voxels in both images are defined in the same way.

        This code takes a point in the warped image, and computes the corresponding point in the original image.
        The displacement vector is the difference between the original image point and the warped image point.

        """



        # Save warped image sizes in the class object
        self.warped_image_size_x, self.warped_image_size_y, self.warped_image_size_z = warped_image_size

        # Pre-allocate displacement vector
        self.ddf = np.zeros((self.warped_image_size_x, self.warped_image_size_y, self.warped_image_size_z, 3))

        # Loop through all points in the warped image to
        # undo the translation by subtracting the translation vector from the point,
        # then undo the rotation by applying the inverse rotation matrix
        # the result is the point in the original image
        for x in range(self.warped_image_size_x):
            for y in range(self.warped_image_size_y):
                for z in range(self.warped_image_size_z):
                    # Original image point
                    original_coords = np.matmul(np.linalg.inv(self.rot_vec), np.array([x, y, z]) - self.trans_vec)
                    # Displacement vector
                    self.ddf[x, y, z, :] = np.array([x, y, z]) - original_coords
    

    # Implement a class member function warp which takes an image volume and returns the warped volume
    def warp(self, volume):
        """
        This function takes an image volume and returns the warped volume.
        To warp the image, the warped image coordinates are computed by adding the displacement vector field to the original image coordinates.
        The warped image volume is then computed by interpolating the original image volume at the warped image coordinates.

        volume: 3D numpy array of shape (image_size_x, image_size_y, image_size_z)

        returns: 3D warped volume of shape (warped_image_size_x, warped_image_size_y, warped_image_size_z)

        """
        # Pre-allocate warped volume ddf
        # warped_image_coords = []
        # for x in range(volume.shape[0]):
        #     for y in range(volume.shape[1]):
        #         for z in range(volume.shape[2]):
        #             # Compute new voxel coordinates in the warped image
        #             # by applying the rotation matrix and then the translation vector
        #             warped_image_coords.append(np.matmul(self.rot_vec, np.array([x, y, z])) + self.trans_vec)
        # warped_image_coords = np.array(warped_image_coords)
        

        # # Interpolate the image volume at the new coordinates to get the warped image volume
        # warped_volume = interpn((np.arange(volume.shape[0]), np.arange(volume.shape[1]), np.arange(volume.shape[2])),
        #                         volume, warped_image_coords, method='nearest',
        #                         bounds_error=False, fill_value=None)

        


        # Implement interpolation using map_coordinates
        #warped_volume = map_coordinates(volume, [warped_image_coords.T[0], warped_image_coords.T[1], warped_image_coords.T[2]], order=3, mode='nearest', cval=np.NaN, prefilter=False)
        
        ##---- Interpolate using scipy.interpolate.griddata ---------
        ## Prepare inputs according to the documentation:
        ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html

        # Create a grid of coordinates of the original volume of shape (n, Dim)
        X, Y, Z = np.meshgrid(np.arange(volume.shape[0]), np.arange(volume.shape[1]), np.arange(volume.shape[2]), indexing='ij')
        grid_coords = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=1)

        # Compute new voxel coordinates in the warped image: which is the same as adding
        # the dense displacement field to the original image coordinates ie grid_coords + ddf
        warped_image_coords = grid_coords + self.ddf.reshape(grid_coords.shape)
        self.warped_image_coords = warped_image_coords

        # Interpolate the image volume at the new coordinates to get the warped image volume
        # volume is flattened to a 1D array of shape (n,)
        # warped image coordinates are already in the correct shape (n, Dim)
        warped_volume = griddata(grid_coords, volume.ravel(), warped_image_coords, method='nearest')

        # Reshape the warped image volume to the correct shape
        warped_volume = warped_volume.reshape(volume.shape)




        return warped_volume

    # Implement a class member function compose which takes another RigidTransform object and returns a new RigidTransform object
    # which is the composition of the two transformations
    def compose(self, rotations2, translations2, image_size):

        """
        This function takes a second set of rotations and translations and returns a new RigidTransform object which
        is the composition of the two transformations.Applying the combined transformation is equivalent to first applying
        them sequentially, the first transformation followed by the second transformation.

        rotations2: a 3x3 rotation matrix
        translations2: a 3x1 translation vector
        image_size: a 3x1 vector containing the size of the image in each dimension

        returns: a new RigidTransform object which is the composition of the two transformations


        """

        # Create a RigidTransform object from second set of rotations and translations
        composed_transform = RigidTransform(rotations2, translations2, image_size = image_size, flag_composing_ddf = self.flag_composing_ddf)

        # Update rotation matrix in the correct order: R2*R1
        composed_transform.rot_vec = np.matmul(composed_transform.rot_vec, self.rot_vec)

        # Update translation vector: R2*t1 + t2
        composed_transform.trans_vec = np.matmul(composed_transform.rot_vec, self.trans_vec) + np.array(translations2)
        
        if self.flag_composing_ddf == True: 
            # Update DDF using a different algorithm: compose_ddfs
            composed_transform.ddf = self.composing_ddfs(self.ddf, composed_transform.ddf)
        else:
            # Update the DDF using compute_ddf:
            composed_transform.compute_ddf(image_size)

        return composed_transform
        
    # A different algorithm for computing the DDF
    def composing_ddfs(self, ddf1, ddf2):
        """
        This function takes two DDFs and returns a new DDF which is the composition of the two transformations.
        This function assumes that the DDFs are defined in the same spatial grid.

        ddf1: dense displacement field from the first transformation
        ddf2: dense displacement field from the second transformation

        """
        composed_ddf = ddf1 + ddf2

        return composed_ddf


   


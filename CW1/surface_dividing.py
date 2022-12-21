# Constructing and dividing triangulated meshes
def surface_dividing(triangulated_surface, sagittal_plane):
    """
    This is a function that takes a triangulated surface (represented by a list of vertices and a list of triangles) and
    a sagittal plane (scalar value), andreturns two lists representing two triangulated surfaces separated by the sagittal plane.
    The triangles are represented by a list of indices of the vertices that make up each triangle.
    In the case the plane intersects a triangle, new triangles are formed.

    Example of usage:
    # Define function inputs
    vertices = [(0, 0, 0), (1, 1, 0), (2, 0, 0), (0, 1, 0), (2, 1, 0)]
    triangles = [(0, 3, 4), (0, 1, 2)]
    sagittal_plane = 1.5
    triangulated_surface = [vertices, triangles]

    # Call function
    triangulated_surfaces_left, triangulated_surfaces_right = surface_dividing(triangulated_surface, sagittal_plane) 
    """
    vertices = triangulated_surface[0]
    triangles = triangulated_surface[1]
    
    # Create empty lists of vertices and triangles for the left and right surfaces that are split by the sagittal plane
    left_surface_vertices = []
    right_surface_vertices = []
    left_surface_triangles = []
    right_surface_triangles = []

    # Split the vertices into left or right based on their relative position wrt sagittal plane
    # sagittal plane is oriented along the x-axis hence x component of vertices is used for comparison
    for vertex in vertices:
        if vertex[0] < sagittal_plane:
            left_surface_vertices.append(tuple(vertex))
        else:
            right_surface_vertices.append(tuple(vertex))
    
    # Determine the position of triangles by checking which side of the sagittal plane each vertex lies
    for triangle in triangles:
        # Extract the respective vertices of the triangle
        triangle_vertices = [vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]]

        # Add counters for determining the position of vertices
        left_of_sagittal = 0
        right_of_sagittal = 0
        for vertex in triangle_vertices:
            if vertex[0] < sagittal_plane:
                left_of_sagittal += 1
            else:
                right_of_sagittal += 1
            
        # If the triangle is fully on one side of the sagittal plane, add it to the appropriate list of triangles
        if left_of_sagittal == 3:
            # Index of the vertices that make up the triangle
            left_indices = []
            for i in range(len(triangle_vertices)):
                left_indices.append(left_surface_vertices.index(tuple(triangle_vertices[i])))
            left_surface_triangles.append(tuple(left_indices))

        elif right_of_sagittal == 3:
            right_indices = []
            for i in range(len(triangle_vertices)):
                right_indices.append(right_surface_vertices.index(tuple(triangle_vertices[i])))
            right_surface_triangles.append(tuple(right_indices))

        else:
        # When a plane is intersected by a triangle, the triangle is split into two triangles
        # Find the indices of the vertices in the left and right lists
            left_indices = []
            right_indices = []
        # Empty array to store intersection points
            intersection_points = []

        # Determine the point of intersection between a triangles edges and the sagittal plane
            for i in range(len(triangle_vertices)):
                v1 = tuple(triangle_vertices[i])
                v2 = tuple(triangle_vertices[(i+1) % 3])
            # Check whether the saggital plane intersects with a triangle's edge
                if v1[0] < sagittal_plane < v2[0] or v2[0] < sagittal_plane < v1[0]:
                    edge_ip = (sagittal_plane - v1[0]) / (v2[0] - v1[0])
                    intersection_point = (sagittal_plane, v1[1] + edge_ip * (v2[1] - v1[1]), v1[2] + edge_ip * (v2[2] - v1[2]))
                    if intersection_point != v1 and intersection_point != v2:
                        intersection_points.append(intersection_point)

            # Add the intersection points to the left and right vertices lists and ensure they do not exist yet
            for i in range(len(intersection_points)):
                if intersection_points[i] not in left_surface_vertices:
                    left_surface_vertices.append(tuple(intersection_points[i]))
                if intersection_points[i] not in right_surface_vertices:
                    right_surface_vertices.append(tuple(intersection_points[i]))
                
            

        # For each vertex of the triangle , determine its index either in the left or in the right lists of vertices
            for vertex in triangle_vertices:
                if vertex[0] < sagittal_plane:
                    left_indices.append(left_surface_vertices.index(tuple(vertex)))
                else:
                    right_indices.append(right_surface_vertices.index(tuple(vertex)))

        # Create new triangles by combining initial vertices with newly formed intersection points
            if right_of_sagittal == 1:
                right_surface_triangles.append((right_indices[0], right_surface_vertices.index((intersection_points[0])), right_surface_vertices.index((intersection_points[1]))))
            else:
                right_surface_triangles.append((right_indices[0], right_surface_vertices.index((intersection_points[0])), right_surface_vertices.index((intersection_points[1]))))
                right_surface_triangles.append((right_indices[1], right_surface_vertices.index((intersection_points[0])), right_surface_vertices.index((intersection_points[1]))))
                right_surface_triangles.append((right_indices[0], right_indices[1] , right_surface_vertices.index((intersection_points[0]))))
                right_surface_triangles.append((right_indices[0], right_indices[1] , right_surface_vertices.index((intersection_points[1]))))
                
            
        # Create new triangles by combining initial vertices with newly formed intersection points
            if left_of_sagittal == 1:
                left_surface_triangles.append((left_indices[0], left_surface_vertices.index((intersection_points[0])), left_surface_vertices.index((intersection_points[1]))))
            else:
                left_surface_triangles.append((left_indices[0], left_surface_vertices.index((intersection_points[0])), left_surface_vertices.index((intersection_points[1]))))
                left_surface_triangles.append((left_indices[1], left_surface_vertices.index((intersection_points[0])), left_surface_vertices.index((intersection_points[1]))))
                left_surface_triangles.append((left_indices[0], left_indices[1] , left_surface_vertices.index((intersection_points[0]))))
                left_surface_triangles.append((left_indices[0], left_indices[1] , left_surface_vertices.index((intersection_points[1]))))
                

    # Return two lists of left triangulated surfaces and right triangulated surfaces wrt to sagittal plane
    return [left_surface_vertices, left_surface_triangles], [right_surface_vertices, right_surface_triangles]



# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy
from skimage import measure

# Load the data
label_train00 = np.load('/workspaces/pgrf/CW1/label_train00.npy').T


vertices, triangles, _, _ = measure.marching_cubes(label_train00, 0, spacing = (0.5, 0.5, 2))

triangulated_surface = [vertices[:5], triangles[:3]]
# vertices = np.asarray([(0, 0, 0), (1, 1, 0), (2, 0, 0), (0, 1, 0), (2, 1, 0)], dtype = np.float64)
# triangles = np.asarray([(0, 3, 4), (0, 1, 2)], dtype = np.int32)
# #sagittal_plane = 1.5
# triangulated_surface = [vertices, triangles]
left_surfaces, right_surfaces = surface_dividing(triangulated_surface, sagittal_plane = 7.5)
# # Eric
# Constructing and dividing triangulated meshes
def surface_dividing(vertices, triangles, sagittal_plane):
    # vertices = triangulated_surface[0]
    # triangles = triangulated_surface[1]
    # The vertices and triangles for the left and right surfaces that are split by the sagittal plane.
    left_vertices = []
    right_vertices = []
    left_triangles = []
    right_triangles = []

    # Split the vertices into left or right based on the sagittal plane (sagittal plane is oriented along the x-axis)
    for vertex in vertices:
        if vertex[0] < sagittal_plane:
            left_vertices.append(vertex)
        else:
            right_vertices.append(vertex)
    
    # Split the triangles into left or right by checking which side of the sagittal plane each vertex lies
    for triangle in triangles:
        triangle_vertices = [vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]]
        left_count = 0
        right_count = 0
        for vertex in triangle_vertices:
            if vertex[0] < sagittal_plane:
                left_count += 1
            else:
                right_count += 1
            
        # If the triangle is fully on one side of the sagittal plane, add it to the appropriate list of triangles
        if left_count == 3:
            left_indices = []
            for i in range(3):
                left_indices.append(left_vertices.index(triangle_vertices[i]))
            left_triangles.append(tuple(left_indices))
        elif right_count == 3:
            right_indices = []
            for i in range(3):
                right_indices.append(right_vertices.index(triangle_vertices[i]))
            right_triangles.append(tuple(right_indices))
        else:
        # When a plane is intersected by a triangle, the triangle is split into two triangles
        # Find the indices of the vertices in the left and right lists
            left_indices = []
            right_indices = []

            intersection_points = []
            for i in range(len(triangle_vertices)):
                v1 = triangle_vertices[i]
                v2 = triangle_vertices[(i+1) % 3]
                if v1[0] < sagittal_plane < v2[0] or v2[0] < sagittal_plane < v1[0]:
                    t = (sagittal_plane - v1[0]) / (v2[0] - v1[0])
                    intersection_point = (sagittal_plane, v1[1] + t * (v2[1] - v1[1]), v1[2] + t * (v2[2] - v1[2]))
                    if intersection_point != v1 and intersection_point != v2:
                        intersection_points.append(intersection_point)

            # Add the intersection points to the left and right vertices lists
            for i in range(len(intersection_points)):
                if intersection_points[i] not in left_vertices:
                    left_vertices.append(tuple(intersection_points[i]))
                if intersection_points[i] not in right_vertices:
                    right_vertices.append(tuple(intersection_points[i]))
                
            # right_vertices.append(tuple(intersection_points[0]))
            # left_vertices.append(tuple(intersection_points[1]))
            # right_vertices.append(tuple(intersection_points[1]))

            




            for vertex in triangle_vertices:
                if vertex[0] < sagittal_plane:
                    left_indices.append(left_vertices.index(vertex))
                else:
                    right_indices.append(right_vertices.index(vertex))

            # Create the two new triangles by combining the vertices on opposite sides of the sagittal plane
            if right_count == 1:
                right_triangles.append((right_indices[0], right_vertices.index((intersection_points[0])), right_vertices.index((intersection_points[1]))))
            else:
                right_triangles.append((right_indices[0], right_vertices.index((intersection_points[0])), right_vertices.index((intersection_points[1]))))
                right_triangles.append((right_indices[1], right_vertices.index((intersection_points[0])), right_vertices.index((intersection_points[1]))))
                right_triangles.append((right_indices[0], right_indices[1] , right_vertices.index((intersection_points[0]))))
                right_triangles.append((right_indices[0], right_indices[1] , right_vertices.index((intersection_points[1]))))
                
            
            # Create the two new triangles by combining the vertices on opposite sides of the sagittal plane
            if left_count == 1:
                left_triangles.append((left_indices[0], left_vertices.index((intersection_points[0])), left_vertices.index((intersection_points[1]))))
            else:
                left_triangles.append((left_indices[0], left_vertices.index((intersection_points[0])), left_vertices.index((intersection_points[1]))))
                left_triangles.append((left_indices[1], left_vertices.index((intersection_points[0])), left_vertices.index((intersection_points[1]))))
                left_triangles.append((left_indices[0], left_indices[1] , left_vertices.index((intersection_points[0]))))
                left_triangles.append((left_indices[0], left_indices[1] , left_vertices.index((intersection_points[1]))))
                

    return [left_vertices, left_triangles], [right_vertices, right_triangles]
            
            
# def surface_dividing(vertices, triangles, sagittal_plane):
#     # Create empty lists for the vertices and triangles of the two resulting surfaces
#     surface1_vertices = []
#     surface2_vertices = []
#     surface1_triangles = []
#     surface2_triangles = []

#     # Iterate through the triangles
#     for triangle in triangles:
#         # Initialize variables to track which vertices are on which side of the sagittal plane
#         surface1_count = 0
#         surface2_count = 0

#         # Iterate through the vertices of the current triangle
#         for vertex in triangle:
#             # Check if the vertex is on the left or right side of the sagittal plane
#             if vertices[vertex][0] < sagittal_plane:
#                 surface1_count += 1
#             else:
#                 surface2_count += 1

#         # If all vertices are on the same side of the sagittal plane, add the triangle to the corresponding list
#         if surface1_count == 3:
#             surface1_triangles.append(triangle)
#         elif surface2_count == 3:
#             surface2_triangles.append(triangle)
#         # If the vertices are on both sides of the sagittal plane, we need to split the triangle
#         else:
#             # Create a list of the indices of the vertices on each side of the sagittal plane
#             surface1_indices = []
#             surface2_indices = []
#             for i, vertex in enumerate(triangle):
#                 if vertices[vertex][0] < sagittal_plane:
#                     surface1_indices.append(i)
#                 else:
#                     surface2_indices.append(i)

#             # Split the triangle into two new triangles
#             triangle1 = [triangle[surface1_indices[0]], triangle[surface2_indices[0]], triangle[surface2_indices[1]]]
#             triangle2 = [triangle[surface1_indices[1]], triangle[surface2_indices[0]], triangle[surface2_indices[1]]]

#             # Add the new triangles to the corresponding lists
#             surface1_triangles.append(triangle1)
#             surface2_triangles.append(triangle2)

#     # Create the lists of vertices for the two resulting surfaces by adding any new vertices created when splitting triangles
#     for triangle in surface1_triangles:
#         for vertex in triangle:
#             if vertex not in surface1_vertices:
#                 surface1_vertices.append(vertex)
#     for triangle in surface2_triangles:
#         for vertex in triangle:
#             if vertex not in surface2_vertices:
#                 surface2_vertices.append(vertex)

#     return surface1_vertices, surface1_triangles, surface2_vertices, surface2_triangles
    
vertices = [(0, 0, 0), (1, 1, 0), (2, 0, 0), (0, 1, 0), (2, 1, 0)]#, (0, 0, 1), (1, 0, 1), (2, 0, 1)]
triangles = [(0, 3, 4), (0, 1, 2)]#, (4, 1, 0), (1, 4, 5), (5, 2, 1), (3, 6, 7), (7, 4, 3), (4, 7, 8), (8, 5, 4), (2, 5, 8), (8, 7, 2), (0, 1, 2), (2, 3, 0)]
sagittal_plane = 1.5

surface_dividing(vertices, triangles, sagittal_plane) 

    
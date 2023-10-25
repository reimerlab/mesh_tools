'''


These functions will help with skeletonization


'''
import copy
from copy import deepcopy
import h5py
import ipyvolume as ipv
import itertools
import matplotlib.pyplot as plt
import meshparty
import networkx as nx
from datasci_tools import numpy_dep as np
import os
import pathlib
from pathlib import Path
from pykdtree.kdtree import KDTree
import random
import scipy
from scipy.spatial.distance import pdist,squareform
from shutil import rmtree
import time
import trimesh



max_stitch_distance_default = 10000

#--- for mesh subtraction ----
mesh_subtraction_distance_threshold_default = 8_000
mesh_subtraction_buffer_default = 1_000

def compare_endpoints(endpioints_1,endpoints_2,**kwargs):
    """
    comparing the endpoints of a graph: 
    
    Ex: 
    from datasci_tools import networkx_utils as xu
    xu = reload(xu)mess
    end_1 = np.array([[2,3,4],[1,4,5]])
    end_2 = np.array([[1,4,5],[2,3,4]])

    xu.compare_endpoints(end_1,end_2)
    """
    #this older way mixed the elements of the coordinates together to just sort the columns
    #return np.array_equal(np.sort(endpoints_1,axis=0),np.sort(endpoints_2,axis=0))
    
    #this is correct way to do it (but has to be exact to return true)
    #return np.array_equal(nu.sort_multidim_array_by_rows(endpoints_1),nu.sort_multidim_array_by_rows(endpoints_2))

    return nu.compare_threshold(nu.sort_multidim_array_by_rows(endpoints_1),
                                nu.sort_multidim_array_by_rows(endpoints_2),
                                **kwargs)

def save_skeleton_cgal(surface_with_poisson_skeleton,largest_mesh_path):
    """
    surface_with_poisson_skeleton (np.array) : nx2 matrix with the nodes
    """
    first_node = surface_with_poisson_skeleton[0][0]
    end_nodes =  surface_with_poisson_skeleton[:,1]
    
    skeleton_to_write = str(len(end_nodes) + 1) + " " + str(first_node[0]) + " " +  str(first_node[1]) + " " +  str(first_node[2])
    
    for node in end_nodes:
        skeleton_to_write +=  " " + str(node[0]) + " " +  str(node[1]) + " " +  str(node[2])
    
    output_file = largest_mesh_path
    if output_file[-5:] != ".cgal":
        output_file += ".cgal"
        
    f = open(output_file,"w")
    f.write(skeleton_to_write)
    f.close()
    return 

#read in the skeleton files into an array
def read_skeleton_edges_coordinates(file_path):
    if type(file_path) == str or type(file_path) == type(Path()):
        file_path = [file_path]
    elif type(file_path) == list:
        pass
    else:
        raise Exception("file_path not a string or list")
    new_file_path = []
    for f in file_path:
        if type(f) == type(Path()):
            new_file_path.append(str(f.absolute()))
        else:
            new_file_path.append(str(f))
    file_path = new_file_path
    
    total_skeletons = []
    for fil in file_path:
        try:
            with open(fil) as f:
                bones = np.array([])
                for line in f.readlines():
                    #print(line)
                    line = (np.array(line.split()[1:], float).reshape(-1, 3))
                    #print(line[:-1])
                    #print(line[1:])

                    #print(bones.size)
                    if bones.size <= 0:
                        bones = np.stack((line[:-1],line[1:]),axis=1)
                    else:
                        bones = np.vstack((bones,(np.stack((line[:-1],line[1:]),axis=1))))
                    #print(bones)
                total_skeletons.append(np.array(bones).astype(float))
        except:
            print(f"file {fil} not found so skipping")
    
    return stack_skeletons(total_skeletons)
#     if len(total_skeletons) > 1:
#         returned_skeleton = np.vstack(total_skeletons)
#         return returned_skeleton
#     if len(total_skeletons) == 0:
#         print("There was no skeletons found for these files")
#     return np.array(total_skeletons).reshape(-1,2,3)

#read in the skeleton files into an array
def read_skeleton_verts_edges(file_path):
    with open(file_path) as f:
        bones = np.array([])
        for line in f.readlines():
            #print(line)
            line = (np.array(line.split()[1:], float).reshape(-1, 3))
            #print(line[:-1])
            #print(line[1:])

            #print(bones.size)
            if bones.size <= 0:
                bones = np.stack((line[:-1],line[1:]),axis=1)
            else:
                bones = np.vstack((bones,(np.stack((line[:-1],line[1:]),axis=1))))
            #print(bones)
    
    bones_array = np.array(bones).astype(float)
    
    #unpacks so just list of vertices
    vertices_unpacked  = bones_array.reshape(-1,3)

    #reduce the number of repeat vertices and convert to list
    unique_rows = np.unique(vertices_unpacked, axis=0)
    unique_rows_list = unique_rows.tolist()

    #assigns the number to the vertex (in the original vertex list) that corresponds to the index in the unique list
    vertices_unpacked_coefficients = np.array([unique_rows_list.index(a) for a in vertices_unpacked.tolist()])

    #reshapes the vertex list to become an edge list (just with the labels so can put into netowrkx graph)
    edges_with_coefficients =  np.array(vertices_unpacked_coefficients).reshape(-1,2)

    return unique_rows, edges_with_coefficients


def skeleton_unique_coordinates(skeleton):
    return np.unique(skeleton.reshape(-1,3),axis=0)

def convert_nodes_edges_to_skeleton(nodes,edges):
    return nodes[edges]

def convert_skeleton_to_nodes(skeleton):
    #unpacks so just list of vertices
    if len(skeleton) == 0:
        return []
    vertices_unpacked  = skeleton.reshape(-1,3)

    #reduce the number of repeat vertices and convert to list
    unique_rows = np.unique(vertices_unpacked, axis=0)
    return unique_rows

convert_skeleton_to_coordinates = convert_skeleton_to_nodes
    
    
def convert_skeleton_to_nodes_edges(
    skeleton,
    verbose = False,):
    
    if verbose:
        st = time.time()
    all_skeleton_vertices = skeleton.reshape(-1,3)
    unique_rows,indices = np.unique(all_skeleton_vertices,return_inverse=True,axis=0)

    #need to merge unique indices so if within a certain range of each other then merge them together
    reshaped_indices = indices.reshape(-1,2)
    
    if verbose:
        print(f"# of unique noes = {len(unique_rows)}")
        print(f"# of edges = {len(reshaped_indices)}")
        print(f"Total time = {time.time() - st}")
    
    return unique_rows,reshaped_indices

convert_skeleton_to_nodes_edges_optimized = convert_skeleton_to_nodes_edges

def convert_skeleton_to_nodes_edges_old(
    bones_array,
    verbose = False,
    ):
    st = time.time()
    #unpacks so just list of vertices
    bones_array = np.array(bones_array)
    
    vertices_unpacked  = bones_array.reshape(-1,3)

    #reduce the number of repeat vertices and convert to list
    unique_rows = np.unique(vertices_unpacked, axis=0)
    unique_rows_list = unique_rows.tolist()

    #assigns the number to the vertex (in the original vertex list) that corresponds to the index in the unique list
    vertices_unpacked_coefficients = np.array([unique_rows_list.index(a) for a in vertices_unpacked.tolist()])

    #reshapes the vertex list to become an edge list (just with the labels so can put into netowrkx graph)
    edges_with_coefficients =  np.array(vertices_unpacked_coefficients).reshape(-1,2)

    if verbose:
        print(f"# of unique noes = {len(unique_rows)}")
        print(f"# of edges = {len(edges_with_coefficients)}")
        print(f"Total time = {time.time() - st}")
        
        
    return unique_rows, edges_with_coefficients

def convert_skeleton_to_nodes_edges_optimized_old(array):
    """
    Purpose: will return the nodes and edges but without making the nodes unique
    (so will have some repeats)
    
    """
    unique_rows  = array.reshape(-1,3)
    curr_edges = np.arange(len(unique_rows)).reshape(-1,2)
    return unique_rows, curr_edges

def calculate_skeleton_segment_distances(my_skeleton,cumsum=True):
    segment_distances = np.sqrt(np.sum((my_skeleton[:,0] - my_skeleton[:,1])**2,axis=1)).astype("float")
    if cumsum:
        return np.cumsum(segment_distances)
    else:
        return segment_distances
    
def calculate_coordinate_distances_cumsum(skeleton):
    return np.concatenate([[0],sk.calculate_skeleton_segment_distances(skeleton,cumsum=True)])

def calculate_skeleton_distance(my_skeleton):
    if len(my_skeleton) == 0:
        return 0
    total_distance = np.sum(np.sqrt(np.sum((my_skeleton[:,0] - my_skeleton[:,1])**2,axis=1)))
    return float(total_distance)

calculate_skeletal_distance = calculate_skeleton_distance
calculate_skeletal_length = calculate_skeleton_distance

def plot_ipv_mesh(mesh,color=[1.,0.,0.,0.2],
                 flip_y=True):
    
    if mesh is None or len(mesh.vertices) == 0:
        return
    
    if flip_y:
        #print("inside elephant flipping copy")
        elephant_mesh_sub = mesh.copy()
        elephant_mesh_sub.vertices[...,1] = -elephant_mesh_sub.vertices[...,1]
    else:
        elephant_mesh_sub = mesh
    
    
    #check if the color is a dictionary
    if type(color) == dict:
        #get the type of values stored in there
        labels = list(color.items())
        
        #if the labels were stored as just numbers/decimals
        if type(labels[0]) == int or type(labels[0]) == float:
            #get all of the possible labels
            unique_labels = np.unique(labels)
            #get random colors for all of the labels
            colors_list =  mu.generate_color_list(n_colors)
            for lab,curr_color in zip(unique_labels,colors_list):
                #find the faces that correspond to that label
                faces_to_keep = [k for k,v in color.items() if v == lab]
                #draw the mesh with that color
                curr_mesh = elephant_mesh_sub.submesh([faces_to_keep],append=True)
                
                mesh4 = ipv.plot_trisurf(elephant_mesh_sub.vertices[:,0],
                               elephant_mesh_sub.vertices[:,1],
                               elephant_mesh_sub.vertices[:,2],
                               triangles=elephant_mesh_sub.faces)
                mesh4.color = curr_color
                mesh4.material.transparent = True
    else:          
        mesh4 = ipv.plot_trisurf(elephant_mesh_sub.vertices[:,0],
                                   elephant_mesh_sub.vertices[:,1],
                                   elephant_mesh_sub.vertices[:,2],
                                   triangles=elephant_mesh_sub.faces)
        mesh4.color = color
        mesh4.material.transparent = True
        

def plot_ipv_skeleton(edge_coordinates,color=[0,0.,1,1],
                     flip_y=True):
    
    if len(edge_coordinates) == 0:
        #print("Edge coordinates in plot_ipv_skeleton were of 0 length so returning")
        return []
    
    if flip_y:
        edge_coordinates = edge_coordinates.copy()
        edge_coordinates[...,1] = -edge_coordinates[...,1] 
    
    #print(f"edge_coordinates inside after change = {edge_coordinates}")
    unique_skeleton_verts_final,edges_final = convert_skeleton_to_nodes_edges_optimized(edge_coordinates)
    mesh2 = ipv.plot_trisurf(unique_skeleton_verts_final[:,0], 
                            unique_skeleton_verts_final[:,1], 
                            unique_skeleton_verts_final[:,2], 
                            lines=edges_final)
    #print(f"color in ipv_skeleton = {color}")
    mesh2.color = color 
    mesh2.material.transparent = True
    
    #print(f"Color in skeleton ipv plot = {color}")

    if flip_y:
        unique_skeleton_verts_final[...,1] = -unique_skeleton_verts_final[...,1]
    
    return unique_skeleton_verts_final

def plot_ipv_scatter(scatter_points,scatter_color=[1.,0.,0.,0.5],
                    scatter_size=0.4,
                    flip_y=True):
    
    if len(scatter_points) == 0:
        return 
    
    scatter_points = (np.array(scatter_points).reshape(-1,3)).astype("float")
    if flip_y:
        scatter_points = scatter_points.copy()
        scatter_points[...,1] = -scatter_points[...,1]
#     print(f"scatter_points[:,0] = {scatter_points[:,0]}")
#     print(f"scatter_points[:,1] = {scatter_points[:,1]}")
#     print(f"scatter_points[:,2] = {scatter_points[:,2]}")
#     print(f"scatter_size = {scatter_size}")
#     print(f"scatter_color = {scatter_color}")
    mesh_5 = ipv.scatter(
            scatter_points[:,0], 
            scatter_points[:,1],
            scatter_points[:,2], 
            size=scatter_size, 
            color=scatter_color,
            marker="sphere")
    mesh_5.material.transparent = True

def graph_skeleton_and_mesh(main_mesh_verts=[],
                            main_mesh_faces=[],
                            unique_skeleton_verts_final=[],
                            edges_final=[],
                            edge_coordinates=[],
                            other_meshes=[],
                            other_meshes_colors =  [],
                            mesh_alpha=0.2,
                            other_meshes_face_components = [],
                            other_skeletons = [],
                            other_skeletons_colors =  [],
                            return_other_colors = False,
                            main_mesh_color = [0.,1.,0.,0.2],
                            main_skeleton_color = [0,0.,1,1],
                            main_mesh_face_coloring = [],
                            other_scatter=[],
                            scatter_size = 0.3,
                            other_scatter_colors=[],
                            main_scatter_color="red",#[1.,0.,0.,0.5],
                            scatter_with_widgets = False,
                            buffer=1000,
                           axis_box_off=True,
                           html_path="",
                           show_at_end=True,
                           append_figure=False,
                            set_zoom = True,
                           flip_y=True,
                           adaptive_min_max_limits = True):
    """
    Graph the final result of skeleton and mesh
    
    Pseudocode on how to do face colorings :
    could get a dictionary mapping faces to colors or groups
    - if mapped to groups then do random colors (and generate them)
    - if mapped to colors then just do submeshes and send the colors
    """
    #print(f"other_scatter = {other_scatter}")
    #print(f"mesh_alpha = {mesh_alpha}")
    
    if not append_figure:
        ipv.figure(figsize=(15,15))
    
    main_mesh_vertices = []
    
    
    #print("Working on main skeleton")
    if (len(unique_skeleton_verts_final) > 0 and len(edges_final) > 0) or (len(edge_coordinates)>0):
        if flip_y:
            edge_coordinates = edge_coordinates.copy()
            edge_coordinates[...,1] = -edge_coordinates[...,1]
        if (len(edge_coordinates)>0):
            unique_skeleton_verts_final,edges_final = convert_skeleton_to_nodes_edges_optimized(edge_coordinates)
        mesh2 = ipv.plot_trisurf(unique_skeleton_verts_final[:,0], 
                                unique_skeleton_verts_final[:,1], 
                                unique_skeleton_verts_final[:,2], 
                                lines=edges_final, color='blue')

        mesh2.color = main_skeleton_color 
        mesh2.material.transparent = True
        
        if flip_y:
            unique_skeleton_verts_final[...,1] = -unique_skeleton_verts_final[...,1]
            
        main_mesh_vertices.append(unique_skeleton_verts_final)
    
    #print("Working on main mesh")
    if len(main_mesh_verts) > 0 and len(main_mesh_faces) > 0:
        if len(main_mesh_face_coloring) > 0:
            #will go through and color the faces of the main mesh if any sent
            for face_array,face_color in main_mesh_face_coloring:
                curr_mesh = main_mesh.submesh([face_array],append=True)
                plot_ipv_mesh(curr_mesh,face_color,flip_y=flip_y)
        else:
            if flip_y:
                main_mesh_verts = main_mesh_verts.copy()
                main_mesh_verts[...,1] = -main_mesh_verts[...,1]
            
            main_mesh = trimesh.Trimesh(vertices=main_mesh_verts,faces=main_mesh_faces)

            mesh3 = ipv.plot_trisurf(main_mesh.vertices[:,0],
                                   main_mesh.vertices[:,1],
                                   main_mesh.vertices[:,2],
                                   triangles=main_mesh.faces)
            
            mesh3.color = main_mesh_color
            mesh3.material.transparent = True
            
            #flipping them back
            if flip_y:
                main_mesh_verts[...,1] = -main_mesh_verts[...,1]
            
        main_mesh_vertices.append(main_mesh_verts)
        
    
    # cast everything to list type
    if type(other_meshes) != list and type(other_meshes) != np.ndarray:
        other_meshes = [other_meshes]
    if type(other_meshes_colors) != list and type(other_meshes_colors) != np.ndarray:
        other_meshes_colors = [other_meshes_colors]
    if type(other_skeletons) != list and type(other_skeletons) != np.ndarray:
        other_skeletons = [other_skeletons]
    if type(other_skeletons_colors) != list and type(other_skeletons_colors) != np.ndarray:
        other_skeletons_colors = [other_skeletons_colors]
        
#     if type(other_scatter) != list and type(other_scatter) != np.ndarray:
#         other_scatter = [other_scatter]
#     if type(other_scatter_colors) != list and type(other_scatter_colors) != np.ndarray:
#         other_scatter_colors = [other_scatter_colors]

    if not nu.is_array_like(other_scatter):
        other_scatter = [other_scatter]
    if not nu.is_array_like(other_scatter_colors):
        other_scatter_colors = [other_scatter_colors]
    
        
    
    
    if len(other_meshes) > 0:
        if len(other_meshes_face_components ) > 0:
            other_meshes_colors = other_meshes_face_components
        elif len(other_meshes_colors) == 0:
            other_meshes_colors = [main_mesh_color]*len(other_meshes)
        else:
            #get the locations of all of the dictionaries
            if "random" in other_meshes_colors:
                other_meshes_colors = mu.generate_color_list(
                            user_colors=[], #if user sends a prescribed list
                            n_colors=len(other_meshes),
                            #colors_to_omit=["green","blue"], #because that is the one used for the main mesh
                            alpha_level=mesh_alpha)
            else:
                other_meshes_colors = mu.generate_color_list(
                            user_colors=other_meshes_colors, #if user sends a prescribed list
                            n_colors=len(other_meshes),
                            #colors_to_omit=["green","blue"], #because that is the one used for the main mesh
                            alpha_level=mesh_alpha)
            
    
       
    #print("Working on other meshes")
    for curr_mesh,curr_color in zip(other_meshes,other_meshes_colors):
        #print(f"flip_y = {flip_y}")
        plot_ipv_mesh(curr_mesh,color=curr_color,flip_y=flip_y)
        
        if curr_mesh is not None:
            main_mesh_vertices.append(curr_mesh.vertices)
    
    
    #print("Working on other skeletons")
    if len(other_skeletons) > 0:
        if len(other_skeletons_colors) == 0:
            other_skeletons_colors = [main_skeleton_color]*len(other_skeletons)
        elif "random" in other_skeletons_colors:
            other_skeletons_colors = mu.generate_color_list(
                        user_colors=[], #if user sends a prescribed list
                        n_colors=len(other_skeletons),
                        #colors_to_omit=["green","blue"], #because that is the one used for the main mesh
                        alpha_level=1)
        else:
            
            other_skeletons_colors = mu.generate_color_list(
                        user_colors=other_skeletons_colors, #if user sends a prescribed list
                        n_colors=len(other_skeletons),
                        #colors_to_omit=["green","blue"], #because that is the one used for the main mesh
                        alpha_level=1)
            #print(f"user colors picked for other_skeletons_colors = {other_skeletons_colors}")
    
        
    for curr_sk,curr_color in zip(other_skeletons,other_skeletons_colors):
        sk_vertices = plot_ipv_skeleton(curr_sk,color=curr_color,flip_y=flip_y)
        
        main_mesh_vertices.append(sk_vertices)
        
        
    #printing the scatter plots
    #print("Working on other scatter plots")
    if len(other_scatter) > 0 and len(other_scatter_colors) == 0:
        other_scatter_colors = [main_scatter_color]*len(other_scatter)
        
    while len(other_scatter_colors) < len(other_scatter):
        other_scatter_colors += other_scatter_colors
        
        
    if not nu.is_array_like(scatter_size):
        scatter_size = [scatter_size]*len(other_scatter)
        
    
    for curr_scatter,curr_color,curr_size in zip(other_scatter,other_scatter_colors,scatter_size):
#         print(f"curr_scatter = {curr_scatter}")
#         print(f"curr_color = {curr_color}")
#         print(f"curr_size= {curr_size}")
        if not scatter_with_widgets:
            plot_ipv_scatter(curr_scatter,scatter_color=curr_color,
                        scatter_size=curr_size,flip_y=flip_y)
        main_mesh_vertices.append(curr_scatter)
            
    if scatter_with_widgets:
        ipvu.plot_multi_scatters(
            other_scatter,other_scatter_colors,scatter_size,
            show_at_end = False,
            new_figure = False,
            flip_y = flip_y,
        )


    #create the main mesh vertices for setting the bounding box
    if len(main_mesh_vertices) == 0:
        raise Exception("No meshes or skeletons passed to the plotting funciton")
    elif len(main_mesh_vertices) == 1:
        main_mesh_vertices = main_mesh_vertices[0]
    else:
        #get rid of all empt
        #print(f"main_mesh_vertices = {main_mesh_vertices}")
        main_mesh_vertices = np.vstack([k.reshape(-1,3) for k in main_mesh_vertices if len(k)>0])
    
    if len(main_mesh_vertices) == 0:
        print("***There is nothing to plot***")
        return
    

#     print(f"main_mesh_vertices = {main_mesh_vertices}")
    
    if flip_y:
        main_mesh_vertices = main_mesh_vertices.copy()
        main_mesh_vertices = main_mesh_vertices.reshape(-1,3)
        main_mesh_vertices[...,1] = -main_mesh_vertices[...,1]
    

    volume_max = np.max(main_mesh_vertices.reshape(-1,3),axis=0)
    volume_min = np.min(main_mesh_vertices.reshape(-1,3),axis=0)
    
#     if len(main_mesh_vertices) < 10:
#         print(f"main_mesh_vertices = {main_mesh_vertices}")
#     print(f"volume_max= {volume_max}")
#     print(f"volume_min= {volume_min}")

    ranges = volume_max - volume_min
    index = [0,1,2]
    max_index = np.argmax(ranges)
    min_limits = [0,0,0]
    max_limits = [0,0,0]


    for i in index:
        if i == max_index or not adaptive_min_max_limits:
            min_limits[i] = volume_min[i] - buffer
            max_limits[i] = volume_max[i] + buffer 
            continue
        else:
            difference = ranges[max_index] - ranges[i]
            min_limits[i] = volume_min[i] - difference/2  - buffer
            max_limits[i] = volume_max[i] + difference/2 + buffer
            
#     print(f"min_limits= {min_limits}")
#     print(f"max_limits= {max_limits}")

    #ipv.xyzlim(-2, 2)
    
    if set_zoom:
        ipv.xlim(min_limits[0],max_limits[0])
        ipv.ylim(min_limits[1],max_limits[1])
        ipv.zlim(min_limits[2],max_limits[2])


        ipv.style.set_style_light()
        if axis_box_off:
            ipv.style.axes_off()
            ipv.style.box_off()
        else:
            ipv.style.axes_on()
            ipv.style.box_on()
        
    if show_at_end:
        ipv.show()
    
    if html_path != "":
        ipv.pylab.save(html_path)
    
    if return_other_colors:
        return other_meshes_colors
        


""" ------------------- Mesh subtraction ------------------------------------"""
#make sure pip3 install trimesh --upgrade so can have slice

try:
    import calcification_Module as cm
except:
    pass


#  Utility functions
angle = np.pi/2
rotation_matrix = np.array([[np.cos(angle),-np.sin(angle),0],
                            [np.sin(angle),np.cos(angle),0],
                            [0,0,1]
                           ])

def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q
def change_basis_matrix(v):
    """
    This just gives change of basis matrix for a basis 
    that has the vector v as its 3rd basis vector
    and the other 2 vectors orthogonal to v 
    (but not necessarily orthogonal to each other)
    *** not make an orthonormal basis ***
    
    -- changed so now pass the non-orthogonal components
    to the QR decomposition to get them as orthogonal
    
    """
    a,b,c = v
    #print(f"a,b,c = {(a,b,c)}")
    if np.abs(c) > 0.00001:
        v_z = v/np.linalg.norm(v)
        v_x = np.array([1,0,-a/c])
        #v_x = v_x/np.linalg.norm(v_x)
        v_y = np.array([0,1,-b/c])
        #v_y = v_y/np.linalg.norm(v_y)
        v_x, v_y = gram_schmidt_columns(np.vstack([v_x,v_y]).T).T
        return np.vstack([v_x,v_y,v_z])
    else:
        #print("Z coeffienct too small")
        #v_z = v
        v[2] = 0
        #print(f"before norm v_z = {v}")
        v_z = v/np.linalg.norm(v)
        #print(f"after norm v_z = {v_z}")
        
        v_x = np.array([0,0,1])
        v_y = rotation_matrix@v_z
        
    return np.vstack([v_x,v_y,v_z])

def mesh_subtraction_by_skeleton_old(main_mesh,edges,
                                 buffer=0.01,
                                bbox_ratio=1.2,
                                 distance_threshold=2000,
                             significance_threshold=500,
                                print_flag=False):
    """
    Purpose: Will return significant mesh pieces that are
    not already accounteed for by the skeleton
    
    Example of how to run
    
    main_mesh_path = Path("./Dustin/Dustin.off")
    main_mesh = trimesh.load_mesh(str(main_mesh_path.absolute()))
    skeleton_path = main_mesh_path.parents[0] / Path(main_mesh_path.stem + "_skeleton.cgal")
    edges = sk.read_skeleton_edges_coordinates(str(skeleton_path.absolute()))

    # turn this into nodes and edges
    main_mesh_nodes, main_mesh_edges = sk.read_skeleton_verts_edges(str(skeleton_path.absolute()))
    sk.graph_skeleton_and_mesh(
                main_mesh_verts=main_mesh.vertices,
                main_mesh_faces=main_mesh.faces,
                unique_skeleton_verts_final = main_mesh_nodes,
                edges_final=main_mesh_edges,
                buffer = 0
                              )
                              
    leftover_pieces =  mesh_subtraction_by_skeleton(main_mesh,edges,
                                 buffer=0.01,
                                bbox_ratio=1.2,
                                 distance_threshold=500,
                             significance_threshold=500,
                                print_flag=False)
                                
    # Visualize the results: 
    pieces_mesh = trimesh.Trimesh(vertices=np.array([]),
                                 faces=np.array([]))

    for l in leftover_pieces:
        pieces_mesh += l

    sk.graph_skeleton_and_mesh(
                main_mesh_verts=pieces_mesh.vertices,
                main_mesh_faces=pieces_mesh.faces,
                unique_skeleton_verts_final = main_mesh_nodes,
                edges_final=main_mesh_edges,
                buffer = 0
                              )
    
    """
    
    skeleton_nodes = edges.reshape(-1,3)
    skeleton_bounding_corners = np.vstack([np.max(skeleton_nodes,axis=0),
               np.min(skeleton_nodes,axis=0)])
    
    main_mesh_bbox_restricted, faces_bbox_inclusion = tu.bbox_mesh_restriction(main_mesh,
                                                                        skeleton_bounding_corners,
                                                                        bbox_ratio)

    if type(main_mesh_bbox_restricted) == type(trimesh.Trimesh()):
        print(f"Inside mesh subtraction, len(main_mesh_bbox_restricted.faces) = {len(main_mesh_bbox_restricted.faces)}")
    else:
        print("***** Bounding Box Restricted Mesh is empty ****")
        main_mesh_bbox_restricted = main_mesh
        faces_bbox_inclusion = np.arange(0,len(main_mesh.faces))
    
    start_time = time.time()

    #face_subtract_color = []
    face_subtract_indices = []

    #distance_threshold = 2000
    
    edge_loop_print=False
    for i,ex_edge in tqdm(enumerate(edges)):
        #print("\n------ New loop ------")
        #print(ex_edge)
        
        # ----------- creating edge and checking distance ----- #
        loop_start = time.time()
        
        edge_line = ex_edge[1] - ex_edge[0]
        sum_threshold = 0.001
        if np.sum(np.abs(edge_line)) < sum_threshold:
            if edge_loop_print:
                print(f"edge number {i}, {ex_edge}: has sum less than {sum_threshold} so skipping")
            continue
#         if edge_loop_print:
#             print(f"Checking Edge Distance = {time.time()-loop_start}")
#         loop_start = time.time()
        
        cob_edge = change_basis_matrix(edge_line)
        
#         if edge_loop_print:
#             print(f"Change of Basis Matrix calculation = {time.time()-loop_start}")
#         loop_start - time.time()
        
        #get the limits of the example edge itself that should be cutoff
        edge_trans = (cob_edge@ex_edge.T)
        #slice_range = np.sort((cob_edge@ex_edge.T)[2,:])
        slice_range = np.sort(edge_trans[2,:])

        # adding the buffer to the slice range
        slice_range_buffer = slice_range + np.array([-buffer,buffer])
        
#         if edge_loop_print:
#             print(f"Calculate slice= {time.time()-loop_start}")
#         loop_start = time.time()

        # generate face midpoints from the triangles
        #face_midpoints = np.mean(main_mesh_bbox_restricted.vertices[main_mesh_bbox_restricted.faces],axis=1) # Old way
        face_midpoints = main_mesh_bbox_restricted.triangles_center
        
#         if edge_loop_print:
#             print(f"Face midpoints= {time.time()-loop_start}")
#         loop_start = time.time()
        
        #get the face midpoints that fall within the slice (by lookig at the z component)
        fac_midpoints_trans = cob_edge@face_midpoints.T
        
#         if edge_loop_print:
#             print(f"Face midpoints transform= {time.time()-loop_start}")
#         loop_start = time.time()
        
        
        
#         if edge_loop_print:
#             print(f"edge midpoint= {time.time()-loop_start}")
#         loop_start = time.time()
        
        slice_mask_pre_distance = ((fac_midpoints_trans[2,:]>slice_range_buffer[0]) & 
                      (fac_midpoints_trans[2,:]<slice_range_buffer[1]))

#         if edge_loop_print:
#             print(f"Applying slice restriction = {time.time()-loop_start}")
#         loop_start = time.time()
        
        
        """ 6/18 change
        # apply the distance threshold to the slice mask
        edge_midpoint = np.mean(ex_edge,axis=0)
        #raise Exception("Add in part for distance threshold here")
        distance_check = np.linalg.norm(face_midpoints[:,:2] - edge_midpoint[:2],axis=1) < distance_threshold
        
        """
        
#         edge_midpoint = np.mean(cob_edge.T,axis=0)
#         distance_check = np.linalg.norm((fac_midpoints_trans.T)[:,:2] - edge_midpoint[:2],axis=1) < distance_threshold

        edge_midpoint = np.mean(edge_trans.T,axis=0)
        distance_check = np.linalg.norm((fac_midpoints_trans.T)[:,:2] - edge_midpoint[:2],axis=1) < distance_threshold
        

        slice_mask = slice_mask_pre_distance & distance_check
        
#         if edge_loop_print:
#             print(f"Applying distance restriction= {time.time()-loop_start}")
#         loop_start = time.time()


        face_list = np.arange(0,len(main_mesh_bbox_restricted.faces))[slice_mask]

        #get the submesh of valid faces within the slice
        if len(face_list)>0:
            main_mesh_sub = main_mesh_bbox_restricted.submesh([face_list],append=True)
        else:
            main_mesh_sub = []
        
        

        if type(main_mesh_sub) != type(trimesh.Trimesh()):
            if edge_loop_print:
                print(f"THERE WERE NO FACES THAT FIT THE DISTANCE ({distance_threshold}) and Z transform requirements")
                print("So just skipping this edge")
            continue

#         if edge_loop_print:
#             print(f"getting submesh= {time.time()-loop_start}")
#         loop_start = time.time()
        
        #get all disconnected mesh pieces of the submesh and the face indices for lookup later
        sub_components,sub_components_face_indexes = tu.split(main_mesh_sub,only_watertight=False)
        if type(sub_components) != type(np.array([])) and type(sub_components) != list:
            #print(f"meshes = {sub_components}, with type = {type(sub_components)}")
            if type(sub_components) == type(trimesh.Trimesh()) :
                sub_components = [sub_components]
            else:
                raise Exception("The sub_components were not an array, list or trimesh")
        
#         if edge_loop_print:
#             print(f"splitting the mesh= {time.time()-loop_start}")
#         loop_start = time.time()

        #getting the indices of the submeshes whose bounding box contain the edge 
        """ 6-19: might want to use bounding_box_oriented? BUT THIS CHANGE COULD SLOW IT DOWN
        contains_points_results = np.array([s_comp.bounding_box_oriented.contains(ex_edge.reshape(-1,3)) for s_comp in sub_components])
        """
        contains_points_results = np.array([s_comp.bounding_box.contains(ex_edge.reshape(-1,3)) for s_comp in sub_components])
        
        containing_indices = (np.arange(0,len(sub_components)))[np.sum(contains_points_results,axis=1) >= len(ex_edge)]
        
#         if edge_loop_print:
#             print(f"containing indices= {time.time()-loop_start}")
#         loop_start = time.time()

        try:
            if len(containing_indices) != 1: 
                if edge_loop_print:
                    print(f"--> Not exactly one containing mesh: {containing_indices}")
                if len(containing_indices) > 1:
                    sub_components_inner = sub_components[containing_indices]
                    sub_components_face_indexes_inner = sub_components_face_indexes[containing_indices]
                else:
                    sub_components_inner = sub_components
                    sub_components_face_indexes_inner = sub_components_face_indexes

                #get the center of the edge
                edge_center = np.mean(ex_edge,axis=0)
                #print(f"edge_center = {edge_center}")

                #find the distance between eacch bbox center and the edge center
                bbox_centers = [np.mean(k.bounds,axis=0) for k in sub_components_inner]
                #print(f"bbox_centers = {bbox_centers}")
                closest_bbox = np.argmin([np.linalg.norm(edge_center-b_center) for b_center in bbox_centers])
                #print(f"bbox_distance = {closest_bbox}")
                edge_skeleton_faces = faces_bbox_inclusion[face_list[sub_components_face_indexes_inner[closest_bbox]]]

    #             if edge_loop_print:
    #                 print(f"finding closest box when 0 or 2 or more containing boxes= {time.time()-loop_start}")
    #             loop_start = time.time()
            elif len(containing_indices) == 1:# when only one viable submesh piece and just using that sole index
                #print(f"only one viable submesh piece so using index only number in: {containing_indices}")

                edge_skeleton_faces = faces_bbox_inclusion[face_list[sub_components_face_indexes[containing_indices[0]].astype("int")]]
    #             if edge_loop_print:
    #                 print(f"only 1 containig face getting the edge_skeleton_faces= {time.time()-loop_start}")
    #             loop_start = time.time()
            else:
                raise Exception("No contianing indices")
        except:
            from datasci_tools import system_utils as su
            su.compressed_pickle(main_mesh_sub,"main_mesh_sub")
            su.compressed_pickle(ex_edge,"ex_edge")
            su.compressed_pickle(sub_components_face_indexes,"sub_components_face_indexes")
            su.compressed_pickle(containing_indices,"containing_indices")
            su.compressed_pickle(face_list,"face_list")
            su.compressed_pickle(faces_bbox_inclusion,"faces_bbox_inclusion")
            
            raise Exception("Indexing not work in mesh subtraction")

            
            
            
            

        if len(edge_skeleton_faces) < 0:
            print(f"****** Warning the edge index {i}: had no faces in the edge_skeleton_faces*******")
        face_subtract_indices.append(edge_skeleton_faces)
#         if edge_loop_print:
#                 print(f"check and append for face= {time.time()-loop_start}")
        #face_subtract_color.append(viable_colors[i%len(viable_colors)])
        
    print(f"Total Mesh subtraction time = {np.round(time.time() - start_time,4)}")
    
    if len(face_subtract_indices)>0:
        all_removed_faces = np.concatenate(face_subtract_indices)

        unique_removed_faces = set(all_removed_faces)

        faces_to_keep = set(np.arange(0,len(main_mesh.faces))).difference(unique_removed_faces)
        new_submesh = main_mesh.submesh([list(faces_to_keep)],only_watertight=False,append=True)
    else:
        new_submesh = main_mesh
    
    significant_pieces = split_significant_pieces(new_submesh,
                                                         significance_threshold,
                                                         print_flag=False)


    return significant_pieces

""" ------------------- End of Mesh Subtraction ------------------------------------"""



""" ----------Start of Surface Skeeltonization -- """



# # Older version that was not working properly
# def generate_surface_skeleton(vertices,
#                               faces, 
#                               surface_samples=1000,
#                           print_flag=False):
    
#     #return surface_with_poisson_skeleton,path_length
    
#     mesh = trimesh.Trimesh(vertices=vertices,
#                                   faces = faces,
#                            )


#     start_time = time.time()

#     ga = nx.from_edgelist(mesh.edges)

#     if surface_samples<len(vertices):
#         k = surface_samples
#     else:
#         k = len(vertices)
#     sampled_nodes = random.sample(ga.nodes, k)


#     lp_end_list = []
#     lp_magnitude_list = []

#     for s in sampled_nodes: 
#         sp_dict = nx.single_source_shortest_path_length(ga,s)

#         list_keys = list(sp_dict.keys())
#         longest_path_node = list_keys[len(list_keys)-1]
#         longest_path_magnitude = sp_dict[longest_path_node]


#         lp_end_list.append(longest_path_node)
#         lp_magnitude_list.append(longest_path_magnitude)

#     #construct skeleton from shortest path
#     final_start = sampled_nodes[np.argmax(lp_magnitude_list)]
#     final_end = sampled_nodes[np.argmax(lp_end_list)]

#     node_list = nx.shortest_path(ga,final_start,final_end)
#     if len(node_list) < 2:
#         print("node_list len < 2 so returning empty list")
#         return np.array([])
#     #print("node_list = " + str(node_list))

#     final_skeleton = mesh.vertices[np.vstack([node_list[:-1],node_list[1:]]).T]
#     if print_flag:
#         print(f"   Final Time for surface skeleton with sample size = {k} = {time.time() - start_time}")

#     return final_skeleton


def generate_surface_skeleton_slower(vertices,
                              faces=None, 
                              surface_samples=1000,
                              n_surface_downsampling=0,
                          print_flag=False):
    """
    Purpose: Generates a surface skeleton without using
    the root method and instead just samples points
    """
    
    #return surface_with_poisson_skeleton,path_length
    if not tu.is_mesh(vertices):
        mesh = trimesh.Trimesh(vertices=vertices,
                                  faces = faces,
                           )
    else:
        mesh = vertices
        vertices = mesh.vertices
        faces = mesh.faces


    start_time = time.time()

    ga = nx.from_edgelist(mesh.edges)

    if surface_samples<len(vertices):
        sampled_nodes = np.random.choice(len(vertices),surface_samples , replace=False)
    else:
        if print_flag:
            print("Number of surface samples exceeded number of vertices, using len(vertices)")
        sampled_nodes = np.arange(0,len(vertices))
        
    lp_end_list = []
    lp_magnitude_list = []

    for s in sampled_nodes: 
        #gives a dictionary where the key is the end node and the value is the number of
        # edges on the shortest path
        sp_dict = nx.single_source_shortest_path_length(ga,s)

        #
        list_keys = list(sp_dict.keys())
        
        #gets the end node that would make the longest shortest path 
        longest_path_node = list_keys[-1]
        
        #get the number of edges for the path
        longest_path_magnitude = sp_dict[longest_path_node]


        #add the ending node and the magnitude of it to lists
        lp_end_list.append(longest_path_node)
        lp_magnitude_list.append(longest_path_magnitude)

    lp_end_list = np.array(lp_end_list)
    #construct skeleton from shortest path
    max_index = np.argmax(lp_magnitude_list)
    final_start = sampled_nodes[max_index]
    final_end = lp_end_list[max_index]

    node_list = nx.shortest_path(ga,final_start,final_end)
    if len(node_list) < 2:
        print("node_list len < 2 so returning empty list")
        return np.array([])
    #print("node_list = " + str(node_list))

    final_skeleton = mesh.vertices[np.vstack([node_list[:-1],node_list[1:]]).T]
    if print_flag:
        print(f"   Final Time for surface skeleton with sample size = {k} = {time.time() - start_time}")
        
    for i in range(n_surface_downsampling):
        final_skeleton = downsample_skeleton(final_skeleton)

    return final_skeleton


#from meshparty_skeletonize import *
def setup_root(mesh, is_soma_pt=None, soma_d=None, is_valid=None):
    """ function to find the root index to use for this mesh
    
    Purpose: The output will be used to find the path for a 
    surface skeletonization (aka: longest shortest path)
    
    The output:
    1) root: One of the end points
    2) target: The other endpoint:
    3) root_ds: (N,) matrix of distances from root to all other vertices
    4) : predecessor matrix for root to shortest paths of all other vertices
    --> used to find surface path
    5) valid: boolean mask (NOT USED)
    
    """
    if is_valid is not None:
        valid = np.copy(is_valid)
    else:
        valid = np.ones(len(mesh.vertices), np.bool)
    assert(len(valid) == mesh.vertices.shape[0])

    root = None
    # soma mode
    if is_soma_pt is not None:
        # pick the first soma as root
        assert(len(soma_d) == mesh.vertices.shape[0])
        assert(len(is_soma_pt) == mesh.vertices.shape[0])
        is_valid_root = is_soma_pt & valid
        valid_root_inds = np.where(is_valid_root)[0]
        if len(valid_root_inds) > 0:
            min_valid_root = np.nanargmin(soma_d[valid_root_inds])
            root = valid_root_inds[min_valid_root]
            root_ds, pred = m_sk.sparse.csgraph.dijkstra(mesh.csgraph,
                                                    directed=False,
                                                    indices=root,
                                                    return_predecessors=True)
        else:
            start_ind = np.where(valid)[0][0]
            root, target, pred, dm, root_ds = m_sk.utils.find_far_points(mesh,
                                                                    start_ind=start_ind)
        valid[is_soma_pt] = False

    if root is None:
        # there is no soma close, so use far point heuristic
        start_ind = np.where(valid)[0][0]
        root, target, pred, dm, root_ds = m_sk.utils.find_far_points(
            mesh, start_ind=start_ind)
    valid[root] = False
    assert(np.all(~np.isinf(root_ds[valid])))
    return root, target,root_ds, pred, valid

def surface_skeleton(mesh,
                     plot = False,
                    **kwargs):
    return_sk = generate_surface_skeleton(mesh.vertices,
                              mesh.faces, 
                                     **kwargs)
    if plot:
        ipvu.plot_objects(
            mesh,
            return_sk,
            buffer = 0,
        )
        
    return return_sk
def generate_surface_skeleton(vertices,
                              faces, 
                              surface_samples=1000,
                              n_surface_downsampling=0,
                          print_flag=False,
                                 ):
    """
    Purpose: To generate a surface skeleton
    
    Specifics: New implementation that uses meshparty 
    method of finding root that optimally finds 
    longest shortest path
    
    """
    
    meshparty_skeleton_time = time.time()
    branch_obj_tr_io  = meshparty.trimesh_io.Mesh(vertices = vertices,
                                   faces=faces)
    
    root, target,root_ds, root_pred, valid = setup_root(branch_obj_tr_io)

    current_path = m_sk.utils.get_path(root,target,root_pred)

    surface_sk_edges = np.vstack([current_path[:-1],current_path[1:]]).T
    meshparty_surface_skeleton = branch_obj_tr_io.vertices[surface_sk_edges]
    
    if print_flag: 
        print(f"Total time for surface skeletonization = {time.time() - meshparty_skeleton_time}")
    
    for i in range(n_surface_downsampling):
        meshparty_surface_skeleton = downsample_skeleton(meshparty_surface_skeleton)
    
    return meshparty_surface_skeleton


def downsample_skeleton(current_skeleton):
    #print("current_skeleton = " + str(current_skeleton.shape))
    """
    Downsamples the skeleton by 50% number of edges
    """
    extra_segment = []
    if current_skeleton.shape[0] % 2 != 0:
        extra_segment = np.array([current_skeleton[0]])
        current_skeleton = current_skeleton[1:]
        #print("extra_segment = " + str(extra_segment))
        #print("extra_segment.shape = " + str(extra_segment.shape))
    else:
        #print("extra_segment = " + str(extra_segment))
        pass

    even_indices = [k for k in range(0,current_skeleton.shape[0]) if k%2 == 0]
    odd_indices = [k for k in range(0,current_skeleton.shape[0]) if k%2 == 1]
    even_verts = current_skeleton[even_indices,0,:]
    odd_verts = current_skeleton[odd_indices,1,:]

    downsampled_skeleton = np.hstack([even_verts,odd_verts]).reshape(even_verts.shape[0],2,3)
    #print("dowsampled_skeleton.shape = " + str(downsampled_skeleton.shape))
    if len(extra_segment) > 0:
        #print("downsampled_skeleton = " + str(downsampled_skeleton.shape))
        final_downsampled_skeleton = np.vstack([extra_segment,downsampled_skeleton])
    else:
        final_downsampled_skeleton = downsampled_skeleton
    return final_downsampled_skeleton


# ----- Stitching Algorithm ----- #


def stitch_skeleton(
                                          staring_edges,
                                          max_stitch_distance=max_stitch_distance_default,
                                          stitch_print = False,
                                          main_mesh = []
                                        ):
    
    print(f"max_stitch_distance = {max_stitch_distance}")
    stitched_time = time.time()

    stitch_start = time.time()

    all_skeleton_vertices = staring_edges.reshape(-1,3)
    unique_rows,indices = np.unique(all_skeleton_vertices,return_inverse=True,axis=0)
    edges_with_coefficients = indices.reshape(-1,2)

    if stitch_print:
        print(f"Getting the unique rows and indices= {time.time()-stitch_start}")
    stitch_start = time.time()

    #create the graph from the edges
    #B = nx.Graph() #old way
    B = xu.GraphOrderedEdges()
    B.add_nodes_from([(x,{"coordinates":y}) for x,y in enumerate(unique_rows)])
    
    
    B.add_edges_from(edges_with_coefficients)
    
    if stitch_print:
        print(f"Putting edges into networkx graph= {time.time()-stitch_start}")
    stitch_start = time.time()

    # find the shortest distance between the two different subgraphs:
    from scipy.spatial import distance_matrix

    # UG = B.to_undirected() #no longer need this
    UG = B
    
    UG.edges_ordered()
    
    if stitch_print:
        print(f"Making undirected graph= {time.time()-stitch_start}")
    stitch_start = time.time()

    #get all of the coordinates

    print("len_subgraphs AT BEGINNING of the loop")
    counter = 0
    print_flag = True

    n_components, labels = scipy.sparse.csgraph.connected_components(csgraph=nx.adjacency_matrix(UG), directed=False, return_labels=True)
    #print(f"At beginning n_components = {n_components}, unique labels = {np.unique(labels)}")
    
    
    
    subgraph_components = np.where(labels==0)[0]
    outside_components = np.where(labels !=0)[0]

    for j in tqdm(range(n_components)):
        
        counter+= 1
        if stitch_print:
            print(f"Starting Loop {counter}")
        start_time = time.time()
        """
        1) Get the indexes of the subgraph
        2) Build a KDTree from those not in the subgraph (save the vertices of these)
        3) Query against the nodes in the subgraph  and get the smallest distance
        4) Create this new edge

        """
        stitch_time = time.time()
        #1) Get the indexes of the subgraph
        #n_components, labels = scipy.sparse.csgraph.connected_components(csgraph=nx.adjacency_matrix(UG), directed=False, return_labels=True)
        if stitch_print:
            print(f"Finding Number of Connected Components= {time.time()-stitch_start}")
        stitch_start = time.time()

        subgraph_components = np.where(labels==0)[0]

        if stitch_print:
            print(f"Faces belonging to largest component= {time.time()-stitch_start}")
        stitch_start = time.time()
        #print("subgraph_components = " + str(subgraph_components))
        if len(subgraph_components) == len(UG.nodes):
            print("all graph is one component!")
            #print(f"unique labels = {np.unique(labels)}")
            break

        if stitch_print:
            print(f"n_components = {n_components}")

        outside_components = np.where(labels !=0)[0]

        if stitch_print:
            print(f"finding faces not in largest component= {time.time()-stitch_start}")
        stitch_start = time.time()
        #print("outside_components = " + str(outside_components))

        #2) Build a KDTree from those not in the subgraph (save the vertices of these)
        mesh_tree = KDTree(unique_rows[outside_components])
        if stitch_print:
            print(f"Building KDTree= {time.time()-stitch_start}")
        stitch_start = time.time()


        #3) Query against the nodes in the subgraph  and get the smallest distance
        """
        Conclusion:
        Distance is of the size of the parts that are in the KDTree
        The closest nodes represent those that were queryed

        """
        distances,closest_node = mesh_tree.query(unique_rows[subgraph_components])
        if stitch_print:
            print(f"Mesh Tree query= {time.time()-stitch_start}")
        stitch_start = time.time()
        min_index = np.argmin(distances)
        
        #check if the distance is too far away 
        if distances[min_index] > max_stitch_distance:
            print(f"**** The distance exceeded max stitch distance of {max_stitch_distance}"
                   f" and still had {n_components} left\n"
                  f"   Actual distance was {distances[min_index]} ")
        

        if stitch_print:
            print(f"Finding closest distance= {time.time()-stitch_start}")
        stitch_start = time.time()


        closest_outside_node = outside_components[closest_node[min_index]]
        closest_subgraph_node = subgraph_components[min_index]

        if stitch_print:
            print(f"Getting nodes to be paired up= {time.time()-stitch_start}")
        stitch_start = time.time()

        
        
        #get the edge distance of edge about to create:

    #         graph_coordinates=nx.get_node_attributes(UG,'coordinates')
    #         prospective_edge_length = np.linalg.norm(np.array(graph_coordinates[closest_outside_node])-np.array(graph_coordinates[closest_subgraph_node]))
    #         print(f"Edge distance going to create = {prospective_edge_length}")

        #4) Create this new edge
        UG.add_edge(closest_subgraph_node,closest_outside_node)

        #get the label of the closest outside node 
        closest_outside_label = labels[closest_outside_node]

        #get all of the nodes with that same label
        new_subgraph_components = np.where(labels==closest_outside_label)[0]

        #relabel the nodes so now apart of the largest component
        labels[new_subgraph_components] = 0

        #move the newly relabeled nodes out of the outside components into the subgraph components
        ## --- SKIP THIS ADDITION FOR RIGHT NOW -- #


        if stitch_print:
            print(f"Adding Edge = {time.time()-stitch_start}")
        stitch_start = time.time()

        n_components -= 1

        if stitch_print:
            print(f"Total Time for loop = {time.time() - start_time}")


    # get the largest subgraph!!! in case have missing pieces

    #add all the new edges to the 

#     total_coord = nx.get_node_attributes(UG,'coordinates')
#     current_coordinates = np.array(list(total_coord.values()))

    current_coordinates = unique_rows
    
    try:
        #total_edges_stitched = current_coordinates[np.array(list(UG.edges())).reshape(-1,2)] #old way of edges
        total_edges_stitched = current_coordinates[UG.edges_ordered().reshape(-1,2)]
    except:
        print("getting the total edges stitched didn't work")
        print(f"current_coordinates = {current_coordinates}")
        print(f"UG.edges_ordered() = {list(UG.edges_ordered())} with type = {type(list(UG.edges_ordered()))}")
        print(f"np.array(UG.edges_ordered()) = {UG.edges_ordered()}")
        print(f"np.array(UG.edges_ordered()).reshape(-1,2) = {UG.edges_ordered().reshape(-1,2)}")
        
        raise Exception(" total_edges_stitched not calculated")
        #print("returning ")
        #total_edges_stitched
    

    print(f"Total time for skeleton stitching = {time.time() - stitched_time}")
    
    return total_edges_stitched


def stack_skeletons(sk_list,graph_cleaning=False):
    list_of_skeletons = [np.array(k).reshape(-1,2,3) for k in sk_list if len(k)>0]
    if len(list_of_skeletons) == 0:
        print("No skeletons to stack so returning empty list")
        return []
    elif len(list_of_skeletons) == 1:
        #print("only one skeleton so no stacking needed")
        return np.array(list_of_skeletons).reshape(-1,2,3)
    else:
        final_sk = (np.vstack(list_of_skeletons)).reshape(-1,2,3)
        if graph_cleaning:
            final_sk = sk.convert_graph_to_skeleton(sk.convert_skeleton_to_graph(final_sk))
        return final_sk

#------------ The actual skeletonization from mesh contraction----------- #
try:
    from calcification_param_Module import calcification_param
except:
    pass
def calcification(
                    location_with_filename,
                    max_triangle_angle =1.91986,
                    quality_speed_tradeoff=0.2,#0.1,
                    medially_centered_speed_tradeoff=0.2,#0.2,
                    area_variation_factor=0.0001,#0.0001,
                    max_iterations=500,#500,
                    is_medially_centered=True,
                    min_edge_length = 75,
                    edge_length_multiplier = 0.002,
                    print_parameters=True,
                ):

    
    if type(location_with_filename) == type(Path()):
        location_with_filename = str(location_with_filename.absolute())
    
    if location_with_filename[-4:] == ".off":
        location_with_filename = location_with_filename[:-4]
    
    #print(f"location_with_filename = {location_with_filename}")
    print(f"min_edge_length = {min_edge_length}")
    
    return_value = calcification_param(
        location_with_filename,
        max_triangle_angle,
        quality_speed_tradeoff,
        medially_centered_speed_tradeoff,
        area_variation_factor,
        max_iterations,
        is_medially_centered,
        min_edge_length,
        edge_length_multiplier,
        print_parameters
    )
    
    return return_value,location_with_filename+"_skeleton.cgal"

cgal_skeletonization_parameters = dict(
    max_triangle_angle =1.91986,
    quality_speed_tradeoff=0.1,
    medially_centered_speed_tradeoff=0.2,#0.2,
    area_variation_factor=0.0001,#0.0001,
    max_iterations=500,#500,
    is_medially_centered=True,
    min_edge_length = 0,
    edge_length_multiplier = 0.002,
)
def skeleton_cgal(
    mesh=None,
    mesh_path=None,
    path_to_write=None,
    filepath = "./temp.off",
    remove_mesh_temp_file=True,
    remove_skeleton_temp_file=False,
    manifold_fix=True,
    verbose=False,
    return_skeleton_file_path_and_status=False,
    cgal_original_parameters = False,
    plot = False,
    **kwargs):
    """
    Pseudocode: 
    1) Write the mesh to a file
    2) Pass the file to the calcification
    3) Delete the temporary file
    
    -- will now check and fix mesh manifoldness
    
    
    Polyhedron mesh;
    if (!input)
    {
        std::cerr << "Cannot open file " << std::endl;
        return 2;
    }
    std::vector<K::Point_3> points;
    std::vector< std::vector<std::size_t> > polygons;
    if (!CGAL::read_OFF(input, points, polygons))
    {
        std::cerr << "Error parsing the OFF file " << std::endl;
        return 3;
    }
    CGAL::Polygon_mesh_processing::orient_polygon_soup(points, polygons);
    CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points, polygons, mesh);
    if(!CGAL::is_closed(mesh)){
        std::cerr << "Not closed mesh"<<std::endl;
        return 4;
    }
    """
    
    #----------- 1/15 Addition: Will fix manifold issues
    if manifold_fix:
        if verbose:
            print("")
        if mesh is None:
            mesh = tu.load_mesh_no_processing(mesh_path)
        
        if not tu.is_manifold(mesh): #should only be one piece
            mesh = tu.poisson_surface_reconstruction(mesh,
                                                 return_largest_mesh=True,
                                                 manifold_clean=True)
        if not tu.is_watertight(mesh):
            mesh = tu.fill_holes(mesh)
            
        print(f"Manifold status before skeletonization = {tu.is_manifold(mesh)}")
        print(f"Watertight status before skeletonization = {tu.is_watertight(mesh)}")
        
        
    tu.write_neuron_off(mesh,"mesh_watertight.off")
    
    #1) Write the mesh to a file
    if not mesh is None:
        written_path = write_neuron_off(mesh,filepath)
    else:
        written_path = mesh_path
    
    #2) Pass the file to the calcification
    sk_time = time.time()
    
    if not cgal_original_parameters:
        skeleton_results,sk_file = calcification(written_path,**kwargs)
    else:
        skeleton_results,sk_file = calcification(
            written_path,
            **cgal_skeletonization_parameters
        )
    
    
    #3) Delete the temporary file
    if remove_mesh_temp_file:
        Path(written_path).unlink()
    

    if verbose:
        print(f"Total time for skeletonizing {time.time() - sk_time}")
        
    if plot:
        significant_poisson_skeleton = read_skeleton_edges_coordinates([sk_file])
        ipvu.plot_objects(
            mesh,
            significant_poisson_skeleton,
            buffer = 0,
        )
        
    if return_skeleton_file_path_and_status:
        return skeleton_results,sk_file
    else:
        significant_poisson_skeleton = read_skeleton_edges_coordinates([sk_file])
        if verbose:
            print(f"Returning skeleton of size {significant_poisson_skeleton.shape}")
        
        
        if remove_skeleton_temp_file:
            Path(sk_file).unlink()
            
        return significant_poisson_skeleton
    
def skeleton_cgal_original_parameters(
    mesh=None,
    cgal_original_parameters = True,
    plot = False,
    **kwargs):
    
    curr_kwargs = cgal_skeletonization_parameters.copy()
    
    curr_kwargs.update(kwargs)
        
    return_sk = skeleton_cgal(
        mesh = mesh,
        **curr_kwargs,
    )
    
    if plot:
        ipvu.plot_objects(
            mesh,
            return_sk,
            buffer=0
        )
        
    return return_sk
        
        




def example_cgal_skeletonization_original_params(
    filename = "./elephant.off",
    plot = True,
    ):
    mesh = tu.load_mesh_no_processing(filename)
    elephant_sk = sk.skeleton_cgal_original_parameters(
        mesh,
        verbose = True,
        medially_centered_speed_tradeoff=0.5,
    #     max_triangle_angle =1.91986,
        quality_speed_tradeoff=5,#0.1,
    #     medially_centered_speed_tradeoff=0.5,
    #     area_variation_factor=0.0001,#0.0001,
    #     max_iterations=5000,#500,
    #     is_medially_centered=True,
    #     min_edge_length =  meshu.bounding_box_diagonal(mesh)*edge_length_multiplier,
    #     edge_length_multiplier = 0,#edge_length_multiplier,
    #     print_parameters = True,
        remove_skeleton_temp_file = True,
        plot = plot,
    )

    
        
    return elephant_sk

# ---------- Does the cleaning of the skeleton -------------- #

#old way that didnt account for the nodes that are close together
def convert_skeleton_to_graph_old(staring_edges,
                             stitch_print=False):
    stitch_start = time.time()

    all_skeleton_vertices = staring_edges.reshape(-1,3)
    unique_rows,indices = np.unique(all_skeleton_vertices,return_inverse=True,axis=0)
    #need to merge unique indices so if within a certain range of each other then merge them together
    
    edges_with_coefficients = indices.reshape(-1,2)

    if stitch_print:
        print(f"Getting the unique rows and indices= {time.time()-stitch_start}")
    stitch_start = time.time()

    #create the graph from the edges
    #B = nx.Graph() #old way
    B = xu.GraphOrderedEdges()
    B.add_nodes_from([(int(x),{"coordinates":y}) for x,y in enumerate(unique_rows)])
    #print("just added the nodes")
    
    #print(f"xu.get_edge_attributes(B)= {xu.get_edge_attributes(B)}")
    
    #B.add_edges_from(edges_with_coefficients) #older weights without weights
    #adds weights for the edges
    weights = np.linalg.norm(unique_rows[edges_with_coefficients[:,0]] - unique_rows[edges_with_coefficients[:,1]],axis=1)
    edges_with_weights = np.hstack([edges_with_coefficients,weights.reshape(-1,1)])
    B.add_weighted_edges_from(edges_with_weights)
    #print("right after add_weighted_edges_from")
    #print(f"xu.get_edge_attributes(B)= {xu.get_edge_attributes(B)}")

    print(f"len(B.edges()) = {len(B.edges())}")
    if stitch_print:
        print(f"Putting edges into networkx graph= {time.time()-stitch_start}")
    stitch_start = time.time()

    # find the shortest distance between the two different subgraphs:
    from scipy.spatial import distance_matrix
    #print(f"B.__class__ = {B.__class__}")
    #print(f"xu.get_edge_attributes(B)= {xu.get_edge_attributes(B)}")
    UG = B
    #UG = B.to_undirected()
    
    if stitch_print:
        print(f"Making undirected graph= {time.time()-stitch_start}")
    stitch_start = time.time()
    
    #UG.remove_edges_from(nx.selfloop_edges(UG))
    UG = xu.remove_selfloops(UG)
    print(f"len(UG.edges()) = {len(UG.edges())}")
    #print(f"UG.__class__ = {UG.__class__}")
    #make sure the edges are ordered 
    UG.reorder_edges()
    print(f"len(UG.edges()) = {len(UG.edges())}")
    #print(f"UG.__class__ = {UG.__class__}")
    return UG


def convert_skeleton_to_graph(
    staring_edges=None,
    stitch_print=False,
    combine_node_dist = 0.0001,
    node_matching_size_threshold=10000,
    vertices = None,
    edges = None):
    """
    Purpose: To automatically convert a skeleton to a graph
    
    * 7/9 adjustments: make so slight differences in coordinates not affect the graph
    
    Pseudocode for how you could apply the closeness to skeletons
    1) Get the unique rows
    2) See if there are any rows that are the same (that gives you what to change them to)
    3) put those into a graph and find the connected components
    4) Pick the first node in the component to be the dominant one
    a. for each non-dominant node, replace every occurance of the non-dominant one with the dominant one in indices
    - add the non-dominant ones to a list to delete 
    
    ** this will result in an indices that doesn't have any of the repeat nodes, but the repeat nodes are still 
    taking up the numbers that they were originally in order with ***
    
    np.delete(x,[1,3],axis=0)) # to delete the rows 
    
    5) remap the indices and delete the unique rows that were not used
    
    
    5) Do everything like normal

"""
    stitch_start = time.time()

    if vertices is None or edges is None:
        all_skeleton_vertices = staring_edges.reshape(-1,3)
        unique_rows,indices = np.unique(all_skeleton_vertices,return_inverse=True,axis=0)

        #need to merge unique indices so if within a certain range of each other then merge them together
        reshaped_indices = indices.reshape(-1,2)
    else:
        reshaped_indices = edges
        unique_rows = vertices
    
    #This is fine because know there might be but fix it later on! (in terms of concept graph)
    if len(reshaped_indices) != len(np.unique(reshaped_indices,axis=0)):
        print("**** Warning: There were redundant edges in the skeleton*****")
    
    #part where will combine nodes that are very close
    
    #only do this if small enough, if too big then must skip (because will get memory error)
    if len(unique_rows) < node_matching_size_threshold:
    
        matching_nodes = nu.get_matching_vertices(unique_rows,equiv_distance=combine_node_dist)

        if len(matching_nodes) > 0:
            """
            Overall this loop will change the unique_rows and indices to account for nodes that should be merged
            """
            # Example graph for finding components
            ex_edges = matching_nodes.reshape(-1,2)
            ex_graph = nx.from_edgelist(ex_edges)


            #get the connected components
            all_conn_comp = list(nx.connected_components(ex_graph))

            to_delete_nodes = []
            for c_comp in all_conn_comp:
                curr_comp = list(c_comp)
                dom_node = curr_comp[0]
                non_dom_nodes = curr_comp[1:]
                for n_dom in non_dom_nodes:
                    indices[indices==n_dom] = dom_node
                    to_delete_nodes.append(n_dom)

            unique_leftovers = np.sort(np.unique(indices.ravel()))
            #construct a dictionary for mapping
            map_dict = dict([(v,k) for k,v in enumerate(unique_leftovers)])

            print(f"Gettng rid of {len(to_delete_nodes)} nodes INSIDE SKELETON TO GRAPH CONVERSION")

            def vec_translate(a):    
                return np.vectorize(map_dict.__getitem__)(a)

            indices = vec_translate(indices)

            #now delete the rows that were ignored
            unique_rows = np.delete(unique_rows,to_delete_nodes,axis=0)

            #do a check to make sure everything is working
            if len(np.unique(indices.ravel())) != len(unique_rows) or max(np.unique(indices.ravel())) != len(unique_rows) - 1:
                raise Exception("The indices list does not match the size of the unique rows"
                               f"np.unique(indices.ravel()) = {np.unique(indices.ravel())}, len(unique_rows)= {len(unique_rows) }")
    
    #resume regular conversion
    edges_with_coefficients = indices.reshape(-1,2)
    
    

    if stitch_print:
        print(f"INSIDE CONVERT_SKELETON_TO_GRAPH Getting the unique rows and indices= {time.time()-stitch_start}")
    stitch_start = time.time()

    #create the graph from the edges
    #B = nx.Graph() #old way
    B = xu.GraphOrderedEdges()
    B.add_nodes_from([(int(x),{"coordinates":y}) for x,y in enumerate(unique_rows)])
    #print("just added the nodes")
    #print(f"xu.get_edge_attributes(B)= {xu.get_edge_attributes(B)}")
    
    #B.add_edges_from(edges_with_coefficients) #older weights without weights
    #adds weights for the edges
    weights = np.linalg.norm(unique_rows[edges_with_coefficients[:,0]] - unique_rows[edges_with_coefficients[:,1]],axis=1)
    edges_with_weights = np.hstack([edges_with_coefficients,weights.reshape(-1,1)])
    B.add_weighted_edges_from(edges_with_weights)
    #print("right after add_weighted_edges_from")
    #print(f"xu.get_edge_attributes(B)= {xu.get_edge_attributes(B)}")

    if stitch_print:
        print(f"Putting edges into networkx graph= {time.time()-stitch_start}")
    stitch_start = time.time()

    # find the shortest distance between the two different subgraphs:
    from scipy.spatial import distance_matrix
    #print(f"B.__class__ = {B.__class__}")
    #print(f"xu.get_edge_attributes(B)= {xu.get_edge_attributes(B)}")
    UG = B
    #UG = B.to_undirected()
    
    if stitch_print:
        print(f"Making undirected graph= {time.time()-stitch_start}")
    stitch_start = time.time()
    
    #UG.remove_edges_from(nx.selfloop_edges(UG))
    UG = xu.remove_selfloops(UG)
    #print(f"UG.__class__ = {UG.__class__}")
    #make sure the edges are ordered 
    UG.reorder_edges()
    #print(f"UG.__class__ = {UG.__class__}")
    return UG


def convert_graph_to_skeleton(UG):
    UG = nx.convert_node_labels_to_integers(UG)
    total_coord = nx.get_node_attributes(UG,'coordinates')
    current_coordinates = np.array(list(total_coord.values()))
    
    try:
        #total_edges_stitched = current_coordinates[np.array(list(UG.edges())).reshape(-1,2)] # old way
        total_edges_stitched = current_coordinates[UG.edges_ordered().reshape(-1,2)]
    except:
        UG.edges_ordered()
        print("getting the total edges stitched didn't work")
        print(f"current_coordinates = {current_coordinates}")
        print(f"UG.edges() = {UG.edges_ordered()} with type = {type(UG.edges_ordered)}")
        print(f"np.array(UG.edges()) = {UG.edges_ordered()}")
        print(f"np.array(UG.edges()).reshape(-1,2) = {UG.edges_ordered().reshape(-1,2)}")
        
        raise Exception(" total_edges_stitched not calculated")
        
    return total_edges_stitched

def list_len_measure(curr_list,G):
    return len(curr_list)

def skeletal_distance(curr_list,G,coordinates_dict):
    
    #clean_time = time.time()
    #coordinates_dict = nx.get_node_attributes(G,'coordinates')
    #print(f"Extracting attributes = {time.time() - clean_time}")
    #clean_time = time.time()
    coor = [coordinates_dict[k] for k in curr_list]
    #print(f"Reading dict = {time.time() - clean_time}")
    #clean_time = time.time()
    norm_values =  [np.linalg.norm(coor[i] - coor[i-1]) for i in range(1,len(coor))]
    #print(f"Calculating norms = {time.time() - clean_time}")
    #print(f"norm_values = {norm_values}")
    return np.sum(norm_values)


def clean_skeleton(
    G,
    distance_func=None,
    min_distance_to_junction = 3,
    return_skeleton=True,
    endpoints_must_keep = None, #must be the same size as soma_border_vertices
    print_flag=False,
    return_removed_skeletons=False,
    error_if_endpoints_must_keep_not_endnode=True,
    verbose=False,
    ):
    """
    Example of how to use: 
    
    Simple Example:  
    def distance_measure_func(path,G=None):
    #print("new thing")
    return len(path)

    new_G = clean_skeleton(G,distance_measure_func,return_skeleton=False)
    nx.draw(new_G,with_labels=True)
    
    More complicated example:
    
    from mesh_tools import skeleton_utils as sk
    from importlib import reload
    sk = reload(sk)

    from pathlib import Path
    test_skeleton = Path("./Dustin_vp6/Dustin_soma_0_branch_0_0_skeleton.cgal")
    if not test_skeleton.exists():
        print(str(test_skeleton)[:-14])
        file_of_skeleton = sk.calcification(str(test_skeleton.absolute())[:-14])
    else:
        file_of_skeleton = test_skeleton

    # import the skeleton
    test_sk = sk.read_skeleton_edges_coordinates(test_skeleton)
    import trimesh
    test_mesh = trimesh.load_mesh(str(str(test_skeleton.absolute())[:-14] + ".off"))
    sk.graph_skeleton_and_mesh(test_mesh.vertices,
                              test_mesh.faces,
                              edge_coordinates=test_sk)

    # clean the skeleton and then visualize
    import time
    clean_time = time.time()
    cleaned_skeleton = clean_skeleton(test_sk,
                        distance_func=skeletal_distance,
                  min_distance_to_junction=10000,
                  return_skeleton=True,
                  print_flag=True)
    print(f"Total time for skeleton clean {time.time() - clean_time}")

    # see what skeleton looks like now
    test_mesh = trimesh.load_mesh(str(str(test_skeleton.absolute())[:-14] + ".off"))
    sk.graph_skeleton_and_mesh(test_mesh.vertices,
                              test_mesh.faces,
                              edge_coordinates=cleaned_skeleton)
                              
                              
    --------------- end of example -----------------
    """
    
    
    """ --- old way which had index error when completley straight line 
    def end_node_path_to_junciton(curr_G,end_node):
        curr_node = end_node
        node_list = [curr_node]
        for i in range(len(curr_G)):
            neighbors = list(curr_G[curr_node])
            if len(neighbors) <= 2:
                curr_node = [k for k in neighbors if k not in node_list][0]
                node_list.append(curr_node)
                #print(f"node_list = {node_list}")
            else:
                break
        return node_list
    """
    
    if distance_func is None:
        distance_func = sk.skeletal_distance
    
    def end_node_path_to_junciton(curr_G,end_node):
        curr_node = end_node
        node_list = [curr_node]
        for i in range(len(curr_G)):
            #print(f"\nloop #{i} with curr_node = {curr_node}")

            neighbors = list(curr_G[curr_node])
            #print(f"neighbors = {neighbors}")
            #print(f"node_list = {node_list}")
            if len(neighbors) <= 2:
                #print(f"[k for k in neighbors if k not in node_list] = {[k for k in neighbors if k not in node_list]}")
                possible_curr_nodes = [k for k in neighbors if k not in node_list]
                if len(possible_curr_nodes) <= 0: #this is when it is just one straight line
                    break
                else:
                    curr_node = possible_curr_nodes[0]
                    node_list.append(curr_node)
                    #print(f"node_list = {node_list}")
            else:
                break
        return node_list
    
    if print_flag:
        print(f"Using Distance measure {distance_func.__name__}")
    
    
    if type(G) not in [type(nx.Graph()),type(xu.GraphOrderedEdges())]:
        G = convert_skeleton_to_graph(G)
        
    kwargs = dict()
    kwargs["coordinates_dict"] = nx.get_node_attributes(G,'coordinates')
    
    
    end_nodes = np.array([k for k,n in dict(G.degree()).items() if n == 1])
    
    
    if (not endpoints_must_keep is None) and len(endpoints_must_keep)>0:
        if print_flag:
            print(f"endpoints_must_keep = {endpoints_must_keep}")
        all_single_nodes_to_eliminate = []
        endpoints_must_keep = np.array(endpoints_must_keep).reshape(-1,3)
        
        if print_flag:
            print(f"Number of end_nodes BEFORE filtering = {len(end_nodes)}")
        end_nodes_coordinates = xu.get_node_attributes(G,node_list=end_nodes)
        
        for end_k in endpoints_must_keep:
            end_node_idx = xu.get_nodes_with_attributes_dict(G,dict(coordinates=end_k))
            if len(end_node_idx)>0:
                end_node_idx = end_node_idx[0]
                try:
                    end_node_must_keep_idx = np.where(end_nodes==end_node_idx)[0][0]
                except:
                    if error_if_endpoints_must_keep_not_endnode:
                        if print_flag:
                            print(f"end_nodes = {end_nodes}")
                            print(f"end_node_idx = {end_node_idx}")
                        raise Exception("Something went wrong when trying to find end nodes")
                    else:
                        if print_flag:
                            print(f"end_nodes = {end_nodes} wasn't endnode but continuing anyway")
                        continue
                all_single_nodes_to_eliminate.append(end_node_must_keep_idx)
            else:
                raise Exception("Passed end node to keep that wasn't in the graph")
            
        if print_flag:
            print(f"all_single_nodes_to_eliminate = {all_single_nodes_to_eliminate}")
        new_end_nodes = np.array([k for i,k in enumerate(end_nodes) if i not in all_single_nodes_to_eliminate])

        #doing the reassigning
        end_nodes = new_end_nodes
        
            
    #clean_time = time.time()
    paths_to_j = [end_node_path_to_junciton(G,n) for n in end_nodes]
    #print(f"Total time for node path to junction = {time.time() - clean_time}")
    #clean_time = time.time()
    end_nodes_dist_to_j = np.array([distance_func(n,G,**kwargs) for n in paths_to_j])
    #print(f"Calculating distances = {time.time() - clean_time}")
    #clean_time = time.time()
    
    end_nodes = end_nodes[end_nodes_dist_to_j<min_distance_to_junction]
    end_nodes_dist_to_j = end_nodes_dist_to_j[end_nodes_dist_to_j<min_distance_to_junction]
    
    skeletons_removed = []
    if len(end_nodes) == 0 or len(end_nodes_dist_to_j) == 0:
        #no end nodes so need to return 
        if print_flag:
            print("no small end nodes to get rid of so returning whole skeleton")
    else:
        
        
        
        
        current_end_node = end_nodes[np.argmin(end_nodes_dist_to_j)]
        #print(f"Ordering the nodes = {time.time() - clean_time}")
        clean_time = time.time()
        if print_flag:
            print(f"total end_nodes = {end_nodes}")
        #current_end_node = ordered_end_nodes[0]
        paths_removed = 0

        for i in tqdm(range(len(end_nodes))):
            current_path_to_junction = end_node_path_to_junciton(G,current_end_node)
            if print_flag:
                #print(f"ordered_end_nodes = {ordered_end_nodes}")
                print(f"\n\ncurrent_end_node = {current_end_node}")
                print(f"current_path_to_junction = {current_path_to_junction}")
            if distance_func(current_path_to_junction,G,**kwargs) <min_distance_to_junction:
                if print_flag:
                    print(f"the current distance that was below was {distance_func(current_path_to_junction,G,**kwargs)}")
                #remove the nodes
                
                path_to_rem = current_path_to_junction[:-1]
                skeletons_removed.append(convert_graph_to_skeleton( G.subgraph(path_to_rem)))
                
                paths_removed += 1
                G.remove_nodes_from(current_path_to_junction[:-1])
                end_nodes = end_nodes[end_nodes != current_end_node]
                end_nodes_dist_to_j = np.array([distance_func(end_node_path_to_junciton(G,n),G,**kwargs) for n in end_nodes])

                end_nodes = end_nodes[end_nodes_dist_to_j<min_distance_to_junction]
                end_nodes_dist_to_j = end_nodes_dist_to_j[end_nodes_dist_to_j<min_distance_to_junction]

                if len(end_nodes_dist_to_j)<= 0:
                    break
                current_end_node = end_nodes[np.argmin(end_nodes_dist_to_j)]
    #             if print_flag:
    #                 print(f"   insdie if statement ordered_end_nodes = {ordered_end_nodes}")

                #current_end_node = ordered_end_nodes[0]

            else:
                break
            
    G = xu.remove_selfloops(G)
    if print_flag:
        print(f"Done cleaning networkx graph with {paths_removed} paths removed")
    if return_skeleton:
        if return_removed_skeletons:
            return convert_graph_to_skeleton(G),skeletons_removed
        else:
            return convert_graph_to_skeleton(G)
    else:
        if return_removed_skeletons:
            return G,skeletons_removed
        else:
            return G
    

min_distance_to_junction_default = 4001
def filter_away_small_skeleton_offshoots(
    skeleton,
    min_distance_to_junction = min_distance_to_junction_default,
    endpoints_must_keep=None,
    verbose = False,
    return_removed_skeletons=False,
    plot = False,
    **kwargs
    ):
    
    return_sk = clean_skeleton(
        skeleton,
        min_distance_to_junction = min_distance_to_junction,
        endpoints_must_keep=endpoints_must_keep,
        print_flag = verbose,
        return_removed_skeletons=return_removed_skeletons,
        **kwargs
    )
    
    if plot:
        ipvu.plot_objects(
            skeletons=[
                skeleton
            ],
            skeletons_colors=["black"],
            scatters=[return_sk.reshape(-1,3)],
        )
    
    return return_sk

filter_away_leaf_skeletons = filter_away_small_skeleton_offshoots




def clean_skeleton_with_soma_verts(G,
                   distance_func,
                  min_distance_to_junction = 3,
                  return_skeleton=True,
                   soma_border_vertices=None, #should be list of soma vertices
                   distance_to_ignore_end_nodes_close_to_soma_border=5000,
                   skeleton_mesh=None,
                   endpoints_must_keep = None, #must be the same size as soma_border_vertices
                  print_flag=False):
    """
    Example of how to use: 
    
    Simple Example:  
    def distance_measure_func(path,G=None):
    #print("new thing")
    return len(path)

    new_G = clean_skeleton(G,distance_measure_func,return_skeleton=False)
    nx.draw(new_G,with_labels=True)
    
    More complicated example:
    
    from mesh_tools import skeleton_utils as sk
    from importlib import reload
    sk = reload(sk)

    from pathlib import Path
    test_skeleton = Path("./Dustin_vp6/Dustin_soma_0_branch_0_0_skeleton.cgal")
    if not test_skeleton.exists():
        print(str(test_skeleton)[:-14])
        file_of_skeleton = sk.calcification(str(test_skeleton.absolute())[:-14])
    else:
        file_of_skeleton = test_skeleton

    # import the skeleton
    test_sk = sk.read_skeleton_edges_coordinates(test_skeleton)
    import trimesh
    test_mesh = trimesh.load_mesh(str(str(test_skeleton.absolute())[:-14] + ".off"))
    sk.graph_skeleton_and_mesh(test_mesh.vertices,
                              test_mesh.faces,
                              edge_coordinates=test_sk)

    # clean the skeleton and then visualize
    import time
    clean_time = time.time()
    cleaned_skeleton = clean_skeleton(test_sk,
                        distance_func=skeletal_distance,
                  min_distance_to_junction=10000,
                  return_skeleton=True,
                  print_flag=True)
    print(f"Total time for skeleton clean {time.time() - clean_time}")

    # see what skeleton looks like now
    test_mesh = trimesh.load_mesh(str(str(test_skeleton.absolute())[:-14] + ".off"))
    sk.graph_skeleton_and_mesh(test_mesh.vertices,
                              test_mesh.faces,
                              edge_coordinates=cleaned_skeleton)
                              
                              
    --------------- end of example -----------------
    """
    
    
    """ --- old way which had index error when completley straight line 
    def end_node_path_to_junciton(curr_G,end_node):
        curr_node = end_node
        node_list = [curr_node]
        for i in range(len(curr_G)):
            neighbors = list(curr_G[curr_node])
            if len(neighbors) <= 2:
                curr_node = [k for k in neighbors if k not in node_list][0]
                node_list.append(curr_node)
                #print(f"node_list = {node_list}")
            else:
                break
        return node_list
    """
    
    def end_node_path_to_junciton(curr_G,end_node):
        curr_node = end_node
        node_list = [curr_node]
        for i in range(len(curr_G)):
            #print(f"\nloop #{i} with curr_node = {curr_node}")

            neighbors = list(curr_G[curr_node])
            #print(f"neighbors = {neighbors}")
            #print(f"node_list = {node_list}")
            if len(neighbors) <= 2:
                #print(f"[k for k in neighbors if k not in node_list] = {[k for k in neighbors if k not in node_list]}")
                possible_curr_nodes = [k for k in neighbors if k not in node_list]
                if len(possible_curr_nodes) <= 0: #this is when it is just one straight line
                    break
                else:
                    curr_node = possible_curr_nodes[0]
                    node_list.append(curr_node)
                    #print(f"node_list = {node_list}")
            else:
                break
        return node_list
    
    print(f"Using Distance measure {distance_func.__name__}")
    
    
    if type(G) not in [type(nx.Graph()),type(xu.GraphOrderedEdges())]:
        G = convert_skeleton_to_graph(G)
        
    kwargs = dict()
    kwargs["coordinates_dict"] = nx.get_node_attributes(G,'coordinates')
    
    
    end_nodes = np.array([k for k,n in dict(G.degree()).items() if n == 1])
    """ 9/16 Addition: Will ignore certain end nodes whose distance is too close to soma border"""
    if not soma_border_vertices is None: #assuming that has the correct length
        if len(end_nodes) > 0:


            """
            OLD METHOD THAT DID NOT TRAVERSE ACROSS MESH GRAPH 
            
            Pseducode:
            1) Get the coordinates for all of the end nodes
            2) Put the soma border vertices in a KDTree
            3) Query the KDTree with the node endpoints coordinates
            4) Filter endpoints for only those farther than distance_to_ignore_end_nodes_close_to_soma_border
            
            
            print(f"Number of end_nodes BEFORE filtering = {len(end_nodes)}")
            end_nodes_coordinates = xu.get_node_attributes(G,node_list=end_nodes)
            soma_KD = KDTree(soma_border_vertices)
            distances,closest_nodes = soma_KD.query(end_nodes_coordinates)
            end_nodes = end_nodes[distances>distance_to_ignore_end_nodes_close_to_soma_border]
            print(f"Number of end_nodes AFTER filtering = {len(end_nodes)}")
        
            """
            print(f"Going to ignore certain endnodes that are {distance_to_ignore_end_nodes_close_to_soma_border} nm close to soma border vertices")
            #New method that traverses across mesh graph
            if skeleton_mesh is None:
                raise Exception("Skeleton_mesh is None when trying to account for soma_border_vertices in cleaning")
                
            print(f"Number of end_nodes BEFORE filtering = {len(end_nodes)}")
            end_nodes_coordinates = xu.get_node_attributes(G,node_list=end_nodes)
            
            #0) Get the mesh vertices and create a KDTree from them
            mesh_KD = KDTree(skeleton_mesh.vertices)
            
            #3) Create Weighted Graph from vertex edges
            vertex_graph = tu.mesh_vertex_graph(skeleton_mesh)
            
            all_single_nodes_to_eliminate = []
            
            if endpoints_must_keep is None:
                endpoints_must_keep = [[]]*len(soma_border_vertices)
            for sm_idx, sbv in soma_border_vertices.items():
                # See if there is a node for use to keep
                
                """ OLD WAY BEFORE MAKING MULTIPLE POSSIBLE SOMA TOUCHES
                end_k = endpoints_must_keep[sm_idx]
                """
                
                end_k_list = endpoints_must_keep[sm_idx]
                for end_k in end_k_list:
                    if not end_k is None:
                        end_node_must_keep = xu.get_nodes_with_attributes_dict(G,dict(coordinates=end_k))[0]
                        end_node_must_keep_idx = np.where(end_nodes==end_node_must_keep)[0][0]
                        all_single_nodes_to_eliminate.append(end_node_must_keep_idx)
                        print(f"Using an already specified end node: {end_node_must_keep} with index {end_node_must_keep_idx}"
                             f"checking was correct node end_nodes[index] = {end_nodes[end_node_must_keep_idx]}")
                    continue
            
                #1) Map the soma border vertices to the mesh vertices
                soma_border_distances,soma_border_closest_nodes = mesh_KD.query(sbv[0].reshape(-1,3))

                #2) Map the endpoints to closest mesh vertices
                end_nodes_distances,end_nodes_closest_nodes = mesh_KD.query(end_nodes_coordinates)



                #4) For each endpoint, find shortest distance from endpoint to a soma border along graph
                # for en,en_mesh_vertex in zip(end_nodes,end_nodes_closest_nodes):
                #     #find the shortest path to a soma border vertex
                node_idx_to_keep = []
                node_idx_to_eliminate = []
                node_idx_to_eliminate_len = []
                for en_idx,en in enumerate(end_nodes_closest_nodes):
                    try:
                        path_len, path = nx.single_source_dijkstra(vertex_graph,
                                                                   source = en,
                                                                   target=soma_border_closest_nodes[0],
                                                                   cutoff=distance_to_ignore_end_nodes_close_to_soma_border
                                                                  )
                    except:
                        node_idx_to_keep.append(en_idx)
                    else: #a valid path was found
                        node_idx_to_eliminate.append(en_idx)
                        node_idx_to_eliminate_len.append(path_len)
                        print(f"May Eliminate end_node {en_idx}: {end_nodes[en_idx]} because path_len to soma border was {path_len}")

                if len(node_idx_to_eliminate_len) > 0:
                    #see if there matches a node that we must keep
                    
                    
                    single_node_idx_to_eliminate = node_idx_to_eliminate[np.argmin(node_idx_to_eliminate_len)]
                    print(f"single_node_to_eliminate = {single_node_idx_to_eliminate}")
                    all_single_nodes_to_eliminate.append(single_node_idx_to_eliminate)
                else:
                    print("No close endpoints to choose from for elimination")
            
            print(f"all_single_nodes_to_eliminate = {all_single_nodes_to_eliminate}")
            new_end_nodes = np.array([k for i,k in enumerate(end_nodes) if i not in all_single_nodes_to_eliminate])

            #doing the reassigning
            end_nodes = new_end_nodes
            
    #clean_time = time.time()
    paths_to_j = [end_node_path_to_junciton(G,n) for n in end_nodes]
    #print(f"Total time for node path to junction = {time.time() - clean_time}")
    #clean_time = time.time()
    end_nodes_dist_to_j = np.array([distance_func(n,G,**kwargs) for n in paths_to_j])
    #print(f"Calculating distances = {time.time() - clean_time}")
    #clean_time = time.time()
    
    end_nodes = end_nodes[end_nodes_dist_to_j<min_distance_to_junction]
    end_nodes_dist_to_j = end_nodes_dist_to_j[end_nodes_dist_to_j<min_distance_to_junction]
    
    if len(end_nodes) == 0 or len(end_nodes_dist_to_j) == 0:
        #no end nodes so need to return 
        print("no small end nodes to get rid of so returning whole skeleton")
    else:
        
        
        
        
        current_end_node = end_nodes[np.argmin(end_nodes_dist_to_j)]
        #print(f"Ordering the nodes = {time.time() - clean_time}")
        clean_time = time.time()
        if print_flag:
            print(f"total end_nodes = {end_nodes}")
        #current_end_node = ordered_end_nodes[0]
        paths_removed = 0

        for i in tqdm(range(len(end_nodes))):
            current_path_to_junction = end_node_path_to_junciton(G,current_end_node)
            if print_flag:
                #print(f"ordered_end_nodes = {ordered_end_nodes}")
                print(f"\n\ncurrent_end_node = {current_end_node}")
                print(f"current_path_to_junction = {current_path_to_junction}")
            if distance_func(current_path_to_junction,G,**kwargs) <min_distance_to_junction:
                if print_flag:
                    print(f"the current distance that was below was {distance_func(current_path_to_junction,G,**kwargs)}")
                #remove the nodes
                paths_removed += 1
                G.remove_nodes_from(current_path_to_junction[:-1])
                end_nodes = end_nodes[end_nodes != current_end_node]
                end_nodes_dist_to_j = np.array([distance_func(end_node_path_to_junciton(G,n),G,**kwargs) for n in end_nodes])

                end_nodes = end_nodes[end_nodes_dist_to_j<min_distance_to_junction]
                end_nodes_dist_to_j = end_nodes_dist_to_j[end_nodes_dist_to_j<min_distance_to_junction]

                if len(end_nodes_dist_to_j)<= 0:
                    break
                current_end_node = end_nodes[np.argmin(end_nodes_dist_to_j)]
    #             if print_flag:
    #                 print(f"   insdie if statement ordered_end_nodes = {ordered_end_nodes}")

                #current_end_node = ordered_end_nodes[0]

            else:
                break
            
    G = xu.remove_selfloops(G)
    if print_flag:
        print(f"Done cleaning networkx graph with {paths_removed} paths removed")
    if return_skeleton:
        return convert_graph_to_skeleton(G)
    else:
        return G
    
def combine_close_branch_points(skeleton=None,
                                combine_threshold = 700,
                               print_flag=False,
                                skeleton_branches=None,):
    """
    Purpose: To take a skeleton or graph and return a skelton/graph 
    where close branch points are combined
    
    
    Example Code of how could get the orders: 
    # How could potentially get the edge order we wanted
    endpoint_neighbors_to_order_map = dict()
    for k in endpoint_neighbors:
        total_orders = []
        total_orders_neighbors = []
        for j in p:
            try:
                total_orders.append(curr_sk_graph[j][k]["order"])
                total_orders_neighbors.append(j)
            except:
                pass
        order_index = np.argmin(total_orders)
        endpoint_neighbors_to_order_map[(k,total_orders_neighbors[order_index])] = total_orders[order_index]
    endpoint_neighbors_to_order_map
    
    
    
    Ex: 
    sk = reload(sk)
    from datasci_tools import numpy_utils as nu
    nu = reload(nu)
    branch_skeleton_data_cleaned = []
    for i,curr_sk in enumerate(branch_skeleton_data):
        print(f"\n----- Working on skeleton {i} ---------")
        new_sk = sk.combine_close_branch_points(curr_sk,print_flag=True)
        print(f"Original Sk = {curr_sk.shape}, Cleaned Sk = {new_sk.shape}")
        branch_skeleton_data_cleaned.append(new_sk)
        
        
        
    """
    print(f"combine_threshold = {combine_threshold}")
    
    debug_time = False
    combine_close_time = time.time()
    
    branches_flag = False    
    if not skeleton_branches is None:
        #Create an array that maps the branch idx to the endpoints and make a copy
        branch_idx_to_endpoints = np.array([find_branch_endpoints(k) for k in skeleton_branches])
        branch_idx_to_endpoints_original = copy.deepcopy(branch_idx_to_endpoints)
        branch_keep_idx = np.arange(0,len(skeleton_branches))
        
        skeleton = sk.stack_skeletons(skeleton_branches)
        branches_flag = True
    
    
    convert_back_to_skeleton = False
    #1) convert the skeleton to a graph
    
    if nu.is_array_like(skeleton):
        curr_sk_graph = sk.convert_skeleton_to_graph(skeleton)
        convert_back_to_skeleton=True
    else:
        curr_sk_graph = skeleton

    #2) Get all of the high degree nodes
    high_degree_nodes = np.array(xu.get_nodes_greater_or_equal_degree_k(curr_sk_graph,3))
    
    """
    Checked that thes high degree nodes were correctly retrieved
    high_degree_coordinates = xu.get_node_attributes(curr_sk_graph,node_list = high_degree_nodes)
    sk.graph_skeleton_and_mesh(other_skeletons=[curr_sk],
                              other_scatter=[high_degree_coordinates])

    """
    
    #3) Get paths between all of them high degree nodes
    valid_paths = []
    valid_path_lengths = []
    for s in high_degree_nodes:
        degree_copy = high_degree_nodes[high_degree_nodes != s]

        for t in degree_copy:
            try:
                path_len, path = nx.single_source_dijkstra(curr_sk_graph,source = s,target=t,cutoff=combine_threshold)
            except:
                continue
            else: #a valid path was found
                degree_no_endpoints = degree_copy[degree_copy != t]

                if len(set(degree_no_endpoints).intersection(set(path))) > 0:
                    continue
                else:
                    match_path=False
                    for v_p in valid_paths:
                        if set(v_p) == set(path):
                            match_path = True
                            break
                    if not match_path:
                        valid_paths.append(np.array(list(path)))
                        valid_path_lengths.append(path_len)
                        
                    
                    
#                     sorted_path = np.sort(path)
#                     try:
                        
                        
#                         if len(nu.matching_rows(valid_paths,sorted_path)) > 0:
#                             continue
#                         else:
#                             valid_paths.append(sorted_path)
#                     except:
#                         print(f"valid_paths = {valid_paths}")
#                         print(f"sorted_path = {sorted_path}")
#                         print(f"nu.matching_rows(valid_paths,sorted_path) = {nu.matching_rows(valid_paths,sorted_path)}")
#                         raise Exception()

    if print_flag:
        print(f"Found {len(valid_paths)} valid paths to replace")
        print(f"valid_paths = {(valid_paths)}")
        print(f"valid_path_lengths = {valid_path_lengths}")
                     
    if debug_time:
        print(f"Finding all paths = { time.time() - combine_close_time}")
        combine_close_time = time.time()
        
    if len(valid_paths) == 0:
        if print_flag:
            print("No valid paths found so just returning the original")
        if branches_flag:
            skeleton_branches,branch_keep_idx
        else:
            return skeleton
    
    # Need to combine paths if there is any overlapping:
    valid_paths

    """
    # --------------If there were valid paths found -------------
    
    
    """
    curr_sk_graph_cp = copy.deepcopy(curr_sk_graph)
    if debug_time:
        print(f"Copying graph= { time.time() - combine_close_time}")
        combine_close_time = time.time()
        
    print(f"length of Graph = {len(curr_sk_graph_cp)}")
    for p_idx,p in enumerate(valid_paths):
        
        
        path_degrees = xu.get_node_degree(curr_sk_graph_cp,p)
        if print_flag:
            print(f"Working on path {p}")
            print(f"path_degrees = {path_degrees}")
        
        p_end_nodes = p[[0,-1]]
        #get endpoint coordinates
        path_coordinates = xu.get_node_attributes(curr_sk_graph_cp,node_list=p)
        end_coordinates = np.array([path_coordinates[0],path_coordinates[-1]]).reshape(-1,3)
        
        if debug_time:
            print(f"Getting coordinates = { time.time() - combine_close_time}")
            combine_close_time = time.time()
        
#         print(f"end_coordinates = {end_coordinates}")
#         print(f"branch_idx_to_endpoints = {branch_idx_to_endpoints}")
#         print(f"branch_idx_to_endpoints.shape = {branch_idx_to_endpoints.shape}")


        debug = False
        if debug:
            end_coordinates_try_2 = xu.get_node_attributes(curr_sk_graph_cp,node_list=p_end_nodes)
            print(f"end_coordinates = {end_coordinates}")
            print(f"end_coordinates_try_2 = {end_coordinates_try_2}")
            print(f"branch_idx_to_endpoints = {branch_idx_to_endpoints}")
            
        if branches_flag:
            #find the branch_idx with the found endpoints (if multiple then continue)
            branch_idxs = nu.find_matching_endpoints_row(branch_idx_to_endpoints,end_coordinates)
        
            if len(branch_idxs)>1:
                continue
            elif len(branch_idxs) == 0:
                raise Exception("No matching endpoints for branch")
            else:
                branch_position_to_delete = branch_idxs[0]
        
        if debug_time:
            print(f"Finding matching endpoints = { time.time() - combine_close_time}")
            combine_close_time = time.time()
    
        #get the coordinates of the path and average them for new node
        average_endpoints = np.mean(path_coordinates,axis=0)
        
        #replace the old end nodes with the new one
        curr_sk_graph_cp,new_node_id = xu.add_new_coordinate_node(curr_sk_graph_cp,node_coordinate=average_endpoints,replace_nodes=p_end_nodes)
        
        if debug_time:
            print(f"Adding new coordinate node = { time.time() - combine_close_time}")
            combine_close_time = time.time()
        
        #go through and change all remaining paths to now include the new combined node id
        for p_idx_curr in range(p_idx+1,len(valid_paths)):
            if debug:
                print(f"valid_paths[p_idx_curr] = {valid_paths[p_idx_curr]}")
                print(f"p_end_nodes = {p_end_nodes}")
                print(f"new_node_id = {new_node_id}")
                print(f"(valid_paths[p_idx_curr]==p_end_nodes[0]) | (valid_paths[p_idx_curr]==p_end_nodes[1]) = {(valid_paths[p_idx_curr]==p_end_nodes[0]) | (valid_paths[p_idx_curr]==p_end_nodes[1])}")
                
                
            valid_paths[p_idx_curr][(valid_paths[p_idx_curr]==p_end_nodes[0]) | (valid_paths[p_idx_curr]==p_end_nodes[1])] = new_node_id
        
        if debug_time:
            print(f"Changing all remaining paths = { time.time() - combine_close_time}")
            combine_close_time = time.time()
        
        if branches_flag:
            #delete the branch id from the index
            branch_idx_to_endpoints = np.delete(branch_idx_to_endpoints, branch_position_to_delete, 0)
            branch_keep_idx = np.delete(branch_keep_idx,branch_position_to_delete)



            #go through and replace and of the endpoints list that were the old endpoints now with the new one
            match_1 = (branch_idx_to_endpoints.reshape(-1,3) == end_coordinates[0]).all(axis=1).reshape(-1,2)
            match_2 = (branch_idx_to_endpoints.reshape(-1,3) == end_coordinates[1]).all(axis=1).reshape(-1,2)
            replace_mask = match_1 | match_2
            branch_idx_to_endpoints[replace_mask] = average_endpoints
            
        if debug_time:
            print(f"Replacing branch index = { time.time() - combine_close_time}")
            combine_close_time = time.time()
        
        #delete the nodes that were on the path
        curr_sk_graph_cp.remove_nodes_from(p)
        
        if debug_time:
            print(f"Removing nodes = { time.time() - combine_close_time}")
            combine_close_time = time.time()
        
    
    
    if branches_flag:
        """
        1) Find the branches that were not filtered away
        2) Get the new and original endpoints of the filtered branches
        3) For all filtered branches:
           i) convert skeleton into a graph
           For each endpoint
               ii) send the original endpoint to get replaced by new one
           iii) convert back into skeleton and save

        """
        #1) Find the branches that were not filtered away
        skeleton_branches = np.array(skeleton_branches)
        filtered_skeleton_branches = skeleton_branches[branch_keep_idx]
        
        #2) Get the new and original endpoints of the filtered branches
        original_endpoints = branch_idx_to_endpoints_original[branch_keep_idx]
        new_endpoints = branch_idx_to_endpoints
        
        #3) For all filtered branches:
        edited_skeleton_branches = []
        for f_sk,f_old_ep,f_new_ep in zip(filtered_skeleton_branches,original_endpoints,new_endpoints):
            f_sk_graph = convert_skeleton_to_graph(f_sk)
            for old_ep,new_ep in zip(f_old_ep,f_new_ep):
                f_sk_graph = xu.add_new_coordinate_node(f_sk_graph,
                                           node_coordinate=new_ep,
                                           replace_coordinates=old_ep,
                                          return_node_id=False)
            edited_skeleton_branches.append(convert_graph_to_skeleton(f_sk_graph))
        
        if debug_time:
            print(f"Filtering branches = { time.time() - combine_close_time}")
            combine_close_time = time.time()
        
        return edited_skeleton_branches,branch_keep_idx
        

    if convert_back_to_skeleton:
        return sk.convert_graph_to_skeleton(curr_sk_graph_cp)
    else:
        return curr_sk_graph_cp






def old_combine_close_branch_points(skeleton,
                                combine_threshold = 700,
                               print_flag=False):
    """
    Purpose: To take a skeleton or graph and return a skelton/graph 
    where close branch points are combined
    
    
    Example Code of how could get the orders: 
    # How could potentially get the edge order we wanted
    endpoint_neighbors_to_order_map = dict()
    for k in endpoint_neighbors:
        total_orders = []
        total_orders_neighbors = []
        for j in p:
            try:
                total_orders.append(curr_sk_graph[j][k]["order"])
                total_orders_neighbors.append(j)
            except:
                pass
        order_index = np.argmin(total_orders)
        endpoint_neighbors_to_order_map[(k,total_orders_neighbors[order_index])] = total_orders[order_index]
    endpoint_neighbors_to_order_map
    
    
    
    Ex: 
    sk = reload(sk)
    from datasci_tools import numpy_utils as nu
    nu = reload(nu)
    branch_skeleton_data_cleaned = []
    for i,curr_sk in enumerate(branch_skeleton_data):
        print(f"\n----- Working on skeleton {i} ---------")
        new_sk = sk.combine_close_branch_points(curr_sk,print_flag=True)
        print(f"Original Sk = {curr_sk.shape}, Cleaned Sk = {new_sk.shape}")
        branch_skeleton_data_cleaned.append(new_sk)
        
        
        
    """
    
    convert_back_to_skeleton = False
    #1) convert the skeleton to a graph
    
    if nu.is_array_like(skeleton):
        curr_sk_graph = sk.convert_skeleton_to_graph(skeleton)
        convert_back_to_skeleton=True
    else:
        curr_sk_graph = skeleton

    #2) Get all of the high degree nodes
    high_degree_nodes = np.array(xu.get_nodes_greater_or_equal_degree_k(curr_sk_graph,3))
    
    """
    Checked that thes high degree nodes were correctly retrieved
    high_degree_coordinates = xu.get_node_attributes(curr_sk_graph,node_list = high_degree_nodes)
    sk.graph_skeleton_and_mesh(other_skeletons=[curr_sk],
                              other_scatter=[high_degree_coordinates])

    """
    
    #3) Get paths between all of them high degree nodes
    valid_paths = []
    valid_path_endpoints = []
    valid_path_lengths = []
    for s in high_degree_nodes:
        degree_copy = high_degree_nodes[high_degree_nodes != s]

        for t in degree_copy:
            try:
                path_len, path = nx.single_source_dijkstra(curr_sk_graph,source = s,target=t,cutoff=combine_threshold)
            except:
                continue
            else: #a valid path was found
                degree_no_endpoints = degree_copy[degree_copy != t]

                if len(set(degree_no_endpoints).intersection(set(path))) > 0:
                    continue
                else:
                    match_path=False
                    for v_p in valid_paths:
                        if set(v_p) == set(path):
                            match_path = True
                            break
                    if not match_path:
                        valid_paths.append(np.sort(path))
                        valid_path_lengths.append(path_len)
                        valid_path_endpoints.append([s,t])
                        
                    
                    
#                     sorted_path = np.sort(path)
#                     try:
                        
                        
#                         if len(nu.matching_rows(valid_paths,sorted_path)) > 0:
#                             continue
#                         else:
#                             valid_paths.append(sorted_path)
#                     except:
#                         print(f"valid_paths = {valid_paths}")
#                         print(f"sorted_path = {sorted_path}")
#                         print(f"nu.matching_rows(valid_paths,sorted_path) = {nu.matching_rows(valid_paths,sorted_path)}")
#                         raise Exception()

    if print_flag:
        print(f"Found {len(valid_paths)} valid paths to replace")
        print(f"valid_paths = {(valid_paths)}")
        print(f"valid_path_lengths = {valid_path_lengths}")
                        
    if len(valid_paths) == 0:
        if print_flag:
            print("No valid paths found so just returning the original")
        return skeleton
    
    # Need to combine paths if there is any overlapping:
    valid_paths

    """
    # --------------If there were valid paths found -------------
    
    5) For the paths that past the thresholding:
    With a certain path
    a. take the 2 end high degree nodes and get all of their neighbors
    a2. get the coordinates of the endpoints and average them for new node
    b. Create a new node with all the neighbors and averaged coordinate
    c. replace all of the other paths computed (if they had the 2 end high degree nodes) replace with the new node ID
    d. Delete the old high degree ends and all nodes on the path
    Go to another path
    """
    curr_sk_graph_cp = copy.deepcopy(curr_sk_graph)
    for p,p_end_nodes in zip(valid_paths,valid_path_endpoints):
        print(f"Working on path {p}")
        
        #a. take the 2 end high degree nodes and get all of their neighbors
        endpoint_neighbors = np.unique(np.concatenate([xu.get_neighbors(curr_sk_graph_cp,k) for k in p]))
        endpoint_neighbors = np.setdiff1d(endpoint_neighbors,list(p))
        print(f"endpoint_neighbors = {endpoint_neighbors}")
        print(f"p_end_nodes = {p_end_nodes}")
        
        #a2. get the coordinates of the endpoints and average them for new node
        #endpoint_coordinates = np.vstack([xu.get_node_attributes(curr_sk_graph_cp,node_list=k) for k in p])
        path_coordinates = xu.get_node_attributes(curr_sk_graph_cp,node_list=p)
        end_coordinates = np.array([path_coordinates[0],path_coordinates[-1]]).reshape(-1,3)
        print(f"end_coordinates = {end_coordinates}")
#         print(f"endpoint_coordinates = {endpoint_coordinates}")
#         print(f"endpoint_coordinates_try_2 = {endpoint_coordinates_try_2}")
        average_endpoints = np.mean(path_coordinates,axis=0)
    
        #b. Create a new node with all the neighbors and averaged coordinate
        new_node_id = np.max(curr_sk_graph_cp.nodes()) + 1
        curr_sk_graph_cp.add_node(new_node_id,coordinates=average_endpoints)
        
        #c. replace all of the other paths computed (if they had the 2 end high degree nodes) replace with the new node ID
        print(f"endpoint_neighbors = {endpoint_neighbors}")
        curr_sk_graph_cp.add_weighted_edges_from([(new_node_id,k,
                    np.linalg.norm(curr_sk_graph_cp.nodes[k]["coordinates"] - average_endpoints)) for k in endpoint_neighbors])
        
    #d. Delete the old high degree ends and all nodes on the path
    concat_valid_paths = np.unique(np.concatenate(valid_paths))
    print(f"Concatenating all paths and deleting: {concat_valid_paths}")
    
    curr_sk_graph_cp.remove_nodes_from(concat_valid_paths)
    
    if convert_back_to_skeleton:
        return sk.convert_graph_to_skeleton(curr_sk_graph_cp)
    else:
        return curr_sk_graph_cp

    
    
    
# ---------------------- Full Skeletonization Function --------------------- #



try:
    from mesh_tools import meshlab
except:
    pass


def skeletonize_connected_branch_meshparty(mesh,
                                           segment_size = 100,
                                           invalidation_d = 1200,
                                           combine_close_skeleton_nodes_threshold = 0,
                                           filter_end_nodes = True,
                                           filter_end_node_length=4000,
                                           root = None,
                                           verbose=False,
                                           only_skeleton = False,
                                           plot = False,
                                           **kwargs
                                          ):
    """
    Purpose: To do the meshparty skeletonization and skeleton procedure
    
    Example: Applying skeletonization to an axon mesh
    
    axon_mesh = neuron_obj.axon_mesh
    meshparty_sk,sk_meshparty_obj = sk.skeletonize_connected_branch_meshparty(axon_mesh,
                                                            root=neuron_obj.axon_starting_coordinate,
                                                            invalidation_d=1200,
                                                            #combine_close_skeleton_nodes_threshold=20000,
                                                                              filter_end_node_length=3000,
                                                                             )

    ipvu.plot_objects(axon_mesh,meshparty_sk)
    
    
    """

    fusion_time = time.time()

    # --------------- Part 3: Meshparty skeletonization and Decomposition ------------- #
    sk_meshparty_obj = m_sk.skeletonize_mesh_largest_component(mesh,
                                                            root=root,
                                                              filter_mesh=False,
                                                               invalidation_d=invalidation_d,
                                                              **kwargs)

    if verbose:
        print(f"meshparty_segment_size = {meshparty_segment_size}")



    new_skeleton = m_sk.skeleton_obj_to_branches(sk_meshparty_obj,
                                                          mesh = mesh,
                                                          meshparty_segment_size=segment_size,
                                                          return_skeleton_only=True,
                                                 combine_close_skeleton_nodes_threshold=combine_close_skeleton_nodes_threshold,
                                                 filter_end_nodes=filter_end_nodes,
                                                 filter_end_node_length=filter_end_node_length,
                                                            **kwargs)

    if verbose:
        print(f"Time meshparty skeletonization: {time.time() - fusion_time }")
        
    if plot:
        print(f"Plotting the skeleton")
        ipvu.plot_objects(mesh,
                         new_skeleton)

    if only_skeleton:
        return new_skeleton,sk_meshparty_obj
    else:
        return new_skeleton



def skeletonize_connected_branch(current_mesh,
                        output_folder="./temp",
                        delete_temp_files=True,
                        name="None",
                        surface_reconstruction_size=50,
                        surface_reconstruction_width = None,#250,
                        poisson_stitch_size = 4000,
                        n_surface_downsampling = 1,
                        n_surface_samples=1000,
                        skeleton_print=False,
                        mesh_subtraction_distance_threshold=mesh_subtraction_distance_threshold_default,
                        mesh_subtraction_buffer=mesh_subtraction_buffer_default,#50,
                        max_stitch_distance =10_000,# 18000,
                        current_min_edge = 75,
                        close_holes=True,
                        limb_name=None,
                        use_surface_after_CGAL = True,
                        remove_cycles=True,
                        connectivity="vertices",
                        verbose=False,
                        remove_mesh_interior_face_threshold=0,
                        error_on_bad_cgal_return=False,
                        ):
    """
    Purpose: To take a mesh and construct a full skeleton of it
    (Assuming the Soma is already extracted)
    
    1) Poisson Surface Reconstruction
    2) CGAL skeletonization of all signfiicant pieces 
        (if above certain size ! threshold) 
                --> if not skip straight to surface skeletonization
    3) Using CGAL skeleton, find the leftover mesh not skeletonized
    4) Do surface reconstruction on the parts that are left over
    - with some downsampling
    5) Stitch the skeleton 
    """
    debug = False
    skeleton_print = False
    export_leftover_debug = False
    
        
    if use_surface_after_CGAL:
        restriction_threshold = surface_reconstruction_size
    else:
        restriction_threshold = poisson_stitch_size
    if verbose:
        print(f"restriction_threshold = {restriction_threshold}")
    
    if debug:
        print(f"connectivity = {connectivity}")
    
    
    print(f"inside skeletonize_connected_branch and use_surface_after_CGAL={use_surface_after_CGAL}, surface_reconstruction_size={surface_reconstruction_size}")
    #check that the mesh is all one piece

    if remove_mesh_interior_face_threshold > 0:
        if verbose:
            print(f"Removing interior meshes greater than {remove_mesh_interior_face_threshold}")
        current_mesh = tu.remove_mesh_interior(current_mesh,size_threshold_to_remove=remove_mesh_interior_face_threshold,
                        verbose = True)
    
    #splitting the mesh into significant pieces
    current_mesh_splits_pre = np.array(split_significant_pieces(current_mesh,
                               significance_threshold=1,
                                                  connectivity=connectivity))
    
    face_lens = np.array([len(k.faces) for k in current_mesh_splits_pre]) 
    keep_idx = np.where(face_lens > restriction_threshold)[0]
    
    print(f"keep_idx = {keep_idx}")
    
    if len(keep_idx) <= 0:
        keep_idx=[0]
        
    current_mesh_splits = current_mesh_splits_pre[keep_idx]
    
    
#     if len(current_mesh_splits) > 1:
#         print(f"The mesh passed has {len(current_mesh_splits)} pieces so just taking the largest one {current_mesh_splits[0]}")
#         current_mesh = current_mesh_splits[0]

    # check the size of the branch and if small enough then just do
    # Surface Skeletonization
    
    skeleton_ready_for_stitching_total = []
    for mesh_idx,current_mesh in enumerate(current_mesh_splits):
        
        
        if len(current_mesh.faces) < surface_reconstruction_size:
            #do a surface skeletonization
            print("Doing skeleton surface reconstruction")
#             surf_sk = generate_surface_skeleton(current_mesh.vertices,
#                                         current_mesh.faces,
#                                         surface_samples=n_surface_samples,
#                                                  n_surface_downsampling=n_surface_downsampling )

            surf_sk = m_sk.skeletonize(current_mesh)
            skeleton_ready_for_stitching =  sk.stack_skeletons([surf_sk])
        else:

            #if can't simply do a surface skeletonization then 
            #use cgal method that requires temp folder

            if type(output_folder) != type(Path()):
                output_folder = Path(str(output_folder))
                output_folder.mkdir(parents=True,exist_ok=True)

            # CGAL Step 1: Do Poisson Surface Reconstruction
            Poisson_obj = meshlab.Poisson(output_folder,overwrite=True)


            skeleton_start = time.time()
            print("     Starting Screened Poisson")
            new_mesh,output_subprocess_obj = Poisson_obj(   
                                        vertices=current_mesh.vertices,
                                         faces=current_mesh.faces,
                                        mesh_filename=name + ".off",
                                         return_mesh=True,
                                         delete_temp_files=delete_temp_files,
                                        )

            if close_holes: 
                print("Using the close holes feature")

                new_mesh = tu.fill_holes(new_mesh)
                """
                Old Way 

                FillHoles_obj = meshlab.FillHoles(output_folder,overwrite=True)

                new_mesh,output_subprocess_obj = FillHoles_obj(   
                                                    vertices=new_mesh.vertices,
                                                     faces=new_mesh.faces,
                                                     return_mesh=True,
                                                     delete_temp_files=delete_temp_files,
                                                    )
                """


            print(f"-----Time for Screened Poisson= {time.time()-skeleton_start}")


            #2) Filter away for largest_poisson_piece:

            #the connectivity her HAS to edges or could send a non-manifold mesh to calcification
            mesh_pieces = split_significant_pieces(new_mesh,
                                                significance_threshold=restriction_threshold,
                                                  connectivity="edges")

            if skeleton_print:
                print(f"Signifiant mesh pieces of {surface_reconstruction_size} size "
                     f"after poisson = {len(mesh_pieces)}")
            skeleton_ready_for_stitching = np.array([])
            skeleton_files = [] # to be erased later on if need be
            if len(mesh_pieces) <= 0:
                if skeleton_print:
                    print("No signficant skeleton pieces so just doing surface skeletonization")
                # do surface skeletonization on all of the pieces
                surface_mesh_pieces = split_significant_pieces(new_mesh,
                                                significance_threshold=2,
                                                              connectivity=connectivity)

                #get the skeletons for all those pieces
#                 current_mesh_skeleton_list = [
#                     generate_surface_skeleton(p.vertices,
#                                         p.faces,
#                                         surface_samples=n_surface_samples,
#                                         n_surface_downsampling=n_surface_downsampling )
#                     for p in surface_mesh_pieces
#                 ]
                
                current_mesh_skeleton_list = [
                    m_sk.skeletonize(p)
                    for p in surface_mesh_pieces
                ]
                

                skeleton_ready_for_stitching = stack_skeletons(current_mesh_skeleton_list)

                #will stitch them together later
            else: #if there are parts that can do the cgal skeletonization
                skeleton_start = time.time()
                print(f"mesh_pieces = {mesh_pieces}")
                print("     Starting Calcification (Changed back where stitches large poissons)")
                for zz,piece in enumerate(mesh_pieces):

                    """ Old way that didnt' check for manifoldness

                    current_mesh_path = output_folder / f"{name}_{zz}"
                    #if skeleton_print:

                    print(f"current_mesh_path = {current_mesh_path}")
                    written_path = write_neuron_off(piece,current_mesh_path)

                    returned_value, sk_file_name = calcification(written_path,
                                                                   min_edge_length = current_min_edge)
                    """
                    #print(f"Path sending to calcification = {written_path[:-4]}")

                    returned_value, sk_file_name = sk.skeleton_cgal(mesh=piece,
                                                                    filepath= output_folder / f"{name}_{zz}",
                                         return_skeleton_file_path_and_status=True,
                                    min_edge_length = current_min_edge)

                    if skeleton_print:
                        print(f"returned_value = {returned_value}")
                        print(f"sk_file_name = {sk_file_name}")
                        
                        
                    if error_on_bad_cgal_return and returned_value != 0:
                        su.compressed_pickle(piece,f"{sk_file_name}_not_sk")
                        print(f"******\n\n\n\n\n\n\n\n\n\n\n\n   SKELETON NOT FORMED \n\n\n\n\n\n\n\n\n\n\n\n")
                        raise Exception(f"Bad skeleton cgal return value = {sk_file_name} for {sk_file_name} ")
                        
                    #print(f"Time for skeletonizatin = {time.time() - skeleton_start}")
                    skeleton_files.append(sk_file_name)

                if skeleton_print:
                    print(f"-----Time for Running Calcification = {time.time()-skeleton_start}")

                #collect the skeletons and subtract from the mesh

                significant_poisson_skeleton = read_skeleton_edges_coordinates(skeleton_files)
                
                if debug:
                    print(f"skeleton_files = {skeleton_files}")
                    print(f"significant_poisson_skeleton = {significant_poisson_skeleton}")
                    su.compressed_pickle(significant_poisson_skeleton,f"significant_poisson_skeleton_{mesh_idx}")


                if len(significant_poisson_skeleton) == 0:
    #                 if not use_surface_after_CGAL:
    #                     surf_sk = generate_surface_skeleton(m.vertices,
    #                                                m.faces,
    #                                                surface_samples=n_surface_samples,
    #                                     n_surface_downsampling=n_surface_downsampling )
    #                     return surf_sk
    #                     raise gu.CGAL_skel_error(f"No CGAL skeleton was generated when the {use_surface_after_CGAL} flag was set")

                    """------------------ 1 / 2 /2021 Addition ------------------------"""

                    print("No recorded skeleton so skipping"
                         " to meshparty skeletonization")

                    skeleton_ready_for_stitching = sk.skeletonize_connected_branch_meshparty(current_mesh)[0]

                    #leftover_meshes_sig = [current_mesh]
                else:
                    if remove_cycles:
                        significant_poisson_skeleton = remove_cycles_from_skeleton(significant_poisson_skeleton)


                    if use_surface_after_CGAL:
                        boolean_significance_threshold=5

                        print(f"Before mesh subtraction number of skeleton edges = {significant_poisson_skeleton.shape[0]+1}")
                        mesh_pieces_leftover =  mesh_subtraction_by_skeleton(current_mesh,
                                                                    significant_poisson_skeleton,
                                                                    buffer=mesh_subtraction_buffer,
                                                                    distance_threshold=mesh_subtraction_distance_threshold,
                                                                    #significance_threshold=boolean_significance_threshold,
                                                                   )

                        # *****adding another significance threshold*****

                        leftover_meshes_sig = [k for k in mesh_pieces_leftover if len(k.faces) > surface_reconstruction_size]
                    else:
                        leftover_meshes_sig = []

                    if debug:
                        su.compressed_pickle(leftover_meshes_sig,f"leftover_meshes_sig_before_filter_{mesh_idx}")
                    #want to filter these significant pieces for only those below a certain width
                    
                    """
                    This part has been deprecated because would not help catch those segments where the 
                    cgal skeletonization failed
                    
                    
                    """
                    surface_reconstruction_width = None
                    if not surface_reconstruction_width is None and len(leftover_meshes_sig) > 0:
                        if skeleton_print:
                            print("USING THE SDF WIDTHS TO FILTER SURFACE SKELETONS")
                            print(f"leftover_meshes_sig before filtering = {len(leftover_meshes_sig)}")
                        leftover_meshes_sig_new = []
                        from trimesh.ray import ray_pyembree
                        ray_inter = ray_pyembree.RayMeshIntersector(current_mesh)
                        """
                        Pseudocode:
                        For each leftover significant mesh
                        1) Map the leftover piece back to the original face
                        2) Get the widths fo the piece
                        3) get the median of the non-zero values
                        4) if greater than the surface_reconstruction_width then add to list

                        """
                        for lm in leftover_meshes_sig:
                            face_indices_leftover_0 = tu.original_mesh_faces_map(current_mesh,lm)
                            curr_width_distances = tu.ray_trace_distance(mesh=current_mesh,
                              face_inds=face_indices_leftover_0,
                                                     ray_inter=ray_inter
                            )
                            filtered_widths = curr_width_distances[curr_width_distances>0]
                            if len(filtered_widths) == 0:
                                continue
                            if np.mean(filtered_widths) < surface_reconstruction_width:
                                leftover_meshes_sig_new.append(lm)

                        leftover_meshes_sig = leftover_meshes_sig_new
                        if skeleton_print:
                            print(f"leftover_meshes_sig AFTER filtering = {len(leftover_meshes_sig)}")

                    if debug:
                        su.compressed_pickle(leftover_meshes_sig,f"leftover_meshes_sig_after_filter_{mesh_idx}")
                    leftover_meshes = combine_meshes(leftover_meshes_sig)
                
                    print(f"len(leftover_meshes_sig) = {leftover_meshes_sig}")

                    if skeleton_print:
                        if export_leftover_debug:
                            for zz,curr_m in enumerate(leftover_meshes_sig):
                                debug_folder_name = "leftover_test"
                                Path(f"./{debug_folder_name}").mkdir(exist_ok=True)
                                tu.write_neuron_off(curr_m,f"./{debug_folder_name}/limb_{limb_name}_{zz}.off")

                    leftover_meshes_sig_surf_sk = []
                    for m in tqdm(leftover_meshes_sig):
    #                     surf_sk = generate_surface_skeleton(m.vertices,
    #                                                    m.faces,
    #                                                    surface_samples=n_surface_samples,
    #                                         n_surface_downsampling=n_surface_downsampling )
                        surf_sk = m_sk.skeletonize(m)
                        if len(surf_sk) > 0:
                            leftover_meshes_sig_surf_sk.append(surf_sk)
                    leftovers_stacked = stack_skeletons(leftover_meshes_sig_surf_sk)
                    #print(f"significant_poisson_skeleton = {significant_poisson_skeleton}")
                    #print(f"leftover_meshes_sig_surf_sk = {leftover_meshes_sig_surf_sk}")
                    if debug:
                        su.compressed_pickle(leftovers_stacked,f"leftovers_stacked_{mesh_idx}")


                    print(f"STacking the leftover and significant poisson skeleton")
                    skeleton_ready_for_stitching = stack_skeletons([significant_poisson_skeleton,leftovers_stacked])
                
        skeleton_ready_for_stitching_total.append(skeleton_ready_for_stitching)
    
    if debug:
        su.compressed_pickle(skeleton_ready_for_stitching_total,"skeleton_ready_for_stitching_total")
            
    skeleton_ready_for_stitching = sk.stack_skeletons([skeleton_ready_for_stitching_total])
    #now want to stitch together whether generated from 
    if skeleton_print:
        print(f"After cgal process the un-stitched skeleton has shape {skeleton_ready_for_stitching.shape}")
        #su.compressed_pickle(skeleton_ready_for_stitching,"sk_before_stitiching")

    # Now want to always do the skeleton stitching
    #if use_surface_after_CGAL:
    stitched_skeletons_full = stitch_skeleton(
                                              skeleton_ready_for_stitching,
                                              max_stitch_distance=max_stitch_distance,
                                              stitch_print = False,
                                              main_mesh = []
                                            )
    #else:
    #stitched_skeletons_full = skeleton_ready_for_stitching

    #stitched_skeletons_full_cleaned = clean_skeleton(stitched_skeletons_full)

    # erase the skeleton files if need to be
    if delete_temp_files:
        for sk_fi in skeleton_files:
            if Path(sk_fi).exists():
                Path(sk_fi).unlink()

    # if created temp folder then erase if empty
    if str(output_folder.absolute()) == str(Path("./temp").absolute()):
        print("The process was using a temp folder")
        if len(list(output_folder.iterdir())) == 0:
            print("Temp folder was empty so deleting it")
            if output_folder.exists():
                rmtree(str(output_folder.absolute()))

    return stitched_skeletons_full
    
def soma_skeleton_stitching(total_soma_skeletons,soma_mesh):
    """
    Purpose: Will stitch together the meshes that are touching
    the soma 
    
    Pseudocode: 
    1) Compute the soma mesh center point
    2) For meshes that were originally connected to soma
    a. Find the closest skeletal point to soma center
    b. Add an edge from closest point to soma center
    3) Then do stitching algorithm on all of remaining disconnected
        skeletons
    
    
    """
    # 1) Compute the soma mesh center point
    soma_center = np.mean(soma_mesh.vertices,axis=0)
    
    soma_connecting_skeleton = []
    for skel in total_soma_skeletons:
        #get the unique vertex points
        unique_skeleton_nodes = np.unique(skel.reshape(-1,3),axis=0)
        
        # a. Find the closest skeletal point to soma center
        # b. Add an edge from closest point to soma center
        mesh_tree = KDTree(unique_skeleton_nodes)
        distances,closest_node = mesh_tree.query(soma_center.reshape(-1,3))
        closest_skeleton_vert = unique_skeleton_nodes[closest_node[np.argmin(distances)]]
        soma_connecting_skeleton.append(np.array([closest_skeleton_vert,soma_center]).reshape(-1,2,3))
    
    print(f"soma_connecting_skeleton[0].shape = {soma_connecting_skeleton[0].shape}")
    print(f"total_soma_skeletons[0].shape = {total_soma_skeletons[0].shape}")
    # stith all of the ekeletons together
    soma_stitched_sk = stack_skeletons(total_soma_skeletons + soma_connecting_skeleton)
    
    return soma_stitched_sk



# ------ Functions to help with the compartment ---- #
# converted into a function

def get_ordered_branch_nodes_coordinates(skeleton_graph,nodes=False,coordinates=True):

    """Purpose: want to get ordered skeleton coordinates:
    1) get both end nodes
    2) count shortest path between them (to get order)
    3) then use the get node attributes function

    """
    #find the 2 endpoints:
    sk_graph_clean = xu.remove_selfloops(skeleton_graph)
    enpoints = [k for k,v in dict(sk_graph_clean.degree).items() if v == 1]
    #print(f"enpoints= {enpoints}")
    if len(enpoints) != 2:
        su.compressed_pickle(skeleton_graph,"skeleton_graph")
        nx.draw(sk_graph_clean)
        print(f"sk_graph_clean.degree = {dict(sk_graph_clean.degree).items() }")
        nx.draw(skeleton_graph,with_labels=True)
        plt.show()
        raise Exception("The number of endpoints was not 2 for a branch")

    # gets the shortest path
    shortest_path = nx.shortest_path(sk_graph_clean,enpoints[0],enpoints[1],weight="weight")
    #print(f"shortest_path = {shortest_path}")

    skeleton_node_coordinates = xu.get_node_attributes(skeleton_graph,node_list=shortest_path)
    #print(f"skeleton_node_coordinates = {skeleton_node_coordinates}")

    if nodes==False and coordinates==True:
        return skeleton_node_coordinates
    elif nodes==True and coordinates==False:
        return shortest_path
    elif nodes==True and coordinates==True:
        return shortest_path,skeleton_node_coordinates
    else:
        raise Exception("neither nodes or coordinates set to return from get_ordered_branch_nodes_coordinates")


def split_skeleton_into_edges(current_skeleton):
    """
    Purpose: Will split a skeleton into a list of skeletons where each skeleton is just
    one previous edge of the skeleton before
    
    Example of how to use: 
    
    returned_split = split_skeleton_into_edges(downsampled_skeleton)
    print(len(returned_split), downsampled_skeleton.shape)
    returned_split
    
    """
    
    total_skeletons = [k for k in current_skeleton]
    return total_skeletons
    
        
def decompose_skeleton_to_branches(current_skeleton,
                                   max_branch_distance=-1,
                                  skip_branch_threshold=20000,
                                  return_indices=False,
                                  remove_cycles=True):
    """
    Example of how to run: 
    elephant_skeleton = sk.read_skeleton_edges_coordinates("../test_neurons/elephant_skeleton.cgal")
    elephant_skeleton_branches = sk.decompose_skeleton_to_branches(elephant_skeleton)
    sk.graph_skeleton_and_mesh(other_skeletons=[sk.stack_skeletons(elephant_skeleton_branches)])
    
    ***** Future error possibly: there could be issues in the future where had triangles of degree > 2 in your skeleton******
    """
    
    if type(current_skeleton) not in [type(nx.Graph()),type(xu.GraphOrderedEdges())]:
        el_sk_graph = convert_skeleton_to_graph(current_skeleton)
    else:
        el_sk_graph = current_skeleton
    
    
    
    el_sk_graph = xu.remove_selfloops(el_sk_graph)
    degree_dict = dict(el_sk_graph.degree)
    branch_nodes = [k for k,v in degree_dict.items() if v <= 2]
    seperated_branch_graph = el_sk_graph.subgraph(branch_nodes)
    
    branch_skeletons = []
    branch_skeleton_indices = []
    max_cycle_iterations = 1000

    seperated_branch_graph_comp = list(nx.connected_components(seperated_branch_graph))
    # now add back the nodes that were missing for each branch and collect all of the skeletons
    for curr_branch in seperated_branch_graph_comp:
        """
        new method of decomposing that avoids keeping loops (but will error if getting rid of large loop)
        
        # old way 
        all_neighbors = [xu.get_neighbors(el_sk_graph,n) for n in curr_branch] 
        all_neighbors.append(list(curr_branch))
        total_neighbors = np.unique(np.hstack(all_neighbors))
        branch_subgraph = el_sk_graph.subgraph(total_neighbors)
        branch_skeletons.append(sk.convert_graph_to_skeleton(branch_subgraph))
        
        New method: only if the two endpoints are connected together, then we give 
        back a skeleton just of those endpoints (so this will skip the current branch alltogether)
        --> but if skipping a branch that is too big then error
        - else do the old method
        
        """
        
        all_neighbors = [xu.get_neighbors(el_sk_graph,n) for n in curr_branch] 
        all_neighbors.append(list(curr_branch))
        total_neighbors = np.unique(np.hstack(all_neighbors))
        
        #check to see if the t junctions are connected
        high_degree_neigh = [k for k in total_neighbors if degree_dict[k]>2]
        if len(high_degree_neigh) > 2:
            raise Exception("Too many high degree nodes found in branch of decomposition")
        if len(high_degree_neigh) == 2:
            if high_degree_neigh[1] in xu.get_neighbors(el_sk_graph,high_degree_neigh[0]):
                print("high-degree endpoints were connected so just using that connection")
                
                #check that what skipping isn't too big
                print(f"curr_branch = {curr_branch}")
                if len(curr_branch) >= 2:
                    branch_subgraph = el_sk_graph.subgraph(list(curr_branch))
                    skip_distance = sk.calculate_skeleton_distance( sk.convert_graph_to_skeleton(branch_subgraph))
                    if  skip_distance > skip_branch_threshold:
                        raise Exception(f"Branch that we are skipping is too large with skip distance: {skip_distance}")

                #save this for later when add back all high degree branches that are connected
#                 branch_skeletons.append((xu.get_node_attributes(el_sk_graph,attribute_name="coordinates"
#                                                                 ,node_list=high_degree_neigh,
#                                                                return_array=True)).reshape(1,2,3))
                continue

        
        
        
        
        branch_subgraph = el_sk_graph.subgraph(total_neighbors)
        
        #12/17 NO LONGER attempting to eliminate any cycles
        if remove_cycles:
            branch_subgraph = xu.remove_cycle(branch_subgraph)
        
        branch_skeletons.append(sk.convert_graph_to_skeleton(branch_subgraph))
        branch_skeleton_indices.append(list(branch_subgraph.nodes()))
        
    #observation: seem to be losing branches that have two high degree nodes connected to each other and no other loop around it
        
    #add back all of the high degree branches that form subgraphs
    high_degree_branch_nodes = [k for k,v in degree_dict.items() if v > 2]
    seperated_branch_graph = el_sk_graph.subgraph(high_degree_branch_nodes)
    #get the connected components
    high_degree_conn_comp = nx.connected_components(seperated_branch_graph)
    
    """
    Here is where need to make a decision about what to do with high degree nodes: 
    I say just split all of the edges just into branches and then problem is solved (1 edge branches)
    """
    
    
    for c_comp in high_degree_conn_comp:
        if len(c_comp) >= 2:
            #add the subgraph to the branches
            branch_subgraph = el_sk_graph.subgraph(list(c_comp))
            branch_subgraph = nx. nx.minimum_spanning_tree(branch_subgraph)
            #and constant loop that check for cycle in this complexand if there is one then delete a random edge from the cycle

            """
            Might have to add in more checks for more complicated high degree node complexes

            """
            
            #new method that will delete any cycles might find in the branches
        
            #12/17 NO LONGER attempting to eliminate any cycles
            if remove_cycles:
                branch_subgraph = xu.remove_cycle(branch_subgraph)
        
            high_degree_branch_complex = sk.convert_graph_to_skeleton(branch_subgraph)
            seperated_high_degree_edges = split_skeleton_into_edges(high_degree_branch_complex)
                    
            #branch_skeletons.append(sk.convert_graph_to_skeleton(branch_subgraph)) #old way
            branch_skeletons += seperated_high_degree_edges
            branch_skeleton_indices += list(branch_subgraph.edges())
            
            
            #check if there every was a cycle: 
            

    for br in branch_skeletons:
        try:
            #print("Testing for cycle")
            edges_in_cycle = nx.find_cycle(sk.convert_skeleton_to_graph(br))
        except:
            pass
        else:
            raise Exception("There was a cycle found in the branch subgraph")
        
    branch_skeletons = [b.reshape(-1,2,3) for b in branch_skeletons]
    
    if return_indices:
        return branch_skeletons,branch_skeleton_indices
    else:
        return branch_skeletons

def convert_branch_graph_to_skeleton(skeleton_graph):
    """ Want an ordered skeleton that is only a line 
    Pseudocode: 
    1) Get the ordered node coordinates
    2) Create an edge array like [(0,1),(1,2).... (n_nodes-1,n_nodes)]
    3) index the edges intot he node coordinates and return
    """
    skeleton_node_coordinates = get_ordered_branch_nodes_coordinates(skeleton_graph)
    #print(f"skeleton_node_coordinates.shape = {skeleton_node_coordinates.shape}")
    s = np.arange(0,len(skeleton_node_coordinates)).T
    edges = np.vstack([s[:-1],s[1:]]).T
    return skeleton_node_coordinates[edges]    


# def divide_branch(curr_branch_skeleton,
#                            segment_width):
#     """
#     When wanting to divide up one branch into multiple branches that 
#     don't exceed a certain size
    
#     Pseudocode: 
#     1) Resize the skee
    
#     """

def resize_skeleton_branch(
                            curr_branch_skeleton,
                           segment_width = 0,
                          n_segments = 0,
                            print_flag=False,
                          try_order_skeleton_from_original=True):
    
    """
    sk = reload(sk)
    cleaned_skeleton = sk.resize_skeleton_branch(curr_branch_skeleton,segment_width=1000)

    sk.graph_skeleton_and_mesh(other_meshes=[curr_branch_mesh],
                              other_skeletons=[cleaned_skeleton])
    """
    
    if segment_width<=0 and n_segments<=0:
        raise Exception("Both segment_width and n_segments are non-positive")
    
    
    #curr_branch_nodes_coordinates = np.vstack([curr_branch_skeleton[:,0,:].reshape(-1,3),curr_branch_skeleton[-1,1,:].reshape(-1,3)])
    #print(f"curr_branch_nodes_coordinates = {curr_branch_nodes_coordinates}")  

    #final product of this is it gets a skeleton that goes in a line from one endpoint to the other 
    #(because skeleton can possibly be not ordered)
    skeleton_graph = sk.convert_skeleton_to_graph(curr_branch_skeleton)
    skeleton_node_coordinates = get_ordered_branch_nodes_coordinates(skeleton_graph)
    cleaned_skeleton = convert_branch_graph_to_skeleton(skeleton_graph)
    
    if print_flag:
        print(f"cleaned_skeleton size = {sk.calculate_skeleton_distance(cleaned_skeleton)}")

    # #already checked that these were good                 
    # print(f"skeleton_node_coordinates = {skeleton_node_coordinates}")
    # print(f"cleaned_skeleton = {cleaned_skeleton}")


    # gets the distance markers of how far have traveled from end node for each node
    seg_bins = np.hstack([np.array([0]),sk.calculate_skeleton_segment_distances(cleaned_skeleton,cumsum=True)])
    
    if print_flag:
        print(f"seg_bins= {seg_bins}")

    if n_segments > 0:
            segment_width = seg_bins[-1]/n_segments #sets the width to 
            if print_flag:
                print(f"segment_width = {segment_width}")
    else:
        if segment_width>seg_bins[-1]:
            #print("Skeletal width required was longer than the current skeleton so just returning the endpoints")
            return np.vstack([cleaned_skeleton[0][0],cleaned_skeleton[-1][-1]]).reshape(1,2,3)
    

    #gets the distance of each segment
    segment_widths = sk.calculate_skeleton_segment_distances(cleaned_skeleton,cumsum=False)
    
    if print_flag:
        print(f"segment_widths = {segment_widths}")
        
    #print(f"total_distance = {sk.calculate_skeleton_distance(cleaned_skeleton)}")

    n_full_segs = int(seg_bins[-1]/segment_width)
    new_seg_endpoints = np.arange(segment_width,segment_width*n_full_segs+0.000000001,segment_width)

    if new_seg_endpoints[-1] > seg_bins[-1]:
        if np.absolute(new_seg_endpoints[-1] - seg_bins[-1]) < 0.000001:
            new_seg_endpoints[-1] = seg_bins[-1]
        else:
            raise Exception("End of new_seg_endpoints is greater than original skeleton ")

    #accounts for the perfect fit
    if new_seg_endpoints[-1] == seg_bins[-1]:
        #print("exact match so eliminating last new bin")
        new_seg_endpoints = new_seg_endpoints[:-1] #remove the last one because will readd it back

    #print(f"seg_bins = {seg_bins}")
    #print(f"new_seg_endpoints = {new_seg_endpoints}")

    #getting the vertices

    """
    3) for each new segment endpoint, 
    a) calculate between which two existing skeleton segment end points it would exist
    (using just a distnace measurement from each end point to the next)
    b)calculate the coordinate that is a certain distance in middle based on which endpoints between

    new_vector * (new_seg_endpoint - lower_bin_distance)/seg_width + lower_bin_vector
    # """

    bin_indices = np.digitize(new_seg_endpoints, seg_bins)
    #print(f"bin_indices = {bin_indices}")
    # print(f"bin_indices = {bin_indices}")
    # print(f"seg_bins[bin_indices-1]= {seg_bins[bin_indices-1]}")
    # print(f"new_seg_endpoints - seg_bins[bin_indices-1] = {(new_seg_endpoints - seg_bins[bin_indices-1]).astype('int')}")
    #print(f"skeleton_node_coordinates (SHOULD BE ORDERED) = {skeleton_node_coordinates}")
    new_coordinates = (((skeleton_node_coordinates[bin_indices] - skeleton_node_coordinates[bin_indices-1])
                       *((new_seg_endpoints - seg_bins[bin_indices-1])/segment_widths[bin_indices-1]).reshape(-1,1)) + skeleton_node_coordinates[bin_indices-1])

    #print(f"new_coordinates = {new_coordinates.shape}")

    #add on the ending coordinates
    final_new_coordinates = np.vstack([skeleton_node_coordinates[0].reshape(-1,3),new_coordinates,skeleton_node_coordinates[-1].reshape(-1,3)])
    #print(f"final_new_coordinates = {final_new_coordinates.shape}")

    #make a new skeleton from the coordinates
    new_skeleton = np.stack((final_new_coordinates[:-1],final_new_coordinates[1:]),axis=1)
    if print_flag:
        print(f"new_skeleton = {new_skeleton.shape}")

        
    if try_order_skeleton_from_original:
        try:
            new_skeleton = sk.order_skeleton(new_skeleton,
                                        start_endpoint_coordinate = curr_branch_skeleton[0][0],
                                            error_on_non_start_coordinate=True)
        except:
            pass
    return new_skeleton


def skeleton_graph_nodes_to_group(skeleton_grpah):
    """
    Checks that no nodes in graph are in the same coordinates and need to be combined
    
    Example Use Case: 
    
    example_skeleton = current_mesh_data[0]["branch_skeletons"][0]
    skeleton_grpah = sk.convert_skeleton_to_graph(example_skeleton)
    limb_nodes_to_group = sk.skeleton_graph_nodes_to_group(skeleton_grpah)
    limb_nodes_to_group

    #decompose the skeleton and then recompose and see if any nodes to group
    decomposed_branches = sk.decompose_skeleton_to_branches(example_skeleton)
    decomposed_branches_stacked = sk.stack_skeletons(example_skeleton)
    restitched_decomposed_skeleton = sk.convert_graph_to_skeleton(sk.convert_skeleton_to_graph(decomposed_branches_stacked))
    sk.skeleton_graph_nodes_to_group(restitched_decomposed_skeleton)

    #shows that the restitched skeleton is still just one connected componet
    connected_components = nx.connected_components(sk.convert_skeleton_to_graph(decomposed_branches_stacked))
    len(list(connected_components))

    sk.graph_skeleton_and_mesh(other_skeletons = [restitched_decomposed_skeleton])
    
    
    """
    if type(skeleton_grpah)  not in [type(nx.Graph()),type(xu.GraphOrderedEdges())]:
        skeleton_grpah = convert_skeleton_to_graph(skeleton_grpah)
    #get all of the vertices
    coordinates = xu.get_node_attributes(skeleton_grpah,attribute_name="coordinates")
    #get the distances between coordinates
    distance_matrix = nu.get_coordinate_distance_matrix(coordinates)
    
    #great a graph out of the distance matrix with a value of 0
    nodes_to_combine = nx.from_edgelist(np.array(np.where(distance_matrix==0)).T)
    #clean graph for any self loops
    nodes_to_combine  = xu.remove_selfloops(nodes_to_combine)
    
    grouped_nodes = nx.connected_components(nodes_to_combine)
    nodes_to_group = [k for k in list(grouped_nodes) if len(k)>1]
    
    return nodes_to_group

def recompose_skeleton_from_branches(decomposed_branches):
    """
    Takes skeleton branches and stitches them back together without any repeating nodes
    """
    decomposed_branches_stacked = sk.stack_skeletons(decomposed_branches)
    restitched_decomposed_skeleton = sk.convert_graph_to_skeleton(sk.convert_skeleton_to_graph(decomposed_branches_stacked))
    return restitched_decomposed_skeleton

def clean_skeleton_with_decompose(distance_cleaned_skeleton):
    """
    Purpose: to eliminate the loops that are cleaned in the decompose process from the skeleton and then reconstruct
    Pseudocode: 
    1) decompose skeleton
    2) recompose skeleton (was checked that no nodes to recombine)
    
    """
    branches = decompose_skeleton_to_branches(distance_cleaned_skeleton)
    return recompose_skeleton_from_branches(branches)

def divide_branch(curr_branch_skeleton,
                            segment_width = 1000,
                           equal_width=True,
                           n_segments = 0):


    """
    When wanting to divide up one branch into multiple branches that 
    don't exceed a certain size

    Example of how to use: 
    
    sk = reload(sk)

    curr_index = 1
    ex_branch = total_branch_skeletons[curr_index]
    ex_mesh = total_branch_meshes[curr_index]
    # sk.graph_skeleton_and_mesh(other_skeletons=[ex_branch],
    #                           other_meshes=[ex_mesh])



    #there were empty arrays which is causing the error!
    returned_branches = sk.divide_branch(curr_branch_skeleton=ex_branch,
                                segment_width = 1000,
                                equal_width=False,
                                n_segments = 0)

    print(len(returned_branches))
    lengths = [sk.calculate_skeleton_distance(k) for k in returned_branches]
    print(f"lengths = {lengths}")


    sk.graph_skeleton_and_mesh(
                                other_skeletons=returned_branches[:10],
                            other_skeletons_colors=["black"],
                              #other_skeletons=[ex_branch],
                              other_meshes=[ex_mesh])

    """

    if segment_width<=0 and n_segments<=0:
        raise Exception("Both segment_width and n_segments are non-positive")

    skeleton_graph = sk.convert_skeleton_to_graph(curr_branch_skeleton)
    skeleton_node_coordinates = get_ordered_branch_nodes_coordinates(skeleton_graph)
    cleaned_skeleton = convert_branch_graph_to_skeleton(skeleton_graph)

    seg_bins = np.hstack([np.array([0]),sk.calculate_skeleton_segment_distances(cleaned_skeleton,cumsum=True)])



    if n_segments > 0:
            segment_width = seg_bins[-1]/n_segments
    else:
        if segment_width>seg_bins[-1]:
            #print("Skeletal width required was longer than the current skeleton so just returning the endpoints")
            return [np.vstack([cleaned_skeleton[0][0],cleaned_skeleton[-1][-1]]).reshape(1,2,3)]


    segment_widths = sk.calculate_skeleton_segment_distances(cleaned_skeleton,cumsum=False)
    #print(f"total_distance = {sk.calculate_skeleton_distance(cleaned_skeleton)}")

    if equal_width and n_segments <= 0:
        #print("making all of the branch segments equal width")
        n_segments_that_fit = seg_bins[-1]/segment_width
        #print(f"n_segments_that_fit = {n_segments_that_fit}")
        if n_segments_that_fit > int(n_segments_that_fit): #if there is some leftover 
            segment_width = seg_bins[-1]/np.ceil(n_segments_that_fit)
            #print(f"New segment width in order to make them equal = {segment_width}\n")

    n_full_segs = int(seg_bins[-1]/segment_width)
    #print(f"n_full_segs = {n_full_segs}")

    #old way
    new_seg_endpoints = np.arange(segment_width,segment_width*n_full_segs+0.01,segment_width)
    
    #print(f"new_seg_endpoints[-1] - seg_bins[-1] = {new_seg_endpoints[-1] - seg_bins[-1]}")
    if new_seg_endpoints[-1] > seg_bins[-1]:
        if new_seg_endpoints[-1] - seg_bins[-1] > 0.01:
            raise Exception("End of new_seg_endpoints is greater than original skeleton ")
        else:
            new_seg_endpoints[-1] =  seg_bins[-1]

    #accounts for the perfect fit
    if new_seg_endpoints[-1] == seg_bins[-1]:
        #print("exact match so eliminating last new bin")
        new_seg_endpoints = new_seg_endpoints[:-1] #remove the last one because will readd it back

    #print(f"seg_bins = {seg_bins}")
    #print(f"new_seg_endpoints = {new_seg_endpoints}")

    #getting the vertices

    """
    3) for each new segment endpoint, 
    a) calculate between which two existing skeleton segment end points it would exist
    (using just a distnace measurement from each end point to the next)
    b)calculate the coordinate that is a certain distance in middle based on which endpoints between

    new_vector * (new_seg_endpoint - lower_bin_distance)/seg_width + lower_bin_vector
    # """

    bin_indices = np.digitize(new_seg_endpoints, seg_bins)

    new_coordinates = (((skeleton_node_coordinates[bin_indices] - skeleton_node_coordinates[bin_indices-1])
                       *((new_seg_endpoints - seg_bins[bin_indices-1])/segment_widths[bin_indices-1]).reshape(-1,1)) + skeleton_node_coordinates[bin_indices-1])

    #these should be the same size
    #     print(f"bin_indices = {bin_indices}")
    #     print(f"new_coordinates = {new_coordinates}")
    #     return bin_indices,new_coordinates

    """
    Using the bin_indices and new_coordinates construct a list of branches with the original vertices plus the new cuts
    Pseudocode:

    indices mean that they are greater than or equal to the bin below but absolutely less than the bin indices value
    --> need to make sure that the new cut does not fall on current cut
    --> do this by checking that the last node before the cut isn't equal to the cut

    1) include all of the skeleton points but not including the bin idexed numer
    """
    returned_branches = []
    skeleton_node_coordinates #these are the original coordinates
    for z,(curr_bin,new_c) in enumerate(zip(bin_indices,new_coordinates)):
        if z==0:
#             print(f"curr_bin = {curr_bin}")
#             print(f"bin_indices = {bin_indices}")
            
            previous_nodes = skeleton_node_coordinates[:curr_bin]
#             print(f"previous_nodes = {previous_nodes}")
#             print(f"previous_nodes[-1] = {previous_nodes[-1]}")
#             print(f"new_c = {new_c}")
#             print(f"np.linalg.norm(previous_nodes[:-1]- new_c) = {np.linalg.norm(previous_nodes[-1]- new_c)}")
            if np.linalg.norm(previous_nodes[-1]- new_c) > 0.001:
                #print("inside linalg_norm")
                previous_nodes = np.vstack([previous_nodes,new_c.reshape(-1,3)])
            
            #print(f"previous_nodes = {previous_nodes}")
            #now create the branch
            returned_branches.append(np.stack((previous_nodes[:-1],previous_nodes[1:]),axis=1).reshape(-1,2,3))
            #print(f"returned_branches = {returned_branches}")
        else:
            #if this was not the first branch
            previous_nodes = new_coordinates[z-1].reshape(-1,3)
            if curr_bin > bin_indices[z-1]:
                previous_nodes = np.vstack([previous_nodes,skeleton_node_coordinates[bin_indices[z-1]:curr_bin].reshape(-1,3)])
            if np.linalg.norm(previous_nodes[-1]- new_c) > 0.001:
                previous_nodes = np.vstack([previous_nodes,new_c.reshape(-1,3)])

            returned_branches.append(np.stack((previous_nodes[:-1],previous_nodes[1:]),axis=1).reshape(-1,2,3))


    #     if np.array_equal(returned_branches[-1],np.array([], dtype="float64").reshape(-1,2,3)):
    #         print(f"previous_nodes= {previous_nodes}")
    #         print(f"new_c = {new_c}")
    #         print(f"curr_bin = {curr_bin}")
    #         print(f"bin_indices = {bin_indices}")
    #         print(f"z = {z}")
    #         raise Exception("stopping")

    #add this last section to the skeleton
    if np.linalg.norm(new_c - skeleton_node_coordinates[-1]) > 0.001: #so the last node has not been added yet
        previous_nodes = new_coordinates[-1].reshape(-1,3)
        if bin_indices[-1]<len(seg_bins):
            previous_nodes = np.vstack([previous_nodes,skeleton_node_coordinates[bin_indices[-1]:len(skeleton_node_coordinates)].reshape(-1,3)])
        else:
            previous_nodes = np.vstack([previous_nodes,skeleton_node_coordinates[-1].reshape(-1,3)])
        returned_branches.append(np.stack((previous_nodes[:-1],previous_nodes[1:]),axis=1).reshape(-1,2,3))
    
    #check 1: that the first and last of original branch is the same as the decomposed
    first_coord = returned_branches[0][0][0]
    last_coord = returned_branches[-1][-1][-1]
    
#     print(f"first original coord = {skeleton_node_coordinates[0]}")
#     print(f"last original coord = {skeleton_node_coordinates[-1]}")
#     print(f"first_coord = {first_coord}")
#     print(f"last_coord = {last_coord}")
    
    
    if not np.array_equal(skeleton_node_coordinates[0],first_coord):
        print(f"skeleton_node_coordinates = {skeleton_node_coordinates}")
        print(f"first_coord = {first_coord}")
        raise Exception("First coordinate does not match")
        
    if not np.array_equal(skeleton_node_coordinates[-1],last_coord):
        print(f"skeleton_node_coordinates = {skeleton_node_coordinates}")
        print(f"last_coord = {last_coord}")
        raise Exception("Last coordinate does not match")

    
    #check 2: that it is all one connected branch
    total_skeleton = sk.stack_skeletons(returned_branches)
    total_skeleton_graph = sk.convert_skeleton_to_graph(total_skeleton)
    n_comps = nx.number_connected_components(total_skeleton_graph)
    
    
    #print(f"Number of connected components is {n_comps}")
    
    if n_comps > 1:
        raise Exception(f"Number of connected components is {n_comps}")

    print(f"Total number of returning branches = {len(returned_branches)}")
    return returned_branches

# -------- for the mesh correspondence -------
# def waterfill_labeling(
#                 total_mesh_correspondence,
#                  submesh_indices,
#                  total_mesh=None,
#                 total_mesh_graph=None,
#                  propagation_type="random",
#                 max_iterations = 1000,
#                 max_submesh_threshold = 1000
#                 ):
#     """
#     Pseudocode:
#     1) check if the submesh you are propagating labels to is too large
#     2) for each unmarked face get the neighbors of all of the faces, and for all these neighbors get all the labels
#     3) if the neighbors label is not empty. depending on the type of progation type then pick the winning label
#     a. random: just randomly choose from list
#     b. .... not yet implemented
#     4) revise the faces that are still empty and repeat process until all faces are empty (have a max iterations number)
#     """
    
#     if not total_mesh_graph:
#         #finding the face adjacency:
#         total_mesh_graph = nx.from_edgelist(total_mesh.face_adjacency)
    
    
    
#     if len(submesh_indices)> max_submesh_threshold:
#         raise Exception(f"The len of the submesh ({len(submesh_indices)}) exceeds the maximum threshold of {max_submesh_threshold} ")
    
#     #check that these are unmarked
#     curr_unmarked_faces = [k for k in submesh_indices if total_mesh_correspondence[k] == -1] 
    
    
#     if len(curr_unmarked_faces)<len(submesh_indices):
#         raise Exception(f"{len(submesh_indices)-len(curr_unmarked_faces)} submesh faces were already labeled before waterfill_labeling started")
    
#     for i in range(max_iterations):
#         #s2) for each unmarked face get the neighbors of all of the faces, and for all these neighbors get all the labels
#         unmarked_faces_neighbors = [xu.get_neighbors(total_mesh_graph,j) for j in curr_unmarked_faces] #will be list of lists
#         #print(f"unmarked_faces_neighbors = {unmarked_faces_neighbors}")
#         unmarked_face_neighbor_labels = [np.array([total_mesh_correspondence[curr_neighbor] for curr_neighbor in z]) for z in unmarked_faces_neighbors]
#         #print(f"unmarked_face_neighbor_labels = {unmarked_face_neighbor_labels}")
        
#         if len(unmarked_face_neighbor_labels) == 0:
#             print(f"curr_unmarked_faces = {curr_unmarked_faces}")
#             print(f"i = {i}")
#             print(f"unmarked_faces_neighbors = {unmarked_faces_neighbors}")
#             print(f"unmarked_face_neighbor_labels = {unmarked_face_neighbor_labels}")
            
#         #check if there is only one type of label and if so then autofil
#         total_labels = list(np.unique(np.concatenate(unmarked_face_neighbor_labels)))
        
#         if -1 in total_labels:
#             total_labels.remove(-1)
        
#         if len(total_labels) == 0:
#             raise Exception("total labels does not have any marked neighbors")
#         elif len(total_labels) == 1:
#             print("All surrounding labels are the same so autofilling the remainder of unlabeled labels")
#             for gg in curr_unmarked_faces:
#                 total_mesh_correspondence[gg] = total_labels[0]
#             break
#         else:
#             #if there are still one or more labels surrounding our unlabeled region
#             for curr_face,curr_neighbors in zip(curr_unmarked_faces,unmarked_face_neighbor_labels):
#                 curr_neighbors = curr_neighbors[curr_neighbors != -1]
#                 if len(curr_neighbors) > 0:
#                     if propagation_type == "random":
#                         total_mesh_correspondence[curr_face] = np.random.choice(curr_neighbors)
#                     else:
#                         raise Exception("Not implemented propagation_type")
        
#         # now replace the new curr_unmarked faces
#         curr_unmarked_faces = [k for k in submesh_indices if total_mesh_correspondence[k] == -1] #old dict way
        
        
#         if len(curr_unmarked_faces) == 0:
#             print(f"breaking out of loop because zero unmarked faces left after {i} iterations")
#             break
        
    
#     #check that no more unmarked faces or error
#     end_unmarked_faces = [k for k in submesh_indices if total_mesh_correspondence[k] == -1]
    
#     if len(end_unmarked_faces) > 0:
#         raise Exception(f"After {i+1} iterations (with max_iterations = {max_iterations} there were still {len(end_unmarked_faces)} faces")
        
    
#     return total_mesh_correspondence


# ----- functions to help with the Neuron class ---- #
def find_branch_endpoints(db,order_by_coordinate = True):
    db_graph = sk.convert_skeleton_to_graph(db)
    end_node_coordinates = xu.get_node_attributes(db_graph,node_list=xu.get_nodes_of_degree_k(db_graph,1))

    if len(end_node_coordinates) != 2:
        raise Exception("Not exactly 2 end nodes in the passed branch")
    else:
        if order_by_coordinate:
            end_node_coordinates = nu.sort_multidim_array_by_rows(
                end_node_coordinates,
                descending=False
            )
        return end_node_coordinates
    
def compare_skeletons_ordered(skeleton_1,skeleton_2,
                             edge_threshold=0.01, #how much the edge distances can vary by
                              node_threshold = 0.01, #how much the nodes can vary by
                              print_flag = False
                             ):
    """
    Purpose: To compare skeletons where the edges are ordered (not comparing overall skeletons)
    Those would be isomorphic graphs (not yet developed)
    
    Example of how to use: 
    skeletons_idx_to_stack = [0,1,2,3]
    total_skeleton = sk.stack_skeletons([double_soma_obj.concept_network.nodes["L1"]["data"].concept_network.nodes[k]["data"].skeleton for k in skeletons_idx_to_stack])
    #sk.graph_skeleton_and_mesh(other_skeletons=[total_skeleton])
    
    skeleton_1 = copy.copy(total_skeleton)
    skeleton_2 = copy.copy(total_skeleton)
    skeleton_1[0][0] = np.array([558916.8, 1122107. ,  842972.8]) #change so there will be error
    
    sk.compare_skeletons_ordered(skeleton_1,
                          skeleton_2,
                             edge_threshold=0.01, #how much the edge distances can vary by
                              node_threshold = 0.01, #how much the nodes can vary by
                              print_flag = True
                             )

    
    """
    sk_1_graph = convert_skeleton_to_graph(skeleton_1)
    sk_2_graph = convert_skeleton_to_graph(skeleton_2)

    return xu.compare_networks(sk_1_graph,sk_2_graph,print_flag=print_flag,
                     edge_comparison_threshold=edge_threshold,
                     node_comparison_threshold=node_threshold)
    
    
# ----------------- 7/22 Functions made to help with graph searching and visualizaiton ------------ #
def skeleton_n_components(curr_skeleton):
    """
    Purpose: To get the number of connected components represented by 
    the current skeleton
    """
    cleaned_branch_components = nx.number_connected_components(convert_skeleton_to_graph(curr_skeleton))
    return cleaned_branch_components

def check_skeleton_one_component(curr_skeleton):
    cleaned_branch_components = skeleton_n_components(curr_skeleton)
    if cleaned_branch_components > 1:
        raise Exception(f"Skeleton is not one component: n_components = {cleaned_branch_components}")
    

# ---------------- 9/17: Will help with creating branch points extending towards soma if not already exist ---

def create_soma_extending_branches(
    current_skeleton, #current skeleton that was created
    skeleton_mesh, #mesh that was skeletonized
    soma_to_piece_touching_vertices,#dictionary mapping a soma it is touching to the border vertices,
    return_endpoints_must_keep=True,
    return_created_branch_info=False,
    try_moving_to_closest_sk_to_endpoint=True, #will try to move the closest skeleton point to an endpoint
    distance_to_move_point_threshold = 1500, #maximum distance willling to move closest skeleton point to get to an endpoint
    check_connected_skeleton=True
                                    ):
    """
    Purpose: To make sure there is one singular branch extending towards the soma
    
    Return value:
    endpoints_must_keep: dict mapping soma to array of the vertex points that must be kept
    because they are the soma extending branches of the skeleton
    
    Pseudocode: 
    Iterating through all of the somas and all of the groups of touching vertices
    a) project the skeleton and the soma boundary vertices on to the vertices of the mesh
    b) Find the closest skeleton point to the soma boundary vetices
    c) check the degree of the closest skeleton point:
    - if it is a degree one then leave alone
    - if it is not a degree one then create a new skeleton branch from the 
    closest skeleton point and the average fo the border vertices and add to 
    
    Extension: (this is the same method that would be used for adding on a floating skeletal piece)
    If we made a new skeleton branch then could pass back the closest skeleton point coordinates
    and the new skeleton segment so we could:
    1) Find the branch that it was added to 
    2) Divide up the mesh correspondnece between the new resultant branches
    -- would then still reuse the old mesh correspondence
    
    """
    endpoints_must_keep = dict()
    new_branches = dict()
    
    #0) Create a graph of the mesh from the vertices and edges and a KDTree
    start_time = time.time()
    vertex_graph = tu.mesh_vertex_graph(skeleton_mesh)
    mesh_KD = KDTree(skeleton_mesh.vertices)
    print(f"Total time for mesh KDTree = {time.time() - start_time}")
    
    for s_index,v in soma_to_piece_touching_vertices.items():
        
        endpoints_must_keep[s_index] = []
        new_branches[s_index]=[]
        
        for j,sbv in enumerate(v):


            #1)  Project all skeleton points and soma boundary vertices onto the mesh
            all_skeleton_points = np.unique(current_skeleton.reshape(-1,3),axis=0)
            sk_points_distances,sk_points_closest_nodes = mesh_KD.query(all_skeleton_points)

            #sbv = soma_to_piece_touching_vertices[s_index]
            print(f"sbv[0].reshape(-1,3) = {sbv[0].reshape(-1,3)}")
            soma_border_distances,soma_border_closest_nodes = mesh_KD.query(sbv[0].reshape(-1,3))

            
            ''' old way that relied on soley paths on the mesh graph
            start_time = time.time()
            #2) Find the closest skeleton point to the soma border (for that soma), find shortest path from many to many
            path,closest_sk_point,closest_soma_border_point = xu.shortest_path_between_two_sets_of_nodes(vertex_graph,sk_points_closest_nodes,soma_border_closest_nodes)
            print(f"Shortest path between 2 nodes = {time.time() - start_time}")

            #3) Find closest skeleton point
            closest_sk_pt = np.where(sk_points_closest_nodes==closest_sk_point)[0][0]
            closest_sk_pt_coord = all_skeleton_points[closest_sk_pt]
            '''
            
            """New Method 10/27
            1) applies a mesh filter for only those within a certian distance along mesh graph (filter)
            2) Of filtered vertices, finds one closest to soma border average
            
            """
            curr_cut_distane = 10000
            
            for kk in range(0,5):
                close_nodes = xu.find_nodes_within_certain_distance_of_target_node(vertex_graph,target_node=soma_border_closest_nodes[0],cutoff_distance=curr_cut_distane)
                filter_1_skeleton_points = np.array([sk_pt for sk_pt,sk_pt_node in zip(all_skeleton_points,sk_points_closest_nodes) if sk_pt_node in close_nodes])
                if len(filter_1_skeleton_points) >0:
                    break
                print(f"On iteration {kk} the filter points were empty with close_nodes len = {len(close_nodes)}, len(all_skeleton_points) = {len(all_skeleton_points)}, len(sk_points_closest_nodes) = {len(sk_points_closest_nodes)}")
                
                curr_cut_distane = curr_cut_distane*3
            
            if len(filter_1_skeleton_points) == 0:
                raise Exception (f"Still No filter nodes with curr_cut_distane = {curr_cut_distane}")
                    
                
            

            border_average_coordinate = np.mean(sbv,axis=0)

            closest_sk_point_idx = np.argmin(np.linalg.norm(filter_1_skeleton_points-border_average_coordinate,axis=1))
            closest_sk_pt_coord = filter_1_skeleton_points[closest_sk_point_idx]
            
            
            
            sk_graph = sk.convert_skeleton_to_graph(current_skeleton)
            
            distance_to_move_point_threshold
            if try_moving_to_closest_sk_to_endpoint:
                print(f"closest_sk_pt_coord BEFORE = {closest_sk_pt_coord}")
                print(f"current_skeleton.shape = {current_skeleton.shape}")
                closest_sk_pt_coord,change_status = move_point_to_nearest_branch_end_point_within_threshold(
                                                        skeleton=current_skeleton,
                                                        coordinate=closest_sk_pt_coord,
                                                        distance_to_move_point_threshold = distance_to_move_point_threshold,
                                                        verbose=True,
                                                        consider_high_degree_nodes=False

                                                        )
                print(f"change_status for create soma extending pieces = {change_status}")
                print(f"closest_sk_pt_coord AFTER = {closest_sk_pt_coord}")
            
            #find the node that has the desired vertices and its' degree
            sk_node = xu.get_nodes_with_attributes_dict(sk_graph,dict(coordinates=closest_sk_pt_coord))[0]
            sk_node_degree = sk_graph.degree()[sk_node]
    

            if sk_node_degree == 0:
                raise Exception("Found 0 degree node in skeleton")
                
            elif sk_node_degree == 1: #3a) If it is a node of degree 1 --> do nothing
                print(f"skipping soma {s_index} because closest skeleton node was already end node")
                endpoints_must_keep[s_index].append(closest_sk_pt_coord)
                new_branches[s_index].append(None)
                continue
            else:
                #3b) If Not endpoint:
                #Add an edge from the closest skeleton point coordinate to vertex average of all soma boundaries
                print("Adding new branch to skeleton")
                print(f"border_average_coordinate = {border_average_coordinate}")
                
                new_branch_sk = np.vstack([closest_sk_pt_coord,border_average_coordinate]).reshape(-1,2,3)
                current_skeleton = sk.stack_skeletons([current_skeleton,new_branch_sk])
                endpoints_must_keep[s_index].append(border_average_coordinate)
                
                #will store the newly added branch and the corresponding border vertices
                new_branches[s_index].append(dict(new_branch = new_branch_sk,border_verts=sbv))
                
        endpoints_must_keep[s_index] = np.array(endpoints_must_keep[s_index])
        
    print(f"endpoints_must_keep = {endpoints_must_keep}")
    #check if skeleton is connected component when finishes
    if check_connected_skeleton:
        if nx.number_connected_components(convert_skeleton_to_graph(current_skeleton)) != 1:
            su.compressed_pickle(current_skeleton,"current_skeleton")
            raise Exception("The skeleton at end wasn't a connected component")
    
    return_value = [current_skeleton]
    
    if return_endpoints_must_keep:
        return_value.append(endpoints_must_keep)
    if return_created_branch_info:
        return_value.append(new_branches)
    return return_value


def find_branch_skeleton_with_specific_coordinate(divded_skeleton,current_coordinate):
    """
    Purpose: From list of skeletons find the ones that have a certain coordinate
    
    Example: 
    curr_limb = current_neuron[0]
    branch_names = curr_limb.get_branch_names(return_int=True)
    curr_limb_divided_skeletons = [curr_limb[k].skeleton for k in branch_names]
    ideal_starting_endpoint = curr_limb.current_starting_coordinate
    
    sk = reload(sk)
    sk.find_branch_skeleton_with_specific_coordinate(curr_limb_divided_skeletons,ideal_starting_endpoint)

    """
    matching_branch = []
    for b_idx,b_sk in enumerate(divded_skeleton):
        match_result = nu.matching_rows(b_sk.reshape(-1,3),current_coordinate)
        #print(f"match_result = {match_result}")
        if len(match_result)>0:
            matching_branch.append(b_idx)
    
    return matching_branch

#----------- 9/24 -------------- #
def find_skeleton_endpoint_coordinates(
    skeleton,
    coordinates_to_exclude = None,
    plot = False,):
    """
    Purpose: To find the endpoint coordinates 
    of a skeleton
    
    Application: 
    1) Can get the endpoints of a skeleton and 
    then check that none of the spines contain 
    an endpoint coordinate to help 
    guard against false spines at the endpoints
    
    Pseudocode:
    1) convert the skeleton to a graph
    2) Find the endpoint nodes of the graph (with degree 1)
    3) return the coordinates of the graph nodes
    
    """
    G = convert_skeleton_to_graph(skeleton)
    endpoint_nodes = xu.get_nodes_of_degree_k(G,degree_choice=1)
    #print(f"endpoint_nodes = {endpoint_nodes}")
    if len(endpoint_nodes) == 0:
        return []
    
    endpoint_coordinates = xu.get_node_attributes(G,node_list=endpoint_nodes)
    
    if coordinates_to_exclude is not None:
        endpoint_coordinates = nu.setdiff2d(np.array(endpoint_coordinates).reshape(-1,3),np.array(coordinates_to_exclude).reshape(-1,3))
        
    if plot:
        ipvu.plot_objects(
            main_skeleton=skeleton,
            scatters=[endpoint_coordinates],
            scatter_size=1
        )
    return endpoint_coordinates


def path_ordered_skeleton(skeleton):
    """
    Purpose: To order the edges in sequential order in
    a skeleton so skeleton[0] is one end edge
    and skeleton[-1] is the other end edge
    
    Pseudocode: 
    How to order a skeleton: 
    1) turn the skeleton into a graph
    2) start at an endpoint node
    3) output the skeleton edges for the edges of the graph until hit the other end node


    
    Ex: 
    skeleton = big_neuron[0][30].skeleton
    new_skeleton_ordered = path_ordered_skeleton(skeleton)
    
    
    
    """

    #1) turn the skeleton into a graph
    G = convert_skeleton_to_graph(skeleton)
    #2) start at an endpoint node
    end_nodes = xu.get_nodes_of_degree_k(G,1)

    sk_node_path = nx.shortest_path(G,source=end_nodes[0],target=end_nodes[-1])
    sk_node_path_coordinates = xu.get_node_attributes(G,node_list=sk_node_path)

    ordered_skeleton = np.hstack([sk_node_path_coordinates[:-1],
                                  sk_node_path_coordinates[1:]]).reshape(-1,2,3)

    return ordered_skeleton

# ----------- 1/6 Addition: To help and not filter away significant skeleton pieces --------- #
def find_end_nodes_with_significant_mesh_correspondence(
    skeleton,
    mesh,
    mesh_threshold = 275,
    skeleton_considered_min = 1600,
    skeleton_considered_max = 6000,
    plot_viable_endpoint_correspondences = False,
    plot_keep_endpoints = False,
    verbose = False):


    current_skeleton = skeleton
    keep_node_coordinates = []

    if len(current_skeleton) == 0:
        return []
        

    #1) Turn the skeleton into a graph
    sk_graph = sk.convert_skeleton_to_graph(current_skeleton)

    #2) Get all of the end nodes
    end_nodes = xu.get_nodes_of_degree_k(sk_graph,1)

    #3) Get all of the high degree nodes in the graph
    high_degree_nodes = xu.get_nodes_greater_or_equal_degree_k(sk_graph,3)

    """
    checking that this went well:

    endnode_coordinates = xu.get_coordinate_by_graph_node(sk_graph,end_nodes)
    high_degree_coordinates = xu.get_coordinate_by_graph_node(sk_graph,high_degree_nodes)

    ipvu.plot_objects(mesh,
                     skeletons=[current_skeleton],
                     scatters=[endnode_coordinates,high_degree_coordinates],
                     scatters_colors=["red","blue"],
                     scatter_size=1)

    """


    if len(high_degree_nodes) > 0 and len(end_nodes)>0:
        viable_end_node = []
        viable_end_node_skeletons = []
        viable_end_node_skeletons_len = []

        for j,e_node in enumerate(end_nodes):

            #a. Find the path to the nearest high degree node
            curr_path,_,_ = xu.shortest_path_between_two_sets_of_nodes(sk_graph,[e_node],high_degree_nodes)

            #b. Get the skeleton of that subgraph
            subskeleton_graph = sk_graph.subgraph(curr_path)
            end_node_skeleton = sk.convert_graph_to_skeleton(subskeleton_graph)

            #c. If length of that skeleton is within a certain range:
            end_node_skeleton_len = sk.calculate_skeleton_distance(end_node_skeleton)
            if end_node_skeleton_len <= skeleton_considered_max and end_node_skeleton_len >= skeleton_considered_min:
                viable_end_node.append(e_node)
                viable_end_node_skeletons.append(end_node_skeleton)
                viable_end_node_skeletons_len.append(end_node_skeleton_len)
            else:
                if verbose:
                    print(f"{j}th endnode ({e_node}) was not checked because length was {end_node_skeleton_len}")


        """
        Checking viable paths were made:

        ipvu.plot_objects(mesh,
                         skeletons=viable_end_node_skeletons,
                         scatters=[endnode_coordinates,high_degree_coordinates],
                         scatters_colors=["red","blue"],
                         scatter_size=1)
        """

        if len(viable_end_node) > 0:
            viable_skeleton_meshes = tu.skeleton_to_mesh_correspondence(mesh = mesh,
                                                            skeletons = viable_end_node_skeletons,
                                                              verbose=verbose
                                               )

            if plot_viable_endpoint_correspondences:
                ipvu.plot_objects(meshes=viable_skeleton_meshes,
                          meshes_colors="random",
                          skeletons=viable_end_node_skeletons,
                         skeletons_colors="random")



            mesh_lengths = np.array([len(k.faces) for k in viable_skeleton_meshes])
            viable_end_node = np.array(viable_end_node)

            viable_end_node_final = viable_end_node[mesh_lengths > mesh_threshold]

            if verbose:
                print(f"Final end nodes to keep = {viable_end_node_final}")
            if len(viable_end_node_final)>0:
                keep_node_coordinates = xu.get_coordinate_by_graph_node(sk_graph,list(viable_end_node_final))

    if plot_keep_endpoints:
        if len(keep_node_coordinates) == 0:
            print("!!! NO KEEP ENDPOINTS FOUND !!!")
        else:
            ipvu.plot_objects(mesh,
                         skeletons=[current_skeleton],
                         scatters=[keep_node_coordinates],
                         scatters_colors=["red"],
                         scatter_size=1)

    return keep_node_coordinates


# ---------------------- For preprocessing of neurons revised ------------------ #
def skeletonize_and_clean_connected_branch_CGAL(mesh,
                       curr_soma_to_piece_touching_vertices=None,
                       total_border_vertices=None,
                        filter_end_node_length=4001,
                       perform_cleaning_checks=False,
                       combine_close_skeleton_nodes = True,
                        combine_close_skeleton_nodes_threshold=700,
                                               verbose=False,
                                                remove_cycles_at_end = True,
                                                remove_mesh_interior_face_threshold=0,
                                                error_on_bad_cgal_return = False,
                                                max_stitch_distance=max_stitch_distance_default,
                                                restrict_end_nodes_filtered_by_corr = False,
                                                **kwargs):
    """
    Purpose: To create a clean skeleton from a mesh
    (used in the neuron preprocessing package)
    """
    
    
    debug = False
    
    branch = mesh
    clean_time = time.time()
    
    if debug is True:
#         print(f"curr_soma_to_piece_touching_vertices = {curr_soma_to_piece_touching_vertices}")
#         print(f"total_border_vertices = {total_border_vertices}")
#         print(f"filter_end_node_length = {filter_end_node_length}")
#         print(f"kwargs = {kwargs}")
        pass
        
    
    current_skeleton = skeletonize_connected_branch(branch,verbose=verbose,
                                                    remove_mesh_interior_face_threshold=remove_mesh_interior_face_threshold,
                                                    error_on_bad_cgal_return=error_on_bad_cgal_return,
                                                    max_stitch_distance=max_stitch_distance,
                                                    **kwargs)
    
    
    print("Checking connected components after skeletonize_connected_branch")
    check_skeleton_connected_component(current_skeleton)

    if not remove_cycles_at_end:
        current_skeleton = remove_cycles_from_skeleton(current_skeleton)
    



#                     sk_debug = True
#                     if sk_debug:
#                         from datasci_tools import system_utils as su
#                         print("**Saving the skeletons**")
#                         su.compressed_pickle(branch,
#                                             "curr_branch_saved")
#                     if sk_debug:
#                         from datasci_tools import system_utils as su
#                         print("**Saving the skeletons**")
#                         su.compressed_pickle(current_skeleton,
#                                             "current_skeleton")

    print(f"    Total time for skeletonizing branch: {time.time() - clean_time}")
    clean_time = time.time()
    
    print("Checking connected components after removing cycles")
    check_skeleton_connected_component(current_skeleton)
    

    
    if not curr_soma_to_piece_touching_vertices is None:

        current_skeleton, curr_limb_endpoints_must_keep = create_soma_extending_branches(
                        current_skeleton=current_skeleton, #current skeleton that was created
                        skeleton_mesh=branch, #mesh that was skeletonized
                        soma_to_piece_touching_vertices=curr_soma_to_piece_touching_vertices,#dictionary mapping a soma it is touching to the border vertices,
                        return_endpoints_must_keep=True,
                                                        )
    else:
        if verbose:
            print("Not Creating soma extending branches because curr_soma_to_piece_touching_vertices is None")
        curr_limb_endpoints_must_keep = None




    print(f"    Total time for Fixing Skeleton Soma Endpoint Extension : {time.time() - clean_time}")
    """  --------- END OF 9/17 Addition:  -------- """

    #                     sk_debug = True
    #                     if sk_debug:
    #                         from datasci_tools import system_utils as su
    #                         print("**Saving the skeletons**")
    #                         su.compressed_pickle(current_skeleton,
    #                                             "current_skeleton_after_addition")



        # --------  Doing the cleaning ------- #
    clean_time = time.time()
    print(f"filter_end_node_length = {filter_end_node_length}")

    """ 9/16 Edit: Now send the border vertices and don't want to clean anyy end nodes that are within certain distance of border"""

    #soma_border_vertices = total_border_vertices,
    #skeleton_mesh=branch,
    
    #gathering the endpoints to send to skeleton cleaning
    if not curr_limb_endpoints_must_keep is None:
        coordinates_to_keep = np.vstack(list(curr_limb_endpoints_must_keep.values())).reshape(-1,3)
    else:
        coordinates_to_keep = None
    
    check_skeleton_connected_component(current_skeleton)
    
    
    
    
    """ ----------- 1/6/21 Addition -----------------
    Can check for end-nodes that should not be filtered away 
    because represent significant portions of mesh
    

    
    """
    if verbose:
        print(f"coordinates_to_keep BEFORE significant mesh check = {coordinates_to_keep}")
    
    if restrict_end_nodes_filtered_by_corr:
        sign_coordinates_from_mesh = sk.find_end_nodes_with_significant_mesh_correspondence(
                                    mesh = mesh,
                                    skeleton = current_skeleton,
                                    skeleton_considered_max=filter_end_node_length+2,
                                    plot_viable_endpoint_correspondences = False,
                                    plot_keep_endpoints = False,
                                    verbose = False)
    else:
        sign_coordinates_from_mesh = []
    
    if len(sign_coordinates_from_mesh)>0:
        if coordinates_to_keep is not None:
            coordinates_to_keep = np.vstack([coordinates_to_keep,sign_coordinates_from_mesh])
        else:
            coordinates_to_keep = sign_coordinates_from_mesh
    
    if verbose:
        print(f"sign_coordinates_from_mesh = {sign_coordinates_from_mesh}")
        print(f"coordinates_to_keep AFTER significant mesh check = {coordinates_to_keep}")

    
    
    new_cleaned_skeleton = clean_skeleton(current_skeleton,
                            distance_func=skeletal_distance,
                      min_distance_to_junction=filter_end_node_length, #this used to be a tuple i think when moved the parameter up to function defintion
                      return_skeleton=True,
#                         soma_border_vertices = total_border_vertices,
#                         skeleton_mesh=branch,
                        endpoints_must_keep = coordinates_to_keep,
                      print_flag=False)

#                     sk_debug = True
#                     if sk_debug:
#                         from datasci_tools import system_utils as su
#                         print("**Saving the skeletons**")
#                         su.compressed_pickle(new_cleaned_skeleton,
#                                             "new_cleaned_skeleton")
    
    print("Checking connected components after clean_skeleton")
    try:
        check_skeleton_connected_component(new_cleaned_skeleton)
    except:
        print("No connected skeleton after cleaning so just going with older skeleton")
        new_cleaned_skeleton = current_skeleton
    
    #--- 1) Cleaning each limb through distance and decomposition, checking that all cleaned branches are connected components and then visualizing
    distance_cleaned_skeleton = new_cleaned_skeleton

    if perform_cleaning_checks:
        #make sure still connected componet
        distance_cleaned_skeleton_components = nx.number_connected_components(convert_skeleton_to_graph(distance_cleaned_skeleton))
        if distance_cleaned_skeleton_components > 1:
            raise Exception(f"distance_cleaned_skeleton {j} was not a single component: it was actually {distance_cleaned_skeleton_components} components")

        print(f"after DISTANCE cleaning limb size of skeleton = {distance_cleaned_skeleton.shape}")

    cleaned_branch = clean_skeleton_with_decompose(distance_cleaned_skeleton)

    if perform_cleaning_checks:
        cleaned_branch_components = nx.number_connected_components(convert_skeleton_to_graph(cleaned_branch))
        if cleaned_branch_components > 1:
            raise Exception(f"BEFORE COMBINE: cleaned_branch {j} was not a single component: it was actually {cleaned_branch_components} components")



    if combine_close_skeleton_nodes:
        print(f"********COMBINING CLOSE SKELETON NODES WITHIN {combine_close_skeleton_nodes_threshold} DISTANCE**********")
        cleaned_branch = combine_close_branch_points(cleaned_branch,
                                                            combine_threshold = combine_close_skeleton_nodes_threshold,
                                                            print_flag=True) 

        
    
    if remove_cycles_at_end:
        cleaned_branch = remove_cycles_from_skeleton(cleaned_branch)
        
    cleaned_branch = clean_skeleton_with_decompose(cleaned_branch)


    if perform_cleaning_checks:
        n_components = nx.number_connected_components(convert_skeleton_to_graph(cleaned_branch)) 
        if n_components > 1:
            raise Exception(f"After combine: Original limb was not a single component: it was actually {n_components} components")
            
        divided_branches = sk.decompose_skeleton_to_branches(cleaned_branch)
        
        #check that when we downsample it is not one component:
        curr_branch_meshes_downsampled = [sk.resize_skeleton_branch(b,n_segments=1) for b in divided_branches]
        downsampled_skeleton = sk.stack_skeletons(curr_branch_meshes_downsampled)
        curr_sk_graph_debug = sk.convert_skeleton_to_graph(downsampled_skeleton)


        con_comp = list(nx.connected_components(curr_sk_graph_debug))
        if len(con_comp) > 1:
            raise Exception(f"There were more than 1 component when downsizing: {[len(k) for k in con_comp]}")

    return cleaned_branch,curr_limb_endpoints_must_keep

def check_skeleton_connected_component(skeleton):
    sk_graph = convert_skeleton_to_graph(skeleton)
    n_comp = nx.number_connected_components(sk_graph)
    if n_comp != 1:
        raise Exception(f"There were {n_comp} number of components detected in the skeleton")

def skeleton_connected_components(skeleton):
    total_limb_sk_graph = sk.convert_skeleton_to_graph(skeleton)
    conn_comp_graph = list(nx.connected_components(total_limb_sk_graph))
    conn_comp_sk = [sk.convert_graph_to_skeleton(total_limb_sk_graph.subgraph(list(k))) for k in conn_comp_graph]
    return conn_comp_sk
        
def remove_cycles_from_skeleton(skeleton,
    max_cycle_distance = 5000,
    verbose = False,
    check_cycles_at_end=True,
    return_original_if_error=False,
    error_on_more_than_two_paths_between_high_degree_nodes=False):
    
    """
    Purpose: To remove small cycles from a skeleton
    

    Pseudocode: How to resolve a cycle
    A) Convert the skeleton into a graph
    B) Find all cycles in the graph

    For each cycle
    -------------
    1) Get the length of the cycle 
    --> if length if too big then skip
    2) If only 1 high degree node, then just delete the other non high degree nodes
    3) Else, there should only be 2 high degree nodes in the vertices of the cycle
    --> if more or less then skip
    3) Get the 2 paths between the high degree nodes
    4) Delete nodes on the path for the longer distance one
    ------------

    C) convert the graph back into a skeleton


    
    
    Ex: 
    remove_cycles_from_skeleton(skeleton = significant_poisson_skeleton)
    
    
    """
    
    try:

        #A) Convert the skeleton into a graph
        skeleton_graph = convert_skeleton_to_graph(skeleton)
        #B) Find all cycles in the graph
        cycles_list = xu.find_all_cycles(skeleton_graph)

        number_skipped = 0
        for j,cyc in enumerate(cycles_list):
            if verbose:
                print(f"\n ---- Working on cycle {j}: {cyc} ----")
            #1) Get the length of the cycle 
            #--> if length if too big then skip
            cyc = np.array(cyc)
            
            if len(np.setdiff1d(cyc,skeleton_graph.nodes()))>0:
                print(f"--- cycle {j} has nodes that don't exist anymore so skipping --")
                continue

            sk_dist_of_cycle = xu.find_skeletal_distance_along_graph_node_path(skeleton_graph,cyc)

            if max_cycle_distance < sk_dist_of_cycle:
                if verbose:
                    print(f"Skipping cycle {j} because total distance ({sk_dist_of_cycle}) is larger than max_cycle_distance ({max_cycle_distance}): {cyc} ")
                number_skipped += 1
                continue


            #Find the degrees of all of the nodes
            node_degrees = np.array([xu.get_node_degree(skeleton_graph,c) for c in cyc])
            print(f"node_degrees = {node_degrees}")

            #2) If only 1 high degree node, then just delete the other non high degree nodes
            if np.sum(node_degrees>2) == 1:
                if verbose:
                    print(f"Deleting non-high degree nodes in cycle {j}: {cyc} becuase there was only one high degree node: {node_degrees}")
                nodes_to_delete = cyc[np.where(node_degrees<=2)[0]]

                skeleton_graph.remove_nodes_from(nodes_to_delete)
                continue


            #3) Else, there should only be 2 high degree nodes in the vertices of the cycle
            #--> if more or less then skip

            if np.sum(node_degrees>2) > 2:
                if verbose:
                    print(f"Skipping cycle {j} because had {np.sum(node_degrees>2)} number of high degree nodes: {node_degrees} ")
                number_skipped += 1
                continue

            high_degree_nodes = cyc[np.where(node_degrees>2)[0]]
            
            if len(high_degree_nodes) == 0:
                print("No higher degree (above 2) nodes detected")
                continue
            
            
            cycle_graph = skeleton_graph.subgraph(cyc)

            #3) Get the 2 paths between the high degree nodes
            both_paths = list(nx.all_simple_paths(cycle_graph,high_degree_nodes[0],high_degree_nodes[1],len(cycle_graph)))

            if len(both_paths) != 2:
                if error_on_more_than_two_paths_between_high_degree_nodes:
                    su.compressed_pickle(skeleton,"skeleton")
                    raise Exception(f"Did not come up with only 2 paths between high degree nodes: both_paths = {both_paths} ")
                else:
                    print(f"Did not come up with only 2 paths between high degree nodes: both_paths = {both_paths} ")

            path_lengths = [xu.find_skeletal_distance_along_graph_node_path(skeleton_graph,g) for g in both_paths]


            #4) Delete nodes on the path for the longer distance one
            longest_path_idx = np.argmax(path_lengths)
            longest_path = both_paths[longest_path_idx]
            if len(longest_path) <= 2:
                raise Exception(f"Longest path for deletion was only of size 2 or less: both_paths = {both_paths}, longest_path = {longest_path}")

            if verbose:
                print(f"For cycle {j} deleting the following path because longest distance {path_lengths[longest_path_idx]}: {longest_path[1:-1]}")

            skeleton_graph.remove_nodes_from(longest_path[1:-1])


        #C) check that all cycles removed except for those ones
        if check_cycles_at_end:
            cycles_at_end = xu.find_all_cycles(skeleton_graph)
            if number_skipped != len(cycles_at_end):
                print(f"The number of cycles skipped ({number_skipped}) does not equal the number of cycles at the end ({len(cycles_at_end)})")
        #C) convert the graph back into a skeleton
        skeleton_removed_cycles = convert_graph_to_skeleton(skeleton_graph)
        
        if len(skeleton_removed_cycles) == 0:
            #su.compressed_pickle(skeleton,"remove_cycles_skeleton")
            #raise Exception("Removing the cycles made the skeleton of 0 size so returning old skeleton")
            print("Removing the cycles made the skeleton of 0 size so returning old skeleton")
            return skeleton
        
        return skeleton_removed_cycles
    except:
        if return_original_if_error:
            return skeleton
        else:
            su.compressed_pickle(skeleton,"remove_cycles_skeleton")
            raise Exception("Something went wrong in remove_cycles_from_skeleton (12/2 found because had disconnected skeleton)")



def skeleton_list_connectivity(skeletons,
    print_flag = False):
    """
    Will find the edge list for the connectivity of 
    branches in a list of skeleton branches
    
    
    """
    
    sk_endpoints = np.array([find_branch_endpoints(k) for k in skeletons]).reshape(-1,3)
    unique_endpoints,indices,counts= np.unique(sk_endpoints,return_inverse=True,return_counts=True,axis=0)
    
    total_edge_list = []
    repeated_indices = np.where(counts>1)[0]
    for ri in repeated_indices:
        connect_branches = np.where(indices == ri)[0]
        connect_branches_fixed = np.floor(connect_branches/2)
        total_edge_list += list(itertools.combinations(connect_branches_fixed,2))
    total_edge_list = np.array(total_edge_list).astype("int")
    
    return total_edge_list

def skeleton_list_connectivity_slow(
    skeletons,
    print_flag = False
    ):

    """
    Purpose: To determine which skeletons
    branches are connected to which and to
    record an edge list

    Pseudocode:
    For all branches i:
        a. get the endpoints
        For all branches j:
            a. get the endpoints
            b. compare the endpoints to the first
            c. if matching then add an edge


    """
    skeleton_connectivity_edge_list = []
    
    for j,sk_j in enumerate(skeletons):
        
        sk_j_ends = sk.find_branch_endpoints(sk_j)
        for i,sk_i in enumerate(skeletons):
            if i<=j:
                continue
            sk_i_ends = sk.find_branch_endpoints(sk_i)

            stacked_endpoints = np.vstack([sk_j_ends,sk_i_ends])
            endpoints_match = nu.get_matching_vertices(stacked_endpoints)

            if len(endpoints_match)>0:
                skeleton_connectivity_edge_list.append((j,i))

    return skeleton_connectivity_edge_list
            

    
def move_point_to_nearest_branch_end_point_within_threshold(
        skeleton,
        coordinate,
        distance_to_move_point_threshold = 1000,
        return_coordinate=True,
        return_change_status=True,
        verbose=False,
        consider_high_degree_nodes=True,
        possible_node_coordinates=None,
        excluded_node_coordinates=None
        ):
    """
    Purpose: To pick a branch or endpoint node that
    is within a certain a certain distance of the original 
    node (if none in certain distance then return original)
    
    Arguments: 
    possible_node_coordinates: this allows you to specify nodes that you want to select
    
    """
    
    #check that an exlucde point is not already included
    
    
    
    curr_skeleton_MAP = skeleton
    MAP_stitch_point = coordinate

    #get a network of the skeleton
    curr_skeleton_MAP_graph = sk.convert_skeleton_to_graph(curr_skeleton_MAP)
    #get the node where the stitching will take place
    node_for_stitch = xu.get_nodes_with_attributes_dict(curr_skeleton_MAP_graph,dict(coordinates=MAP_stitch_point))[0]
    #get all of the endnodes or high degree nodes
    
    if verbose:
        print(f"node_for_stitch = {node_for_stitch}: {xu.get_coordinate_by_graph_node(curr_skeleton_MAP_graph,node_for_stitch)}")
    # -------- 1/2 Make sure that stitch point is not part of the exclude node point, and if so then move it ---------#
    if not (excluded_node_coordinates is None):
        excluded_node_coordinates = excluded_node_coordinates.reshape(-1,3)
        possible_node_loc_to_exclude = np.array([xu.get_graph_node_by_coordinate(curr_skeleton_MAP_graph,zz,return_neg_one_if_not_find=True) for zz in excluded_node_coordinates])
        node_for_stitch=xu.move_node_from_exclusion_list(curr_skeleton_MAP_graph,
                                        exclusion_list=possible_node_loc_to_exclude,
                                        node=node_for_stitch,
                                        return_coordinate=False,
                                                        )
        # --- 1/4 addition that helped debug where the stitch point was being connected to an exclude node -- #
        MAP_stitch_point = xu.get_coordinate_by_graph_node(curr_skeleton_MAP_graph,node_for_stitch)
    if verbose:
        print(f"node_for_stitch AFTER = {node_for_stitch}: {xu.get_coordinate_by_graph_node(curr_skeleton_MAP_graph,node_for_stitch)}")
    
    # ----- 11/13 addition: Use the node locations sent or just use the high degree or end nodes from the graph
    if possible_node_coordinates is None:
        curr_MAP_end_nodes = xu.get_nodes_of_degree_k(curr_skeleton_MAP_graph,1)
        if consider_high_degree_nodes:
            curr_MAP_branch_nodes = xu.get_nodes_greater_or_equal_degree_k(curr_skeleton_MAP_graph,3)
        else:
            curr_MAP_branch_nodes = []
        possible_node_loc = np.array(curr_MAP_end_nodes + curr_MAP_branch_nodes)
    else:
        possible_node_loc = np.array([xu.get_graph_node_by_coordinate(curr_skeleton_MAP_graph,zz) for zz in possible_node_coordinates])
    
    if verbose:
        print(f"possible_node_loc = {possible_node_loc}")
    #removing the high degree coordinates that should not be there
    
    if not (excluded_node_coordinates is None):
        possible_node_loc = np.setdiff1d(possible_node_loc,possible_node_loc_to_exclude)
        
    if verbose:
        print(f"possible_node_loc AFTER = {possible_node_loc}")
    
    #get the distance along the skeleton from the stitch point to all of the end or branch nodes
    curr_shortest_path,end_node_1,end_node_2 = xu.shortest_path_between_two_sets_of_nodes(curr_skeleton_MAP_graph,
                                                                node_list_1=[node_for_stitch],
                                                                node_list_2=possible_node_loc)

    if verbose:
        print(f"curr_shortest_path = {curr_shortest_path}")

        
    changed_node = False
    if len(curr_shortest_path) == 1:
        if verbose:
             print(f"Current stitch point was a branch or endpoint")
        MAP_stitch_point_new = MAP_stitch_point
    else:
        
        #get the length of the path
        shortest_path_length = nx.shortest_path_length(curr_skeleton_MAP_graph,
                           end_node_1,
                           end_node_2,
                           weight="weight")

        if verbose:
            print(f"Current stitch point was not a branch or endpoint, shortest_path_length to one = {shortest_path_length}")
            
        if shortest_path_length < distance_to_move_point_threshold:
            if verbose:
                print(f"Changing the stitch point becasue the distance to end or branch node was {shortest_path_length}"
                     f"\nNew stitch point has degree {xu.get_node_degree(curr_skeleton_MAP_graph,end_node_2)}")
            
            MAP_stitch_point_new = end_node_2
            changed_node=True
        else:
            MAP_stitch_point_new = MAP_stitch_point
    
    if return_coordinate and changed_node:
        MAP_stitch_point_new = xu.get_node_attributes(curr_skeleton_MAP_graph,node_list=MAP_stitch_point_new)[0]
    
    return_value = [MAP_stitch_point_new]
    if return_change_status:
        return_value.append(changed_node)
        
    return return_value


def cut_skeleton_at_coordinate(skeleton,
                        cut_coordinate,
                              tolerance = 0.001, #if have to find cut point that is not already coordinate
                               verbose=False
                        ):
    """
    Purpose: 
    To cut a skeleton into 2 pieces at a certain cut coordinate
    
    Application: Used when the MP skeleton pieces 
    connect to the middle of a MAP branch and have to split it
    
    Example:
    ex_sk = neuron_obj[1][0].skeleton
    cut_coordinate = np.array([ 560252., 1121040.,  842599.])

    new_sk_cuts = sk.cut_skeleton_at_coordinate(skeleton=ex_sk,
                              cut_coordinate=cut_coordinate)

    ipvu.plot_objects(skeletons=new_sk_cuts,
                     skeletons_colors="random",
                     scatters=[cut_coordinate])
    
    
    """
    curr_MAP_sk_new = []
    #b) Convert the skeleton into a graph
    curr_MAP_sk_graph = sk.convert_skeleton_to_graph(skeleton)
    #c) Find the node of the MAP stitch point (where need to do the breaking)
    
    
    MP_stitch_node = xu.get_nodes_with_attributes_dict(curr_MAP_sk_graph,dict(coordinates=cut_coordinate))
    
    # --------- New Addition that accounts for if cut point is not an actual node but can interpolate between nodes -------#
    if len(MP_stitch_node) == 0: #then have to add the new stitch point
        current_point = cut_coordinate
        winning_edge = None
        
        for node_a,node_b in curr_MAP_sk_graph.edges:
            node_a_coord,node_b_coord = xu.get_node_attributes(curr_MAP_sk_graph,node_list=[node_a,node_b])

            AB = np.linalg.norm(node_a_coord-node_b_coord)
            AC = np.linalg.norm(node_a_coord-cut_coordinate)
            CB = np.linalg.norm(cut_coordinate-node_b_coord)

            if np.abs(AB - AC - CB) < tolerance:
                winning_edge = [node_a,node_b]
                winning_edge_coord = [node_a_coord,node_b_coord]
                if verbose:
                    print(f"Found winning edge: {winning_edge}")
                break
        if winning_edge is None:
            raise Exception("Cut point was neither a matching node nor a coordinate between 2 nodes ")
            
        new_node_name = np.max(curr_MAP_sk_graph.nodes()) + 1

        curr_MAP_sk_graph.add_nodes_from([(new_node_name,{"coordinates":cut_coordinate})])
        curr_MAP_sk_graph.add_weighted_edges_from([(winning_edge[k],
                                            new_node_name,
                                            np.linalg.norm(winning_edge_coord[k] - cut_coordinate)
                                           ) for k in range(0,2)])
        curr_MAP_sk_graph.remove_edge(winning_edge[0],winning_edge[1])

        MP_stitch_node = new_node_name
    else:
        MP_stitch_node = MP_stitch_node[0]
        
    # --------- End of Addition -------#
    
    #d) Find the degree one nodes
    curr_end_nodes_for_break = xu.get_nodes_of_degree_k(curr_MAP_sk_graph,1)

    #e) For each degree one node:
    for e_n in curr_end_nodes_for_break:
        #- Find shortest path from stitch node to end node
        stitch_to_end_path = nx.shortest_path(curr_MAP_sk_graph,MP_stitch_node,e_n)
        #- get a subgraph from that path
        stitch_to_end_path_graph = curr_MAP_sk_graph.subgraph(stitch_to_end_path)
        #- convert graph to a skeleton and save as new skeletons
        new_sk = sk.convert_graph_to_skeleton(stitch_to_end_path_graph)
        curr_MAP_sk_new.append(new_sk)

    return curr_MAP_sk_new



def smooth_skeleton_branch(skeleton,
                    neighborhood=2,
                    iterations=100,
                    coordinates_to_keep=None,
                    keep_endpoints=True,
    ):
    
    """
    Purpose: To smooth skeleton of branch while keeping the same endpoints

    Pseudocode:
    1) get the endpoint coordinates of the skeleton
    2) turn the skeleton into nodes and edges
    - if number of nodes is less than 3 then return
    3) Find the indexes that are the end coordinates
    4) Send the coordinates and edges off to get smoothed
    5) Replace the end coordinate smooth vertices with original
    6) Convert nodes and edges back to a skeleton
    
    Ex: 
    orig_smoothed_sk = smooth_skeleton(neuron_obj[limb_idx][branch_idx].skeleton,
                                  neighborhood=5)

    """
    
    



    #2) turn the skeleton into nodes and edges
    nodes,edges = sk.convert_skeleton_to_nodes_edges(skeleton)

    #- if number of nodes is less than 3 then return
    if len(nodes) < 3:
        print("Only 2 skeleton nodes so cannot do smoothing")
        return skeleton

    if not coordinates_to_keep is None:
        coordinates_to_keep = np.array(coordinates_to_keep).reshape(-1,3)
        
    if keep_endpoints:
        #1) get the endpoint coordinates of the skeleton
        curr_endpoints = sk.find_branch_endpoints(skeleton).reshape(-1,3)
        if not coordinates_to_keep is None:
            coordinates_to_keep = np.vstack([curr_endpoints,coordinates_to_keep]).reshape(-1,3)
        else:
            coordinates_to_keep = curr_endpoints
            

        
    #3) Find the indexes that are the end coordinates
    coordinates_to_keep_idx = [nu.matching_rows(nodes,k)[0] for k in coordinates_to_keep]
        

    #4) Send the coordinates and edges off to get smoothed
    
    smoothed_nodes = m_sk.smooth_graph(nodes,edges,neighborhood=neighborhood,iterations=iterations)

    #5) Replace the end coordinate smooth vertices with original
    for endpt_idx,endpt in zip(coordinates_to_keep_idx,coordinates_to_keep):
        smoothed_nodes[endpt_idx] = endpt

    #6) Convert nodes and edges back to a skeleton
    final_sk_smooth = sk.convert_nodes_edges_to_skeleton(smoothed_nodes,edges)

    return final_sk_smooth



def add_and_smooth_segment_to_branch(skeleton,
                              skeleton_stitch_point=None,
                              new_stitch_point=None,
                              new_seg=None,
                              resize_mult= 0.2,
                               n_resized_cutoff=3,
                               smooth_branch_at_end=True,
                                n_resized_cutoff_to_smooth=None,
                                     smooth_width = 100,
                                max_stitch_distance_for_smoothing=300,
                                verbose=False,
                               **kwargs,
                              ):
    """
    Purpose: To add on a new skeletal segment to a branch that will
    prevent the characteristic hooking when stitching a new point
    
    Pseudocode: 
    1) Get the distance of the stitch point = A
    2) Resize the skeleton to B*A (where B < 1)
    3) Find all nodes that are CA away from the MP stitch point
    4) Delete those nodes (except the last one and set that as the new stitch point)
    5) Make the new stitch

    Ex: When using a new segment
    orig_sk_func_smoothed = add_and_smooth_segment_to_branch(orig_sk,
                           new_seg = np.array([stitch_point_MAP,stitch_point_MP]).reshape(-1,2,3))
    """
    # 12/21 Addition: If the point you are trying to stitch to is already there then just return the skeleton
    if not new_stitch_point is None:
        sk_graph_at_beginning = sk.convert_skeleton_to_graph(skeleton)
        match_nodes_to_new_stitch_point = xu.get_nodes_with_attributes_dict(sk_graph_at_beginning,dict(coordinates=new_stitch_point))
        if len(match_nodes_to_new_stitch_point)>0:
            if verbose:
                print("New stitch point was already on the skeleton so don't need to add it")
            return skeleton
    
    
    if len(skeleton) == 0:
        raise Exception("The skeleton passed to the smoothing function was empty")
    
    orig_sk = skeleton
    orig_sk_segment_width = np.mean(sk.calculate_skeleton_segment_distances(orig_sk,cumsum=False))
    
    if skeleton_stitch_point is None or new_stitch_point is None:
        new_seg_reshaped = new_seg.reshape(-1,3)
        #try to get the stitch points from the new seg
        if ((len(sk.find_branch_skeleton_with_specific_coordinate([orig_sk],new_seg_reshaped[0])) > 0) and
            (len(sk.find_branch_skeleton_with_specific_coordinate([orig_sk],new_seg_reshaped[1])) == 0)):
            stitch_point_MP = new_seg_reshaped[0]
            stitch_point_MAP = new_seg_reshaped[1]
        elif ((len(sk.find_branch_skeleton_with_specific_coordinate([orig_sk],new_seg_reshaped[1])) > 0) and
            (len(sk.find_branch_skeleton_with_specific_coordinate([orig_sk],new_seg_reshaped[0])) == 0)):
            stitch_point_MP = new_seg_reshaped[1]
            stitch_point_MAP = new_seg_reshaped[0]
        else:
            raise Exception("Could not find a stitch point that was on the existing skeleton and one that was not")
    else:
        stitch_point_MAP = new_stitch_point
        stitch_point_MP = skeleton_stitch_point

    #1) Get the distance of the stitch point = A
    stitch_distance = np.linalg.norm(stitch_point_MAP-stitch_point_MP)
    if stitch_distance > max_stitch_distance_for_smoothing:
        if verbose:
            print(f"Using max stitch distance ({max_stitch_distance_for_smoothing}) for smoothing because stitch_distance greater ({stitch_distance}) ")
        stitch_distance = max_stitch_distance_for_smoothing
    
        

    #2) Resize the skeleton to B*A (where B < 1)
    
    orig_sk_resized = sk.resize_skeleton_branch(orig_sk,segment_width = resize_mult*stitch_distance)

    #3) Find all nodes that are CA away from the MP stitch point
    orig_resized_graph = sk.convert_skeleton_to_graph(orig_sk_resized)
    MP_stitch_node = xu.get_nodes_with_attributes_dict(orig_resized_graph,dict(coordinates=stitch_point_MP))

    if len(MP_stitch_node) == 1:
        MP_stitch_node = MP_stitch_node[0]
    else:
        raise Exception(f"MP_stitch_node not len = 1: len = {len(MP_stitch_node)}")

    nodes_within_dist = gu.dict_to_array(xu.find_nodes_within_certain_distance_of_target_node(orig_resized_graph,
                                                         target_node=MP_stitch_node,
                                                        cutoff_distance=n_resized_cutoff*stitch_distance,
                                                                            return_dict=True))
    farthest_node_idx = np.argmax(nodes_within_dist[:,1])
    farthest_node = nodes_within_dist[:,0][farthest_node_idx]
    new_stitch_point_MP = xu.get_node_attributes(orig_resized_graph,node_list=farthest_node)[0]

    #need to consider if the farthest node is an endpoint
    farthest_node_degree = xu.get_node_degree(orig_resized_graph,farthest_node)

    keep_branch = None
    if farthest_node_degree > 1:#then don't have to worry about having reached branch end
        cut_branches = sk.cut_skeleton_at_coordinate(orig_sk,cut_coordinate=new_stitch_point_MP)
        #find which branch had the original cut point
        branch_to_keep_idx = 1 - sk.find_branch_skeleton_with_specific_coordinate(cut_branches,stitch_point_MP)[0]
        keep_branch = cut_branches[branch_to_keep_idx]
    else:
        keep_branch = np.array([])

    # nodes_to_delete = np.delete(nodes_within_dist[:,0],farthest_node_idx)
    new_seg = np.array([[new_stitch_point_MP],[stitch_point_MAP]]).reshape(-1,2,3)
    final_sk = sk.stack_skeletons([new_seg,keep_branch]) 
    final_sk=sk.convert_graph_to_skeleton(sk.convert_skeleton_to_graph(final_sk))
    
    

    if smooth_branch_at_end:
        #resize the skeleton
        coordinates_to_keep=None
        #skeleton_reshaped = sk.resize_skeleton_branch(final_sk,segment_width=smooth_width)
        skeleton_reshaped = final_sk
        
        if len(keep_branch)>0:
            if n_resized_cutoff_to_smooth is None:
                n_resized_cutoff_to_smooth = n_resized_cutoff + 2
  
            """
            Pseudocode for smoothing only a certain portion:
            1) Convert the branch to a graph
            2) Find the node with the MP stitch point
            3) Find all nodes within n_resized_cutoff_to_smooth*stitch_distance
            4) Get all nodes not in that list
            5) Get all the coordinates of those nodes
            """
            #1) Convert the branch to a graph
            sk_gr = convert_skeleton_to_graph(skeleton_reshaped)
            #2) Find the node with the MP stitch point
            MP_stitch_node = xu.get_graph_node_by_coordinate(sk_gr,new_stitch_point_MP)
            MAP_stitch_node = xu.get_graph_node_by_coordinate(sk_gr,stitch_point_MAP)

            #3) Find all nodes within n_resized_cutoff_to_smooth*stitch_distance
            distance_to_smooth = n_resized_cutoff_to_smooth*stitch_distance
            nodes_to_smooth_pre = xu.find_nodes_within_certain_distance_of_target_node(sk_gr,target_node=MP_stitch_node,
                                                                cutoff_distance=distance_to_smooth)
            
            #need to add in nodes to endpoint in case the distance_to_smooth doesn't exend there
            nodes_to_MAP = nx.shortest_path(sk_gr,MP_stitch_node,MAP_stitch_node)

            nodes_to_smooth = np.unique(list(nodes_to_smooth_pre) + list(nodes_to_MAP))
            #print(f"nodes_to_smooth = {nodes_to_smooth}")

            #4) Get all nodes not in that list
            nodes_to_not_smooth = np.setdiff1d(list(sk_gr.nodes()),list(nodes_to_smooth))
            #print(f"nodes_to_smooth = {nodes_to_not_smooth}")

            #5) Get all the coordinates of those nodes
            coordinates_to_keep = xu.get_node_attributes(sk_gr,node_list=nodes_to_not_smooth)
            
        
        final_sk = smooth_skeleton_branch(skeleton_reshaped,coordinates_to_keep=coordinates_to_keep,**kwargs)
        
    #need to resize the final_sk
    if len(final_sk) == 0:
        """
        Pseudocode: 
        3) Create a skeleton segment from the skeleton_stitch_point to the new point
        4) Stack the skeletons
        5) Return 
        
        """
        print("The Skeleton at the end of smoothing was empty so just going to stitch the new point to skeleton without stitching")
        
        
        new_sk_seg = np.array([skeleton_stitch_point,new_stitch_point])
        final_sk = sk.stack_skeletons([skeleton,new_sk_seg])
        return final_sk
        
    else: 
        final_sk = sk.resize_skeleton_branch(final_sk,segment_width=orig_sk_segment_width)
        return final_sk

def number_connected_components(skeleton):
    """
    Will find the number of connected components in a whole skeleton
    
    """
    return nx.number_connected_components(convert_skeleton_to_graph(skeleton))

def number_connected_components_branches(skeleton_branches):
    """
    Will find the number of connected components in a list of skeleton branches
    
    """
    return nx.number_connected_components(convert_skeleton_to_graph(stack_skeletons(skeleton_branches)))

    
# ---------------- 11/26 Extra Utils for the Error Detection------------------
def endpoint_connectivity(endpoints_1,endpoints_2,
                         exceptions_flag=True,
                          return_coordinate=False,
                         print_flag=False):
    """
    Pupose: To determine where the endpoints of two branches are connected
    
    Example: 
    end_1 = np.array([[759621., 936916., 872083.],
       [790891., 913598., 806043.]])
    end_2 = np.array([[790891., 913598., 806043.],
       [794967., 913603., 797825.]])
       
    endpoint_connectivity(end_1,end_2)
    >> {0: 1, 1: 0}
    """
    connections_dict = dict()
    
    stacked_endpoints = np.vstack([endpoints_1,endpoints_2])
    endpoints_match = nu.get_matching_vertices(stacked_endpoints)
    
    if len(endpoints_match) == 0:
        print_string = f"No endpoints matching: {endpoints_match}"
        if exceptions_flag:
            raise Exception(print_string)
        else:
            print(print_string)
        return connections_dict
    
    if len(endpoints_match) > 1:
        print_string = f"Multiple endpoints matching: {endpoints_match}"
        if exceptions_flag:
            raise Exception(print_string)
        else:
            print(print_string)
    
    
    #look at the first connection
    first_match = endpoints_match[0]
    first_endpoint_match = first_match[0]
    
    if print_flag:
        print(f"first_match = {first_match}")
        print(f"first_endpoint_match = {endpoints_1[first_endpoint_match]}")
    
    if return_coordinate:
        return endpoints_1[first_endpoint_match]
    
    if 0 != first_endpoint_match and 1 != first_endpoint_match:
        raise Exception(f"Non 0,1 matching node in first endpoint: {first_endpoint_match}")
    else:
        connections_dict.update({0:first_endpoint_match})
        
    second_endpoint_match = first_match[-1]
    
    if 2 != second_endpoint_match and 3 != second_endpoint_match:
        raise Exception(f"Non 2,3 matching node in second endpoint: {second_endpoint_match}")
    else:
        connections_dict.update({1:second_endpoint_match-2})
    
    return connections_dict

def shared_endpoint(skeleton_1,skeleton_2,return_possibly_two=False):
    """
    Will return the endpoint that joins two branches
    """
    end_1 = find_branch_endpoints(skeleton_1)
    end_2 = find_branch_endpoints(skeleton_2)
    try:
        node_connectivity = endpoint_connectivity(end_1,end_2,print_flag=False,return_coordinate=True)
    except:
        if return_possibly_two:
            return np.unique(np.vstack([end_1,end_2]),axis=0)
        else:
            raise Exception("Not exactly one shared endpoint")
    else:
        return node_connectivity
    
def matching_endpoint_singular(
    array_1,
    array_2,
    return_indices = False,
    verbose = False,
    ):
    
    """
    purpose: To find the one matching coordinate between two arrays
    and can return the index of each
    """
    shared_coord = sk.shared_endpoint(array_1,
                                   array_2)
    if verbose:
        print(f"shared_coord = {shared_coord}")
    if return_indices:
        array_1_idx = nu.matching_row_index(array_1,shared_coord)
        array_2_idx = nu.matching_row_index(array_2,shared_coord)
        return array_1_idx,array_2_idx
    else:
        return shared_coord
    

def flip_skeleton(current_skeleton):
    """
    Will flip the absolute order of a skeleton
    """
    new_sk = np.flip(current_skeleton,0)
    return np.flip(new_sk,1)


def order_skeleton(skeleton,start_endpoint_coordinate=None,verbose=False,return_indexes=False,
                  error_on_non_start_coordinate = False):
    """
    Purpose: to get the skeleton in ordered vertices
    1) Convert to graph
    2) Find the endpoint nodes
    3) Find the shortest path between endpoints
    4) Get the coordinates of all of the nodes
    5) Create the skeleton by indexing into the coordinates by the order of the path

    """
    if start_endpoint_coordinate is None:
        start_endpoint_coordinate = sk.find_branch_endpoints(skeleton)[0]
        
    #1) Convert to graph
    sk_graph = convert_skeleton_to_graph(skeleton)
    #2) Find the endpoint nodes
    sk_graph_endpt_nodes = np.array(xu.get_nodes_of_degree_k(sk_graph,1))
    if verbose:
        print(f"sk_graph_endpt_nodes = {sk_graph_endpt_nodes}")
    
    #2b) If a starting endpoint coordinate was picked then use that
    if not start_endpoint_coordinate is None:
        if verbose:
            print(f"Using start_endpoint_coordinate = {start_endpoint_coordinate}")
        curr_st_node = xu.get_graph_node_by_coordinate(sk_graph,start_endpoint_coordinate)
        start_node_idx = np.where(sk_graph_endpt_nodes==curr_st_node)[0]
        if len(start_node_idx) == 0:
            if error_on_non_start_coordinate:
                raise Exception(f"The start endpoint was not an end node: {start_endpoint_coordinate}")
            else:
                print(f"Warning: start endpoint was not an end node: {start_endpoint_coordinate} but not erroring")
            first_start_node = curr_st_node
        else:
            if verbose:
                print(f"start_node_idx = {start_node_idx}")
            start_node_idx = start_node_idx[0]
            first_start_node = sk_graph_endpt_nodes[start_node_idx]
    else:
        start_node_idx = 0
        first_start_node = sk_graph_endpt_nodes[start_node_idx]
        
    
    leftover_start_nodes = sk_graph_endpt_nodes[sk_graph_endpt_nodes!=first_start_node]
    if len(leftover_start_nodes) == 1:
        other_end_node = leftover_start_nodes[0]
    else:
        #find the 
        shortest_path,orig_st,other_end_node = xu.shortest_path_between_two_sets_of_nodes(sk_graph,[first_start_node],list(leftover_start_nodes))

    #3) Find the shortest path between endpoints
    shortest_path = np.array(nx.shortest_path(sk_graph,first_start_node,other_end_node)).astype("int")
    
    if verbose:
        print(f"shortest_path = {shortest_path}")


    #4) Get the coordinates of all of the nodes
    node_coordinates = xu.get_node_attributes(sk_graph,node_list = shortest_path)

    #5) Create the skeleton by indexing into the coordinates by the order of the path
    
    ordered_skeleton = np.stack((node_coordinates[:-1],node_coordinates[1:]),axis=1)
    
    if return_indexes:
        new_edges = np.sort(np.stack((shortest_path[:-1],shortest_path[1:]),axis=1),axis=1)
        original_edges = np.sort(sk_graph.edges_ordered(),axis=1)
        
        orig_indexes = [nu.matching_rows_old(original_edges,ne)[0] for ne in new_edges]
        return ordered_skeleton,orig_indexes
    
    return ordered_skeleton


def order_skeletons_connecting_endpoints(skeletons,
                                         starting_endpoint_coordinate
                        ):
    """
    Purpose: To get a list of the endpoints in the order
    that the list of skeletons are connected
    
    Ex: 
    order_skeletons_connecting_endpoints([neuron_obj[0][k].skeleton for
                                                              k in ex_path],
                                    starting_endpoint_coordinate=st_coord)
    
    
    """
    total_skeleton = sk.stack_skeletons([sk.resize_skeleton_branch(s,
                                                              n_segments=1) for s in skeletons])
    return sk.order_skeleton(total_skeleton,start_endpoint_coordinate=starting_endpoint_coordinate)
    


def align_skeletons_at_connectivity(sk_1,sk_2):
    """
    To align 2 skeletons where both starts with the endpoint
    that they share
    
    Ex: 
    
    
    """
    common_coordinate = shared_endpoint(sk_1,sk_2)
    sk_1 = order_skeleton(sk_1,start_endpoint_coordinate=common_coordinate)
    sk_2 = order_skeleton(sk_2,start_endpoint_coordinate=common_coordinate)
    return sk_1,sk_2




def restrict_skeleton_from_start(skeleton,
                     cutoff_distance,
                    subtract_cutoff = False,
                    return_indexes = True,
                    return_success = True,
                    resize_skeleton_to_help_success = False,
                    tolerance = 10,
                    starting_coordinate = None):
    """
    To restrict a skeleton to a certain cutoff distance from the start
    which keeps that distance or subtracts it (and does not resize or reorder the skeleton but keeps the existing segment lengths)
    
    Ex: 
    restrict_skeleton_from_start(skeleton = base_skeleton_ordered,
    cutoff_distance = offset)

    **** warning only works if skeletal segment *****\
    """
    
    #handling if the cutof is 0
    if cutoff_distance <= 0:
            return_values = [skeleton]
            if return_indexes:
                return_values.append(np.arange(0,len(skeleton)))
            if return_success:
                return_values.append(True)
            return return_values
        
    if starting_coordinate is not None:
        skeleton = sk.order_skeleton(skeleton,starting_coordinate)

    if resize_skeleton_to_help_success:
        skeleton = sk.resize_skeleton_branch(skeleton,
                                    segment_width = tolerance)
        
    distance_of_segs = calculate_skeleton_segment_distances(skeleton,cumsum=False)
    offset_idxs = np.where(np.cumsum(distance_of_segs)>=(cutoff_distance-tolerance))[0]
    if len(offset_idxs)>0:
        offset_idxs = offset_idxs[1:]

    subtract_idxs = np.delete(np.arange(len(distance_of_segs)),offset_idxs)
        
    subtract_sk = skeleton[subtract_idxs]
    subtract_sk_len = calculate_skeleton_distance(subtract_sk)
#     print(f"subtract_sk_len = {subtract_sk_len}")
#     print(f"(cutoff_distance-tolerance) = {(cutoff_distance-tolerance)}")
    success_subtraction = subtract_sk_len >= (cutoff_distance-tolerance)
#     print(f"success_subtraction = {success_subtraction}")

    #flip the indexes if want to keep the segment
    if not subtract_cutoff: 
        keep_indexes = np.delete(np.arange(len(distance_of_segs)),offset_idxs)
    else:
        keep_indexes = offset_idxs

    #restrict the skeleton
    return_sk = skeleton[keep_indexes]
    
    if len(return_sk) == 0:
        return_values = [skeleton]
        if return_indexes:
            return_values.append(np.arange(0,len(skeleton)))
        if return_success:
            return_values.append(False)
        return return_values

    return_values = [return_sk]

    if return_indexes:
        return_values.append(keep_indexes)
    if return_success:
        return_values.append(success_subtraction)

    return return_values



def matching_skeleton_branches_by_vertices(branches):
    
    decomposed_branches = branches
    kdtree_branches = [KDTree(k.reshape(-1,3)) for k in decomposed_branches]
    matching_edges_kdtree = []
    for i,d_br_1 in tqdm(enumerate(decomposed_branches)):
        for j,d_br_2 in enumerate(decomposed_branches):
            if i < j:
                dist, nearest = kdtree_branches[i].query(d_br_2.reshape(-1,3))
                if sum(dist) == 0:
                    matching_edges_kdtree.append([i,j])
                    
    return matching_edges_kdtree
                    
                    
def matching_skeleton_branches_by_endpoints(branches):
    matching_edges = []
    decomposed_branches = branches
    
    for i,d_br_1 in tqdm(enumerate(decomposed_branches)):
        for j,d_br_2 in enumerate(decomposed_branches):
            if i < j:
                c_t = time.time()
                br_1_end = sk.find_branch_endpoints(d_br_1)
                br_2_end = sk.find_branch_endpoints(d_br_2)
                #print(f"branch: {time.time() - c_t}")
                c_t = time.time()
                if sk.compare_endpoints(br_1_end,br_2_end):
                    matching_edges.append([i,j])
    return matching_edges
    

def check_correspondence_branches_have_2_endpoints(correspondence,
                                                  verbose=True,
                                                  raise_error= True):
    """
    Purpose: check that all branches have 2 endpoints
    """

    irregular_branches = []
    for piece_idx,piece_correspondence in correspondence.items():
        
        if "branch_skeleton" in piece_correspondence.keys():
            k = piece_idx
            v = piece_correspondence
            
            curr_sk = v["branch_skeleton"]
            curr_sk_endpoint_coord = sk.find_skeleton_endpoint_coordinates(curr_sk)
            if len(curr_sk_endpoint_coord) != 2:
                if verbose:
                    print(f"Branch {k} had {len(curr_sk_endpoint_coord)} endpoints")
                irregular_branches.append([piece_idx,k,len(curr_sk_endpoint_coord)])
        else:
            for k,v in piece_correspondence.items():
                curr_sk = v["branch_skeleton"]
                curr_sk_endpoint_coord = sk.find_skeleton_endpoint_coordinates(curr_sk)
                if len(curr_sk_endpoint_coord) != 2:
                    if verbose:
                        print(f"Piece {piece_idx}, Branch {k} had {len(curr_sk_endpoint_coord)} endpoints")
                    irregular_branches.append([piece_idx,k,len(curr_sk_endpoint_coord)])
    if raise_error and len(irregular_branches)>0:
        raise Exception(f"Found the following irregular branches: {irregular_branches}")
        
    return irregular_branches


# ---------------- 12/23 -------------------- #
def offset_skeletons_aligned_at_shared_endpoint(skeletons,
                                               offset=1000,
                                            comparison_distance=2000,
                                                min_comparison_distance=1000,
                                            verbose=True,
                                                common_endpoint=None,
                                                comparison_endpoints = None,
                                                subtract_cutoff=True
                                               ):

    """
    Pseudocode: 

    1) Get the shared endpoint of the branches
    2) Reorder the branches so both start with the endpoint and then resize

    For each edge skeleton (in order to get the final edge skeletons):
    3) Use the restrict skeelton function to subtract the offset
    - if not then add whole skeleton to final skeleton
    4) if it was a sucess, see if the distance is greater than comparison distance
    - if not then add current skeleton to final
    5) Use the subtract skeleton to only keep the comparison distance of skeleton
    6) Add to final skeleton

    offset_skeletons_aligned_at_shared_endpoint()
    
    Ex: 
    vis_branches_idx = [7,9]
    vis_branches = [curr_limb[k] for k in vis_branches_idx]
    vis_branches


    curr_skeletons = [k.skeleton for k in vis_branches]
    stripped_skeletons = sk.offset_skeletons_aligned_at_shared_endpoint(curr_skeletons)

    curr_colors = ["red","black"]
    ipvu.plot_objects(meshes=[k.mesh for k in vis_branches],
                      meshes_colors=curr_colors,
                      skeletons=stripped_skeletons,
                      skeletons_colors=curr_colors,
                      scatters=[np.array([stripped_skeletons[0][-1][-1],stripped_skeletons[1][-1][-1]])],
                      scatter_size=1
                     )


    sk.parent_child_skeletal_angle(stripped_skeletons[1],stripped_skeletons[0])


    """
    edge_skeletons = skeletons
    seg_size = 100

    if common_endpoint is None:
        common_endpoint = sk.shared_endpoint(edge_skeletons[0],edge_skeletons[1],return_possibly_two=True)
        if common_endpoint.ndim > 1:
            print("More than one common endpoint so just choosing the first")
            common_endpoint = common_endpoint[0]
    
    
    if comparison_endpoints is None:
        edge_skeletons_ordered = [sk.order_skeleton(sk.resize_skeleton_branch(e,seg_size),common_endpoint) for e in edge_skeletons]
    else:
        edge_skeletons_ordered = [sk.order_skeleton(sk.resize_skeleton_branch(e,seg_size),cm) for e,comm in zip(edge_skeletons,comparison_endpoints)]
    
    final_skeletons = []
    for e in edge_skeletons_ordered:
        
        # -------- Making sure that we don't take off too much so it's just a spec
        original_sk_length = sk.calculate_skeleton_distance(e)
        if original_sk_length < offset + min_comparison_distance:
            offset_adjusted = original_sk_length - min_comparison_distance
            if offset_adjusted < 0:
                offset_adjusted = 0
                
            #print(f" Had to Adjust Offset to {offset_adjusted}")
        else:
            offset_adjusted  = offset
            
        ret_sk,_,success = sk.restrict_skeleton_from_start(e,
                                                           cutoff_distance = offset_adjusted,
                                                           subtract_cutoff=True)
        if not success:
            final_skeletons.append(e)
        else:
            if sk.calculate_skeleton_distance(ret_sk) > comparison_distance:
                ret_sk,_,success = sk.restrict_skeleton_from_start(ret_sk,
                                                           cutoff_distance = comparison_distance,
                                                           subtract_cutoff=False)
            final_skeletons.append(ret_sk)
    return final_skeletons


def parent_child_skeletal_angle(parent_skeleton,child_skeleton):
    """
    To find the angle from continuation that the
    second skeleton deviates from the parent angle 
    
    angles are just computed from the vectors of the endpoints
    
    """
    up_sk = parent_skeleton
    d_sk = child_skeleton
    
    up_sk_flipped = sk.flip_skeleton(up_sk)

    up_vec = up_sk_flipped[-1][-1] - up_sk_flipped[0][0] 
    d_vec_child = d_sk[-1][-1] - d_sk[0][0]

    parent_child_angle = np.round(nu.angle_between_vectors(up_vec,d_vec_child),2)
    return parent_child_angle




def offset_skeletons_aligned_parent_child_skeletal_angle(skeleton_1,skeleton_2,
                                                        offset=1000,
                                                        comparison_distance=2000,
                                                        min_comparison_distance=1000):
    """
    Purpose: To determine the parent child skeletal angle
    of 2 skeletons while using the offset and comparison distance
    
    
    
    """
    
    edge_skeletons = [skeleton_1,skeleton_2]
    aligned_sk_parts = sk.offset_skeletons_aligned_at_shared_endpoint(edge_skeletons,
                                                                     offset=offset,
                                                        comparison_distance=comparison_distance,
                                                        min_comparison_distance=min_comparison_distance)


    curr_angle = sk.parent_child_skeletal_angle(aligned_sk_parts[0],aligned_sk_parts[1])
    return curr_angle



def map_between_branches_lists(branches_1,branches_2,check_all_matched=True,
                              min_to_match = 2):
    """
    Purpose: 
    Will create a unique mapping of a branch
    in the first list to the best fitting branch in the second
    in terms of the most matching coordinates with a distance of 0
    
    min_to_match is the number of vertices that must match in order to
    be considered for the matching 
    Ex:
    cleaned_branches = sk.decompose_skeleton_to_branches(curr_limb_sk_cleaned)
    original_branches = [k.skeleton for k in curr_limb]
    map_between_branches_lists(original_branches,cleaned_branches)
    """
    original_branches = branches_1
    cleaned_branches = branches_2
    
    reshape_skeletons = [k.reshape(-1,3) for k in cleaned_branches]
    reshaped_skeleton_labels = np.hstack([[i]*len(k) for i,k in enumerate(reshape_skeletons)])
    stacked_reshaped_skeletons = np.vstack(reshape_skeletons)

    old_to_new_branch_mapping = []

    for o_br in tqdm(original_branches):
        o_br_kd = KDTree(o_br.reshape(-1,3))

        zero_distance_points = np.where(o_br_kd.query(stacked_reshaped_skeletons)[0]==0)[0]
        zero_distance_branch_idx = reshaped_skeleton_labels[zero_distance_points]

        n_matches = [len(np.where(zero_distance_branch_idx==k)[0]) for k,br_data in enumerate(cleaned_branches)]
        
        max_matched_index = np.argmax(n_matches)
        max_matched_index_number = n_matches[max_matched_index]
        
        if max_matched_index_number >= min_to_match:
            old_to_new_branch_mapping.append(max_matched_index)
        else:
            old_to_new_branch_mapping.append(-1)
        
    old_to_new_branch_mapping = np.array(old_to_new_branch_mapping)
    if check_all_matched:
        if len(np.unique(old_to_new_branch_mapping[old_to_new_branch_mapping!=-1])) < len(branches_2):
            raise Exception("Not all of the new branches had at least one mapping")
    return old_to_new_branch_mapping

def map_between_branches_lists_old(branches_1,branches_2,check_all_matched=True,
                              min_to_match = 2):
    """
    Purpose: 
    Will create a unique mapping of a branch
    in the first list to the best fitting branch in the second
    in terms of the most matching coordinates with a distance of 0
    
    min_to_match is the number of vertices that must match in order to
    be considered for the matching 
    Ex:
    cleaned_branches = sk.decompose_skeleton_to_branches(curr_limb_sk_cleaned)
    original_branches = [k.skeleton for k in curr_limb]
    map_between_branches_lists(original_branches,cleaned_branches)
    """
    original_branches = branches_1
    cleaned_branches = branches_2
    
    cleaned_branches_endpoints = [np.array(sk.find_skeleton_endpoint_coordinates(k)).reshape(-1,3) for k in cleaned_branches]
    
    old_to_new_branch_mapping = []

    for o_br in tqdm(original_branches):
        o_br_kd = KDTree(o_br.reshape(-1,3))

        #n_matches = [len(np.where(o_br_kd.query(c_br.reshape(-1,3))[0]==0)[0]) for c_br in cleaned_branches]
        n_matches = [len(np.where(o_br_kd.query(c_br)[0]==0)[0]) for c_br in cleaned_branches_endpoints]
        
        max_matched_index = np.argmax(n_matches)
        max_matched_index_number = n_matches[max_matched_index]
        
        if max_matched_index_number >= min_to_match:
            old_to_new_branch_mapping.append(max_matched_index)
        else:
            old_to_new_branch_mapping.append(-1)
        
    old_to_new_branch_mapping = np.array(old_to_new_branch_mapping)
    if check_all_matched:
        if len(np.unique(old_to_new_branch_mapping[old_to_new_branch_mapping!=-1])) < len(branches_2):
            raise Exception("Not all of the new branches had at least one mapping")
    return old_to_new_branch_mapping



# ---------------- 1/17: Helper function with the axon identification----- #

# ---------------- 12/23 -------------------- #
def restrict_skeleton_from_start_plus_offset(skeleton,
                                               offset=1000,
                                            comparison_distance=2000,
                                                min_comparison_distance=1000,
                                            verbose=True,
                                             start_coordinate=None,
                                             skeleton_resolution=100,
                                             plot = False,
                                               ):

    """
    Purpose: Will get a portion of the skeleton relative the start 
    that 
    1) subtracts off the offset
    2) keeps the next comparison distance length
    
    Pseudocode: 

    3) Use the restrict skeelton function to subtract the offset
    - if not then add whole skeleton to final skeleton
    4) if it was a sucess, see if the distance is greater than comparison distance
    - if not then add current skeleton to final
    5) Use the subtract skeleton to only keep the comparison distance of skeleton
    6) Add to final skeleton

    offset_skeletons_aligned_at_shared_endpoint()
    
    Ex: 
    vis_branches_idx = [7,9]
    vis_branches = [curr_limb[k] for k in vis_branches_idx]
    vis_branches


    curr_skeletons = [k.skeleton for k in vis_branches]
    stripped_skeletons = sk.offset_skeletons_aligned_at_shared_endpoint(curr_skeletons)

    curr_colors = ["red","black"]
    ipvu.plot_objects(meshes=[k.mesh for k in vis_branches],
                      meshes_colors=curr_colors,
                      skeletons=stripped_skeletons,
                      skeletons_colors=curr_colors,
                      scatters=[np.array([stripped_skeletons[0][-1][-1],stripped_skeletons[1][-1][-1]])],
                      scatter_size=1
                     )


    sk.parent_child_skeletal_angle(stripped_skeletons[1],stripped_skeletons[0])


    """
    
    if start_coordinate is not None:
        skeleton = sk.order_skeleton(skeleton,start_coordinate)
        
    if skeleton_resolution is not None:
        skeleton = sk.resize_skeleton_branch(skeleton,skeleton_resolution)
    
    e = skeleton
        
    # -------- Making sure that we don't take off too much so it's just a spec
    original_sk_length = sk.calculate_skeleton_distance(e)
    if original_sk_length < offset + min_comparison_distance:
        offset_adjusted = original_sk_length - min_comparison_distance
        if offset_adjusted < 0:
            offset_adjusted = 0

        #print(f" Had to Adjust Offset to {offset_adjusted}")
    else:
        offset_adjusted  = offset

    ret_sk,_,success = sk.restrict_skeleton_from_start(e,
                                                       cutoff_distance = offset_adjusted,
                                                       subtract_cutoff=True)
    if not success:
        ret_sk = e
    else:
        if sk.calculate_skeleton_distance(ret_sk) > comparison_distance:
            ret_sk,_,success = sk.restrict_skeleton_from_start(ret_sk,
                                                       cutoff_distance = comparison_distance,
                                                       subtract_cutoff=False)
    if sk.calculate_skeleton_distance(ret_sk) == 0:
        if verbose:
            print("Ended with empty skeleton at end so returning original skeleton")
            
        ret_sk = skeleton
        
    if plot:
        if len(ret_sk) > 0:
            ipvu.plot_objects(
                                 skeletons=[ret_sk,skeleton],
                                 skeletons_colors=["green","blue"],
                                  scatters=[start_coordinate,ret_sk[0][0],ret_sk.reshape(-1,3)],
                                  scatters_colors=["red","green","green"],
                                scatter_size = [0.5,0.5,0.2],
                                 )
        else:
            print(f"No skeleton to plot")
        
    return ret_sk


def skeleton_endpoint_vector(skeleton,
                             normalize_vector=True,
                             starting_coordinate = None):
    """
    Purpose: To get the vector made by the endpoints of a skeleton
    
    Pseudocode: 
    1) if the starting coordinate is specified then order the skeleton 
    according to that starting coordinate
    2) Get the endpoints of the skeleton
    3) Subtract the endpoints from each other (Normalize if requested)
    
    
    """
    if starting_coordinate is not None:
        restricted_skeleton = sk.order_skeleton(skeleton,
                                                            start_endpoint_coordinate=starting_coordinate)
    else:
        restricted_skeleton = skeleton
    
    restricted_skeleton_endpoints_sk = np.array([restricted_skeleton[0][0],restricted_skeleton[-1][-1]]).reshape(-1,2,3)
    
    restricted_skeleton_vector = np.array(restricted_skeleton[-1][-1]-restricted_skeleton[0][0])
    
    if normalize_vector:
        restricted_skeleton_vector = restricted_skeleton_vector/np.linalg.norm(restricted_skeleton_vector)
        
    return restricted_skeleton_vector

    
    
# ----------- 1 /21  -------------------- #
def high_degree_coordinates_on_path(limb_obj,curr_path_to_cut,
                                   degree_to_check=4):
    """
    Purpose: Find coordinates on a skeleton of the path speciifed (in terms of node ids)
    that are above the degree_to_check (in reference to the skeleton)
    
    
    """
    path_divergent_points = [sk.find_branch_endpoints(limb_obj[k].skeleton) for k in curr_path_to_cut]
    endpoint_coordinates = np.unique(np.concatenate(path_divergent_points),axis=0)

    limb_sk_gr = sk.convert_skeleton_to_graph(limb_obj.skeleton)
    endpoint_degrees = xu.get_coordinate_degree(limb_sk_gr,endpoint_coordinates)
    high_degree_endpoint_coordinates = endpoint_coordinates[endpoint_degrees>=degree_to_check]
    
    return high_degree_endpoint_coordinates


# ---------- 2/15: Help with Getting path from synapses from to soma ------- #



def skeleton_path_between_skeleton_coordinates(
    starting_coordinate,
    destination_coordinate=None,
    skeleton=None,
    skeleton_graph = None,
    destination_node = None,
    only_skeleton_distance = False,
    plot_skeleton_path = False,
    return_singular_node_path_if_no_path = False,
    verbose = False,
    ):
    """ 

    Purpose: To find the skeleton_path_between two coordinates (that lie on a skeleton)
    (can just request the distances and not the path)


    Pseudocode:
    0) Convert the skeleton into a graph
    1) Find the graph node of the destination coordinate
    2) For each coordinate:
    - find the node in the skeleton graph

    a. if skeleton path is requested
    - find the shortest path between the start and destiation node (and convert )
    - find the shortest path length between start node and destination node

    b. if only skeleton distance is requested
    - find shortest path distance between 2 nodes

    3) return list of skeleton paths or skeleton path distances


    Ex:
    skeleton_path_between_skeleton_coordinates(
            #skeleton = filtered_neuron[0].skeleton,
    skeleton = None,
            starting_coordinate = skeleton[[10000]].reshape(-1,3),

            destination_coordinate = None,
            skeleton_graph = limb_graphs[0],
            destination_node = 15310,
            only_skeleton_distance = False,
            plot_skeleton_path = True,
            verbose = True,)
    """
    total_st = time.time()
    st = time.time()
    debug_time = False

    #0) Convert the skeleton into a graph
    if skeleton_graph is None:
        skeleton_graph = sk.convert_skeleton_to_graph(skeleton)
        
        if debug_time:
            print(f"Skeleton Graph: {np.round(time.time() - st,4)}")
            st = time.time()

    #1) Find the graph node of the destination coordinate
    if destination_node is None:
        destination_node= xu.get_graph_node_by_coordinate(
            skeleton_graph,
            destination_coordinate,
            return_single_value=True
        )
        if debug_time:
            print(f"destination_node: {np.round(time.time() - st,4)}")
            st = time.time()

    if verbose:
        print(f"destination_node= {destination_node}")

    singular_output_flag = False
    if starting_coordinate.ndim == 1:
        singular_output_flag = True
        starting_coordinate = [starting_coordinate]

    """
    2) For each coordinate:
    - find the node in the skeleton graph

    a. if skeleton path is requested
    - find the shortest path between the start and destiation node (and convert to a skeleton)

    b. if only skeleton distance is requested
    - find shortest path distance between 2 nodes



    """
    if plot_skeleton_path and skeleton is None:
        skeleton = sk.convert_graph_to_skeleton(skeleton_graph)

    if plot_skeleton_path and destination_coordinate is None:
        destination_coordinate = xu.get_coordinate_by_graph_node(skeleton_graph,
                                                                destination_node)

    output = []
    for st_coord in starting_coordinate:
        st_node = xu.get_graph_node_by_coordinate(skeleton_graph,
                                                    st_coord,
                                                    return_single_value=True)
        
        if debug_time:
            print(f"st_node: {np.round(time.time() - st,4)}")
            st = time.time()

        if only_skeleton_distance:
            # how to generate the shortest path length
            shortest_path_length = nx.shortest_path_length(skeleton_graph,
                                       st_node,
                                       destination_node,
                                       weight="weight")
            output.append(shortest_path_length)
            if verbose:
                print(f"shortest_path_length = {shortest_path_length}")
            
            if debug_time:
                print(f"shortest_path_length: {np.round(time.time() - st,4)}")
                st = time.time()
        else:
            shortest_path = nx.shortest_path(skeleton_graph,
                            st_node,
                            destination_node)
            if verbose:
                print(f"shortest_path = {shortest_path}")
            curr_skeleton = sk.convert_graph_to_skeleton(
                    skeleton_graph.subgraph(shortest_path)
            )
            if len(curr_skeleton) == 0 and return_singular_node_path_if_no_path:
                curr_skeleton = np.vstack([starting_coordinate,starting_coordinate]).reshape(-1,3)
            
            output.append(curr_skeleton)

            if plot_skeleton_path:
                ipvu.plot_objects(main_skeleton=skeleton,
                                 skeletons=[curr_skeleton],
                                 skeletons_colors="red",
                                 scatters=[st_coord,destination_coordinate],
                                 scatters_colors=["red","blue"],
                                 scatter_size=1)
                
    if singular_output_flag:
        output = output[0]
        
    if debug_time:
        print(f"whole time: {np.round(time.time() - total_st,4)}")
        st = time.time()
    return output


def closest_skeleton_coordinate(skeleton,
                               coordinate):
    """
    Purpose: will map a coordinate to the closest skeleton coordinate
    
    Pseudocode:
    1) Turn the skeleton into a KDTree of coordinates
    2) Find the closest coordinate
    """
    skeleton_points = skeleton.reshape(-1,3)
    skeleton_kd = KDTree(skeleton_points)
    dist,closest_coord = skeleton_kd.query(np.array([coordinate]).reshape(-1,3))
    return skeleton_points[closest_coord[0]]

def kd_tree_from_unique_coordinates(skeleton):
    unique_coords = sk.skeleton_unique_coordinates(skeleton)
    return KDTree(unique_coords)

def closest_skeleton_coordinates(
    coordinates,
    skeleton=None,
    radius=None,
    verbose = False,
    plot = False,
    skeleton_coordinates = None,
    return_idx = False,
    return_distances = False,
    ):
    """
    Purpose: will map a coordinate to the closest skeleton coordinate
    
    Pseudocode:
    1) Turn the skeleton into a KDTree of coordinates
    2) Find the closest coordinate
    
    Ex: 
    sk.closest_skeleton_coordinates(
        restr_skeleton,
        coordinates = soma_center,
        plot = True,
        verbose = True,
        radius = 50_000,
    )
    """
    coordinates = np.array(coordinates).reshape(-1,3).astype('float')
    
    if skeleton_coordinates is None:
        unique_coords = sk.skeleton_unique_coordinates(skeleton).astype('float')
    else:
        unique_coords = np.array(skeleton_coordinates).astype("float")
    
    
    if radius is None:
        skeleton_kd = KDTree(unique_coords)
        total_dists,closest_coords_idx = skeleton_kd.query(coordinates)
        
        closest_coords_idx = np.unique(closest_coords_idx)
        closest_coords = unique_coords[closest_coords_idx]
    else:
        closest_coords_idx = []
        total_dists = []
        for c in coordinates:
            dists = np.linalg.norm(unique_coords-c,axis=1)
            closest_coords_idx_curr = np.where(dists < radius)[0]
            if len(closest_coords_idx_curr) > 0:
                closest_coords_idx.append(closest_coords_idx_curr)
                total_dists.append(dists[closest_coords_idx_curr])
        
        if len(closest_coords_idx) > 0:
            closest_coords_idx = np.hstack(closest_coords_idx)
            total_dists = np.hstack(total_dists)
            closest_coords_idx = np.unique(closest_coords_idx)
            
        
            
        closest_coords = unique_coords[closest_coords_idx]
        
    if verbose:
        print(f"Number of coordinates found = {len(closest_coords)}")
        
#     if len(closest_coords) == 0:
#         raise Exception("")
        
    if plot and skeleton is not None:
        coord_color = "blue"
        skeleton_coord_color = "red"
        print(f"coordinates = {coord_color}, found skeleton colors = {skeleton_coord_color}")
        
        ipvu.plot_objects(
            main_skeleton = skeleton,
            scatters = [coordinates,closest_coords],
            scatters_colors = [coord_color,skeleton_coord_color]
        )
        
    if return_idx:
        return closest_coords_idx
    
    if return_distances:
        return total_dists
        
    return closest_coords
        
    

def high_degree_coordinates_on_skeleton(skeleton,
                                       min_degree_to_find=4,
                                             exactly_equal=False,
                                             verbose=False,
                                       plot_high_degree_points=False):
    """
    Purpose: To find the high degree coordinates on a skeleton
    
    """
    if len(skeleton) == 0:
        return []
    
    limb_sk_gr = sk.convert_skeleton_to_graph(skeleton)
    #3) Find all nodes with a degree above five
    if not exactly_equal:
        curr_high_degree_nodes = xu.get_nodes_greater_or_equal_degree_k(limb_sk_gr,min_degree_to_find)
        if verbose:
            print(f"curr_high_degree_nodes for get_nodes_greater_or_equal_degree_k = {curr_high_degree_nodes}")
    else:
        curr_high_degree_nodes = xu.get_nodes_of_degree_k(limb_sk_gr,min_degree_to_find)
        if verbose:
            print(f"curr_high_degree_nodes for get_nodes_of_degree_k = {curr_high_degree_nodes}")


    if len(curr_high_degree_nodes)>0:

        #4) Get the coordinates of all of those nodes
        curr_high_degree_coordinates = xu.get_coordinate_by_graph_node(limb_sk_gr,curr_high_degree_nodes)

        if verbose:
            print(f"curr_high_degree_coordinates = {curr_high_degree_coordinates}")
    else:
        curr_high_degree_coordinates = []
            
            
    if plot_high_degree_points:
        if len(curr_high_degree_coordinates)==0:
            print("*** No HIGH DEGREE COORDINATES TO PLOT ****")
        else:
            ipvu.plot_objects(skeletons = [skeleton],
                             scatters=curr_high_degree_coordinates,
                             scatters_colors="red",
                             scatter_size=0.3)
    return curr_high_degree_coordinates
    
    
def shortest_path_between_two_sets_of_skeleton_coordiantes(
        skeleton,
        coordinates_list_1,
        coordinates_list_2,
        return_closest_coordinates = False,
        return_path_distance = False,
        plot_closest_coordinates = False,
    
        G = None,
    
    ):

    """
    Purpose: to find the shortest skeleton path between 2 sets
    of skeleton coordinates

    Pseudocode: 
    1) convert the skeleton into a graph
    2) Find the nodes of the skeleton coordinates in both groups
    3) Find the shortest path between the nodes (along with the winning nodes)
    4) Find the coordinates of these end nodes (if just asked for this return here)
    5) Get a subgraph of the shortest path and convert back into a skeleton
    
    Ex:
    shortest_path_between_two_sets_of_skeleton_coordiantes(
                    skeleton = limb_skeleton,
                    coordinates_list_1 = [starting_coordinate],
                    coordinates_list_2 = closest_branch_endpoints,
                    return_closest_coordinates = True,
                    plot_closest_coordinates = True)
    """



    coordinates_list_1 = np.array(coordinates_list_1).reshape(-1,3)
    coordinates_list_2 = np.array(coordinates_list_2).reshape(-1,3)

    #1) convert the skeleton into a graph
    if G is None:
        sk_graph = sk.convert_skeleton_to_graph(skeleton)
    else:
        sk_graph = G

    #2) Find the nodes of the skeleton coordinates in both groups
    coordinates_list_1_nodes = xu.get_graph_nodes_by_coordinates(sk_graph,coordinates_list_1)
    coordinates_list_2_nodes = xu.get_graph_nodes_by_coordinates(sk_graph,coordinates_list_2)

    #3) Find the shortest path between the nodes (along with the winning nodes)
    path,node_1,node_2 = xu.shortest_path_between_two_sets_of_nodes(
        sk_graph,
        coordinates_list_1_nodes,
        coordinates_list_2_nodes,
        return_path_distance=return_path_distance)
    
    if return_path_distance:
        return path

    shortest_skeleton_path = None

    if return_closest_coordinates:
        node_1_coordinate = xu.get_coordinate_by_graph_node(sk_graph,node_1)
        node_2_coordinate = xu.get_coordinate_by_graph_node(sk_graph,node_2)
        return_value = [node_1_coordinate,node_2_coordinate ]
    else:
        if plot_closest_coordinates:
            node_1_coordinate = xu.get_coordinate_by_graph_node(sk_graph,node_1)
            node_2_coordinate = xu.get_coordinate_by_graph_node(sk_graph,node_2)
        sk_graph_subgraph = sk_graph.subgraph(path)
        shortest_skeleton_path = sk.convert_graph_to_skeleton(sk_graph_subgraph)
        return_value = shortest_skeleton_path

    if plot_closest_coordinates:
        if shortest_skeleton_path is None:
            sk_graph_subgraph = sk_graph.subgraph(path)
            shortest_skeleton_path = sk.convert_graph_to_skeleton(sk_graph_subgraph)
            
        if skeleton is None:
            skeleton = sk.convert_graph_to_skeleton(sk_graph)
        ipvu.plot_objects(main_skeleton=skeleton,
                         skeletons=[shortest_skeleton_path],
                         skeletons_colors="red",
                         scatters=[node_1_coordinate,node_2_coordinate],
                         scatters_colors=["lime","black"],
                         scatter_size=0.4)
        
    
    return return_value

def restrict_skeleton_to_distance_from_coordinate(skeleton,
                                                     coordinate,
                                                 distance_threshold):
    """
    Purpose:
    To restrict a skeleton to only those points within a certain distance
    of a starting coordinate
    
    Ex:
    retr_skeleton = sk.restrict_skeleton_to_distance_from_coordinate(neuron_obj[0].skeleton,
                                                 neuron_obj[0].current_starting_coordinate,
                                                 10000,)
    ipvu.plot_objects(main_skeleton=neuron_obj[0].skeleton,
                     skeletons=[restr_skeleton],
                     skeletons_colors="red")     
    
    """
    candidate_sk = skeleton
    starting_sk_coord = coordinate
    
    #1) Convert skeleton to graph
    
    candidate_sk_graph = sk.convert_skeleton_to_graph(candidate_sk)
    
    #2) Find all skeleton points that are within a certain distance of the starting coordinate
    starting_sk_node = xu.get_graph_node_by_coordinate(candidate_sk_graph,starting_sk_coord)
    skeletons_nodes_for_comparison = xu.find_nodes_within_certain_distance_of_target_node(
        candidate_sk_graph,
        starting_sk_node,
        distance_threshold)
    np.array(list(skeletons_nodes_for_comparison))
    comparison_subgraph = candidate_sk_graph.subgraph(skeletons_nodes_for_comparison)
    
    return sk.convert_graph_to_skeleton(comparison_subgraph)


def skeleton_coordinate_offset_from_endpoint(skeleton,
                                            endpoint_coordinate,
                                             offset_distance,
                                             plot_coordinate = False,
                                            verbose=False,
                                            subtract_cutoff=True,
                                            return_success = False):
    """
    Purpose: Will return a skeleton coordinate
    that is a certain distance offset from an endpoint coordinate
    
    Ex: 
    new_point = skeleton_coordinate_offset_from_endpoint(v_skeleton,
                                        coord,
                                        #offset_distance=offset_distance_for_points,
                                        offset_distance=50000,
                                        verbose=True,
                                                    plot_coordinate=True)
    """
    skeleton = sk.order_skeleton(skeleton,
                 endpoint_coordinate)
    
    restr_skeleton,success = sk.restrict_skeleton_from_start(skeleton,#.reshape(-1,2,3),
                            cutoff_distance=offset_distance,
                           subtract_cutoff=subtract_cutoff,
                                            return_indexes=False)
    if not success:
        curr_point = skeleton[-1][-1]
    else:
        curr_point = restr_skeleton[0][0]
    
    if verbose:
        print(f"Restriction of skeleton by {offset_distance} was success = {success}"
             f" and point is {curr_point}")
    if plot_coordinate:
        ipvu.plot_objects(skeletons=[skeleton],
                 scatters=[endpoint_coordinate.reshape(-1,3),
                          curr_point.reshape(-1,3)],
                         scatter_size=1,
                         scatters_colors=["red","blue"])
    if return_success:
        return curr_point,success
    else:
        return curr_point

def shared_coordiantes(skeletons,return_one=False):
    """
    Purpose: To return the shared coordinates by skeletons
    
    """
    s_coords = nu.intersect2d_multi_list([k.reshape(-1,3) for k in skeletons])
    if not return_one:
        return s_coords
    else:
        if len(s_coords) != 1:
            raise Exception(f"Number of shared coordinates was not one: {s_coords}")
        else:
            return s_coords[0]
        
        
def skeleton_coordinate_path_from_start(skeleton,
    start_endpoint_coordinate=None):
    """
    Purpose: will just give vertex path
    of skeleton from the first 
    """
    if start_endpoint_coordinate is not None:
        skeleton = sk.order_skeleton(skeleton,start_endpoint_coordinate=start_endpoint_coordinate)
    ordered_path = np.vstack([skeleton[0][0].reshape(-1,3),skeleton[1:,0].reshape(-1,3),skeleton[-1][-1].reshape(-1,3)])
    return ordered_path


def skelton_coordinate_path_to_skeleton(skeleton_coordinate_path):
    x = skeleton_coordinate_path
    return np.hstack([x[:-1],x[1:]]).reshape(-1,2,3)


def closest_distances_from_skeleton_vertices_to_base_skeleton(skeleton,
                                                               base_skeleton,
                                                              verbose = False,
                                                              plot_min_pair = False):
    """
    Purposes: To measure the distance from one skeleton's vertices
    to the closest vertices on a base skeleton
    
    Application: To help determine if a merge
    error has occured where the skeletons pass too close
    to each other
    
    Ex: 
    sk.closest_distances_from_skeleton_vertices_to_base_skeleton(new_sks[0],
                                                          new_sks[1],
                                                          verbose= True,
                                                          plot_min_pair=True)
    """
    sk1_coords = np.unique(base_skeleton.reshape(-1,3),axis=0)
    sk2_coords = np.unique(skeleton.reshape(-1,3),axis=0)
    sk1_kd = KDTree(np.unique(sk1_coords.reshape(-1,3),axis=0))
    dist, closest_index = sk1_kd.query(sk2_coords)
    
    if verbose:
        print(f"Min of {len(dist)} dist = {np.min(dist)}")
        
    if plot_min_pair:
        min_idx = np.argmin(dist)
        sk2_min_coord = sk2_coords[min_idx]
        sk1_min_coord = sk1_coords[closest_index[min_idx]]
        
        ipvu.plot_objects(skeletons = [skeleton,base_skeleton],
                          skeletons_colors=["red","blue"],
                         scatters=[sk2_min_coord,sk1_min_coord],
                         scatters_colors=["red","blue"])

    return dist

def resize_skeleton_with_branching(
    skeleton,
    segment_width,
    optimal_speed = False,
    verbose = False,
    plot = False):
    """
    Purpose: To resize a skeleton that may have
    some branching points
    
    Pseudocode: 
    1) Decompose the skeleton into segments
    2) Resize all of the skeleton segments
    3) Recompose the skeleton
    
    """
    st = time.time()
    sk_branches = sk.decompose_skeleton_to_branches(skeleton)
    sk_branches_resized = [sk.resize_skeleton_branch(k,segment_width=segment_width)
                           for k in sk_branches]
    if optimal_speed:
        return_sk = sk.stack_skeletons(sk_branches_resized)
    else:
        return_sk =  sk.recompose_skeleton_from_branches(sk_branches_resized)
    if verbose:
        print(f"Time for resize_skeleton_with_branching = {time.time() - st}")
    if plot:
        ipvu.plot_objects(main_skeleton = return_sk)
    return return_sk

# --------- 7/29: helped with apical classifications ---------- #
def order_resize_skeleton(skeleton,
                          start_endpoint_coordinate,
                         segment_width=0,
                            n_segments=0,
                          plot_skeleton = False,
                            **kwargs):
    """
    Purpose: To order a skeleton and then resize it
    
    Ex: 
    sk.order_resize_skeleton(branch_obj.skeleton,
                         nru.upstream_endpoint(limb_obj,branch_idx),
                         skeleton_resize_distance,
                         plot_skeleton=True
                        )
    """
    sk_ord = sk.order_skeleton(skeleton, 
                               start_endpoint_coordinate = start_endpoint_coordinate)
    if segment_width == 0 and n_segments == 0:
        sk_ord_resize = sk_ord
    else:
        sk_ord_resize = sk.resize_skeleton_branch(sk_ord,
                                                 segment_width=segment_width,
                                                 n_segments=n_segments,
                                                 **kwargs)
        
    if plot_skeleton:
        ipvu.plot_objects(
                     skeletons=[sk_ord_resize],
                     scatters=[sk_ord_resize[0][0]],
                     scatter_size=1,
                     scatters_colors="red")

    return sk_ord_resize

def skeleton_vectors(skeleton,
                    start_endpoint_coordinate,
                    segment_width=0,
                     n_segments=0,
                     plot_skeleton=False
                    ):
    """
    Purpose: To get skeleton vectors of a reshaped and ordered skeleton
    """
    branch_sk_ord_resized = sk.order_resize_skeleton(skeleton,
                         start_endpoint_coordinate,
                         segment_width=segment_width,
                        n_segments=n_segments,
                         plot_skeleton=plot_skeleton
                        )
    skeleton_vectors = [sk.skeleton_endpoint_vector(k.reshape(-1,2,3)) for k in branch_sk_ord_resized]
    return skeleton_vectors

def angle_between_skeleton_vectors_and_ref_vector(reference_vector,
    skeleton_vectors = None,
    skeleton=None,
    start_endpoint_coordinate=None,
    segment_width=0,
     n_segments=0,
     plot_skeleton=False):
    """
    Purpose: To calculate the angles between a reference 
    vector and certain skeleton vectors created 
    by sections of a skeleton that has
    been ordered and reshaped
    """
    if skeleton_vectors is None:
        sk_vectors = sk.skeleton_vectors(skeleton,
                         start_endpoint_coordinate=start_endpoint_coordinate,
                         segment_width=segment_width,
                        n_segments=n_segments,           
                            plot_skeleton=plot_skeleton)
    
    return np.array([nu.angle_between_vectors(reference_vector,k) for k in sk_vectors])

def percentage_skeleton_match_to_ref_vector(skeleton,
                                         reference_vector,
                                        max_angle,
                                        min_angle=None,
                                         #arguments for computing the vectors
                                          start_endpoint_coordinate=None,
                                        segment_width=0,
                                         n_segments=0,
                                         plot_skeleton=False,
                                             
                                             verbose = False,
                                          #argmuments for what will be returned
                                             return_match_length = False
                                         ):
    """
    Find the percentage (and distance) that match angle requirements to a given vector
    
    Ex: 
    sk.percentage_skeleton_match_to_ref_vector(branch_obj.skeleton,
                                          reference_vector=reference_vector,
                                           max_angle=angle_threshold,
                                          start_endpoint_coordinate=nru.upstream_endpoint(limb_obj,branch_idx),
                                        segment_width=skeleton_resize_distance,
                                         plot_skeleton=True,
                                          verbose=True,
                                          return_match_length = True)
    
    """
    sk_angles_between_ref_vector = sk.angle_between_skeleton_vectors_and_ref_vector(reference_vector,
                                                                                    skeleton=skeleton,
                                                                                   start_endpoint_coordinate=start_endpoint_coordinate,
                         segment_width=segment_width,
                        n_segments=n_segments,           
                            plot_skeleton=plot_skeleton)
    
    if min_angle is None:
        min_angle = -1
    if max_angle is None:
        max_angle = np.inf
        
    skeleton_segment_matches = np.where((sk_angles_between_ref_vector > min_angle) & 
                                 (sk_angles_between_ref_vector < max_angle))[0]
    
    if verbose:
        print(f"sk_angles_between_ref_vector = {sk_angles_between_ref_vector}")
        print(f"with min_angle = {min_angle} and max_angle = {max_angle}")
        print(f"# of matches = {len(skeleton_segment_matches)} : skeleton_segment_matches = {skeleton_segment_matches}")
    
    perc_match = len(skeleton_segment_matches)/len(sk_angles_between_ref_vector)
    
    if verbose:
        print(f"perc_match = {perc_match}")
    
    if return_match_length:
        match_length = perc_match*(sk.calculate_skeleton_distance(skeleton))
        if verbose:
            print(f"match_length = {match_length}")
        return perc_match,match_length
    else:
        return perc_match
    
    
def bounding_box_corners(skeleton):
    """
    Purpose: To find the minimum and maximum 
    coordinates of the skeleton bounding box
    """
    sk_nodes = sk.convert_skeleton_to_nodes(skeleton)
    return nu.min_max_3D_coordinates(sk_nodes)

def bbox_volume(skeleton):
    sk_nodes = sk.convert_skeleton_to_nodes(skeleton)
    return nu.bounding_box_volume(sk_nodes)


def coordinates_along_skeleton_offset_from_start(
    skeleton,
    starting_coordinate,
    n_points,
    offset,
    verbose = False,
    plot_starting_skeleton = False,
    plot_restricted_skeleton = False,
    plot_points = False,
    ):

    """
    Purpose: To generate a certain number of points from the start of a skeleton.
    Ideally the points are a certain distance away from the start , but if there
    is not enough distance to do this then just space the points out as much as possible

    Pseudocode:  
    1) Restrict the skeleton to n_points*offset from the start
    For the number of points going to take
    2) Resize the skeleton
    
    Ex: 
    from mesh_tools import skeleton_utils as sk
    from datasci_tools import numpy_dep as np


    skeleton = neuron_obj[0][42].skeleton
    starting_coordinate =  neuron_obj[0].current_starting_coordinate


    sk.coordinates_along_skeleton_offset_from_start(
    skeleton,
    starting_coordinate,
        offset = 3000,
        n_points = 10,
        plot_points=True
    )
    """
    # skeleton = sk.resize_skeleton_branch(skeleton,
    #                                     segment_width = 20)

    if plot_starting_skeleton:
        ipvu.plot_objects(main_skeleton = skeleton,
                         scatters=[starting_coordinate],
                         scatter_size=1)
    if verbose:
        print(f"skeleton distance = {sk.calculate_skeleton_distance(skeleton)}")

    skeleton_rest = sk.restrict_skeleton_from_start(
        skeleton = skeleton,
        cutoff_distance = n_points*offset,
        return_indexes=False,
        return_success = False,
        resize_skeleton_to_help_success = True,
        starting_coordinate = starting_coordinate,
        )[0]

    curr_skeleton_dist = sk.calculate_skeleton_distance(skeleton_rest)
    if verbose:
        print(f"skeleton_rest distance = {curr_skeleton_dist}")

    if plot_restricted_skeleton:
        ipvu.plot_objects(main_skeleton = skeleton,
                          skeletons = [skeleton_rest],
                          skeletons_colors="red",
                         scatters=[starting_coordinate],
                         scatter_size=1)


    #skeleton_rest_resize = sk.resize_skeleton_branch(skeleton_rest,n_segments=n_points)
    skeleton_ordered = sk.order_skeleton(
        skeleton_rest,
        start_endpoint_coordinate = starting_coordinate,
    )

    #print(f"skeleton_ordered  = {skeleton_ordered}")
    #Simpler method: Could maybe skip this process with the following; 
    new_sk = sk.resize_skeleton_branch(skeleton_ordered,
                                       n_segments=n_points,
                                      )
    #print(f"new_sk = {new_sk}")
    if n_points == 1:
        points_array = [new_sk[0][-1]]
    else:
        points_array = new_sk[:,1].reshape(-1,3)
        #points_array = sk.convert_skeleton_to_nodes(new_sk[1:])

    '''
    #calculate the middle points
    last_point = skeleton_ordered[-1][-1]
    points_array = []
    new_offset = curr_skeleton_dist/n_points

    if verbose:
        print(f"New offset = {new_offset}")

    curr_skeleton = skeleton_ordered
    for i in range(1,n_points):
        curr_rest = sk.restrict_skeleton_from_start_plus_offset(
        skeleton = skeleton,
        offset = i*new_offset,
        min_comparison_distance = 0,
        start_coordinate = starting_coordinate,
        skeleton_resolution = 10,
        comparison_distance = np.inf
        )


        print(sk.calculate_skeleton_distance(curr_rest))

        new_point = curr_rest[0][0]
        curr_skeleton = curr_rest
        points_array.append(new_point)

    points_array.append(last_point)
    '''

    if plot_points:
        start_point_color = "blue"
        points_color = "red"
        print(f"start_point_color = {start_point_color}, points_color = {points_color}")
        ipvu.plot_objects(main_skeleton = skeleton,
                      #skeletons = [skeleton_rest],
                      skeletons_colors=["red"],
                     scatters=[starting_coordinate,points_array],
                      scatters_colors=[start_point_color,
                                       points_color],
                     scatter_size=2)

    if verbose:
        print(f"# of points layed down = {len(points_array)} (desired {n_points})")
    return points_array


def skeleton_from_h5py(file,
                      verbose = False):
    with h5py.File(file, 'r') as hf:
        vertices = hf['vertices'][()].astype(np.float64)
        edges = hf['edges'][()].astype(np.uint32)

    skeleton_manual = vertices[edges]
    if verbose:
        print(f"skeleton shape = {skeleton_manual.shape}")
    return skeleton_manual



def mesh_subtraction_by_skeleton(

    mesh,
    edges,

    buffer=mesh_subtraction_buffer_default,
    bbox_ratio=None,#,1.2
    distance_threshold=mesh_subtraction_distance_threshold_default,
    resize_length = 2000,
    verbose =False,
    initial_plotting = False,

    edge_idx_to_process = None,
    edge_loop_print=False,
    edge_loop_plotting_slice = False,
    edge_loop_plotting = False,


    edge_length_to_process = 0.001,
    submesh_face_min = 200,

    edge_loop_plot_winning_faces= False,
    return_subtracted_mesh = False,

    plot_final_mesh = False,
    final_split_n_faces_min = 500,
    ):



    """
    Purpose: Will return significant mesh pieces that are
        not already accounteed for by the skeleton

    Psuedocode: 
    1) Restrict the mesh with a bounding box if requested
    2) Optionally downsample the skeleton
    Iterate through all of the edges in skeleton and find matching faces
    3) Concatenate into one mesh
    
    ------Example 1:-----
    segment_id = 31504318800

    neuron_obj = hdju.neuron_obj_from_table(
        segment_id=segment_id,
        table = h01auto.Decomposition.Object(),
        verbose = True,
        return_one=True,
    )

    from neurd import neuron_visualizations as nviz
    nviz.visualize_neuron(neuron_obj,limb_branch_dict="all")


    ret_mesh = sk.mesh_subtraction_by_skeleton(

        mesh = neuron_obj[0].mesh,
        edges = neuron_obj[0][3].skeleton,

        buffer=2000,
        bbox_ratio=None,#,1.2
        distance_threshold=8_000,
        resize_length = 2000,
        verbose =True,
        initial_plotting = False,

        #edge_idx_to_process = [0,3],
        edge_loop_print=False,
        edge_loop_plotting_slice = False,
        edge_loop_plotting = False,


        edge_length_to_process = 0.001,
        submesh_face_min = 200,

        edge_loop_plot_winning_faces= False,
        return_subtracted_mesh = True,

        plot_final_mesh = True,
        final_split_n_faces_min = 500,
        )

    


    """
    main_mesh_bbox_restricted = None
    if bbox_ratio is not None:
        main_mesh_bbox_restricted, faces_bbox_inclusion = tu.bbox_mesh_restriction(mesh,
                                                                            nu.bouning_box_corners(edges.reshape(-1,3)),
                                                                                   bbox_ratio)
        if type(main_mesh_bbox_restricted) == type(trimesh.Trimesh()):
            if verbose:
                print(f"Inside mesh subtraction, len(main_mesh_bbox_restricted.faces) = {len(main_mesh_bbox_restricted.faces)}")
        else:
            main_mesh_bbox_restricted = None

    if main_mesh_bbox_restricted is None:
        main_mesh_bbox_restricted = mesh
        faces_bbox_inclusion = np.arange(0,len(mesh.faces))

    if initial_plotting:
        print(f"Before Resize")
        ipvu.plot_objects(main_mesh_bbox_restricted,
                         main_skeleton = edges)  

    if resize_length is not None:
        try: 
            edges = sk.resize_skeleton_with_branching(edges,resize_length)
        except:
            pass
        else:
            if verbose:
                print(f"Was able to resize skeleton to length: {resize_length}")

    if initial_plotting:
        print(f"After Resize")
        ipvu.plot_objects(main_mesh_bbox_restricted,
                         main_skeleton = edges)  


    # Iterate through edges of the skeleton: 
    start_time = time.time()

    #face_subtract_color = []
    face_subtract_indices = []



    for i,ex_edge in tqdm(enumerate(edges)):
        if edge_loop_print:
            print(f"\n--- Working on Edge {i}---")
        #print("\n------ New loop ------")
        #print(ex_edge)
        if edge_idx_to_process is not None:
            if i not in edge_idx_to_process:
                continue

        # ----------- creating edge and checking distance ----- #
        loop_start = time.time()
        base = ex_edge[0]
        edge_line = ex_edge[1] - ex_edge[0]

        #a) Skip the very small edges
        if np.sum(np.abs(edge_line)) < edge_length_to_process:
            if edge_loop_print:
                print(f"edge number {i}, {ex_edge}: has sum less than {sum_threshold} so skipping")
            continue

        #b) Create change of basis matrix that will align z axis with vector
        cob_edge = nu.change_basis_matrix(edge_line)

        #c) Find range of the edge slice
        edge_trans = (cob_edge@(ex_edge-base).T) #this will be [0,magnitude of vector]
        slice_range = np.sort(edge_trans[2,:]) 
        slice_range_buffer = slice_range + np.array([-buffer,buffer])

        if edge_loop_print:
            print(f"slice_range_buffer={slice_range_buffer}")

        #d) transform the face midpoints of mesh
        face_midpoints = main_mesh_bbox_restricted.triangles_center - base
        fac_midpoints_trans = cob_edge@face_midpoints.T

        #e) find the mask for indices within the buffer along the z axis
        slice_mask_z_idx = np.where((fac_midpoints_trans[2,:]>slice_range_buffer[0]) & 
                          (fac_midpoints_trans[2,:]<slice_range_buffer[1]))[0]
        if edge_loop_print:
            print(f"len(slice_mask_z_idx) = {len(slice_mask_z_idx)}")
        if edge_loop_plotting_slice:
            print("slice_mask_z_idx")
            ipvu.plot_objects(main_mesh_bbox_restricted,
                             meshes=[main_mesh_bbox_restricted.submesh([slice_mask_z_idx],append=True,only_watertight=False)],
                             meshes_colors="red",
                              skeletons=[ex_edge]
                             )


        #f) find the mask for the indices within the buffer on the x,y plane
        slice_mask_xy_idx = np.where(np.linalg.norm(fac_midpoints_trans[:2,:],axis=0) < distance_threshold)[0]
        if edge_loop_print:
            print(f"len(slice_mask_xy_idx) = {len(slice_mask_xy_idx)}")
        if edge_loop_plotting_slice:
            print("slice_mask_xy_idx")
            ipvu.plot_objects(main_mesh_bbox_restricted,
                             meshes=[main_mesh_bbox_restricted.submesh([slice_mask_xy_idx],append=True,only_watertight=False)],
                             meshes_colors="red",
                              skeletons=[ex_edge]
                             )

        #g) Combine the x/y and z distance masks
        slice_idx = np.intersect1d(slice_mask_xy_idx,slice_mask_z_idx)
        if edge_loop_print:
            print(f"len(slice_idx) = {len(slice_idx)}")
        if edge_loop_plotting_slice:
            print("slice_idx")
            ipvu.plot_objects(main_mesh_bbox_restricted,
                             meshes=[main_mesh_bbox_restricted.submesh([slice_idx],append=True,only_watertight=False)],
                             meshes_colors="red",
                              skeletons=[ex_edge]
                             )

        #h) Continue to next edge if no faces in the search
        if len(slice_idx) == 0:
            if edge_loop_print:
                print(f"slice_idx empty so continuing")
            continue


        #i) Get the submesh and split into connected components (and filter for significant ones)
        face_list = slice_idx
        main_mesh_sub = main_mesh_bbox_restricted.submesh([face_list],append=True,only_watertight=False)
        sub_components,sub_components_face_indexes = tu.split(main_mesh_sub,
                                                              only_watertight=False,
                                                             return_mesh_list = True)
        sub_components = np.array(sub_components)
        sub_components_face_indexes= np.array(sub_components_face_indexes)

        if edge_loop_print:
            print(f"sub_components ({len(sub_components)})= {sub_components}")
        if submesh_face_min > 0:
            filt_idx = tu.filter_meshes_by_size(sub_components,
                                             size_threshold=submesh_face_min,
                                             return_indices = True)
            if len(filt_idx) > 0:
                sub_components = sub_components[filt_idx]
                sub_components_face_indexes = sub_components_face_indexes[filt_idx]

            if edge_loop_print:
                print(f"sub_components AFTER FILTERING ({len(sub_components)})= {sub_components}")


        if edge_loop_plotting:
            print(f"sub_components")
            ipvu.plot_objects(meshes=sub_components,
                             meshes_colors="random",
                             main_skeleton=ex_edge.reshape(-1,2,3))

        #j) Filter the sub components to only one based on 1) bbox containing vertices and then 
        #   closest bounding box center to skeleton center to find winning component idx

        if len(sub_components) == 1:
            winning_subcomponent_idx = 0
        else:
            """
            Psuedocode: 
            1) Test to see which meshes have a bounding box that contains the skeleton
            2a) only one then make that the winner
            2b) Find the mesh with the closest mesh center

            """
            containing_indices = np.array([i for i,k in enumerate(sub_components)
                                  if tu.mesh_bbox_contains_skeleton(k,ex_edge)])

            if edge_loop_print:
                print(f"containing_indices = {containing_indices}")

            if len(containing_indices) == 1: 
                if edge_loop_print:
                    print(f"Only one mesh with bbox containing skeleton")
                winning_subcomponent_idx = containing_indices[0]
            else:
                if edge_loop_print:
                    print(f"Resorting to finding mesh with the closest bounding box center")

                if len(containing_indices) == 0:
                    containing_indices = np.arange(len(sub_components))

                edge_center = np.mean(ex_edge,axis=0)
                closest_containing_idx = tu.closest_mesh_to_coordinate(mesh_list=sub_components[containing_indices],
                                              coordinate=edge_center,
                                              verbose = False,
                                              distance_method="mesh_center",
                                              return_mesh = False
                                                )

                winning_subcomponent_idx = containing_indices[closest_containing_idx]

        if edge_loop_print:
            print(f"winning_subcomponent_idx = {winning_subcomponent_idx}")

        #k) Find the faces that belong to the winning mesh    
        edge_skeleton_faces = faces_bbox_inclusion[face_list[sub_components_face_indexes[winning_subcomponent_idx]]]
        submesh_by_faces_idx = mesh.submesh([edge_skeleton_faces],append=True,only_watertight=False)
        submesh_by_trimesh = sub_components[winning_subcomponent_idx]

        if edge_loop_print:
            print(f"submesh_by_faces_idx= {submesh_by_faces_idx},submesh_by_trimesh = {submesh_by_trimesh}")

        if edge_loop_plot_winning_faces:
            print(f"Winning mesh without faces")
            ipvu.plot_objects(submesh_by_trimesh)
            print(f"Winning mesh WITH faces")
            ipvu.plot_objects(submesh_by_faces_idx)



        #l) Append the faces to running list
        if len(edge_skeleton_faces) <= 0 and edge_loop_print:
            print(f"****** Warning the edge index {i}: had no faces in the edge_skeleton_faces*******")

        face_subtract_indices.append(edge_skeleton_faces)

    if verbose:
        print(f"Total Mesh subtraction time = {np.round(time.time() - start_time,4)}")


    #m) Find the final submesh and split



    if len(face_subtract_indices)>0:
        all_removed_faces = np.concatenate(face_subtract_indices)
        unique_removed_faces = np.unique(all_removed_faces)

        faces_to_keep = np.delete(np.arange(0,len(mesh.faces)),unique_removed_faces)
        if return_subtracted_mesh:
            faces_to_keep = unique_removed_faces
        else:
            faces_to_keep = np.delete(np.arange(0,len(mesh.faces)),unique_removed_faces)
    else:
        if return_subtracted_mesh:
            faces_to_keep= []
        else:
            faces_to_keep = np.arange(0,len(mesh.faces))

    new_submesh = tu.submesh(mesh,faces_to_keep)

    if plot_final_mesh:
        print(f"plot_final_mesh")
        ipvu.plot_objects(new_submesh)


    if final_split_n_faces_min > 0 and not return_subtracted_mesh:
        significant_pieces = tu.split_significant_pieces(new_submesh,
                                                             final_split_n_faces_min,
                                                             print_flag=False)
        if plot_final_mesh:
            print(f"significant_pieces")
            ipvu.plot_objects(meshes=significant_pieces,
                             meshes_colors="random")

        return_value = significant_pieces
    else:
        return_value = new_submesh
        
    return return_value


def jitter_skeleton_from_coordinate(
    coordinate,
    jitter = None,
    random_noise = True,
    verbose = True,
    ):
    """
    Purpose: generate a skeleton segment to
    slightly shift and endpoint and return the skeleton

    Pseudocode: 
    1) Compute the new jittered endpoint
    2) Create the new skeleton segment
    """
    if jitter is None:
        jitter = 2

    jitter= nu.convert_to_array_like(jitter)
    
    if len(jitter) != 3:
        jitter = np.concatenate([jitter,jitter,jitter])
        if random_noise:
            jitter = np.random.rand(3)*jitter

    jitter= np.array(jitter)

    if verbose:
        print(f"jitter = {jitter}")

    new_skeleton = np.array([coordinate,jitter + coordinate])
    if verbose:
        print(f"new_skeleton: {new_skeleton}")

    return new_skeleton


def vector_away_from_endpoint(
    skeleton,
    endpoint,    
    offset=500,
    comparison_distance=3_000,#2_000,
    skeleton_resolution=100,
    min_comparison_distance = 1000,
    plot_restricted_skeleton = False,
    normalize_vector = True,
    verbose = False,
    ):
    
    #plot_restricted_skeleton = True

    """
    Purpose: To get the upstream or downstream vector
    from the offest start of a skeleton

    Result: The vector naturally points away from the start coordinate
    
    Ex: 
    sk.vector_away_from_endpoint(
        skeleton = branch_obj.skeleton,
        endpoint = np.array([2504610. ,  480431. ,   33741.2]),
        verbose = True,
        plot_restricted_skeleton = True
    )
    """


    restr_skeleton = sk.restrict_skeleton_from_start_plus_offset(skeleton,
                                                       offset=offset,
                                                    comparison_distance=comparison_distance,
                                                        min_comparison_distance=min_comparison_distance,
                                                    verbose=verbose,
                                                     start_coordinate=endpoint,
                                                    skeleton_resolution = skeleton_resolution,
                                                    plot = plot_restricted_skeleton
                                                       )

    sk_vec = sk.skeleton_endpoint_vector(
        restr_skeleton,
        normalize_vector = normalize_vector,)

    if verbose:
        print(f"sk_vec = {sk_vec}")
        
    return sk_vec


def coordinates_from_downstream_dist(
    skeleton,
    donwstream_dists,
    start_endpoint_coordinate = None,
    verbose = False,
    segment_width = 20,
    plot = False
    ):
    """
    Purpose: Get skeleton coordinates as dictated by the skeletal lengths downstream
    - would be nice to chop up a skeleton by lengths from upstream

    Psueodocode: 
    1) Order skeleton
    2) Compute the skeletal distances of each coordinate
    3) Get the coordinates that are closest in downstream dist to those sent
    """    
    
    curr_skeleton = sk.order_resize_skeleton(
        skeleton,
        start_endpoint_coordinate = start_endpoint_coordinate,
        segment_width=segment_width)
    
    skeletal_distances = sk.calculate_coordinate_distances_cumsum(
        curr_skeleton)
    
    sk_kd = KDTree(skeletal_distances)
    dist,closest_idx = sk_kd.query(donwstream_dists)
    
    sk_coords = sk.skeleton_coordinate_path_from_start(curr_skeleton)
    
    coordinates = sk_coords[closest_idx]
    
    if verbose:
        print(f"coordinates = {coordinates.shape}")
        
    if plot:
        from datasci_tools import matplotlib_utils as mu
        ipvu.plot_objects(
            scatters=[k.reshape(-1,3) for k in coordinates],
            scatters_colors=list(mu.color_transition(n=len(coordinates))),
            scatter_size=0.5
        )
        
    return coordinates



def coordinate_restriction_to_skeleton(
    skeleton,
    coordinates,
    distance_threshold = 15_000,
    keep_skeleton_within_threshold = False,
    plot = False,
    verbose = False,  
    return_graph = False,
    ):
    """Purpose: To keep or eliminate all skeletal 
    points within a certain distance of a coordinate: 

    1) convert skeleton to graph
    2) find all node coordinates and names
    3) Find the closest distance from coordinates to cancel points
    4) Either take the subgraph of nodes with range or outside of range
    
    Ex: 
    sk.coordinate_restriction_to_skeleton(
        skeleton = skeleton_proof_stitch,
        coordinates = hdju.nucleus_center_from_segment_id(segment_id_raw),
        distance_threshold = 40_000,
        keep_skeleton_within_threshold = False,
        plot = True,
        verbose = True,  
        )
    """
    if len(skeleton) == 0:
        return skeleton
    
    coordinates = np.array(coordinates).reshape(-1,3)

    #1) Turn skeleton into nodes
    G = sk.convert_skeleton_to_graph(skeleton)

    #2) Find skeleton coordinates
    node_names = np.array(list(G.nodes()))
    node_coords = np.array([G.nodes[k]["coordinates"] for k in node_names])

    kd_obj = KDTree(coordinates)
    dist,_ = kd_obj.query(node_coords)

    if keep_skeleton_within_threshold:
        node_mask = dist <=  distance_threshold
    else:
        node_mask = dist > distance_threshold

    
    G_sub = G.subgraph(node_names[node_mask]).copy()
    if verbose:
        print(f"# of nodes in original = {len(G.nodes())}")
        print(f"# of nodes after restriction = {len(G_sub.nodes())}")
        
    
    skeleton_new = sk.convert_graph_to_skeleton(G_sub)

    if plot:
        original_color = "red"
        final_color  = "black"
        print(f"original_color = {original_color}, fianl_color = {final_color}")

        ipvu.plot_objects(
                skeletons=[skeleton,skeleton_new],
            skeletons_colors=[original_color,final_color]

        )
        
    if return_graph:
        return G_sub
    
    return skeleton_new

def node_names_and_coord_from_G(G=None,skeleton=None):
    if G is None:
        G = sk.convert_skeleton_to_graph(skeleton)
    
    node_names = np.array(list(G.nodes()))
    node_coords = np.array([G.nodes[k]["coordinates"] for k in node_names])
    return node_names,node_coords

def subtract_exact_coordinates_from_skeleton(
    skeleton,
    coordinates,
    return_graph = False,
    verbose= False,
    ):
    
    return sk.coordinate_restriction_to_skeleton(
        skeleton,
        coordinates,
        distance_threshold = 0,
        verbose = verbose,
        return_graph = return_graph
        
    )

def empty_skeleton():
    return np.array([]).reshape(-1,2,3)

def subskeleton_from_vertices_idx(
    vertices_idx,
    skeleton = None,
    vertices=None,
    edges=None,
    plot = False,
    ):
    """
    Purpose: To get a skeleton from a list of indices
    referencing a subset of the original vertices
    
    Ex: 
    sk.subskeleton_from_vertices_idx(
        vertices_idx = np.arange(0,1000),
        skeleton = truth_skeleton,
        plot=True
    )
    """
    if vertices is None or edges is None:
        vertices,edges = sk.convert_skeleton_to_nodes_edges(skeleton)
    
    if len(vertices) == 0:
        return sk.empty_skeleton()
    
    G = nx.from_edgelist(edges)
    G_sub = G.subgraph(vertices_idx).copy()

    G_sub_edges = np.array(list(G_sub.edges()))
    if len(G_sub_edges) > 0:
        new_skeleton = vertices[G_sub_edges]
    else:
        new_skeleton = sk.empty_skeleton()
    
    if plot:
        original_sk = vertices[edges]
        
        original_color = "black"
        sub_color = "red"
        print(f"Original skeleton ({original_color}), subskeleton color ({sub_color})")
        ipvu.plot_objects(
            skeletons = [original_sk,new_skeleton],
            skeletons_colors = [original_color,sub_color]
        )
        
    return new_skeleton

def mesh_restriction_to_skeleton(
    skeleton,
    mesh,
    distance_threshold=6_000,
    keep_skeleton_within_threshold=True,
    plot=False,
    verbose=False,
    return_graph=False,
    ):

    new_sk = sk.coordinate_restriction_to_skeleton(
        skeleton,
        coordinates = mesh.vertices,
        distance_threshold = distance_threshold,
        keep_skeleton_within_threshold = keep_skeleton_within_threshold,
        verbose=verbose,
        return_graph = return_graph,
    )
    
    if plot:
        if return_graph:
            curr_sk = sk.convert_graph_to_skeleton(new_sk)
        else:
            curr_sk = new_sk
            
        ipvu.plot_objects(
            mesh,
            new_sk,
        )
    
    return new_sk

def accuracy_stats_from_true_predicted_skeleton(
    true_skeleton,
    predicted_skeleton,
    #exclude_skeleton = None,
    search_radius = 5000,
    include_skeletons = True,
    plot_subskeleton = False,
    plot_skeleton_categories = False,
    mesh = None,
    verbose = False,
    
    ):

    """
    Purpose: To compute all TP/FP... skeletons and skeletal lengths
    and all other stats: precision, recall between 2 skeletons

    What to do with an exclude skeleton? It should not be included in the false negative
    """

    manual_skeleton = true_skeleton
    auto_skeleton = predicted_skeleton
    
    result_dict = dict()

    # --- Do the TP and FP
    manual_vertices,manual_edges = sk.convert_skeleton_to_nodes_edges(manual_skeleton)
    auto_vertices,auto_edges = sk.convert_skeleton_to_nodes_edges(auto_skeleton)


    if len(auto_vertices) > 0:
        manual_kd = KDTree(manual_vertices)
        dist,closest_nodes = manual_kd.query(auto_vertices)
        tp_mask = dist < search_radius
        all_idx = np.arange(len(auto_vertices))
        valid_idx = all_idx[tp_mask]
        invalid_idx = all_idx[~tp_mask]

    else:
        valid_idx = []
        invalid_idx = []


    tp_skeleton = sk.subskeleton_from_vertices_idx(
            vertices = auto_vertices,
            edges = auto_edges,
            vertices_idx = valid_idx,
            plot=plot_subskeleton
        )
    result_dict["TP_skeleton"] = tp_skeleton

    fp_skeleton = sk.subskeleton_from_vertices_idx(
            vertices = auto_vertices,
            edges = auto_edges,
            vertices_idx = invalid_idx,
            plot=plot_subskeleton
        )
    result_dict["FP_skeleton"] = fp_skeleton

    # ----- Doing the FN skeletons ------------
    if len(auto_vertices) > 0:
        auto_kd = KDTree(auto_vertices)
        dist,closest_nodes = auto_kd.query(manual_vertices)
        tp_mask = dist < search_radius
        all_idx = np.arange(len(manual_vertices))
        valid_idx = all_idx[tp_mask]
        invalid_idx = all_idx[~tp_mask]

    else:
        valid_idx = np.arange(len(manual_vertices))
        invalid_idx = []

    fn_skeleton = sk.subskeleton_from_vertices_idx(
            vertices = manual_vertices,
            edges = manual_edges,
            vertices_idx = invalid_idx,
            plot=plot_subskeleton
        )
    result_dict["FN_skeleton"] = fn_skeleton


    # plotting the different categories
    if plot_skeleton_categories:
        for k,curr_sk in result_dict.items():
            print(f"\n---Plotting {k}")
            ipvu.plot_objects(mesh,curr_sk)


    # calculates the skeletal length
    for k in list(result_dict.keys()):
        result_dict[f"{k}_length"] = sk.calculate_skeleton_distance(result_dict[k])

    result_dict["total_length"] = sk.calculate_skeleton_distance(predicted_skeleton)

    if verbose:
        print(f"results: {dict([(k,v) for k,v in result_dict.items() if 'length' in k])}")

    #compute the precision, recall and f1
    pre_recall_score = stu.calculate_scores(
            TP = result_dict[f"TP_skeleton_length"],
            FP = result_dict[f"FP_skeleton_length"],
            FN = result_dict[f"FN_skeleton_length"],
    )
    if verbose:
        print(f"pre_recall_score = {pre_recall_score}")

    result_dict.update(
        pre_recall_score
    )

    if not include_skeletons:
        for k in list(result_dict.keys()):
            if not ('length' in k):
                del result_dict[k] 

    return result_dict

def center_skeleton(skeleton,center=None):
    if center is None:
        center = np.mean(skeleton.reshape(-1,3),axis=0)

    return skeleton - center
    

def resize_center_and_coordinates_from_skeleton(
    skeleton,
    resize_skeleton = True,
    segment_width = 5_000,
    plot_resized_skeleton = False,
    
    center_coordinates = True,
    center = None,
    verbose = False,
    
    return_coordinates = True,
    ):
    """
    Purpose: To resize, center export as 
    coordinates
    """
    #2) Resize the skeleton
    if resize_skeleton:
        skeleton_scaled = sk.resize_skeleton_with_branching(
            skeleton,
            segment_width =segment_width,
            optimal_speed = True,
            verbose = verbose,
            plot = plot_resized_skeleton
        )
    else:
        skeleton_scaled = skeleton
    
    
    """
    4) If requested to center coordinates
    a. Pull down the soma center if None given
    b. Subtract soma center from coordinates
    """
    if center_coordinates:
        if center is None:
            center = np.mean(skeleton_scaled.reshape(-1,3),axis=0)
            
        skeleton_scaled = skeleton_scaled - center
        
    if return_coordinates:
        #3) Convert into coordinates
        skeleton_scatter = sk.convert_skeleton_to_coordinates(skeleton_scaled)
    else:
        skeleton_scatter = skeleton_scaled
        
    return skeleton_scatter

def resize_and_center_skeleton(
    skeleton,
    resize_skeleton = True,
    segment_width = 5_000,
    plot_resized_skeleton = False,
    
    center_coordinates = True,
    center = None,
    verbose = False,
    
    ):
    
    return resize_center_and_coordinates_from_skeleton(
        skeleton,
        resize_skeleton = resize_skeleton,
        segment_width = segment_width,
        plot_resized_skeleton = plot_resized_skeleton,

        center_coordinates = center_coordinates,
        center = center,
        verbose = verbose,

        return_coordinates = False,
        )

def intersect_skeleton(
    skeleton_1,
    skeleton_2,
    distance_threshold=5000,
    **kwargs
    ):
    
    if len(skeleton_1) == 0 or len(skeleton_2) == 0:
        return sk.empty_skeleton()
    
    coordinates = sk.convert_skeleton_to_coordinates(skeleton_2)
    
    sk.coordinate_restriction_to_skeleton(
        skeleton=skeleton_1,
        coordinates=coordinates,
        distance_threshold=distance_threshold,
        keep_skeleton_within_threshold=True,
        **kwargs
    )
    
def setdiff_skeleton(
    skeleton_1,
    skeleton_2,
    distance_threshold=5000,
    **kwargs
    ):
    
    if len(skeleton_1) == 0:
        return sk.empty_skeleton()
    if len(skeleton_2) == 0:
        return skeleton_1.copy()
    
    coordinates = sk.convert_skeleton_to_coordinates(skeleton_2)
    
    return sk.coordinate_restriction_to_skeleton(
        skeleton=skeleton_1,
        coordinates=coordinates,
        distance_threshold=distance_threshold,
        keep_skeleton_within_threshold=False,
        **kwargs
    )


#--- from mesh_tools ---
from . import meshparty_skeletonize as m_sk
from . import trimesh_utils as tu
from .trimesh_utils import split_significant_pieces,split,combine_meshes,write_neuron_off


#--- from datasci_tools ---
from datasci_tools import general_utils as gu
from datasci_tools import ipyvolume_utils as ipvu
from datasci_tools import matplotlib_utils as mu
from datasci_tools import mesh_utils as meshu
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_utils as nu
from datasci_tools import statistics_utils as stu
from datasci_tools import system_utils as su
from datasci_tools.tqdm_utils import tqdm

unique_vertices_edges_from_vertices_edges = xu.unique_vertices_edges_from_vertices_edges
graph_from_unique_vertices_edges = xu.graph_from_unique_vertices_edges
graph_from_non_unique_vertices_edges= xu.graph_from_non_unique_vertices_edges

from . import skeleton_utils as sk
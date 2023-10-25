'''


These functions just help with generically 
helping with trimesh mesh manipulation


'''
import copy
import h5py
import itertools
import logging
import matplotlib.pyplot as plt
import networkx as nx
from trimesh.grouping import *
from trimesh.graph import *
from datasci_tools import numpy_dep as np
import open3d as o3d
import pandas as pd
from pathlib import Path
from pykdtree.kdtree import KDTree
import pymeshfix
import time
import trimesh
from trimesh.path.exchange.misc import faces_to_path
from trimesh import triangles

try:
    import cgal_Segmentation_Module as csm
except:
    pass


#loading a mesh safely without any processing to mess up the vertices/faces
def load_mesh_no_processing(current_mesh_file):
    """
    will load a mesh from .off file format
    """
    if type(current_mesh_file) == type(Path()):
        current_mesh_file = str(current_mesh_file.absolute())
    if current_mesh_file[-4:] != ".off":
        current_mesh_file += ".off"
    return trimesh.load_mesh(current_mesh_file,process=False)

# --------- Dealing with h5 files
def load_mesh_no_processing_h5(current_mesh_file):
    """
    Will load a mesh from h5py file format
    
    """
    if type(current_mesh_file) == type(Path()):
        current_mesh_file = str(current_mesh_file.absolute())
    if current_mesh_file[-3:] != ".h5":
        current_mesh_file += ".h5"
        
    with h5py.File(current_mesh_file, 'r') as hf:
        vertices = hf['vertices'][()].astype(np.float64)
        faces = hf['faces'][()].reshape(-1, 3).astype(np.uint32)
        
    return trimesh.Trimesh(vertices=vertices,faces=faces)

def mesh_from_vertices_faces(vertices,faces):
    vertices = vertices.astype('float64')
    faces = faces.astype('int')
    return trimesh.Trimesh(vertices=vertices,faces=faces)

def write_h5_file(mesh=None,vertices=None,faces=None,segment_id=12345,
                  filepath="./",
                 filename=None,
                 return_file_path=True):
    """
    Purpose: Will write a h5 py file to store a mesh
    
    Pseudocode:
    1) Extract the vertices and the faces
    2) Create the complete file path with the write extension
    3) Write the .h5 file
    4) return the filepath 
    """
    
    #1) Extract the vertices and the faces
    if (vertices is None) or (faces is None):
        if mesh is None:
            raise Exception("mesh none and vertices or faces are none ")
        vertices=mesh.vertices
        faces=mesh.faces
        
    #2) Create the complete file path with the write extension
    curr_path = Path(filepath)
    
    assert curr_path.exists()
    
    if filename is None:
        filename = f"{segment_id}.h5"
    
    if str(filename)[-3:] != ".h5":
        filename = str(filename) + ".h5"
    
    total_path = str((curr_path / Path(filename)).absolute())
    
    with h5py.File(total_path, 'w') as hf:
        hf.create_dataset('segment_id', data=segment_id)
        hf.create_dataset('vertices', data=vertices)
        hf.create_dataset('faces', data=faces)
        
    if return_file_path:
        return total_path
    



# --------- Done with h5 files ---------------- #


def mesh_center_vertex_average(mesh_list):
    if not nu.is_array_like(mesh_list):
        mesh_list = [mesh_list]
    mesh_list_centers = [np.array(np.mean(k.vertices,axis=0)).astype("float")
                           for k in mesh_list]
    if len(mesh_list) == 1:
        return mesh_list_centers[0]
    else:
        return mesh_list_centers
    
    
        
def mesh_center_weighted_face_midpoints(mesh):
    """
    Purpose: calculate a mesh center point
    
    Pseudocode: 
    a) get the face midpoints
    b) get the surface area of all of the faces and total surface area
    c) multiply the surface area percentage by the midpoints
    d) sum up the products
    """
    #a) get the face midpoints
    face_midpoints = mesh.triangles_center
    #b) get the surface area of all of the faces and total surface area
    total_area = mesh.area
    face_areas = mesh.area_faces
    face_areas_prop = face_areas/total_area

    #c) multiply the surface area percentage by the midpoints
    mesh_center = np.sum(face_midpoints*face_areas_prop.reshape(-1,1),axis=0)
    return mesh_center
        

def write_neuron_off(current_mesh,main_mesh_path):
    if type(main_mesh_path) != str:
        main_mesh_path = str(main_mesh_path.absolute())
    if main_mesh_path[-4:] != ".off":
        main_mesh_path += ".off"
    current_mesh.export(main_mesh_path)
    with open(main_mesh_path,"a") as f:
        f.write("\n")
    return main_mesh_path


def combine_meshes(mesh_pieces,merge_vertices=True):
    leftover_mesh = trimesh.Trimesh(vertices=np.array([]),faces=np.array([]))
#     for m in mesh_pieces:
#         leftover_mesh += m

    leftover_mesh = trimesh.util.concatenate( mesh_pieces +  [leftover_mesh])
        
    if merge_vertices:
        leftover_mesh.merge_vertices()
    
    return leftover_mesh

"""
def bbox_mesh_restriction(curr_mesh,bbox_upper_corners,
                         mult_ratio = 1):
    bbox_center = np.mean(bbox_upper_corners,axis=0)
    bbox_distance = np.max(bbox_upper_corners,axis=0)-bbox_center
    
    #face_midpoints = np.mean(curr_mesh.vertices[curr_mesh.faces],axis=1)
    face_midpoints = curr_mesh.triangles_center
    
    sum_totals = np.invert(np.sum((np.abs(face_midpoints-bbox_center)-mult_ratio*bbox_distance) > 0,axis=1).astype("bool").reshape(-1))
    #total_face_indexes = set(np.arange(0,len(sum_totals)))
    faces_bbox_inclusion = (np.arange(0,len(sum_totals)))[sum_totals]
    
    try:
        curr_mesh_bbox_restriction = curr_mesh.submesh([faces_bbox_inclusion],append=True)
        return curr_mesh_bbox_restriction,faces_bbox_inclusion
    except:
        #print(f"faces_bbox_inclusion = {faces_bbox_inclusion}")
        #print(f"curr_mesh = {curr_mesh}")
        #raise Exception("failed bbox_mesh")
        return curr_mesh,np.arange(0,len(curr_mesh.faces))
    
"""

# New bounding box method able to accept multiple
def bbox_mesh_restriction(curr_mesh,bbox_upper_corners,
                         mult_ratio = 1):
    """
    Purpose: Can send multiple bounding box corners to the function
    and it will restrict your mesh to only the faces that are within
    those bounding boxs
    ** currently doing bounding boxes that are axis aligned
    
    -- Future work --
    could get an oriented bounding box by doing
    
    elephant_skeleton_verts_mesh = trimesh.Trimesh(vertices=el_verts,faces=np.array([]))
    elephant_skeleton_verts_mesh.bounding_box_oriented 
    
    but would then have to do a projection into the oriented bounding box
    plane to get all of the points contained within
    
    
    """
    
    

    if type(bbox_upper_corners) != list:
        bbox_upper_corners = [bbox_upper_corners]
    
    sum_totals_list = []
    for bb_corners in bbox_upper_corners:
        
        if tu.is_mesh(bb_corners):
            bb_corners = tu.bounding_box_corners(bb_corners)
    
        bbox_center = np.mean(bb_corners,axis=0)
        bbox_distance = np.max(bb_corners,axis=0)-bbox_center

        #face_midpoints = np.mean(curr_mesh.vertices[curr_mesh.faces],axis=1)
        face_midpoints = curr_mesh.triangles_center

        current_sums = np.invert(np.sum((np.abs(face_midpoints-bbox_center)-mult_ratio*bbox_distance) > 0,axis=1).astype("bool").reshape(-1))
        sum_totals_list.append(current_sums)
    
    sum_totals = np.logical_or.reduce(sum_totals_list)
    #print(f"sum_totals = {sum_totals}")
    
    faces_bbox_inclusion = (np.arange(0,len(sum_totals)))[sum_totals]
    
    try:
        curr_mesh_bbox_restriction = curr_mesh.submesh([faces_bbox_inclusion],append=True,repair=False)
        return curr_mesh_bbox_restriction,faces_bbox_inclusion
    except:
        #print(f"faces_bbox_inclusion = {faces_bbox_inclusion}")
        #print(f"curr_mesh = {curr_mesh}")
        #raise Exception("failed bbox_mesh")
        return curr_mesh,np.arange(0,len(curr_mesh.faces))
    


# -------------- 11/21 More bounding box functions ----- #
def bounding_box(mesh,oriented=False):
    """
    Returns the mesh of the bounding box
    
    Input: Can take in the corners of a bounding box as well
    """
    if nu.is_array_like(mesh):
        mesh = np.array(mesh).reshape(-1,3)
        if len(mesh) != 2:
            raise Exception("did not recieve bounding box corners")
        mesh = trimesh.Trimesh(vertices = np.vstack([mesh,mesh.mean(axis=0).reshape(-1,3)]).reshape(-1,3),
                              faces=np.array([[0,1,2]]))
        mesh = mesh.bounding_box
    
    if oriented:
        return mesh.bounding_box_oriented
    else:
        return mesh.bounding_box

def bounding_box_center(mesh,oriented=False):
    """
    Computed the center of the bounding box
    
    Ex:
    ex_mesh = neuron_obj_with_web[axon_limb_name][9].mesh
    ipvu.plot_objects(ex_mesh,
                      scatters=[tu.bounding_box_center(ex_mesh)],
                      scatter_size=1)
    """
    bb_corners = bounding_box_corners(mesh,oriented = oriented)
    return np.mean(bb_corners,axis=0)
    
def bounding_box_corners(mesh,bbox_multiply_ratio=1,
                        oriented=False):
    #bbox_verts = mesh.bounding_box.vertices
    bbox_verts = bounding_box(mesh,oriented=oriented).vertices
    bb_corners = np.array([np.min(bbox_verts,axis=0),np.max(bbox_verts,axis=0)]).reshape(2,3)
    if bbox_multiply_ratio == 1:
        return bb_corners
    
    bbox_center = np.mean(bb_corners,axis=0)
    bbox_distance = np.max(bb_corners,axis=0)-bbox_center
    new_corners = np.array([bbox_center - bbox_multiply_ratio*bbox_distance,
                            bbox_center + bbox_multiply_ratio*bbox_distance
                           ]).reshape(-1,3)
    return new_corners

def bounding_box_corner_min(mesh,**kwargs):
    return bounding_box_corners(mesh,**kwargs)[0]

def bbox_min_x(mesh,**kwargs):
    return bounding_box_corners(mesh,**kwargs)[0][0]
def bbox_min_y(mesh,**kwargs):
    return bounding_box_corners(mesh,**kwargs)[0][1]
def bbox_min_z(mesh,**kwargs):
    return bounding_box_corners(mesh,**kwargs)[0][2]

def bounding_box_corner_max(mesh,**kwargs):
    return bounding_box_corners(mesh,**kwargs)[1]

def bbox_max_x(mesh,**kwargs):
    return bounding_box_corners(mesh,**kwargs)[1][0]
def bbox_max_y(mesh,**kwargs):
    return bounding_box_corners(mesh,**kwargs)[1][1]
def bbox_max_z(mesh,**kwargs):
    return bounding_box_corners(mesh,**kwargs)[1][2]
        
def check_coordinates_inside_bounding_box(mesh,
                                    coordinates,
                                    bbox_coordinate_divisor=None,
                                          return_inside_indices = True, # or else returns true/false mask of points inside
                                             verbose=False):
    """
    Purpose: To return the indices of points inside of the 
    bounding box of  amesh
    
    Ex:
    soma_mesh = neuron_obj.get_soma_meshes()[0]
    ex_verts = np.vstack([neuron_objs[0][1].mesh.vertices[:5],soma_mesh.vertices[:5]])
    tu.check_coordinates_inside_bounding_box(soma_mesh,ex_verts,return_inside_indices=False)

    """
    if len(mesh.vertices) <= 0:
        if return_inside_indices:
            return []
        else:
            return np.array([False]*len(coordinates))
    
    
    curr_mesh_bounding_box = np.array(tu.bounding_box_corners(mesh))

    if bbox_coordinate_divisor is not None:
        if verbose:
            print(f"curr_mesh_bounding_box before divisor ({bbox_coordinate_divisor}): {curr_mesh_bounding_box}")
        curr_mesh_bounding_box = curr_mesh_bounding_box/np.array(bbox_coordinate_divisor)

        if verbose:
            print(f"curr_mesh_bounding_box AFTER divisor ({bbox_coordinate_divisor}): {curr_mesh_bounding_box}")

    inside_true_false_mask = np.all((coordinates <= curr_mesh_bounding_box[1]) & (coordinates >= curr_mesh_bounding_box[0]),axis=1)
    if return_inside_indices:
        points_inside = np.where(inside_true_false_mask)[0]
        return points_inside
    else:
        return inside_true_false_mask
    
def vertices_mask_inside_mesh_bbox(
    mesh,
    mesh_for_bbox,
    bbox_multiply_ratio=1,
    ):
    main_mesh_bbox_corners = tu.bounding_box_corners(mesh_for_bbox,bbox_multiply_ratio)
    inside_results = trimesh.bounds.contains(main_mesh_bbox_corners,mesh.vertices.reshape(-1,3))
    return inside_results


def vertices_mask_inside_meshes_bbox(
    mesh,
    meshes_for_bbox,
    bbox_multiply_ratio=1,
    ):
    
    mesh_for_bbox = nu.convert_to_array_like(meshes_for_bbox)
    vertices_mask = np.zeros(len(mesh.vertices))
    
    for m in meshes_for_bbox:
        vertices_mask += tu.vertices_mask_inside_mesh_bbox(mesh,m,bbox_multiply_ratio=bbox_multiply_ratio)
    return vertices_mask
    
def n_vertices_inside_mesh_bbox(
    mesh,
    mesh_for_bbox,
    return_inside = True,
    bbox_multiply_ratio=1,
    return_percentage = False,
    verbose = False
    ):
    """
    Purpose: to return the number of faces within the bounding box of another face 
    """
    #1) Get the bounding box corners of the main mesh
    inside_results = tu.vertices_mask_inside_meshes_bbox(
        mesh,
        mesh_for_bbox,
        bbox_multiply_ratio=bbox_multiply_ratio,
        )
    n_inside_vertices = int(np.sum(inside_results))
        
    if not return_inside:
        n_inside_vertices = len(mesh.vertices) - n_inside_vertices
        
    inside_vertices_per = n_inside_vertices/len(mesh.vertices)*100
    
    if verbose:
        print(f"n_inside_vertices = {n_inside_vertices}")
        print(f"inside_vertices_per = {inside_vertices_per}")
    
    if return_percentage:
        return inside_vertices_per
    else:
        return n_inside_vertices    
    
def n_vertices_outside_mesh_bbox(
    mesh,
    mesh_for_bbox,
    bbox_multiply_ratio=1,
    return_percentage = False,
    verbose = False
    ):
    
    return tu.n_vertices_inside_mesh_bbox(
    mesh,
    mesh_for_bbox,
    return_inside = False,
    bbox_multiply_ratio=bbox_multiply_ratio,
    return_percentage = return_percentage,
    verbose = verbose
    )
    


def check_meshes_outside_mesh_bbox(main_mesh,test_meshes,
                                  return_indices=False):
    return check_meshes_inside_mesh_bbox(main_mesh,test_meshes,
                                  return_indices=return_indices,
                                  return_inside=False)



def check_meshes_inside_mesh_bbox(main_mesh,test_meshes,
                                  return_indices=False,
                                  return_inside=True,
                                 bbox_multiply_ratio=1):
    """
    Purpose: Will check to see if any of the vertices
    of the test meshes are inside the bounding box of the main mesh
    
    Pseudocode: 
    1) Get the bounding box corners of the main mesh
    2) For each test mesh
    - send the vertices to see if inside bounding box
    - if any are then add indices to the running list
    
    3) Return either the meshes/indices of the inside/outside pieces
    based on the parameters set
    
    """
    #1) Get the bounding box corners of the main mesh
    main_mesh_bbox_corners = bounding_box_corners(main_mesh,bbox_multiply_ratio)
    
    #2) Iterate through test meshes
    inside_meshes_idx = []
    for j,tm in enumerate(test_meshes):
        inside_results = trimesh.bounds.contains(main_mesh_bbox_corners,tm.vertices.reshape(-1,3))
        if np.any(inside_results):
            inside_meshes_idx.append(j)
    
    #3) Set the return values
    if not return_inside:
        return_idx = np.delete(np.arange(len(test_meshes)),inside_meshes_idx)
    else:
        return_idx = np.array(inside_meshes_idx)
    
    if return_indices:
        return return_idx
    else:
        return [k for i,k in enumerate(test_meshes) if i in return_idx]
    
def check_meshes_outside_multiple_mesh_bbox(main_meshes,test_meshes,
                                  return_indices=False):
    return check_meshes_inside_multiple_mesh_bbox(main_meshes,test_meshes,
                                  return_indices=return_indices,
                                  return_inside=False)

def check_meshes_inside_multiple_mesh_bbox(main_meshes,test_meshes,
                                  return_indices=False,
                                  return_inside=True):
    """
    Purpose: will return all of the pieces inside or outside of 
    multiple seperate main mesh bounding boxes
    
    Pseudocode: 
    For each main mesh
    1) Run the check_meshes_inside_mesh_bbox and collect the resulting indexes
    2) Combine the results based on the following:
    - If outside, then do intersetion of results (becuase need to be outside of all)
    - if inside, then return union of results (because if inside at least one then should be considered inside)
    3) Return either the meshes or indices
    
    Ex: 
    from mesh_tools import trimesh_utils as tu
    tu = reload(tu)
    tu.check_meshes_inside_multiple_mesh_bbox([soma_mesh,soma_mesh,soma_mesh],neuron_obj.non_soma_touching_meshes,
                                 return_indices=False)
    
    """
    if not nu.is_array_like(main_meshes):
        raise Exception("Was expecting a list of main meshes")
    
    #1) Run the check_meshes_inside_mesh_bbox and collect the resulting indexes
    
    all_results = []
    for main_mesh in main_meshes:
        curr_results = check_meshes_inside_mesh_bbox(main_mesh,test_meshes,
                                  return_indices=True,
                                  return_inside=return_inside)
        
        all_results.append(curr_results)
    
    #2) Combine the results based on the following:
    if return_inside:
        joining_function = np.union1d
    else:
        joining_function = np.intersect1d
    
    final_indices = all_results[0]
    
    for i in range(1,len(all_results)):
        final_indices = joining_function(final_indices,all_results[i])
    
    #3) Return either the meshes or indices
    if return_indices:
        return final_indices
    else:
        return [k for i,k in enumerate(test_meshes) if i in final_indices]

    
    

# main mesh cancellation
# --------------- 12/3 Addition: Made the connectivity matrix from the vertices by default ------------- #
def split_significant_pieces_old(new_submesh,
                            significance_threshold=100,
                            print_flag=False,
                            return_insignificant_pieces=False,
                            connectivity="vertices"):
    
    if type(new_submesh) != type(trimesh.Trimesh()):
        print("Inside split_significant_pieces and was passed empty mesh so retruning empty list")
        if return_insignificant_pieces:
            return [],[]
        else:
            return []
    
    if print_flag:
        print("------Starting the mesh filter for significant outside pieces-------")
#     from datasci_tools import system_utils as su
#     su.compressed_pickle(new_submesh,f"new_submesh_{np.random.randint(10,1000)}")
    if connectivity=="edges":
        mesh_pieces = new_submesh.split(only_watertight=False,repair=False)
    else:
        mesh_pieces = split_by_vertices(new_submesh,return_components=False)
        
    if print_flag:
        print(f"Finished splitting mesh_pieces into = {mesh_pieces}")
    if type(mesh_pieces) not in [type(np.ndarray([])),type(np.array([])),list]:
        mesh_pieces = [mesh_pieces]
    
    if print_flag:
        print(f"There were {len(mesh_pieces)} pieces after mesh split")

    significant_pieces = [m for m in mesh_pieces if len(m.faces) >= significance_threshold]
    if return_insignificant_pieces:
        insignificant_pieces = [m for m in mesh_pieces if len(m.faces) < significance_threshold]

    if print_flag:
        print(f"There were {len(significant_pieces)} pieces found after size threshold")
    if len(significant_pieces) <=0:
        print("THERE WERE NO MESH PIECES GREATER THAN THE significance_threshold")
        if return_insignificant_pieces:
            return [],[]
        else:
            return []
    
    #arrange the significant pieces from largest to smallest
    x = [len(k.vertices) for k in significant_pieces]
    sorted_indexes = sorted(range(len(x)), key=lambda k: x[k])
    sorted_indexes = sorted_indexes[::-1]
    sorted_significant_pieces = [significant_pieces[k] for k in sorted_indexes]
    
    if return_insignificant_pieces:
        #arrange the significant pieces from largest to smallest
        x = [len(k.vertices) for k in insignificant_pieces]
        sorted_indexes = sorted(range(len(x)), key=lambda k: x[k])
        sorted_indexes = sorted_indexes[::-1]
        sorted_significant_pieces_insig = [insignificant_pieces[k] for k in sorted_indexes]
    if return_insignificant_pieces:
        return sorted_significant_pieces,sorted_significant_pieces_insig
    else:
        return sorted_significant_pieces
    
def face_idx_map_from_face_idx_list(
    face_idx_list,
    mesh = None,
    n_faces = None,
    default_value = -1
    ):
    """
    Purpose: To turn a list of face idx
    into an overall face idx mapping to each component
    """
    if n_faces is None:
        n_faces = len(mesh.faces)
        
    face_map_idx = np.ones(n_faces)*default_value
    for j,curr_idx in enumerate(face_idx_list):
        face_map_idx[curr_idx] = j
        
    return face_map_idx


def split_significant_pieces(new_submesh,
                            significance_threshold=100,
                            print_flag=False,
                            return_insignificant_pieces=False,
                             return_face_indices = False,
                             return_face_map_for_indices = False,
                            connectivity="vertices"):
    """
    Will split a mesh based on connectivity of edges or vertices
    (can return insiginifcant pieces and their face indices)
    
    Ex: 
    (split_meshes,split_meshes_face_idx,
    split_meshes_insig,split_meshes_face_idx_insig) = tu.split_significant_pieces(main_mesh_total,connectivity="edges",
                                                                                  return_insignificant_pieces=True,
                                                                     return_face_indices=True)
    
    
    
    """

    
    
    if type(new_submesh) != type(trimesh.Trimesh()):
        print("Inside split_significant_pieces and was passed empty mesh so retruning empty list")
        if return_insignificant_pieces:
            return [],[]
        else:
            return []
    
    if print_flag:
        print("------Starting the mesh filter for significant outside pieces-------")
#     from datasci_tools import system_utils as su
#     su.compressed_pickle(new_submesh,f"new_submesh_{np.random.randint(10,1000)}")
    if connectivity=="edges":
        """ Old way that did not have options for the indices 
        mesh_pieces = new_submesh.split(only_watertight=False,repair=False)
        """
        
        mesh_pieces,mesh_pieces_idx = tu.split(new_submesh,connectivity="edges") 
    else:
        mesh_pieces,mesh_pieces_idx = split_by_vertices(new_submesh,return_components=True)
        
    if print_flag:
        print(f"Finished splitting mesh_pieces into = {mesh_pieces}")
    if type(mesh_pieces) not in [type(np.ndarray([])),type(np.array([])),list]:
        mesh_pieces = [mesh_pieces]
    
    if print_flag:
        print(f"There were {len(mesh_pieces)} pieces after mesh split")

    mesh_pieces = np.array(mesh_pieces)
    mesh_pieces_idx = np.array(mesh_pieces_idx)
    
    pieces_len = np.array([len(m.faces) for m in mesh_pieces])
    significant_pieces_idx = np.where(pieces_len >= significance_threshold)[0]
    insignificant_pieces_idx = np.where(pieces_len < significance_threshold)[0]
    
    significant_pieces = mesh_pieces[significant_pieces_idx]
    significant_pieces_face_idx = mesh_pieces_idx[significant_pieces_idx]
    
    if return_insignificant_pieces:
        insignificant_pieces = mesh_pieces[insignificant_pieces_idx]
        insignificant_pieces_face_idx = mesh_pieces_idx[insignificant_pieces_idx]

    if print_flag:
        print(f"There were {len(significant_pieces)} pieces found after size threshold")
        
        
    if len(significant_pieces) <=0:
        #print("THERE WERE NO MESH PIECES GREATER THAN THE significance_threshold")
        if return_insignificant_pieces:
            if return_face_indices:
                return [],[],[],[]
            else:
                return [],[]
        else:
            if return_face_indices:
                return [],[]
            else:
                return []
    
    #arrange the significant pieces from largest to smallest
    sorted_significant_meshes,sorted_significant_meshes_idx = tu.sort_meshes_largest_to_smallest(significant_pieces,
                                                                                                sort_attribute="vertices",
                                                                                                return_idx=True)
    sorted_significant_meshes_face_idx = significant_pieces_face_idx[sorted_significant_meshes_idx]
    
    sorted_significant_meshes = list(sorted_significant_meshes)
    sorted_significant_meshes_face_idx = list(sorted_significant_meshes_face_idx)
    
#     x = [len(k.vertices) for k in significant_pieces]
#     sorted_indexes = sorted(range(len(x)), key=lambda k: x[k])a
#     sorted_indexes = sorted_indexes[::-1]
#     sorted_significant_pieces = [significant_pieces[k] for k in sorted_indexes]
    
    if return_insignificant_pieces:
        sorted_insignificant_meshes,sorted_insignificant_meshes_idx = tu.sort_meshes_largest_to_smallest(insignificant_pieces,
                                                                                                sort_attribute="vertices",
                                                                                                return_idx=True)
        sorted_insignificant_meshes_face_idx = insignificant_pieces_face_idx[sorted_insignificant_meshes_idx]
        
        sorted_insignificant_meshes = list(sorted_insignificant_meshes)
        sorted_insignificant_meshes_face_idx = list(sorted_insignificant_meshes_face_idx)
        
#         #arrange the significant pieces from largest to smallest
#         x = [len(k.vertices) for k in insignificant_pieces]
#         sorted_indexes = sorted(range(len(x)), key=lambda k: x[k])
#         sorted_indexes = sorted_indexes[::-1]
#         sorted_significant_pieces_insig = [insignificant_pieces[k] for k in sorted_indexes]
        
        
    if return_face_map_for_indices and return_insignificant_pieces:
        sorted_significant_meshes_face_idx = tu.face_idx_map_from_face_idx_list(sorted_significant_meshes_face_idx,mesh = new_submesh)
        sorted_insignificant_meshes_face_idx = tu.face_idx_map_from_face_idx_list(sorted_insignificant_meshes_face_idx,mesh = new_submesh)
        
    if return_insignificant_pieces:
        if return_face_indices:
            
            return (sorted_significant_meshes,sorted_significant_meshes_face_idx,
                sorted_insignificant_meshes,sorted_insignificant_meshes_face_idx)
        else:
            return (sorted_significant_meshes,
                sorted_insignificant_meshes)
    else:
        if return_face_indices:
            return (sorted_significant_meshes,sorted_significant_meshes_face_idx)
        else:
            return sorted_significant_meshes


"""
******* 
The submesh function if doesn't have repair = False might
end up adding on some faces that you don't want!
*******
"""

def sort_meshes_largest_to_smallest(meshes,
                                    sort_attribute="faces",
                                    return_idx=False):
    x = [len(getattr(k,sort_attribute)) for k in meshes]
    sorted_indexes = sorted(range(len(x)), key=lambda k: x[k])
    sorted_indexes = sorted_indexes[::-1]
    sorted_meshes = [meshes[k] for k in sorted_indexes]
    if return_idx:
        return sorted_meshes,sorted_indexes
    else:
        return sorted_meshes
    
    

def split(mesh, only_watertight=False,
          adjacency=None,
          engine=None, 
          return_components=True, 
          return_face_idx_map = False,
          connectivity="vertices",
          return_mesh_list = False,
          **kwargs):
    """
    Split a mesh into multiple meshes from face
    connectivity.
    If only_watertight is true it will only return
    watertight meshes and will attempt to repair
    single triangle or quad holes.
    Parameters
    ----------
    mesh : trimesh.Trimesh
    only_watertight: bool
      Only return watertight components
    adjacency : (n, 2) int
      Face adjacency to override full mesh
    engine : str or None
      Which graph engine to use
    Returns
    ----------
    meshes : (m,) trimesh.Trimesh
      Results of splitting
      
    ----------------***** THIS VERSION HAS BEEN ALTERED TO PASS BACK THE COMPONENTS INDICES TOO ****------------------
    
    if return_components=True then will return an array of arrays that contain face indexes for all the submeshes split off
    Ex: 
    
    tu.split(elephant_and_box)
    meshes = array([<trimesh.Trimesh(vertices.shape=(2775, 3), faces.shape=(5558, 3))>,
        <trimesh.Trimesh(vertices.shape=(8, 3), faces.shape=(12, 3))>],
       dtype=object)
    components = array([array([   0, 3710, 3709, ..., 1848, 1847, 1855]),
        array([5567, 5566, 5565, 5564, 5563, 5559, 5561, 5560, 5558, 5568, 5562,
        5569])], dtype=object)
    
    """
    if connectivity == "vertices":
        return split_by_vertices(
            mesh,
            return_components=return_components,
            return_face_idx_map = return_face_idx_map)
    
    if adjacency is None:
        adjacency = mesh.face_adjacency

    # if only watertight the shortest thing we can split has 3 triangles
    if only_watertight:
        min_len = 4
    else:
        min_len = 1
        
    #print(f"only_watertight = {only_watertight}")

    components = connected_components(
        edges=adjacency,
        nodes=np.arange(len(mesh.faces)),
        min_len=min_len,
        engine=engine)
              
    
    #print(f"components = {[c.shape for c in components]}")
    meshes = mesh.submesh(
        components, only_watertight=only_watertight, repair=False, **kwargs)
    #print(f"meshes = {meshes}")
    
    """ 6 19, old way of doing checking that did not resolve anything
    if type(meshes) != type(np.array([])):
        print(f"meshes = {meshes}, with type = {type(meshes)}")
    """
        
    if type(meshes) != type(np.array([])) and type(meshes) != list:
        #print(f"meshes = {sub_components}, with type = {type(sub_components)}")
        if type(meshes) == type(trimesh.Trimesh()) :
            
            print("list was only one so surrounding them with list")
            #print(f"meshes_before = {meshes}")
            #print(f"components_before = {components}")
            meshes = [meshes]
            
        else:
            raise Exception("The sub_components were not an array, list or trimesh")
            
    #make sure they are in order from least to greatest size
#     current_array = [len(c) for c in components]
#     ordered_indices = np.flip(np.argsort(current_array))
    
    # order according to number of faces in meshes (SO DOESN'T ERROR ANYMORE)
    current_array = [len(c.faces) for c in meshes]
    ordered_indices = np.flip(np.argsort(current_array))
              
    
    
    ordered_meshes = np.array([meshes[i] for i in ordered_indices])
    ordered_components = np.array([components[i] for i in ordered_indices])
    
    
    if len(ordered_meshes)>=2:
        if (len(ordered_meshes[0].faces) < len(ordered_meshes[1].faces)) and (len(ordered_meshes[0].vertices) < len(ordered_meshes[1].vertices)) :
            #print(f"ordered_meshes = {ordered_meshes}")
            raise Exception(f"Split is not passing back ordered faces:"
                            f" ordered_meshes = {ordered_meshes},  "
                           f"components= {components},  "
                           f"meshes = {meshes},  "
                            f"current_array={current_array},  "
                            f"ordered_indices={ordered_indices},  "
                           )
    
    #control if the meshes is iterable or not
    try:
        ordered_comp_indices = np.array([k.astype("int") for k in ordered_components])
    except:
        pass
        # from datasci_tools import system_utils as su
        # su.compressed_pickle(ordered_components,"ordered_components")
        # print(f"ordered_components = {ordered_components}")
        # raise Exception("ordered_components")
    
    if return_mesh_list:
        if type(ordered_meshes) != type(np.array([])) and type(ordered_meshes) != list:
            #print(f"meshes = {sub_components}, with type = {type(sub_components)}")
            if type(ordered_meshes) == type(trimesh.Trimesh()) :
                ordered_meshes = [ordered_meshes]
            else:
                raise Exception("The sub_components were not an array, list or trimesh")
    
    if return_face_idx_map:
        return_components = True
        ordered_comp_indices = tu.face_idx_map_from_face_idx_list(ordered_comp_indices,mesh=mesh,)
    if return_components:
        return ordered_meshes,ordered_comp_indices
    else:
        return ordered_meshes

def closest_distance_between_meshes(original_mesh,
                                    submesh,print_flag=False,
                                   attribute_name = "triangles_center"):
    global_start = time.time()
    original_mesh_midpoints = getattr(original_mesh,attribute_name)
    submesh_midpoints = getattr(submesh,attribute_name)
    
    #1) Put the submesh face midpoints into a KDTree
    submesh_mesh_kdtree = KDTree(submesh_midpoints)
    #2) Query the fae midpoints of submesh against KDTree
    distances,closest_node = submesh_mesh_kdtree.query(original_mesh_midpoints)
    
    if print_flag:
        print(f"Total time for mesh distance: {time.time() - global_start}")
        
    return np.min(distances)

def compare_meshes_by_face_midpoints_list(mesh1_list,mesh2_list,**kwargs):
    match_list = []
    for mesh1,mesh2 in zip(mesh1_list,mesh2_list):
        match_list.append(compare_meshes_by_face_midpoints(mesh1,mesh2,**kwargs))
    
    return match_list

def compare_meshes_by_face_midpoints(mesh1,mesh2,match_threshold=0.001,print_flag=False):
    #0) calculate the face midpoints of each of the faces for original and submesh
    debug = False
    global_start = time.time()
    total_faces_greater_than_threshold = dict()
    starting_meshes = [mesh1,mesh2]
    if debug:
        print(f"mesh1.faces.shape = {mesh1.faces.shape},mesh2.faces.shape = {mesh2.faces.shape}")
    for i in range(0,len(starting_meshes)):
        
        original_mesh_midpoints = starting_meshes[i].triangles_center
        submesh_midpoints = starting_meshes[np.abs(i-1)].triangles_center


        #1) Put the submesh face midpoints into a KDTree
        submesh_mesh_kdtree = KDTree(submesh_midpoints)
        #2) Query the fae midpoints of submesh against KDTree
        distances,closest_node = submesh_mesh_kdtree.query(original_mesh_midpoints)

        faces_greater_than_treshold = (np.arange(len(original_mesh_midpoints)))[distances >= match_threshold]
        total_faces_greater_than_threshold[i] = faces_greater_than_treshold
    
    if print_flag:
        print(f"Total time for mesh mapping: {time.time() - global_start}")
    
    
    if len(total_faces_greater_than_threshold[0])>0 or len(total_faces_greater_than_threshold[1])>0:
        if print_flag:
            print(f"{len(total_faces_greater_than_threshold[0])} face midpoints of mesh1 were farther than {match_threshold} "
                  f"from the face midpoints of mesh2")
            
            print(f"{len(total_faces_greater_than_threshold[1])} face midpoints of mesh2 were farther than {match_threshold} "
                  f"from the face midpoints of mesh1")
        if debug:
            mesh1.export("mesh1_failed.off")
            mesh2.export("mesh2_failed.off")
        return False
    else:
        if print_flag:
            print("Meshes are equal!")
        return True
    
def original_mesh_vertices_map(original_mesh, submesh=None,
                               vertices_coordinates=None,
                               matching=True,
                               match_threshold = 0.001,
                               print_flag=False):
    """
    Purpose: Given an original_mesh and either a 
        i) submesh
        ii) list of vertices coordinates
    Find the indices of the original vertices in the
    original mesh
    
    Pseudocode:
    1) Get vertices to map to original
    2) Construct a KDTree of the original mesh vertices
    3) query the closest vertices on the original mesh
    
    
    """
    
    if not submesh is None:
        vertices_coordinates = submesh.vertices
    elif vertices_coordinates is None:
        raise Exception("Both Submesh and vertices_coordinates are None")
    else:
        pass
    
    global_start = time.time()
    #1) Put the submesh face midpoints into a KDTree
    original_mesh_kdtree = KDTree(original_mesh.vertices)
    #2) Query the fae midpoints of submesh against KDTree
    distances,closest_node = original_mesh_kdtree.query(vertices_coordinates)

    #check that all of them matched below the threshold
    if np.any(distances> match_threshold):
        raise Exception(f"There were {np.sum(distances> match_threshold)} faces that did not have an exact match to the original mesh")

    if print_flag:
        print(f"Total time for mesh mapping: {time.time() - global_start}")

    return closest_node
    
def subtract_mesh(original_mesh,subtract_mesh,
                    return_mesh=True,
                    exact_match=True,
                  match_threshold=0.001,
                  error_for_exact_match = True,
                   ):
    if nu.is_array_like(subtract_mesh) and len(subtract_mesh) == 0:
        return original_mesh
    if nu.is_array_like(subtract_mesh):
        subtract_mesh = combine_meshes(subtract_mesh)
        
    if len(subtract_mesh.faces) <= 0:
        return original_mesh
    return original_mesh_faces_map(original_mesh=original_mesh,
                                   submesh=subtract_mesh,
                                   matching=False,
                                   return_mesh=return_mesh,
                                   exact_match=exact_match,
                                   match_threshold=match_threshold,
                                   error_for_exact_match=error_for_exact_match,
                                  )

def restrict_mesh(original_mesh,restrict_meshes,
                    return_mesh=True
                   ):
    
    if nu.is_array_like(restrict_meshes):
        restrict_meshes = combine_meshes(restrict_meshes)
        
    return original_mesh_faces_map(original_mesh=original_mesh,
                                   submesh=restrict_meshes,
                                   matching=True,
                                   return_mesh=return_mesh
                                  )

def original_mesh_faces_map(original_mesh, submesh,
                           matching=True,
                           print_flag=False,
                           match_threshold = 0.001,
                            exact_match=False,
                            error_for_exact_match = True,
                           return_mesh=False,
                           original_mesh_kdtree=None):
    """
    PUrpose: Given a base mesh and mesh that was a submesh of that base mesh
    - find the original face indices of the submesh
    
    Pseudocode: 
    0) calculate the face midpoints of each of the faces for original and submesh
    1) Put the base mesh face midpoints into a KDTree
    2) Query the fae midpoints of submesh against KDTree
    3) Only keep those that correspond to the faces or do not correspond to the faces
    based on the parameter setting
    
    Can be inversed so can find the mapping of all the faces that not match a mesh
    """
    global_start = time.time()
    
    if type(original_mesh) not in  [type(trimesh.Trimesh()),trimesh.primitives.Box]:
        raise Exception("original mesh must be trimesh object")
    
    if type(submesh) != type(trimesh.Trimesh()):
        if not nu.non_empty_or_none(submesh):
            if matching:
                return_faces = np.array([])
                if return_mesh:
                    return trimesh.Trimesh(faces=np.array([]),
                                          vertices=np.array([]))
            else:
                return_faces = np.arange(0,len(original_mesh.faces))
                if return_mesh:
                    return original_mesh
                
            return return_faces
                
        else:
            submesh = combine_meshes(submesh)
    
    #pre-check for emppty meshes
    if len(submesh.vertices) == 0 or len(submesh.faces) == 0:
        if matching:
            return np.array([])
        else:
            return np.arange(0,len(original_mesh.faces))
        
    
    
    #0) calculate the face midpoints of each of the faces for original and submesh
    original_mesh_midpoints = original_mesh.triangles_center
    submesh_midpoints = submesh.triangles_center
    
    if not exact_match:
        #This was the old way which was switching the order the new faces were found
        #1) Put the submesh face midpoints into a KDTree
        submesh_mesh_kdtree = KDTree(submesh_midpoints)
        #2) Query the fae midpoints of submesh against KDTree
        distances,closest_node = submesh_mesh_kdtree.query(original_mesh_midpoints)
    

        if print_flag:
            print(f"Total time for mesh mapping: {time.time() - global_start}")

        #3) Only keep those that correspond to the faces or do not correspond to the faces
        #based on the parameter setting
        if matching:
            return_faces = (np.arange(len(original_mesh_midpoints)))[distances < match_threshold]

        else:
            return_faces = (np.arange(len(original_mesh_midpoints)))[distances >= match_threshold]
    else:        
        #1) Put the submesh face midpoints into a KDTree
        if original_mesh_kdtree is None:
            original_mesh_kdtree = KDTree(original_mesh_midpoints)
            
        #2) Query the fae midpoints of submesh against KDTree
        distances,closest_node = original_mesh_kdtree.query(submesh_midpoints)
        
        #check that all of them matched below the threshold
        if np.any(distances> match_threshold):
            if error_for_exact_match:
                raise Exception(f"There were {np.sum(distances> match_threshold)} faces that did not have an exact match to the original mesh")
            else:
                keep_map = distances <= match_threshold
                distances = distances[keep_map]
                closest_node = closest_node[keep_map]
        
        if print_flag:
            print(f"Total time for mesh mapping: {time.time() - global_start}")
        
        if matching:
            return_faces = closest_node
        else:
            return_faces = nu.remove_indexes(np.arange(len(original_mesh_midpoints)),closest_node)

    if return_mesh:
        return original_mesh.submesh([return_faces],append=True)
    else:
        return return_faces

def shared_edges_between_faces_on_mesh(mesh,faces_a,faces_b,
                                 return_vertices_idx=False):
    """
    Given two sets of faces, find the edges which are in both sets of faces.
    Parameters
    ---------
    faces_a : (n, 3) int
      Array of faces
    faces_b : (m, 3) int
      Array of faces
    Returns
    ---------
    shared : (p, 2) int
      Edges shared between faces
      
      
    Pseudocode:
    1) Get the unique edges of each of the faces
    """
    faces_a_edges = np.unique(mesh.faces_unique_edges[faces_a].ravel())
    faces_b_edges = np.unique(mesh.faces_unique_edges[faces_b].ravel())
    shared_edges_idx = np.intersect1d(faces_a_edges,faces_b_edges)
    
    if return_vertices_idx:
        return np.unique(mesh.edges_unique[shared_edges_idx].ravel())
    else:
        return shared_edges_idx
    
def mesh_pieces_connectivity(
                main_mesh,
                central_piece,
                periphery_pieces,
                connectivity="edges",
                return_vertices=False,
                return_central_faces=False,
                return_vertices_idx = False,
                print_flag=False,
                merge_vertices=False):
    """
    purpose: function that will determine if certain pieces of mesh are touching in reference
    to a central mesh

    Pseudocde: 
    1) Get the original faces of the central_piece and the periphery_pieces
    2) For each periphery piece, find if touching the central piece at all
    
    - get the vertices belonging to central mesh
    - get vertices belonging to current periphery
    - see if there is any overlap
    
    
    2a) If yes then add to the list to return
    2b) if no, don't add to list
    
    Example of How to use it: 
    
    connected_mesh_pieces = mesh_pieces_connectivity(
                    main_mesh=current_mesh,
                    central_piece=seperate_soma_meshes[0],
                    periphery_pieces = sig_non_soma_pieces)
    print(f"connected_mesh_pieces = {connected_mesh_pieces}")

    Application: For finding connectivity to the somas


    Example: How to use merge vertices option
    import time

    start_time = time.time()

    #0) Getting the Soma border

    tu = reload(tu)
    new_mesh = tu.combine_meshes(touching_limbs_meshes + [curr_soma_mesh])

    soma_idx = 1
    curr_soma_mesh = current_neuron[nru.soma_label(soma_idx)].mesh
    touching_limbs = current_neuron.get_limbs_touching_soma(soma_idx)
    touching_limb_objs = [current_neuron[k] for k in touching_limbs]

    touching_limbs_meshes = [k.mesh for k in touching_limb_objs]
    touching_pieces,touching_vertices = tu.mesh_pieces_connectivity(main_mesh=new_mesh,
                                            central_piece = curr_soma_mesh,
                                            periphery_pieces = touching_limbs_meshes,
                                                             return_vertices=True,
                                                            return_central_faces=False,
                                                                    print_flag=False,
                                                                    merge_vertices=True,
                                                                                     )
    limb_to_soma_border = dict([(k,v) for k,v in zip(np.array(touching_limbs)[touching_pieces],touching_vertices)])
    limb_to_soma_border

    print(time.time() - start_time)

    """
    
    """
    # 7-8 change: wanted to adapt so could give face ids as well instead of just meshes
    """
    if merge_vertices:
        main_mesh.merge_vertices()
    
    #1) Get the original faces of the central_piece and the periphery_pieces
    if type(central_piece) == type(trimesh.Trimesh()):
        central_piece_faces = original_mesh_faces_map(main_mesh,central_piece)
    else:
        #then what was passed were the face ids
        central_piece_faces = central_piece.copy()
        
    if print_flag:
        print(f"central_piece_faces = {central_piece_faces}")
    
    periphery_pieces_faces = []
    #periphery_pieces_faces = [original_mesh_faces_map(main_mesh,k) for k in periphery_pieces]
    #print(f"periphery_pieces = {len(periphery_pieces)}")
    for k in periphery_pieces:
        if type(k) == type(trimesh.Trimesh()):
            #print("using trimesh pieces")
            periphery_pieces_faces.append(original_mesh_faces_map(main_mesh,k))
        else:
            #print("just using face idxs")
            periphery_pieces_faces.append(k)
    
    if print_flag:
        print(f"periphery_pieces_faces = {periphery_pieces_faces}")
    
    #2) For each periphery piece, find if touching the central piece at all
    touching_periphery_pieces = []
    touching_periphery_pieces_intersecting_vertices= []
    touching_periphery_pieces_intersecting_vertices_idx = []
    
    #the faces have the vertices indices stored so just comparing vertices indices!
    
    if connectivity!="edges":
        central_p_verts = np.unique(main_mesh.faces[central_piece_faces].ravel())
    
    for j,curr_p_faces in enumerate(periphery_pieces_faces):
        
        if connectivity=="edges": #will do connectivity based on edges
            intersecting_vertices = shared_edges_between_faces_on_mesh(main_mesh,
                                                                       faces_a=central_piece_faces,
                                                                       faces_b=curr_p_faces,
                                                                       return_vertices_idx=True)
            
        else:
            curr_p_verts = np.unique(main_mesh.faces[curr_p_faces].ravel())
            intersecting_vertices = np.intersect1d(central_p_verts,curr_p_verts)
        
        if print_flag:
            print(f"intersecting_vertices = {intersecting_vertices}")
        
        if len(intersecting_vertices) > 0:
            touching_periphery_pieces.append(j)
            touching_periphery_pieces_intersecting_vertices.append(main_mesh.vertices[intersecting_vertices])
            touching_periphery_pieces_intersecting_vertices_idx.append(intersecting_vertices)
    
    
    
    #redoing the return structure
    return_value = [touching_periphery_pieces]
    if return_vertices:
        return_value.append(touching_periphery_pieces_intersecting_vertices)
    if return_central_faces:
        return_value.append(central_piece_faces)
    if return_vertices_idx:
        return_value.append(touching_periphery_pieces_intersecting_vertices_idx)
    
    if len(return_value) == 1:
        return return_value[0]
    else:
        return return_value
    
    
#     if not return_vertices and not return_central_faces:
#         return touching_periphery_pieces
#     else:
#         if return_vertices and return_central_faces:
#             return touching_periphery_pieces,touching_periphery_pieces_intersecting_vertices,central_piece_faces
#         elif return_vertices:
#             return touching_periphery_pieces,touching_periphery_pieces_intersecting_vertices
#         elif return_central_faces:
#             touching_periphery_pieces,central_piece_faces
#         else:
#             raise Exception("Soething messed up with return in mesh connectivity")
            

def two_mesh_list_connectivity(mesh_list_1,mesh_list_2,
                               main_mesh,
                              return_weighted_edges = True,
                              verbose=False):
    """
    Purpose: To find the connectivity between two sets of 
    mesh lists (and possibly return the number of vertices that are connecting them)
    
    Pseudocode:
    1) Stack the somas meshes and other meshes
    2) Create a connectivity pairing to test
    3) Run the mesh connectivity to get the correct pairings and the weights
    4) Return the edges with the weights optionally attached

    **stacked meshes could be faces indices or meshes themselves

    """
    print_optimize = verbose
    main_mesh_total = main_mesh
    
    optimize_time = time.time()

    mesh_list_1_face_idx = tu.convert_meshes_to_face_idxes(mesh_list_1,main_mesh_total)
    mesh_list_2_face_idx = tu.convert_meshes_to_face_idxes(mesh_list_2,main_mesh_total)

    if print_optimize:
        print(f" Converting {time.time()-optimize_time}")
    optimize_time = time.time()

    #1) Stack the somas meshes and other meshes
    stacked_meshes = list(mesh_list_1_face_idx) + list(mesh_list_2_face_idx)

    #2) Create a connectivity pairing to test
    n_list_1 = len(mesh_list_1)
    pais_to_test = nu.unique_pairings_between_2_arrays(np.arange(n_list_1),np.arange(len(mesh_list_2)) + n_list_1)

    if print_optimize:
        print(f" Unique pairing {time.time()-optimize_time}")
    optimize_time = time.time()

    #3) Run the mesh connectivity to get the correct pairings and the weights
    optimize_time = time.time()
    mesh_conn_edges,conn_weights = tu.mesh_list_connectivity(meshes=stacked_meshes,
                             main_mesh = main_mesh_total,
                             pairs_to_test=pais_to_test,
                             weighted_edges=True)

    mesh_conn_edges[:,1] = mesh_conn_edges[:,1]-n_list_1
    if print_optimize:
        print(f" Mesh connectivity {time.time()-optimize_time}")
    optimize_time = time.time()

    if return_weighted_edges:
        return np.vstack([mesh_conn_edges.T,conn_weights]).T
    else:
        return mesh_conn_edges
    
    
    
def convert_meshes_to_face_idxes(mesh_list,
                              main_mesh,
                              exact_match=True,
                                original_mesh_kd=None):
    """
    Purpose: Will convert a list of 
    
    
    """
    
    if original_mesh_kd is None:
        original_mesh_midpoints = main_mesh.triangles_center
        original_mesh_kd = KDTree(original_mesh_midpoints)
    
    
    periphery_pieces_faces = []
    for k in mesh_list:
        if type(k) == type(trimesh.Trimesh()):
            #print("using trimesh pieces")
            periphery_pieces_faces.append(original_mesh_faces_map(main_mesh,k,
                                                                 exact_match=exact_match,
                                                                 original_mesh_kdtree=original_mesh_kd))
        else:
            #print("just using face idxs")
            periphery_pieces_faces.append(k)
    
    return periphery_pieces_faces
    
            
def mesh_list_connectivity(meshes,
                        main_mesh,
                           connectivity="edges",
                           pairs_to_test=None,
                           min_common_vertices=1,
                           weighted_edges=False,
                           return_vertex_connection_groups=False,
                           return_largest_vertex_connection_group=False,
                           return_connected_components=False,
                        print_flag = False,
                          verbose = False):
    """
    Pseudocode:
    1) Build an edge list
    2) Use the edgelist to find connected components

    Arguments:
    - meshes (list of trimesh.Trimesh) #
    - retrun_vertex_connection_groups (bool): whether to return the touching vertices


    """
    if verbose:
        print_flag = verbose

    periphery_pieces = meshes
    meshes_connectivity_edge_list = []
    meshes_connectivity_edge_weights = []
    meshes_connectivity_vertex_connection_groups = []
    
    vertex_graph = None
    
        

    periphery_pieces_faces = []
    #periphery_pieces_faces = [original_mesh_faces_map(main_mesh,k) for k in periphery_pieces]
    #print(f"periphery_pieces = {len(periphery_pieces)}")
    for k in periphery_pieces:
        if type(k) == type(trimesh.Trimesh()):
            #print("using trimesh pieces")
            periphery_pieces_faces.append(original_mesh_faces_map(main_mesh,k))
        else:
            #print("just using face idxs")
            periphery_pieces_faces.append(k)


    
        """
        Pseudocode:
        Iterates through all combinations of meshes
        1) get the faces of both meshes in the pair
        2) using the faces get the shared edges between them (if any)
        3) If there were shared edges then save them as intersecting vertices
        
        
        """
    if pairs_to_test is None:
        pairs_to_test = nu.all_unique_choose_2_combinations(np.arange(len(periphery_pieces_faces)))
    
    for i,j in pairs_to_test:
        central_p_faces = periphery_pieces_faces[j]
        
        if connectivity!="edges":
            central_p_verts = np.unique(main_mesh.faces[central_p_faces].ravel())

        curr_p_faces = periphery_pieces_faces[i]
        if connectivity=="edges": #will do connectivity based on edges
            intersecting_vertices = shared_edges_between_faces_on_mesh(main_mesh,
                                                                       faces_a=central_p_faces,
                                                                       faces_b=curr_p_faces,
                                                                       return_vertices_idx=True)

        else: #then do the vertex way

            curr_p_verts = np.unique(main_mesh.faces[curr_p_faces].ravel())

            intersecting_vertices = np.intersect1d(central_p_verts,curr_p_verts)

        if print_flag:
            print(f"intersecting_vertices = {intersecting_vertices}")


        if len(intersecting_vertices) >= min_common_vertices:
            if return_vertex_connection_groups:

                if vertex_graph is None:
                    vertex_graph = mesh_vertex_graph(main_mesh)

                curr_vertex_connection_groups = split_vertex_list_into_connected_components(
                                                vertex_indices_list=intersecting_vertices,
                                                mesh=main_mesh,
                                                vertex_graph=vertex_graph,
                                                return_coordinates=True,
                                               )
                if return_largest_vertex_connection_group:
                    curr_vertex_connection_groups_len = [len(k) for k in curr_vertex_connection_groups]
                    largest_group = np.argmax(curr_vertex_connection_groups_len)
                    curr_vertex_connection_groups = curr_vertex_connection_groups[largest_group]

                meshes_connectivity_vertex_connection_groups.append(curr_vertex_connection_groups)


            meshes_connectivity_edge_list.append((i,j))
            meshes_connectivity_edge_weights.append(len(intersecting_vertices))
                
                    

    meshes_connectivity_edge_list = nu.sort_elements_in_every_row(meshes_connectivity_edge_list)
    if return_vertex_connection_groups:
        return meshes_connectivity_edge_list,meshes_connectivity_vertex_connection_groups
    elif return_connected_components:
        return xu.connected_components_from_nodes_edges(np.arange(len(meshes)),meshes_connectivity_edge_list)
    else:
        if weighted_edges:
            return meshes_connectivity_edge_list,meshes_connectivity_edge_weights
        else:
            return meshes_connectivity_edge_list

'''
Saved method before added in vertex options


def mesh_list_connectivity(meshes,
                        main_mesh,
                           min_common_vertices=1,
                           return_vertex_connection_groups=False,
                           return_largest_vertex_connection_group=False,
                        print_flag = False):
    """
    Pseudocode:
    1) Build an edge list
    2) Use the edgelist to find connected components

    Arguments:
    - meshes (list of trimesh.Trimesh) #
    - retrun_vertex_connection_groups (bool): whether to return the touching vertices


    """

    periphery_pieces = meshes
    meshes_connectivity_edge_list = []
    meshes_connectivity_vertex_connection_groups = []
    
    vertex_graph = None
    
        

    periphery_pieces_faces = []
    #periphery_pieces_faces = [original_mesh_faces_map(main_mesh,k) for k in periphery_pieces]
    #print(f"periphery_pieces = {len(periphery_pieces)}")
    for k in periphery_pieces:
        if type(k) == type(trimesh.Trimesh()):
            #print("using trimesh pieces")
            periphery_pieces_faces.append(original_mesh_faces_map(main_mesh,k))
        else:
            #print("just using face idxs")
            periphery_pieces_faces.append(k)


    for j,central_p_faces in enumerate(periphery_pieces_faces):
        central_p_verts = np.unique(main_mesh.faces[central_p_faces].ravel())
        for i in range(0,j):
            curr_p_faces = periphery_pieces_faces[i]
            curr_p_verts = np.unique(main_mesh.faces[curr_p_faces].ravel())

            intersecting_vertices = np.intersect1d(central_p_verts,curr_p_verts)
            if print_flag:
                print(f"intersecting_vertices = {intersecting_vertices}")
            
            
            

            if len(intersecting_vertices) >= min_common_vertices:
                if return_vertex_connection_groups:
                    
                    if vertex_graph is None:
                        vertex_graph = mesh_vertex_graph(main_mesh)
                    
                    curr_vertex_connection_groups = split_vertex_list_into_connected_components(
                                                    vertex_indices_list=intersecting_vertices,
                                                    mesh=main_mesh,
                                                    vertex_graph=vertex_graph,
                                                    return_coordinates=True,
                                                   )
                    if return_largest_vertex_connection_group:
                        curr_vertex_connection_groups_len = [len(k) for k in curr_vertex_connection_groups]
                        largest_group = np.argmax(curr_vertex_connection_groups_len)
                        curr_vertex_connection_groups = curr_vertex_connection_groups[largest_group]
                
                    meshes_connectivity_vertex_connection_groups.append(curr_vertex_connection_groups)
                    
                meshes_connectivity_edge_list.append((j,i))

    meshes_connectivity_edge_list = nu.sort_elements_in_every_row(meshes_connectivity_edge_list)
    if return_vertex_connection_groups:
        return meshes_connectivity_edge_list,meshes_connectivity_vertex_connection_groups
    else:
        return meshes_connectivity_edge_list
    






'''
    
    


def split_vertex_list_into_connected_components_old(
    vertex_indices_list, #list of vertices referencing the mesh
    mesh=None, #the main mesh the vertices list references
    vertex_graph=None, # a precomputed vertex graph if available
    return_coordinates=True, #whether to return the groupings as coordinates (if False the returns them as indices)
    ):
    """
    Purpose: 
    Given a list of vertices (in reference to a main mesh),
    returns the vertices divided into connected components on the graph
    
    Pseudocode:
    1) Build graph from vertex and edges of mesh
    2) Create a subgraph from the vertices list
    3) Find the connected components of the subgraph
    4) Either return the vertex coordinates or indices
    """
    if vertex_graph is None:
        #1) Build graph from vertex and edges of mesh
        if mesh is None:
            raise Exception("Neither the vertex graph or mesh argument were non None")
            
        vertex_graph = mesh_vertex_graph(mesh)
    
    #2) Create a subgraph from the vertices list
    vertex_subgraph = vertex_graph.subgraph(vertex_indices_list)
    
    vertex_groups = [np.array(list(k)).astype("int") for k in list(nx.connected_components(vertex_subgraph))]
    
    if return_coordinates:
        return [mesh.vertices[k] for k in vertex_groups]
    else:
        return vertex_groups
    
def vertex_indices_subgraph(
    mesh,
    vertex_indices,
    plot_graph=False,
    verbose = False):
    """
    Purpose: To return a connectivity
    subgraph of the vertex indices
    """
    total_edge_indices = []
    for v_idx in vertex_indices:
        edge_indices = np.where((mesh.edges_unique[:,0]==v_idx ) | 
         (mesh.edges_unique[:,1]==v_idx ))[0]
        total_edge_indices += list(edge_indices)
        
    if verbose:
        print(f"total_edge_indices = {total_edge_indices}")
    
    edges = mesh.edges_unique[total_edge_indices]
    edges_lengths = mesh.edges_unique_length[total_edge_indices]
    
    if len(total_edge_indices) == 0:
        return nx.Graph()
    
    curr_weighted_edges = np.hstack([edges,
                                     edges_lengths.reshape(-1,1)])
    vertex_graph = nx.Graph()  
    vertex_graph.add_weighted_edges_from(curr_weighted_edges)
    curr_G = vertex_graph.subgraph(vertex_indices)
    if plot_graph:
        nx.draw(curr_G,with_labels = True)
    return curr_G
    
    
def split_vertex_list_into_connected_components(
    vertex_indices_list, #list of vertices referencing the mesh
    mesh, #the main mesh the vertices list references
    vertex_graph=None, # NOT USED
    return_coordinates=True, #whether to return the groupings as coordinates (if False the returns them as indices)
    ):
    """
    Purpose: 
    Given a list of vertices (in reference to a main mesh),
    returns the vertices divided into connected components on the graph
    
    Pseudocode:
    1) Get the subgraph of vertices in list
    2) Create a subgraph from the vertices list
    3) Find the connected components of the subgraph
    4) Either return the vertex coordinates or indices
    """

    #2) Create a subgraph from the vertices list
    if vertex_graph is None:
        vertex_graph = tu.vertex_indices_subgraph(mesh,vertex_indices_list)
    
    vertex_subgraph = vertex_graph.subgraph(vertex_indices_list)
    
    vertex_groups = [np.array(list(k)).astype("int") for k in list(nx.connected_components(vertex_subgraph))]
    
    if return_coordinates:
        return [mesh.vertices[k] for k in vertex_groups]
    else:
        return vertex_groups


def split_mesh_into_face_groups(
    base_mesh,face_mapping,
    return_idx=True,
    check_connect_comp = True,
    return_dict=True,
    plot = False):
    """
    Will split a mesh according to a face coloring of labels to split into 
    """
    if type(face_mapping) == dict:
        sorted_dict = dict(sorted(face_mapping.items()))
        face_mapping = list(sorted_dict.values())
    
    if len(face_mapping) != len(base_mesh.faces):
        raise Exception("face mapping does not have same length as mesh faces")
    
    unique_labels = np.sort(np.unique(face_mapping))
    total_submeshes = dict()
    total_submeshes_idx = dict()
    for lab in tqdm(unique_labels):
        faces = np.where(face_mapping==lab)[0]
        total_submeshes_idx[lab] = faces
        if not check_connect_comp:
            total_submeshes[lab] = base_mesh.submesh([faces],append=True,only_watertight=False,repair=False)
        else: 
            curr_submeshes = base_mesh.submesh([faces],append=False,only_watertight=False,repair=False)
            #print(f"len(curr_submeshes) = {len(curr_submeshes)}")
            if len(curr_submeshes) == 1:
                total_submeshes[lab] = curr_submeshes[0]
            else:
                raise Exception(f"Label {lab} has {len(curr_submeshes)} disconnected submeshes"
                                "\n(usually when checking after the waterfilling algorithm)")
    
    if not return_dict:
        total_submeshes = np.array(list(total_submeshes.values()))
        total_submeshes_idx =np.array(list(total_submeshes_idx.values()))
        
    if plot:
        total_submeshes = np.array(list(total_submeshes.values()))
        ipvu.plot_objects(
            meshes = list(total_submeshes),
            meshes_colors = "random",
        )
        
    if return_idx:
        return total_submeshes,total_submeshes_idx
    else:
        return total_submeshes
    
def split_mesh_by_closest_skeleton(mesh,skeletons,return_meshes=False):
    """
    Pseudocode: 
    For each N branch: 
    1) Build a KDTree of the skeleton
    2) query the mesh against the skeleton, get distances

    3) Concatenate all the distances and turn into (DxN) matrix 
    4) Find the argmin of each row and that is the assignment

    """


    dist_list = []
    for s in skeletons:
        sk_kd = KDTree(s.reshape(-1,3))
        dist, _ = sk_kd.query(mesh.triangles_center)
        dist_list.append(dist)

    dist_matrix = np.array(dist_list).T
    face_assignment = np.argmin(dist_matrix,axis=1)
    
    split_meshes_faces = [np.where(face_assignment == s_i)[0] for s_i in range(len(skeletons))]
    
    if return_meshes:
        split_meshes = [mesh.submesh([k],append=True,repair=False) for k in split_meshes_faces]
        return split_meshes,split_meshes_faces 
    else:
        return split_meshes_faces
        
#     split_meshes,split_meshes_faces = tu.split_mesh_into_face_groups(mesh,face_assignment,return_dict=False)
#     return split_meshes,split_meshes_faces 


 
"""
https://github.com/GPUOpen-LibrariesAndSDKs/RadeonProRenderUSD/issues/2


apt-get update
apt-get install -y wget

#explains why has to do this so can see the shared library: 
#https://stackoverflow.com/questions/1099981/why-cant-python-find-shared-objects-that-are-in-directories-in-sys-path
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc 
source ~/.bashrc



https://github.com/embree/embree#linux-and-macos (for the dependencies)

#for the dependencies
sudo apt-get install -y cmake-curses-gui
sudo apt-get install -y libtbb-dev
sudo apt-get install -y libglfw3-dev

Then run the following bash script (bash embree.bash)

trimesh bash file

---------------------------
set -xe
​
# Fetch the archive from GitHub releases.
wget https://github.com/embree/embree/releases/download/v2.17.7/embree-2.17.7.x86_64.linux.tar.gz -O /tmp/embree.tar.gz -nv
echo "2c4bdacd8f3c3480991b99e85b8f584975ac181373a75f3e9675bf7efae501fe  /tmp/embree.tar.gz" | sha256sum --check
tar -xzf /tmp/embree.tar.gz --strip-components=1 -C /usr/local
# remove archive
rm -rf /tmp/embree.tar.gz
​
# Install python bindings for embree (and upstream requirements).
pip3 install --no-cache-dir numpy cython
pip3 install --no-cache-dir https://github.com/scopatz/pyembree/releases/download/0.1.6/pyembree-0.1.6.tar.gz

-------------------------------





"""    
try:
    from trimesh.ray import ray_pyembree
except:
    pass
def ray_trace_distance(mesh, 
                    face_inds=None, 
                   vertex_inds=None,
                   ray_origins=None,
                   ray_directions=None,
                   max_iter=10, 
                   rand_jitter=0.001, 
                   verbose=False,
                   ray_inter=None,
                    replace_zero_values_with_center_distance = False,
                      debug=False):
    """
    Purpose: To calculate the distance from a vertex or face
    midpoint to an intersecting side of the mesh
    - To help with width calculations
    
    Pseudocode: 
    1) Get the ray origins and directions
    2) Create a mask that tells which ray origins we have
    calculated a valid width for and an array to store the widths (start as -1)
    3) Start an iteration loop that will only stop
    when have a valid width for all origin points
        a. get the indices of which points we don't have valid sdfs for
        and restrict the run to only those 
        b. Add some gaussian noise to the normals of these rays
        c. Run the ray intersection to get the (multiple=False)
            - locations of intersections (mx3)
            - index_ray responsible for that intersection (m,)
            - mesh face that was intersected (m,)
        d. Update the width array for the origins that returned
           a valid width (using the diagonal_dot instead of linalg.norm because faster )
        e. Update the mask that shows which ray_origins have yet to be processed
    
    4) Return the width array
    
    
    
    """
    
    if not trimesh.ray.has_embree:
        logging.warning(
            "calculating rays without pyembree, conda install pyembree for large speedup")

    #initializing the obejct that can perform ray tracing
    if ray_inter is None:
        ray_inter = ray_pyembree.RayMeshIntersector(mesh)
    
    if not face_inds is None:
        ray_origins = mesh.triangles_center[face_inds]
        ray_directions = mesh.face_normals[face_inds]
    elif not vertex_inds is None:
        ray_origins = mesh.vertices[vertex_inds]
        ray_directions = mesh.vertex_normals[vertex_inds]
    elif (not ray_origins is None) and (not ray_directions is None):
        pass
    else:
        face_inds = np.arange(0,len(mesh.faces))
        ray_origins = mesh.triangles_center[face_inds]
        ray_directions = mesh.face_normals[face_inds]
        
    
    rs = np.zeros(len(ray_origins)) #array to hold the widths when calculated
    good_rs = np.full(len(rs), False) #mask that keeps track of how many widths have been calculated

    it = 0
    while not np.all(good_rs): #continue until all sdf widths are calculated
        if debug:
            print(f"\n --- Iteration = {it} -----")
        if debug:
            print(f"Number of non_good rs = {np.sum(~good_rs)}")
        
        #this is the indices of where the mask [~good_rs,:] is true
        blank_inds = np.where(~good_rs)[0] #the vertices who still need widths calculated
        
        #
        starts = ray_origins[blank_inds] - ray_directions[blank_inds]
        
        #gets the opposite of the vertex normal so is pointing inwards
        #then adds jitter that gets bigger and bigger
        ray_directions_with_jitter = -ray_directions[blank_inds] \
            + (1.2**it)*rand_jitter*np.random.rand(* #the * is to expand out the shape tuple
                                                   ray_directions[blank_inds].shape)
        
        #computes the locations, index_ray and index of hit mesh
        intersect_locations,intersect_ray_index,intersect_mesh_index = ray_inter.intersects_location(starts, ray_directions_with_jitter, multiple_hits=False)
        
        if debug:
            print(f"len(intersect_locations) = {len(intersect_locations)}")
            
        if len(intersect_locations) > 0:
            
            #rs[blank_inds[intersect_ray_index]] = np.linalg.norm(starts[intersect_ray_index]-intersect_locations,axis=1)
            depths = trimesh.util.diagonal_dot(intersect_locations - starts[intersect_ray_index],
                                      ray_directions_with_jitter[intersect_ray_index])
            if debug:
                print(f"Number of dephts that are 0 = {len(np.where(depths == 0)[0])}")
            rs[blank_inds[intersect_ray_index]] = depths
            
            if debug:
                print(f"Number of rs == 0: {len(np.where(rs==0)[0]) }")
                print(f"np.sum(~good_rs) BEFORE UPDATE= {np.sum(~good_rs) }")
                if len(depths)<400:
                    print(f"depths = {depths}")
                    print(f"blank_inds[intersect_ray_index] = {blank_inds[intersect_ray_index]}")
                    print(f"np.where(rs==0)[0] = {np.where(rs==0)[0]}")
            good_rs[blank_inds[intersect_ray_index]] = True
            if debug:
                print(f"np.sum(~good_rs) AFTER UPDATE = {np.sum(~good_rs) }")
            
        if debug: 
            print(f"np.all(good_rs) = {np.all(good_rs)}")
        it += 1
        if it > max_iter:
            if verbose:
                print(f"hit max iterations {max_iter}")
            break
            
    if replace_zero_values_with_center_distance:
        m_center = tu.mesh_center_weighted_face_midpoints(mesh)
        zero_map = rs <= 0.01
        rs[zero_map] = np.linalg.norm(mesh.triangles_center[zero_map] - m_center,axis=1)
    return rs

def ray_trace_distance_by_mesh_center_dist(mesh):
    """
    Purpose: In case the ray trace comes
    back with an error or a measure of 0
    
    Pseudocode: 
    1) get the center of the mesh
    2) Find the distance of all face midpoints to mesh center
    """
    
    #1) get the center of the mesh
    m_center = tu.mesh_center_weighted_face_midpoints(mesh)
    
    m_dist = np.linalg.norm(mesh.vertices - m_center,axis=1)
    return m_dist
        
def vertices_coordinates_to_vertex_index(mesh,vertex_coordinates,error_on_unmatches=True):
    """
    Purpose: To map the vertex coordinates to vertex indices
    
    """
    m_kd = KDTree(mesh.vertices)
    dist,closest_vertex = m_kd.query(vertex_coordinates)
    zero_dist = np.where(dist == 0)[0]
    if error_on_unmatches:
        mismatch_number = len(vertex_coordinates)-len(zero_dist)
        if mismatch_number > 0:
            raise Exception(f"{mismatch_number} of the vertices coordinates were not a perfect match")
    
    return closest_vertex[zero_dist]

def vertices_coordinates_to_faces(mesh,vertex_coordinates,error_on_unmatches=False,concatenate_unique_list=True):
    vertex_indices = vertices_coordinates_to_vertex_index(mesh,vertex_coordinates,error_on_unmatches)
    return vertices_to_faces(mesh,vertex_indices,concatenate_unique_list)

def vertices_to_faces(current_mesh,vertices,
                     concatenate_unique_list=False):
    """
    Purpose: If have a list of vertex indices, to get the face indices associated with them
    """
    try:
        vertices = np.array(vertices)

        intermediate_face_list = current_mesh.vertex_faces[vertices]
        faces_list = [k[k!=-1] for k in intermediate_face_list]
        if concatenate_unique_list:
            return np.unique(np.concatenate(faces_list))
        else:
            return faces_list
    except:
        print(f"vertices = {vertices}")
        su.compressed_pickle(current_mesh,"current_mesh_error_v_to_f")
        su.compressed_pickle(vertices,"vertices_error_v_to_f")
        raise Exception("Something went wrong in vertices to faces")

def vertices_coordinates_to_faces_old(current_mesh,vertex_coordinates):
    """
    
    Purpose: If have a list of vertex coordinates, to get the face indices associated with them
    
    Example: To check that it worked well with picking out border
    sk.graph_skeleton_and_mesh(other_meshes=[curr_branch.mesh,curr_branch.mesh.submesh([unique_border_faces],append=True)],
                              other_meshes_colors=["red","black"],
                              mesh_alpha=1)

    
    
    """
    try:
        border_vertices_idx = []
        for v in vertex_coordinates:
            curr_match_idx = nu.matching_rows(current_mesh.vertices,v)
            if len(curr_match_idx) > 0:
                border_vertices_idx.append(curr_match_idx)
        border_vertices_idx = np.array(border_vertices_idx)
    except:
        su.compressed_pickle(current_mesh,"current_mesh")
        su.compressed_pickle(vertex_coordinates,"vertex_coordinates")
        raise Exception("Something went from for matching_rows")
        
    border_faces = vertices_to_faces(current_mesh,vertices=border_vertices_idx)
    unique_border_faces = np.unique(np.concatenate(border_faces))
    return unique_border_faces


def mesh_vertex_graph(mesh):
    """
    Purpose: Creates a weighted connectivity graph from the vertices and edges
    
    """
    curr_weighted_edges = np.hstack([mesh.edges_unique,mesh.edges_unique_length.reshape(-1,1)])
    vertex_graph = nx.Graph()  
    vertex_graph.add_weighted_edges_from(curr_weighted_edges)
    return vertex_graph

# ------------ Algorithms used for checking the spines -------- #


def waterfilling_face_idx(mesh,
                      starting_face_idx,
                      n_iterations=10,
                         return_submesh=False,
                         connectivity="vertices"):
    """
    Will extend certain faces by infecting neighbors 
    for a certain number of iterations:
    
    Example:
    curr_border_faces = tu.find_border_faces(curr_branch.mesh)
    expanded_border_mesh = tu.waterfilling_face_idx(curr_branch.mesh,
                                                    curr_border_faces,
                                                     n_iterations=10,
                                                    return_submesh=True)
    sk.graph_skeleton_and_mesh(other_meshes=[curr_branch.mesh,expanded_border_mesh],
                              other_meshes_colors=["black","red"])
    """
    #1) set the starting faces
    final_faces = starting_face_idx
    
    #0) Turn the mesh into a graph
    if connectivity=="edges":
        total_mesh_graph = nx.from_edgelist(mesh.face_adjacency)
        #2) expand the faces
        for i in range(n_iterations):
            final_faces = np.unique(np.concatenate([xu.get_neighbors(total_mesh_graph,k) for k in final_faces]))
    else:
        for i in range(n_iterations):
            final_faces = face_neighbors_by_vertices(mesh,final_faces)

    
    if return_submesh:
        return mesh.submesh([final_faces],append=True,repair=False)
    else:
        return final_faces
    
    
def find_border_vertices(
    mesh,
    return_coordinates=False,
    plot = False):
    if len(mesh.faces) < 3:
        return []

    if mesh.is_watertight:
        return []

    # we know that in a watertight mesh every edge will be included twice
    # thus every edge which appears only once is part of a hole boundary
    boundary_groups = group_rows(
        mesh.edges_sorted, require_count=1
    )

    vertices_idx = mesh.edges_sorted[boundary_groups].ravel()
    
    if plot:
        coords = np.array(mesh.vertices[vertices_idx]).reshape(-1,3)
        ipvu.plot_objects(
            mesh,
            scatters = [coords],
        )
    if return_coordinates:
        return np.array(mesh.vertices[vertices_idx]).reshape(-1,3)
    return vertices_idx

def find_border_faces(mesh):
    border_verts = find_border_vertices(mesh)
    border_faces = np.unique(np.concatenate(vertices_to_faces(mesh,find_border_vertices(mesh).ravel())))
    return border_faces


def find_border_vertex_groups(
    mesh,
    return_coordinates=False,
    return_cycles=False,
    return_sizes=False,
    verbose = False,
    plot = False
    ):
    """
    Will return all borders as faces and grouped together
    """
    if len(mesh.faces) < 3 or mesh.is_watertight:
        if verbose:
            print(f"Returning because watertight or less than 3 edges")
        if return_sizes:
            return [[]],[0]
        else:
            return [[]]


    # we know that in a watertight mesh every edge will be included twice
    # thus every edge which appears only once is part of a hole boundary
    boundary_groups = group_rows(
        mesh.edges_sorted, require_count=1)
    
    if verbose:
        print(f"len(boundary_groups) = {len(boundary_groups)}")

    # mesh is not watertight and we have too few edges
    # edges to do a repair
    # since we haven't changed anything return False
    if len(boundary_groups) < 3:
        return []

    boundary_edges = mesh.edges[boundary_groups]
    index_as_dict = [{'index': i} for i in boundary_groups]

    # we create a graph of the boundary edges, and find cycles.
    g = nx.from_edgelist(
        np.column_stack((boundary_edges,
                         index_as_dict)))
    if return_cycles:
        border_edge_groups = xu.find_all_cycles(g,time_limit=20)
        if len(border_edge_groups)  == 0:
            print("Finding the cycles did not work when doing the border vertex edge so using connected components")
        border_edge_groups = list(nx.connected_components(g))
    else:
        border_edge_groups = list(nx.connected_components(g))

    """
    Psuedocode on converting list of edges to 
    list of faces

    """
    if plot:
        scatters = [mesh.vertices[list(k)] for k in border_edge_groups]
        ipvu.plot_objects(mesh,scatters= scatters,scatters_colors = "random")
    
    if return_coordinates:
        return_value = [mesh.vertices[list(k)] for k in border_edge_groups]
    else:
        return_value = [list(k) for k in border_edge_groups]
    
    if return_sizes:
        border_groups_len = np.array([len(k) for k in return_value])
        return return_value,border_groups_len
    else:
        return return_value
    
    

def find_border_face_groups(mesh,return_sizes=False):
    """
    Will return all borders as faces and grouped together
    """
    if len(mesh.faces) < 3:
        return []

    if mesh.is_watertight:
        return []

    # we know that in a watertight mesh every edge will be included twice
    # thus every edge which appears only once is part of a hole boundary
    boundary_groups = group_rows(
        mesh.edges_sorted, require_count=1)

    # mesh is not watertight and we have too few edges
    # edges to do a repair
    # since we haven't changed anything return False
    if len(boundary_groups) < 3:
        return []

    boundary_edges = mesh.edges[boundary_groups]
    index_as_dict = [{'index': i} for i in boundary_groups]

    # we create a graph of the boundary edges, and find cycles.
    g = nx.from_edgelist(
        np.column_stack((boundary_edges,
                         index_as_dict)))
    border_edge_groups = list(nx.connected_components(g))

    """
    Psuedocode on converting list of edges to 
    list of faces

    """
    border_face_groups = [vertices_to_faces(mesh,list(j),concatenate_unique_list=True) for j in border_edge_groups]
    if return_sizes:
        border_groups_len = np.array([len(k) for k in border_face_groups])
        return border_face_groups,border_groups_len
    else:
        return border_face_groups
    
def border_euclidean_length(border):
    """
    The border does have to be specified as ordered coordinates
    
    """
    ex_border_shift = np.roll(border,1,axis=0)
    return np.sum(np.linalg.norm(border - ex_border_shift,axis=1))

def largest_hole_length(mesh,euclidean_length=True):
    """
    Will find either the vertex count or the euclidean distance
    of the largest hole in a mesh
    
    """
    try:
        border_vert_groups,border_vert_sizes = find_border_vertex_groups(mesh,
                                    return_coordinates=True,
                                     return_cycles=True,
                                    return_sizes=True,
                                                                       )
    except:
        return None
    
    #accounting for if found no holes
    if border_vert_sizes[0] == 0:
        return 0

    if euclidean_length:
        border_lengths = [border_euclidean_length(k) for k in border_vert_groups]
        largest_border_idx = np.argmax(border_lengths)
        largest_border_size = border_lengths[largest_border_idx]
        return largest_border_size
    else:
        return np.max(border_vert_sizes)

def expand_border_faces(mesh,n_iterations=10,return_submesh=True):
    curr_border_faces_groups = find_border_face_groups(mesh)
    expanded_border_face_groups = []
    for curr_border_faces in curr_border_faces_groups:
        expanded_border_mesh = waterfilling_face_idx(mesh,
                                                    curr_border_faces,
                                                     n_iterations=n_iterations,
                                                    return_submesh=return_submesh)
        expanded_border_face_groups.append(expanded_border_mesh)
    return expanded_border_face_groups
    
def mesh_with_ends_cutoff(mesh,n_iterations=5,
                         return_largest_mesh=True,
                         significance_threshold=100,
                         verbose=False):
    """
    Purpose: Will return a mesh with the ends with a border
    that are cut off by finding the border, expanding the border
    and then removing these faces and returning the largest piece
    
    Pseudocode:
    1) Expand he border meshes
    2) Get a submesh without the border faces
    3) Split the mesh into significants pieces
    3b) Error if did not find any significant meshes
    4) If return largest mesh is True, only return the top one
    
    """
    #1) Expand he border meshes
    curr_border_faces = expand_border_faces(mesh,n_iterations=n_iterations,return_submesh=False)
    
    #2) Get a submesh without the border faces
    if verbose:
        print(f"Removing {len(curr_border_faces)} border meshes of sizes: {[len(k) for k in curr_border_faces]} ")
    faces_to_keep = np.delete(np.arange(len(mesh.faces)),np.concatenate(curr_border_faces))
    leftover_submesh = mesh.submesh([faces_to_keep],append=True,repair=False)

    if verbose:
        printf("Leftover submesh size: {leftover_submesh}")
        
    #3) Split the mesh into significants pieces
    sig_leftover_pieces = split_significant_pieces(leftover_submesh,significance_threshold=significance_threshold)
    
    #3b) Error if did not find any significant meshes
    if len(sig_leftover_pieces) <= 0:
        raise Exception("No significant leftover pieces were detected after border subtraction")
        
    #4) If return largest mesh is True, only return the top one
    if return_largest_mesh:
        return sig_leftover_pieces[0]
    else:
        return sig_leftover_pieces
    
'''
# Old method that only computed percentage of total number of border vertices
def filter_away_border_touching_submeshes(
                            mesh,
                            submesh_list,
                            border_percentage_threshold=0.5,#would make 0.00001 if wanted to enforce nullification if at most one touchedss
                            verbose = False,
                            return_meshes=True,
                            ):
    """
    Purpose: Will return submeshes or indices that 
    do not touch a border edge of the parenet mesh

    Pseudocode:
    1) Get the border vertices of mesh
    2) For each submesh
    - do KDTree between submesh vertices and border vertices
    - if one of distances is equal to 0 then nullify

    Ex: 
    
    return_value = filter_away_border_touching_submeshes(
                                mesh = eraser_branch.mesh,
                                submesh_list = eraser_branch.spines,
                                verbose = True,
                                return_meshes=True)
    sk.graph_skeleton_and_mesh(main_mesh_verts=mesh.vertices,
                           main_mesh_faces=mesh.faces,
                            other_meshes=eraser_branch.spines,
                                                  other_meshes_colors="red")
    sk.graph_skeleton_and_mesh(main_mesh_verts=mesh.vertices,
                           main_mesh_faces=mesh.faces,
                            other_meshes=return_value,
                                other_meshes_colors="red")
    """

    #1) Get the border vertices of mesh
    border_verts_idx = find_border_vertices(mesh)
    if len(border_verts_idx) == 0:
        if verbose:
            print("There were no border edges for the main mesh")
        passed_idx = np.arange(len(submesh_list))
    else:
        """
        Want to just find a matching border group and then look 
        at percentage
        """

        passed_idx = []
        for i,subm in enumerate(submesh_list):
            spine_kdtree = KDTree(subm.vertices)
            dist,closest_vert_idx = spine_kdtree.query(mesh.vertices[border_verts_idx])
            
            if len(dist[dist == 0])/len(border_verts_idx) < border_percentage_threshold:
                passed_idx.append(i)


        passed_idx = np.array(passed_idx)

    if return_meshes:
        return [k for i,k in enumerate(submesh_list) if i in passed_idx]
    else:
        return passed_idx
'''

def filter_away_border_touching_submeshes_by_group(
                            mesh,
                            submesh_list,
                            border_percentage_threshold=0.3,#would make 0.00001 if wanted to enforce nullification if at most one touchedss
                            inverse_border_percentage_threshold=0.9,
                            verbose = False,
                            return_meshes=True,
                    
                            ):
    """
    Purpose: Will return submeshes or indices that 
    do not touch a border edge of the parenet mesh

    Pseudocode:
    1) Get the border vertices of mesh grouped
    2) For each submesh
       a. Find which border group the vertices overlap with (0 distances)
       b. For each group that it is touching 
          i) Find the number of overlap
          ii) if the percentage is greater than threshold then nullify
    - 

    Ex: 
    
    return_value = filter_away_border_touching_submeshes(
                                mesh = eraser_branch.mesh,
                                submesh_list = eraser_branch.spines,
                                verbose = True,
                                return_meshes=True)
    sk.graph_skeleton_and_mesh(main_mesh_verts=mesh.vertices,
                           main_mesh_faces=mesh.faces,
                            other_meshes=eraser_branch.spines,
                                                  other_meshes_colors="red")
    sk.graph_skeleton_and_mesh(main_mesh_verts=mesh.vertices,
                           main_mesh_faces=mesh.faces,
                            other_meshes=return_value,
                                other_meshes_colors="red")
                                
    Ex 2:
    tu = reload(tu)
    tu.filter_away_border_touching_submeshes_by_group(
        mesh=curr_branch.mesh,
        submesh_list=curr_branch.spines
    )
    """
    
    if verbose:
        print(f"border_percentage_threshold = {border_percentage_threshold}")

    #1) Get the border vertices of mesh
    border_vertex_groups = find_border_vertex_groups(mesh)
    if len(border_vertex_groups) == 0:
        if verbose:
            print("There were no border edges for the main mesh")
        passed_idx = np.arange(len(submesh_list))
    else:
        """
        Want to just find a matching border group and then look 
        at percentage
        """

        passed_idx = []
        for i,subm in enumerate(submesh_list):
            #creates KDTree for the submesh
            spine_kdtree = KDTree(subm.vertices)
            
            not_touching_significant_border=True
            
            for z,b_verts in enumerate(border_vertex_groups):
                dist,closest_vert_idx = spine_kdtree.query(mesh.vertices[list(b_verts)])
                touching_perc = len(dist[dist == 0])/len(b_verts)
                if verbose:
                    print(f"Submesh {i} touching percentage for border {z} = {touching_perc}")
                if touching_perc > border_percentage_threshold:
                    if verbose:
                        print(f"Submesh {z} was touching a greater percentage ({touching_perc}) of border vertices than threshold ({border_percentage_threshold})")
                    not_touching_significant_border=False
                    break
            
            #apply the spine check that will see if percentage of border vertices of spine touching mesh border vertices
            #is above some threshold
            if inverse_border_percentage_threshold > 0:
                if verbose:
                    print(f"Applying inverse_border_percentage_threshold = {inverse_border_percentage_threshold}")
                    print(f"border_vertex_groups = {border_vertex_groups}")
                all_border_verts = np.concatenate([list(k) for k in border_vertex_groups])
                whole_border_kdtree= KDTree(mesh.vertices[all_border_verts])
                dist,closest_vert_idx = whole_border_kdtree.query(subm.vertices)
                touching_perc = len(dist[dist == 0])/len(dist)
                if touching_perc > inverse_border_percentage_threshold:
                    not_touching_significant_border = False
            
            if not_touching_significant_border:
                passed_idx.append(i)


        passed_idx = np.array(passed_idx)
        if verbose:
            print(f"At end passed_idx = {passed_idx} ")

    if return_meshes:
        return [k for i,k in enumerate(submesh_list) if i in passed_idx]
    else:
        return passed_idx

    
def max_distance_betwee_mesh_vertices(mesh_1,mesh_2,
                                      verbose=False,
                                     max_distance_threshold=None):
    """
    Purpose: Will calculate the maximum distance between vertices of two meshes
    
    Application: Can be used to see how well a poisson reconstruction
    estimate of a soma and the actual soma that was backtracked to 
    the mesh are in order to identify true somas and not
    get fooled by the glia / neural error checks
    
    Pseudocode:
    1) Make a KDTree from the new backtracked soma
    2) Do a query of the poisson soma vertices
    3) If a certain distance is too far then fail
    
    """
    
    #print(f"mesh_1={mesh_1},mesh_2 = {mesh_2}")
    #1) Make a KDTree from the new backtracked soma
    backtrack_mesh_kdtree = KDTree(mesh_1.vertices)
    #2) Do a query of the poisson soma vertices
    check_mesh_distances,closest_nodes = backtrack_mesh_kdtree.query(mesh_2.vertices)
    #print(f"check_mesh_distances = {check_mesh_distances}")
    max_dist = np.max(check_mesh_distances)
    
    if verbose:
        print(f"maximum distance from mesh_2 vertices to mesh_1 vertices is = {max_dist}")
    
    if max_distance_threshold is None:
        return max_dist
    else:
        if max_dist > max_distance_threshold:
            return False
        else:
            return True

try:
    from mesh_tools import meshlab
except:
    pass

def fill_holes(mesh,
              max_hole_size=2000,
              self_itersect_faces=False,
              ):
    
    mesh.merge_vertices()
    
    
    #if len(tu.find_border_face_groups(mesh))==0 and tu.is_manifold(mesh) and tu.is_watertight(mesh):
    if len(tu.find_border_face_groups(mesh))==0 and tu.is_watertight(mesh):
        print("No holes needed to fill and mesh was watertight and no vertex group")
        return mesh
        
    lrg_mesh = mesh
    with meshlab.FillHoles(max_hole_size=max_hole_size,self_itersect_faces=self_itersect_faces) as fill_hole_obj:

        mesh_filled_holes,fillholes_file_obj = fill_hole_obj(   
                                            vertices=lrg_mesh.vertices,
                                             faces=lrg_mesh.faces,
                                             return_mesh=True,
                                             delete_temp_files=True,
                                            )
    return mesh_filled_holes

def filter_meshes_by_containing_coordinates(mesh_list,nullifying_points,
                                                filter_away=True,
                                           method="distance",
                                           distance_threshold=2000,
                                           verbose=False,
                                           return_indices=False):
    """
    Purpose: Will either filter away or keep meshes from a list of meshes
    based on points based to the function
    
    Application: Can filter away spines that are too close to the endpoints of skeletons
    
    Ex: 
    import trimesh
    from datasci_tools import numpy_dep as np
    tu = reload(tu)

    curr_limb = recovered_neuron[2]
    curr_limb_end_coords = find_skeleton_endpoint_coordinates(curr_limb.skeleton)


    kept_spines = []

    for curr_branch in curr_limb:
        #a) get the spines
        curr_spines = curr_branch.spines

        #For each spine:
        if not curr_spines is None:
            curr_kept_spines = tu.filter_meshes_by_bbox_containing_coordinates(curr_spines,
                                                                            curr_limb_end_coords)
            print(f"curr_kept_spines = {curr_kept_spines}")
            kept_spines += curr_kept_spines

    ipvu.plot_objects(meshes=kept_spines)
    """
    if not nu.is_array_like(mesh_list):
        mesh_list = [mesh_list]
        
    nullifying_points = np.array(nullifying_points).reshape(-1,3)
    
    containing_meshes = []
    containing_meshes_idx = []
    
    non_containing_meshes = []
    non_containing_meshes_idx = []
    for j,sp_m in enumerate(mesh_list):
        # tried filling hole and using contains
        #sp_m_filled = tu.fill_holes(sp_m)
        #contains_results = sp_m.bounds.contains(currc_limb_end_coords)

        #tried using the bounds method
        #contains_results = trimesh.bounds.contains(sp_m.bounds,currc_limb_end_coords.reshape(-1,3))

        #final version
        if method=="bounding_box":
            contains_results = sp_m.bounding_box_oriented.contains(nullifying_points.reshape(-1,3))
        elif method == "distance":
            sp_m_kdtree = KDTree(sp_m.vertices)
            distances,closest_nodes = sp_m_kdtree.query(nullifying_points.reshape(-1,3))
            contains_results = distances <= distance_threshold
            if verbose:
                print(f"Submesh {j} ({sp_m}) distances = {distances}")
                print(f"Min distance {np.min(distances)}")
                print(f"contains_results = {contains_results}\n")
        else:
            raise Exception(f"Unimplemented method ({method}) requested")
            
        if verbose:
            print(f"np.sum(contains_results) = {np.sum(contains_results)}")
        if np.sum(contains_results) > 0:
            containing_meshes.append(sp_m)
            containing_meshes_idx.append(j)
            
        else:
            non_containing_meshes.append(sp_m)
            non_containing_meshes_idx.append(j)
    
    if verbose:
        print(f"containing_meshes = {containing_meshes}")
        print(f"non_containing_meshes = {non_containing_meshes}")
    
    if filter_away:
        if return_indices:
            return non_containing_meshes_idx
        else:
            return non_containing_meshes
    else:
        if return_indices:
            return containing_meshes_idx
        else:
            return containing_meshes
    

# --------------- 11/11 ---------------------- #
try:
    from mesh_tools import meshlab 
except:
    pass

def poisson_surface_reconstruction(mesh,
                                   output_folder="./temp",
                                  delete_temp_files=True,
                                   name=None,
                                  verbose=False,
                                  return_largest_mesh=False,
                                   return_significant_meshes=False,
                                   significant_mesh_threshold=1000,
                                  manifold_clean =True):
    if type(output_folder) != type(Path()):
        output_folder = Path(str(output_folder))
        output_folder.mkdir(parents=True,exist_ok=True)

    # CGAL Step 1: Do Poisson Surface Reconstruction
    Poisson_obj = meshlab.Poisson(output_folder,overwrite=True)

    if name is None:
        name = f"mesh_{np.random.randint(10,1000)}"

    
    skeleton_start = time.time()
    
    if verbose:
        print("     Starting Screened Poisson")
    new_mesh,output_subprocess_obj = Poisson_obj(   
                                vertices=mesh.vertices,
                                 faces=mesh.faces,
                                mesh_filename=name + ".off",
                                 return_mesh=True,
                                 delete_temp_files=delete_temp_files,
                                )
    if verbose:
        print(f"-----Time for Screened Poisson= {time.time()-skeleton_start}")
    
    if return_largest_mesh:
        new_mesh = tu.split_significant_pieces(new_mesh,
                                               significance_threshold=1,
                                               connectivity='edges')[0]
        #only want to check manifoldness if it is one piece
        if manifold_clean:
            new_mesh.merge_vertices()
            new_mesh = tu.fill_holes(new_mesh)
            #new_mesh = tu.connected_nondegenerate_mesh(mesh)
            print(f"Mesh manifold status: {tu.is_manifold(new_mesh)}")
            print(f"Mesh watertight status: {tu.is_watertight(new_mesh)}")
    
    if return_significant_meshes:
        return tu.split_significant_pieces(new_mesh,
                                               significance_threshold=significant_mesh_threshold,
                                               connectivity='edges')
    
    
    return new_mesh


def decimate(mesh,
               decimation_ratio=0.25,
               output_folder="./temp",
              delete_temp_files=True,
               name=None,
              verbose=False):
    if type(output_folder) != type(Path()):
        output_folder = Path(str(output_folder))
        output_folder.mkdir(parents=True,exist_ok=True)

    # CGAL Step 1: Do Poisson Surface Reconstruction
    Decimator_obj = meshlab.Decimator(decimation_ratio,output_folder,overwrite=True)

    if name is None:
        name = f"mesh_{np.random.randint(10,1000)}"

    skeleton_start = time.time()
    
    if verbose:
        print("     Starting Screened Poisson")
    #Step 1: Decimate the Mesh and then split into the seperate pieces
    new_mesh,output_obj = Decimator_obj(vertices=mesh.vertices,
             faces=mesh.faces,
             segment_id=None,
             return_mesh=True,
             delete_temp_files=False)
    
    if verbose:
        print(f"-----Time for Screened Poisson= {time.time()-skeleton_start}")
        
    return new_mesh
        


def pymeshfix_clean(mesh,
                    joincomp = True,
                   remove_smallest_components = False,
                   verbose=False):
    """
    Purpose: Will apply the pymeshfix algorithm
    to clean the mesh
    
    Application: Can help with soma identificaiton
    because otherwise nucleus could interfere with the segmentation
    
    
    """
    if verbose:
        print(f"Staring pymeshfix on {mesh}")
    start_time = time.time()

    meshfix = pymeshfix.MeshFix(mesh.vertices,mesh.faces)
    
    meshfix.repair(
                   verbose=False,
                   joincomp=joincomp,
                   remove_smallest_components=remove_smallest_components
                  )
    current_neuron_poisson_pymeshfix = trimesh.Trimesh(vertices=meshfix.v,faces=meshfix.f)

    if verbose:
        print(f"Total time for pymeshfix = {time.time() - start_time}")
    return current_neuron_poisson_pymeshfix




def mesh_segmentation_largest_conn_comp(
        mesh = None,
        filepath = None,
        clusters=2,
        smoothness=0.2,
        cgal_folder = Path("./"),
        return_sdf = True,


        delete_temp_files = True,
        return_meshes = True ,
        check_connect_comp = True, #will only be used if returning meshes
        return_ordered_by_size = True,

        verbose = False,
        plot_segmentation = False,
        return_mesh_idx = False,

    ):
    """
    Function tha segments the mesh and then 
    either returns:
    1) Face indexes of different mesh segments
    2) The cut up mesh into different mesh segments
    3) Can optionally return the sdf values of the different mesh

    Example: 
    tu = reload(tu)

    meshes_split,meshes_split_sdf = tu.mesh_segmentation(
        mesh = real_soma
    )
    
    """
    

    # ------- 1/14 Additon: Going to make sure mesh has no degenerate faces --- #
    mesh_to_segment,faces_kept = tu.connected_nondegenerate_mesh(mesh,
                                                                 return_kept_faces_idx=True,
                                                                 return_removed_faces_idx=False)
    
    #-------- 1/14 Addition: Will check for just a sheet of mesh with a 0 sdf ---------#
    sdf_faces = tu.ray_trace_distance(mesh_to_segment)
    perc_0_faces = np.sum(sdf_faces==0)/len(sdf_faces)
    if verbose:
        print(f"perc_0_faces = {perc_0_faces}")
    if perc_0_faces == 1:
        cgal_data = np.zeros(len(mesh.faces))
        cgal_sdf_data = np.zeros(len(mesh.faces))
        
        delete_temp_files=False
    else:
        if not cgal_folder.exists():
            cgal_folder.mkdir(parents=True,exist_ok=False)

        mesh_temp_file = False
        if filepath is None:
            if mesh is None:
                raise Exception("Both mesh and filepath are None")
            file_dest = cgal_folder / Path(f"{np.random.randint(10,1000)}_mesh.off")
            filepath = write_neuron_off(mesh_to_segment,file_dest)
            mesh_temp_file = True

        filepath = Path(filepath)

        assert(filepath.exists())
        filepath_no_ext = filepath.absolute().parents[0] / filepath.stem


        start_time = time.time()

        if verbose:
            print(f"Going to run cgal segmentation with:"
                 f"\nFile: {str(filepath_no_ext)} \nclusters:{clusters} \nsmoothness:{smoothness}")

        csm.cgal_segmentation(str(filepath_no_ext),clusters,smoothness)

        #read in the csv file
        cgal_output_file = Path(str(filepath_no_ext) + "-cgal_" + str(np.round(clusters,2)) + "_" + "{:.2f}".format(smoothness) + ".csv" )
        cgal_output_file_sdf = Path(str(filepath_no_ext) + "-cgal_" + str(np.round(clusters,2)) + "_" + "{:.2f}".format(smoothness) + "_sdf.csv" )

        cgal_data_pre_filt = np.genfromtxt(str(cgal_output_file.absolute()), delimiter='\n')
        cgal_sdf_data_pre_filt = np.genfromtxt(str(cgal_output_file_sdf.absolute()), delimiter='\n')

        """ 1/14: Need to adjust for the degenerate faces removed
        """
        cgal_data = np.ones(len(mesh.faces))*(np.max(cgal_data_pre_filt)+1)
        cgal_data[faces_kept] = cgal_data_pre_filt

        cgal_sdf_data = np.zeros(len(mesh.faces))
        cgal_sdf_data[faces_kept] = cgal_sdf_data_pre_filt
    
    if mesh is None:
        mesh = load_mesh_no_processing(filepath)
    split_meshes,split_meshes_idx = split_mesh_into_face_groups(mesh,cgal_data,return_idx=True,
                                   check_connect_comp = check_connect_comp,
                                                                  return_dict=False)
        
    if plot_segmentation:
        print(f"Initial segmentation with clusters = {clusters}, smoothness = {smoothness}")
        ipvu.plot_objects(meshes=split_meshes,
                         meshes_colors=mu.generate_non_randon_named_color_list(len(split_meshes)))
    
    if return_meshes:
        
        ''' OLD METHOD NOT DOING SORTING CORRECTLY
        if return_ordered_by_size:
            split_meshes,split_meshes_sort_idx = sort_meshes_largest_to_smallest(split_meshes,return_idx=True)
            split_meshes_idx = split_meshes_idx[split_meshes_sort_idx]

        if return_sdf:
            #will return sdf data for all of the meshes
            sdf_medains_for_mesh = np.array([np.median(cgal_sdf_data[k]) for k in split_meshes_idx])

            if return_ordered_by_size:
                sdf_medains_for_mesh = sdf_medains_for_mesh[split_meshes_sort_idx]
            
            if return_mesh_idx:
                return_value= split_meshes,sdf_medains_for_mesh,split_meshes_idx
            else:
                return_value= split_meshes,sdf_medains_for_mesh,
        else:
            if return_mesh_idx:
                return_value= split_meshes,split_meshes_idx
            else:
                return_value= split_meshes
        '''
        if return_ordered_by_size:
            split_meshes,split_meshes_sort_idx = sort_meshes_largest_to_smallest(split_meshes,return_idx=True)
            split_meshes_idx = split_meshes_idx[split_meshes_sort_idx]

        if return_sdf:
            #will return sdf data for all of the meshes
            sdf_medains_for_mesh = np.array([np.median(cgal_sdf_data[k]) for k in split_meshes_idx])
            
            if return_mesh_idx:
                return_value= split_meshes,sdf_medains_for_mesh,split_meshes_idx
            else:
                return_value= split_meshes,sdf_medains_for_mesh,
        else:
            if return_mesh_idx:
                return_value= split_meshes,split_meshes_idx
            else:
                return_value= split_meshes
            
            
        
    else:
        if return_sdf:
            return_value= cgal_data,cgal_sdf_data
        else:
            return_value= cgal_data

    if delete_temp_files:
        cgal_output_file.unlink()
        cgal_output_file_sdf.unlink()
        if mesh_temp_file:
            filepath.unlink()

    return return_value

def mesh_segmentation(
    mesh,
    filepath = None,
    clusters=2,
    smoothness=0.2,
    cgal_folder = Path("./"),
    return_sdf = True,


    delete_temp_files = True,
    return_meshes = True,
    check_connect_comp = True, #will only be used if returning meshes
    return_ordered_by_size = True,

    verbose = False,

    # -- for the connected components
    connectivity = "vertices",
    min_n_faces_conn_comp = 40,
    #plot_conn_comp_segmentation = False,
    plot_segmentation = False,
    plot_buffer = 0,
    return_mesh_idx = False,
    ):
    """
    Function tha segments the mesh and then 
    either returns:
    1) Face indexes of different mesh segments
    2) The cut up mesh into different mesh segments
    3) Can optionally return the sdf values of the different mesh

    Example: 
    tu = reload(tu)

    meshes_split,meshes_split_sdf = tu.mesh_segmentation(
        mesh = real_soma
    )

    """
    if mesh is None:
        mesh = load_mesh_no_processing(filepath)

    mesh_filtered,nondeg_faces = tu.remove_degenerate_faces(
        mesh,
        return_face_idxs=True
    )


    cgal_data = np.zeros(len(mesh.faces))
    cgal_sdf_data = np.zeros(len(mesh.faces))
    counter = 0
    if len(nondeg_faces) == 0:
        pass
    else:
        conn_mesh,conn_faces = tu.split_significant_pieces(mesh_filtered,
                                significance_threshold=min_n_faces_conn_comp,
                               return_face_indices=True,
                               connectivity=connectivity)
        for j,(mesh_to_segment,faces_kept) in enumerate(zip(conn_mesh,conn_faces)):
            curr_delete_temp_files = delete_temp_files
            #-------- 1/14 Addition: Will check for just a sheet of mesh with a 0 sdf ---------#
    #         if len(mesh_to_segment.faces) < min_n_faces_conn_comp:
    #             if verbose:
    #                 print(f"Skipping mesh {j} because n_faces ({len(mesh_to_segment.faces)}) less than min threshold ({min_n_faces_conn_comp})")
    #             continue
            sdf_faces = tu.ray_trace_distance(mesh_to_segment)
            perc_0_faces = np.sum(sdf_faces==0)/len(sdf_faces)
            if verbose:
                print(f"perc_0_faces = {perc_0_faces}")
            if perc_0_faces == 1:
                cgal_data_pre_filt = np.zeros(len(mesh_to_segment.faces))
                cgal_sdf_data_pre_filt = np.zeros(len(mesh_to_segment.faces))

                curr_delete_temp_files=False
            else:
                if not cgal_folder.exists():
                    cgal_folder.mkdir(parents=True,exist_ok=False)

                mesh_temp_file = False
                if filepath is None:
                    if mesh is None:
                        raise Exception("Both mesh and filepath are None")
                    file_dest = cgal_folder / Path(f"{np.random.randint(10,1000)}_mesh.off")
                    mesh_filepath = tu.write_neuron_off(mesh_to_segment,file_dest)
                    mesh_temp_file = True

                mesh_filepath = Path(mesh_filepath)

                assert(mesh_filepath.exists())
                filepath_no_ext = mesh_filepath.absolute().parents[0] / mesh_filepath.stem


                start_time = time.time()

                if verbose:
                    print(f"Going to run cgal segmentation with:"
                         f"\nFile: {str(filepath_no_ext)} \nclusters:{clusters} \nsmoothness:{smoothness}")

                csm.cgal_segmentation(str(filepath_no_ext),clusters,smoothness)

                #read in the csv file
                cgal_output_file = Path(str(filepath_no_ext) + "-cgal_" + str(np.round(clusters,2)) + "_" + "{:.2f}".format(smoothness) + ".csv" )
                cgal_output_file_sdf = Path(str(filepath_no_ext) + "-cgal_" + str(np.round(clusters,2)) + "_" + "{:.2f}".format(smoothness) + "_sdf.csv" )

                cgal_data_pre_filt = np.genfromtxt(str(cgal_output_file.absolute()), delimiter='\n')
                cgal_sdf_data_pre_filt = np.genfromtxt(str(cgal_output_file_sdf.absolute()), delimiter='\n')

                """ 1/14: Need to adjust for the degenerate faces removed
                """
            cgal_data[nondeg_faces[faces_kept]] = cgal_data_pre_filt + np.max(cgal_data) + 1
            cgal_sdf_data[nondeg_faces[faces_kept]] = cgal_sdf_data_pre_filt


            if curr_delete_temp_files:
                cgal_output_file.unlink()
                cgal_output_file_sdf.unlink()
                if mesh_temp_file:
                    mesh_filepath.unlink()



    split_meshes,split_meshes_idx= tu.split_mesh_into_face_groups(mesh,cgal_data,return_idx=True,
                                   check_connect_comp = check_connect_comp,
                                                                  return_dict=False)

    if plot_segmentation:
        print(f"Initial segmentation with clusters = {clusters}, smoothness = {smoothness}")
        ipvu.plot_objects(meshes=split_meshes,
                         meshes_colors=mu.generate_non_randon_named_color_list(len(split_meshes)),
                         buffer = plot_buffer)

    if return_meshes:
        if return_ordered_by_size:
            split_meshes,split_meshes_sort_idx = tu.sort_meshes_largest_to_smallest(split_meshes,return_idx=True)
            split_meshes_idx = split_meshes_idx[split_meshes_sort_idx]

        if return_sdf:
            #will return sdf data for all of the meshes
            sdf_medains_for_mesh = np.array([np.median(cgal_sdf_data[k]) for k in split_meshes_idx])

            if return_mesh_idx:
                return_value= split_meshes,sdf_medains_for_mesh,split_meshes_idx
            else:
                return_value= split_meshes,sdf_medains_for_mesh,
        else:
            if return_mesh_idx:
                return_value= split_meshes,split_meshes_idx
            else:
                return_value= split_meshes


    else:
        if return_sdf:
            return_value= cgal_data,cgal_sdf_data
        else:
            return_value= cgal_data



    return return_value

"""Purpose: crude check to see if mesh is manifold:

https://gamedev.stackexchange.com/questions/61878/how-check-if-an-arbitrary-given-mesh-is-a-single-closed-mesh
"""

def convert_trimesh_to_o3d(mesh):
    if not type(mesh) == type(o3d.geometry.TriangleMesh()):
        new_o3d_mesh = o3d.geometry.TriangleMesh()
        new_o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        new_o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    else:
        new_o3d_mesh = mesh
    return new_o3d_mesh

def convert_o3d_to_trimesh(mesh):
    if not type(mesh) == type(trimesh.Trimesh()):
        new_mesh = trimesh.Trimesh(
                                    vertices=np.asarray(mesh.vertices),
                                   faces=np.asarray(mesh.triangles),
                                vertex_normals=np.asarray(mesh.vertex_normals)
                                  )
    else:
        new_mesh = mesh
    return new_mesh
    
def mesh_volume_o3d(mesh):
    mesh_o3d = convert_trimesh_to_o3d(mesh)
    return mesh_o3d.get_volume()

    
def is_manifold(mesh):
    #su.compressed_pickle(mesh,"manifold_debug_mesh")
    mesh_o3d = convert_trimesh_to_o3d(mesh)  
    return mesh_o3d.is_vertex_manifold()

def is_watertight(mesh,allow_if_no_border_verts=False):
    wat_tight = mesh.is_watertight
    if not wat_tight and allow_if_no_border_verts:
        if len(tu.find_border_vertices(mesh)) == 0 and mesh.volume > 0:
            return True
        else:
            return False
    else:
        return wat_tight

def get_non_manifold_edges(mesh):
    mesh_o3d = convert_trimesh_to_o3d(mesh)  
    return np.asarray(mesh_o3d.get_non_manifold_edges())

def get_non_manifold_vertices(mesh):
    mesh_o3d = convert_trimesh_to_o3d(mesh)  
    return np.asarray(mesh_o3d.get_non_manifold_vertices())


"""
From trimesh:

self.remove_infinite_values()
self.merge_vertices(**kwargs)
# if we're cleaning remove duplicate
# and degenerate faces

if validate:
    self.remove_duplicate_faces()
    self.remove_degenerate_faces()
    self.fix_normals()


"""
def connected_nondegenerate_mesh(
    mesh,
    return_kept_faces_idx=False,
    return_removed_faces_idx = False,
    connectivity="edges",
    plot = False,
    verbose = False,
    ):
    """
    Purpose: To convert a mesh to a connected non-degenerate mesh
    
    Pseuducode:
    1) Find all the non-degnerate faces indices
    2) Split the mesh with the connectivity and returns the largest mesh
    3) Return a submesh of all non degenerate faces and the split meshes
    
    """
    mesh_filtered,nondeg_faces = tu.remove_degenerate_faces(mesh,
                                                           return_face_idxs=True)
    
    if len(nondeg_faces) == 0:
        raise Exception("No mesh after removing degenerate faces")
    conn_mesh,conn_faces = tu.split_significant_pieces(mesh_filtered,
                                significance_threshold=1,
                               return_face_indices=True,
                               connectivity=connectivity)
    
    kept_faces = nondeg_faces[conn_faces[0]]
    not_kept_faces = np.delete(np.arange(len(mesh.faces)),kept_faces)
    
    if verbose or plot:
        print(f"# of faces kept = {len(kept_faces)}/{len(kept_faces) + len(not_kept_faces)}")
    if plot:
        print(f" --> red = kept mesh")
        ipvu.plot_objects(
            mesh,
            meshes = [conn_mesh[0]],
            meshes_colors = "red"
        )
                                                
    if return_kept_faces_idx and return_removed_faces_idx:
        return conn_mesh[0],kept_faces,not_kept_faces
    elif return_kept_faces_idx:
        return conn_mesh[0],kept_faces
    elif return_removed_faces_idx:
        return conn_mesh[0],not_kept_faces
    else:
        return conn_mesh[0]
    

def find_degenerate_faces(mesh,return_nondegenerate_faces=False):
    nondegenerate = trimesh.triangles.nondegenerate(
                mesh.triangles,
                areas=mesh.area_faces,

                height=trimesh.tol.merge)

    if not return_nondegenerate_faces:
        return np.where(nondegenerate==False)[0]
    else:
        return np.where(nondegenerate==True)[0]
    
def find_nondegenerate_faces(mesh):
    return find_degenerate_faces(mesh,return_nondegenerate_faces=True)

def remove_degenerate_faces(mesh,return_face_idxs=False):
    nondeg_faces = find_nondegenerate_faces(mesh)
    new_mesh = mesh.submesh([nondeg_faces],append=True,repair=False)
    if return_face_idxs:
        return new_mesh,nondeg_faces
    else:
        return new_mesh

def mesh_interior(mesh,
                    return_interior=True,
                    quality_max=0.1,
                  try_hole_close=True,
                      max_hole_size = 10000,
                     self_itersect_faces=False,
                  verbose=True,
                    
                    **kwargs
              ):
    
    if try_hole_close:
        try:
            mesh = fill_holes(mesh,
                  max_hole_size=max_hole_size,
                  self_itersect_faces=self_itersect_faces)
        except:
            if verbose: 
                print("The hole closing did not work so continuing without")
            pass
                
    with meshlab.Interior(return_interior=return_interior,
                                quality_max=quality_max,
                                 **kwargs) as remove_obj:

        mesh_remove_interior,remove_file_obj = remove_obj(   
                                            vertices=mesh.vertices,
                                             faces=mesh.faces,
                                             return_mesh=True,
                                             delete_temp_files=True,
                                            )
    return mesh_remove_interior

def remove_mesh_interior(mesh,
                         inside_pieces=None,
                         size_threshold_to_remove=700,
                         quality_max=0.1,
                         verbose=True,
                         return_removed_pieces=False,
                         connectivity="vertices",
                         try_hole_close=True,
                         return_face_indices=False,
                         **kwargs):
    """
    Will remove interior faces of a mesh with a certain significant size
    
    """
    if inside_pieces is None:
        curr_interior_mesh = mesh_interior(mesh,return_interior=True,
                                           quality_max=quality_max,
                                       try_hole_close=try_hole_close,
                                       **kwargs)
    else:
        print("inside remove_mesh_interior and using precomputed inside_pieces")
        if nu.is_array_like(inside_pieces):
            curr_interior_mesh = tu.combine_meshes(inside_pieces)
        else:
            curr_interior_mesh = inside_pieces
        
    
    sig_inside = tu.split_significant_pieces(curr_interior_mesh,significance_threshold=size_threshold_to_remove,
                                            connectivity=connectivity)
    if len(sig_inside) == 0:
        sig_meshes_no_threshold = split_significant_pieces(curr_interior_mesh,significance_threshold=1)
        meshes_sizes = np.array([len(k.faces) for k in sig_meshes_no_threshold])
        if verbose:
            print(f"No significant ({size_threshold_to_remove}) interior meshes present")
            if len(meshes_sizes)>0:
                print(f"largest is {(np.max(meshes_sizes))}")
        if return_face_indices:
            return_mesh = np.arange(len(mesh.faces))
        else:
            return_mesh= mesh
    else:
        if verbose:
            print(f"Removing the following inside neurons: {sig_inside}")
        
        if return_face_indices:
            return_mesh= subtract_mesh(mesh,sig_inside,
                    return_mesh=False,
                    exact_match=False
                   )
        else:
            return_mesh= subtract_mesh(mesh,sig_inside,
                                      exact_match=False)
        
        
    if return_removed_pieces:
        # --- 11/15: Need to only return inside pieces that are mapped to the original face ---
        sig_inside_remapped = [tu.original_mesh_faces_map(mesh,jj,
                                                          return_mesh=True) for jj in sig_inside]
        sig_inside_remapped = [k for k in sig_inside_remapped if (type(k) == type(trimesh.Trimesh())) and len(k.faces) >= 1] 
        return return_mesh,sig_inside_remapped
    else:
        return return_mesh
    
    
def filter_vertices_by_mesh(mesh,vertices):
    """
    Purpose: To restrict the vertices to those
    that only lie on a mesh
    
    """
    
    #1) Build a KDTree of the mesh
    curr_mesh_tree = KDTree(mesh.vertices)
    
    #2) Query the vertices against the mesh
    dist,closest_nodes = curr_mesh_tree.query(vertices)
    match_verts = vertices[dist==0]
    
    return match_verts
    


def fill_holes_trimesh(mesh):
    mesh_copy = copy.deepcopy(mesh)
    trimesh.repair.fill_holes(mesh_copy)
    return mesh_copy


def mesh_volume_convex_hull(mesh):
    return mesh.convex_hull.volume

def mesh_volume(mesh,
    #watertight_method="trimesh",
    watertight_method="fill_mesh_holes_with_fan",#"convex_hull",
    return_closed_mesh=False,
    zero_out_not_closed_meshes=True,
    poisson_obj=None,
    fill_holes_obj=None,
    convex_hole_backup = True,
    verbose=False,
    default_volume_for_too_small_meshes = 0,
    allow_if_no_border_verts = True,
    ):
    if len(mesh.faces) < 2:
        return default_volume_for_too_small_meshes
    
    # -------------- 1/10 Addition: just adds the quick convex hull calculation ------ #
    if watertight_method == "fill_mesh_holes_with_fan":
        mesh = fill_mesh_holes_with_fan(mesh)
        if not is_watertight(mesh,allow_if_no_border_verts = allow_if_no_border_verts):
            if verbose:
                print(f"Using convex hull method as backup for fan")
            mesh = mesh.convex_hull
        if return_closed_mesh:
            return mesh.volume,mesh
        else:
            return mesh.volume
    
    
    if watertight_method == "convex_hull":
        if return_closed_mesh:
            return mesh.convex_hull.volume,mesh.convex_hull
        else:
            return mesh.convex_hull.volume

    """
    Purpose: To try and compute the volume of spines 
    with an optional argumet to try and close the mesh beforehand
    """
    start_time = time.time()
    if watertight_method is None:
        closed_mesh = mesh
    else:
        try: 
            if watertight_method == "trimesh":
                closed_mesh = fill_holes_trimesh(mesh)
                if not closed_mesh.is_watertight:
                    with su.suppress_stdout_stderr() if (not verbose) else su.dummy_context_mgr():
                        print("Trimesh closing holes did not work so using meshlab fill holes")
                        _, closed_mesh = mesh_volume(mesh=mesh,
                                                     watertight_method="fill_holes",
                                                     return_closed_mesh=True,
                                                     poisson_obj=poisson_obj,
                                                     fill_holes_obj=fill_holes_obj,
                                                     verbose=verbose)
            elif watertight_method == "poisson":
                if poisson_obj is None:
                    with su.suppress_stdout_stderr() if (not verbose) else su.dummy_context_mgr():
                        closed_mesh = poisson_surface_reconstruction(mesh)
                else:
                    with su.suppress_stdout_stderr() if (not verbose) else su.dummy_context_mgr():
                        print("Using premade object for poisson")
                        closed_mesh,output_subprocess_obj = poisson_obj(   
                                vertices=mesh.vertices,
                                 faces=mesh.faces,
                                 return_mesh=True,
                                 delete_temp_files=True,
                                )
                    
            elif watertight_method == "fill_holes":
                try:
                    if fill_holes_obj is None:
                        with su.suppress_stdout_stderr() if (not verbose) else su.dummy_context_mgr():
                            closed_mesh = fill_holes(mesh)
                    else:
                        with su.suppress_stdout_stderr() if (not verbose) else su.dummy_context_mgr():
                            print("Using premade object for fill holes")
                            closed_mesh,fillholes_file_obj = fill_holes_obj(   
                                            vertices=mesh.vertices,
                                             faces=mesh.faces,
                                             return_mesh=True,
                                             delete_temp_files=True,
                                            )
                except:
                    if verbose:
                        print("Filling holes did not work so using poisson reconstruction")
                    if poisson_obj is None:
                        with su.suppress_stdout_stderr() if (not verbose) else su.dummy_context_mgr():
                            closed_mesh = poisson_surface_reconstruction(mesh)
                    else:
                        with su.suppress_stdout_stderr() if (not verbose) else su.dummy_context_mgr():
                            print("Using premade object for poisson")
                            closed_mesh,output_subprocess_obj = Poisson_obj(   
                                    vertices=mesh.vertices,
                                     faces=mesh.faces,
                                     return_mesh=True,
                                     delete_temp_files=True,
                                    )
            else:
                raise Exception(f"The watertight method ({watertight_method}) is not one of implemented ones")
        except:
            print(f"The watertight method {watertight_method} could not run so not closing mesh")
            closed_mesh = mesh
            
    if verbose:
        print(f"Total time for mesh closing = {time.time() - start_time}")
        
    
    if not closed_mesh.is_watertight or closed_mesh.volume < 0:
        if zero_out_not_closed_meshes:
            final_volume = 0
        else:
            if convex_hole_backup:
                final_volume = closed_mesh.convex_hull.volume
            else:
                raise Exception(f"mesh {mesh} was not watertight ({mesh.is_watertight}) or volume is 0, vol = {closed_mesh.volume}")
    else:
        final_volume = closed_mesh.volume
    
    if return_closed_mesh:
        return final_volume,closed_mesh
    else:
        return final_volume
    

def vertex_components(mesh):
    return [list(k) for k in nx.connected_components(mesh.vertex_adjacency_graph)]

def components_to_submeshes(mesh,components,return_components=True,only_watertight=False,**kwargs):
    meshes = mesh.submesh(
        components, only_watertight=only_watertight, repair=False, **kwargs)
    

        
    if type(meshes) != type(np.array([])) and type(meshes) != list:
        #print(f"meshes = {sub_components}, with type = {type(sub_components)}")
        if type(meshes) == type(trimesh.Trimesh()) :
            
            print("list was only one so surrounding them with list")
            #print(f"meshes_before = {meshes}")
            #print(f"components_before = {components}")
            meshes = [meshes]
            
        else:
            raise Exception("The sub_components were not an array, list or trimesh")
            
    # order according to number of faces in meshes (SO DOESN'T ERROR ANYMORE)
    current_array = [len(c.faces) for c in meshes]
    ordered_indices = np.flip(np.argsort(current_array))
    
    
    
    ordered_meshes = np.array([meshes[i] for i in ordered_indices])
    ordered_components = np.array([components[i] for i in ordered_indices])
    
    if len(ordered_meshes)>=2:
        if (len(ordered_meshes[0].faces) < len(ordered_meshes[1].faces)) and (len(ordered_meshes[0].vertices) < len(ordered_meshes[1].vertices)) :
            #print(f"ordered_meshes = {ordered_meshes}")
            raise Exception(f"Split is not passing back ordered faces:"
                            f" ordered_meshes = {ordered_meshes},  "
                           f"components= {components},  "
                           f"meshes = {meshes},  "
                            f"current_array={current_array},  "
                            f"ordered_indices={ordered_indices},  "
                           )
    
    #control if the meshes is iterable or not
    try:
        ordered_comp_indices = np.array([k.astype("int") for k in ordered_components],)
    except:
        pass
    
    if return_components:
        return ordered_meshes,ordered_comp_indices
    else:
        return ordered_meshes


def split_by_vertices(
    mesh,
    return_components=False,
    return_face_idx_map=False,
    verbose=False):
    
    local_time = time.time()
    conn_verts = vertex_components(mesh)
    if verbose:
        print(f"for vertex components = {time.time() - local_time}")
        local_time = time.time()
    faces_per_component = [np.unique(np.concatenate(mesh.vertex_faces[k])) for k in conn_verts]
    if verbose:
        print(f"for faces_per_component = {time.time() - local_time}")
        local_time = time.time()
    
    faces_per_component = [k[k!=-1] for k in faces_per_component]
    if verbose:
        print(f"filtering faces_per_component = {time.time() - local_time}")
        local_time = time.time()
        
    ordered_meshes,ordered_comp_indices = components_to_submeshes(mesh,faces_per_component,return_components=True)
    if verbose:
        print(f"for components_to_submeshes = {time.time() - local_time}")
        local_time = time.time()
    
    if return_face_idx_map:
        return_components = True
        ordered_comp_indices = tu.face_idx_map_from_face_idx_list(ordered_comp_indices,mesh=mesh,)
        
    #print(f"inside vertices split, ordered_comp_indices = {ordered_comp_indices[0].dtype}")
    
    if return_components:
        return ordered_meshes,ordered_comp_indices
    else:
        return ordered_meshes
    
    
def mesh_face_graph_by_vertex(mesh):
    """
    Create a connectivity graph based on the faces that touch the same vertex have a connection edge
    
    """
    faces_adj_by_vertex = np.concatenate([np.array(list(itertools.combinations(k[k!=-1],2))) for k in mesh.vertex_faces if len(k[k!=-1])>1])
    if len(faces_adj_by_vertex) == 0:
        return nx.Graph()
    else:
        unique_edges = np.unique(faces_adj_by_vertex,axis=0)
        return nx.from_edgelist(unique_edges)
    
    
def find_closest_coordinate_to_mesh_faces(mesh,coordinates,return_closest_distance=False,verbose=False):
    """
    Given a list of coordinates will find the closest
    face on a mesh
    
    """
    coordinates = np.array(coordinates).reshape(-1,3)
    #2) get the closest point from the nodes to face centers of mesh
    mesh_kd = KDTree(mesh.triangles_center)
    dist,closest_faces = mesh_kd.query(coordinates)
    
    #3) Get the lowest distance
    closest_index = np.argmin(dist)
    min_distance = dist[closest_index]
    
    if verbose:
        print(f"Closest_distance = {min_distance}")
        
    if return_closest_distance:
        return closest_index,min_distance
    else:
        return closest_index
    
closest_coordinate_to_mesh_faces = find_closest_coordinate_to_mesh_faces

def find_closest_face_to_coordinates(mesh,coordinates,return_closest_distance=False,verbose=False):
    """
    Given a list of coordinates will find the closest
    face on a mesh
    
    """
    coordinates = np.array(coordinates).reshape(-1,3)
    #2) get the closest point from the nodes to face centers of mesh
    mesh_kd = KDTree(mesh.triangles_center)
    dist,closest_faces = mesh_kd.query(coordinates)
    
    #3) Get the lowest distance
    closest_index = np.argmin(dist)
    min_distance = dist[closest_index]
    closest_index = closest_faces[closest_index]
    
    if verbose:
        print(f"Closest_distance = {min_distance}")
        
    if return_closest_distance:
        return closest_index,min_distance
    else:
        return closest_index
    
closest_face_to_coordinates = find_closest_face_to_coordinates

def closest_mesh_coordinate_to_other_mesh(
    mesh,
    other_mesh,
    coordinate_type = "vertices",
    return_closest_distance = False,
    verbose = False,
    plot = False):
    """
    Purpose: To find the mesh coordinate to coordinates
    from aother mehs
    """
    if "face" in coordinate_type:
        coordinate_type = "triangles_center"
        
    mesh_coordinates = getattr(mesh,coordinate_type)
    other_mesh_coordinates = getattr(other_mesh,coordinate_type)
    
    mesh_kd = KDTree(other_mesh_coordinates)
    dist,closest_face = mesh_kd.query(mesh_coordinates)
    
    closest_index = np.argmin(dist)
    closest_dist = dist[closest_index]
    closest_coord = mesh_coordinates[closest_index]
    
    if verbose:
        print(f"Closesst coordinate was {closest_coord} (dist = {closest_dist})")
        
    if plot:
        ipvu.plot_objects(
            meshes = [mesh,other_mesh],
            meshes_colors = ["red","green"],
            scatters = [closest_coord.reshape(-1,3)]
        )
    if return_closest_distance:
        return closest_coord,closest_dist
    else:
        return closest_coord
    
def closest_mesh_vertex_to_other_mesh(
    mesh,
    other_mesh,
    return_closest_distance = False,
    verbose = False,
    plot=False):
    
    return closest_mesh_coordinate_to_other_mesh(
        mesh,
        other_mesh,
        coordinate_type = "vertices",
        return_closest_distance = return_closest_distance,
        verbose = verbose,
        plot=plot,)
    
    
def closest_face_to_coordinate(mesh,coordinate,return_face_coordinate = False):
    """
    To find the closest face midpoint to a coordiante
    
    """
    closest_idx = np.argmin(np.linalg.norm(mesh.triangles_center - coordinate,axis=-1))
    if return_face_coordinate:
        return mesh.triangles_center[closest_idx]
    else:
        return closest_idx
    
def farthest_face_to_coordinate(mesh,coordinate,return_face_coordinate = False):
    """
    To find the closest face midpoint to a coordiante
    
    """
    closest_idx = np.argmax(np.linalg.norm(mesh.triangles_center - coordinate,axis=-1))
    if return_face_coordinate:
        return mesh.triangles_center[closest_idx]
    else:
        return closest_idx
    
def closest_face_to_coordinate_distance(mesh,coordinate):
    return np.linalg.norm(
    tu.closest_face_to_coordinate(mesh,coordinate,return_face_coordinate = True) - 
        coordinate
    )

def farthest_face_to_coordinate_distance(mesh,coordinate):
    return np.linalg.norm(
    tu.farthest_face_to_coordinate(mesh,coordinate,return_face_coordinate = True) - 
        coordinate
    )
    
    
def face_neighbors_by_vertices(mesh,faces_list,
                              concatenate_unique_list=True):
    """
    Find the neighbors of face where neighbors are
    faces that touch the same vertices
    
    Pseudocode: 
    1) Change the faces to vertices
    2) Find all the faces associated with the vertices
    """
    if concatenate_unique_list:
        return vertices_to_faces(mesh,mesh.faces[faces_list].ravel(),concatenate_unique_list=concatenate_unique_list)
    else:
        return [vertices_to_faces(mesh,mesh.faces[k].ravel(),concatenate_unique_list=True) for k in faces_list]
    
    
def face_neighbors_by_vertices_seperate(mesh,faces_list):
    f_verts = mesh.faces[faces_list]
    return [np.unique(k[k!=-1]) for k in mesh.vertex_faces[f_verts]]

def skeleton_to_mesh_correspondence(mesh,
                                    skeletons,
                                    remove_inside_pieces_threshold = 100,
                                    return_meshes=True,
                                    distance_by_mesh_center=True,
                                    connectivity="edges",
                                    verbose=False):
    """
    Purpose: To get the first pass mesh 
    correspondence of a skeleton or list of skeletons
    in reference to a mesh

    Pseudocode: 
    1) If requested, remove the interior of the mesh (if this is set then can't return indices)
    - if return indices is set then error if interior also set
    2) for each skeleton:
        a. Run the mesh correspondence adaptive function
        b. check to see if got any output (if did not then return empty list or empty mesh)
        c. If did add a submesh or indices to the return list


    Example:
    return_value = tu.skeleton_to_mesh_correspondence( mesh = debug_mesh,
                                                skeletons = viable_end_node_skeletons
                                   )

    ipvu.plot_objects(meshes=return_value,
                      meshes_colors="random",
                      skeletons=viable_end_node_skeletons,
                     skeletons_colors="random")
    """

    if type(skeletons) != list:
        skeletons = [skeletons]

    return_indices = []

    if remove_inside_pieces_threshold > 0:
        curr_limb_mesh_indices = tu.remove_mesh_interior(mesh,
                                                 size_threshold_to_remove=remove_inside_pieces_threshold,
                                                 try_hole_close=False,
                                                 return_face_indices=True,
                                                )
        curr_limb_mesh_indices = np.array(curr_limb_mesh_indices)
        curr_mesh = mesh.submesh([curr_limb_mesh_indices],append=True,repair=False)
    else:
        curr_limb_mesh_indices = np.arange(len(mesh.faces))
        curr_mesh = mesh


    #1) Run the first pass mesh correspondence
    for curr_sk in skeletons:
        returned_data = cu.mesh_correspondence_adaptive_distance(curr_sk,
                                  curr_mesh,
                                 skeleton_segment_width = 1000,
                                 distance_by_mesh_center=distance_by_mesh_center,
                                                            connectivity=connectivity)

        if len(returned_data) == 0:
            return_indices.append([])
        else:

            curr_branch_face_correspondence, width_from_skeleton = returned_data
            return_indices.append(curr_limb_mesh_indices[curr_branch_face_correspondence])

    if return_meshes:
        return_value = []
        for ind in return_indices:
            if len(ind)>0:
                return_value.append(mesh.submesh([ind],append=True,repair=False))
            else:
                return_value.append(trimesh.Trimesh(vertices=np.array([]),
                                                   faces=np.array([])))
    else:
        return_value = return_indices

    if verbose:
        if not return_meshes:
            ret_val_sizes = [len(k) for k in return_value]
        else:
            ret_val_sizes = [len(k.faces) for k in return_value]

        print(f"Returned value sizes = {ret_val_sizes}")
        
    return return_value

def mesh_segmentation_from_skeleton(
    mesh,
    skeleton,
    skeleton_segment_width = 0.01,
    initial_distance_threshold = 0.2,
    skeletal_buffer = 0.01,
    backup_distance_threshold = 0.4,
    backup_skeletal_buffer = 0.02,
    plot_correspondence_first_pass=False,
    plot = False,
    ):
    
    """
    Purpose: To turn a skeleton into a mesh
    correspondence dictionary
    
    1) Divide up skeleton
    2) Find mesh that corresponds to branches
    3) Refines the correspondence so only 1 skeletal
    branch matched to each face
    """
    from neurd import preprocessing_vp2 as pre

    local_correspondence = pre.mesh_correspondence_first_pass(
        mesh,
        skeleton,
        skeleton_segment_width = skeleton_segment_width,
        initial_distance_threshold = initial_distance_threshold,
        skeletal_buffer = skeletal_buffer,
        backup_distance_threshold = backup_distance_threshold,
        backup_skeletal_buffer = backup_skeletal_buffer,
        plot = plot_correspondence_first_pass,
    )

    refined_correspondence = pre.correspondence_1_to_1(
        mesh=mesh,
        local_correspondence=local_correspondence,
        plot = plot,
    )

    return refined_correspondence

def skeleton_and_mesh_segmentation(
    mesh = None,
    filepath = None,
    skeleton_kwargs = None,
    skeleton_function = None,
    plot_skeleton = False,
    segmentation_kwargs = None,
    plot_segmentation = False,
    verbose = False,
    ):
    
    """
    tu.skeleton_and_mesh_segmentation(
        filepath = "./elephant.off",
        plot_segmentation = True,
    )
    """
    
    if skeleton_kwargs is None:
        skeleton_kwargs = dict()
        
    if segmentation_kwargs is None:
        segmentation_kwargs = dict()
    
    if skeleton_function is None:
        skeleton_function = sk.skeleton_cgal_original_parameters
    
    if mesh is None:
        mesh = tu.load_mesh_no_processing(filepath)
        
    
    skeleton = skeleton_function(
        mesh,
        verbose = verbose,
        remove_skeleton_temp_file= True,
        plot = plot_skeleton,
        **skeleton_kwargs
    )
    
    correspond = tu.mesh_segmentation_from_skeleton(
        mesh,
        skeleton,
        plot = plot_segmentation,
        **segmentation_kwargs
    )
    
    return correspond
    


'''def find_large_dense_submesh(mesh,
                             glia_pieces=None, #the glia pieces we already want removed
                            verbose = True,
                            large_dense_size_threshold = 600000,
                            large_mesh_cancellation_distance = 3000,
                            filter_away_floating_pieces = True,
                            bbox_filter_away_ratio = 1.7,
                            connectivity="vertices",
                            floating_piece_size_threshold = 130000,
                            remove_large_dense_submesh=True):

    """
    Purpose: Getting all of the points close to glia and removing them

    1) Get the inner glia mesh
    2) Build a KDTree fo the inner glia
    3) Query the faces of the remainingn mesh
    4) Get all the faces beyond a certain distance and get the submesh
    5) Filter away all floating pieces in a certain region of the bounding box
    """
    current_neuron_mesh = mesh

    #1) Get the inner glia mesh

    all_large_dense_pieces = []

    if glia_pieces is None:
        mesh_removed_glia,inside_pieces = tu.remove_mesh_interior(current_neuron_mesh,
                               size_threshold_to_remove=large_dense_size_threshold,
                                connectivity=connectivity,
                                try_hole_close=False,
                                return_removed_pieces =True,
                                 **kwargs
                               )
    else:
        print("using precomputed glia pieces")
        inside_pieces = list(glia_pieces)
        mesh_removed_glia = tu.subtract_mesh(mesh,inside_pieces,
                                            exact_match=False)

    all_large_dense_pieces+= inside_pieces


    for j,inside_glia in enumerate(inside_pieces):
        if verbose:
            print(f"\n ---- Working on inside piece {j} ------")

        #2) Build a KDTree fo the inner glia
        glia_kd = KDTree(inside_glia.triangles_center.reshape(-1,3))


        #3) Query the faces of the remainingn mesh
        dist, _ = glia_kd.query(mesh_removed_glia.triangles_center)

        #4) Get all the faces beyond a certain distance and get the submesh
        within_threshold_faces = np.where(dist>= large_mesh_cancellation_distance)[0]
        removed_threshold_faces = np.where(dist< large_mesh_cancellation_distance)[0]


        if len(within_threshold_faces)>0:
            #gathering pieces to return that were removed
            close_pieces_removed = mesh_removed_glia.submesh([removed_threshold_faces],append=True,repair=False)
            all_large_dense_pieces.append(close_pieces_removed)
            
            mesh_removed_glia = mesh_removed_glia.submesh([within_threshold_faces],append=True,repair=False)


            if verbose:
                print(f"For glia mesh {j} there were {len(within_threshold_faces)} faces within {large_mesh_cancellation_distance} distane")
                print(f"New mesh size is {mesh_removed_glia}")


            if filter_away_floating_pieces:
                floating_sig_pieces, floating_insig_pieces = tu.split_significant_pieces(mesh_removed_glia,
                                                                  significance_threshold = floating_piece_size_threshold,
                                                                  return_insignificant_pieces=True,
                                                                  connectivity=connectivity)

                if len(floating_insig_pieces)>0:
                    #get the 
                    floating_pieces_to_remove = tu.check_meshes_inside_mesh_bbox(inside_glia,floating_insig_pieces,bbox_multiply_ratio=bbox_filter_away_ratio)

                    if verbose:
                        print(f"Found {len(floating_insig_pieces)} and going to remove {len(floating_pieces_to_remove)} that were inside bounding box")

                    if len(floating_pieces_to_remove)>0:

                        mesh_removed_glia = tu.subtract_mesh(mesh_removed_glia,floating_pieces_to_remove,
                                                            exact_match=False)

                        all_large_dense_pieces += floating_pieces_to_remove

                        if verbose:
                            print(f"After removal of floating pieces the mesh is {mesh_removed_glia}")
            """
            To help visualize the floating pieces that were removed

            ipvu.plot_objects(mesh_removed_glia,
                          meshes=floating_insig_pieces,
                         meshes_colors="red")

            """
            
    #compiling the large dense submesh
    if len(all_large_dense_pieces) > 0:
        total_dense_submesh = tu.combine_meshes(all_large_dense_pieces)
    else:
        if verbose: 
            print("There was no large dense submesh")
        total_dense_submesh = trimesh.Trimesh(vertices = np.array([]),
                                             faces=np.array([]))

    if remove_large_dense_submesh:
        return mesh_removed_glia,total_dense_submesh
    else:
        return total_dense_submesh
    '''
    

def find_large_dense_submesh(mesh,
                             glia_pieces=None, #the glia pieces we already want removed
                            verbose = True,
                            large_dense_size_threshold = 400000,
                            large_mesh_cancellation_distance = 3000,
                            filter_away_floating_pieces = True,
                            bbox_filter_away_ratio = 1.7,
                            connectivity="vertices",
                            floating_piece_size_threshold = 130000,
                            remove_large_dense_submesh=True):

    """
    Purpose: Getting all of the points close to glia and removing them

    1) Get the inner glia mesh
    2) Build a KDTree fo the inner glia
    3) Query the faces of the remainingn mesh
    4) Get all the faces beyond a certain distance and get the submesh
    5) Filter away all floating pieces in a certain region of the bounding box
    """
    current_neuron_mesh = mesh

    #1) Get the inner glia mesh

    all_large_dense_pieces = []

    if glia_pieces is None:
        mesh_removed_glia,inside_pieces = tu.remove_mesh_interior(current_neuron_mesh,
                               size_threshold_to_remove=large_dense_size_threshold,
                                connectivity=connectivity,
                                try_hole_close=False,
                                return_removed_pieces =True,
                                 **kwargs
                               )
    else:
        print("using precomputed glia pieces")
        inside_pieces = list(glia_pieces)
        mesh_removed_glia = tu.subtract_mesh(mesh,inside_pieces,
                                            exact_match=False)

    all_large_dense_pieces+= inside_pieces

    
    
    for j,inside_glia in enumerate(inside_pieces):
        if verbose:
            print(f"\n ---- Working on inside piece {j} ------")

        #2) Build a KDTree fo the inner glia
        glia_kd = KDTree(inside_glia.triangles_center.reshape(-1,3))


        #3) Query the faces of the remainingn mesh
        dist, _ = glia_kd.query(mesh_removed_glia.triangles_center)

        #4) Get all the faces beyond a certain distance and get the submesh
        within_threshold_faces = np.where(dist>= large_mesh_cancellation_distance)[0]
        removed_threshold_faces = np.where(dist< large_mesh_cancellation_distance)[0]


        if len(within_threshold_faces)>0:
            #gathering pieces to return that were removed
            close_pieces_removed = mesh_removed_glia.submesh([removed_threshold_faces],append=True,repair=False)
            all_large_dense_pieces.append(close_pieces_removed)
            
            mesh_removed_glia = mesh_removed_glia.submesh([within_threshold_faces],append=True,repair=False)


            if verbose:
                print(f"For glia mesh {j} there were {len(within_threshold_faces)} faces within {large_mesh_cancellation_distance} distane")
                print(f"New mesh size is {mesh_removed_glia}")


                
                
                
                
    if filter_away_floating_pieces:
        
        floating_sig_pieces, floating_insig_pieces = tu.split_significant_pieces(mesh_removed_glia,
                                                              significance_threshold = floating_piece_size_threshold,
                                                              return_insignificant_pieces=True,
                                                              connectivity=connectivity)
        
        for j,inside_glia in enumerate(inside_pieces):
    
            if len(floating_insig_pieces)>0:
                
                floating_pieces_to_remove = tu.check_meshes_inside_mesh_bbox(inside_glia,floating_insig_pieces,bbox_multiply_ratio=bbox_filter_away_ratio)

                if verbose:
                    print(f"Found {len(floating_insig_pieces)} and going to remove {len(floating_pieces_to_remove)} that were inside bounding box")

                if len(floating_pieces_to_remove)>0:

                    mesh_removed_glia = tu.subtract_mesh(mesh_removed_glia,floating_pieces_to_remove,
                                                        exact_match=False)

                    all_large_dense_pieces += floating_pieces_to_remove

                    if verbose:
                        print(f"After removal of floating pieces the mesh is {mesh_removed_glia}")
            """
            To help visualize the floating pieces that were removed

            ipvu.plot_objects(mesh_removed_glia,
                          meshes=floating_insig_pieces,
                         meshes_colors="red")

            """
            
    #compiling the large dense submesh
    if len(all_large_dense_pieces) > 0:
        total_dense_submesh = tu.combine_meshes(all_large_dense_pieces)
    else:
        if verbose: 
            print("There was no large dense submesh")
        total_dense_submesh = None

    if remove_large_dense_submesh:
        return mesh_removed_glia,total_dense_submesh
    else:
        return total_dense_submesh
    
    
    
def empty_mesh():
    return trimesh.Trimesh(vertices=np.array([]),
                          faces=np.array([]))
    
    
        
def percentage_vertices_inside(
                                main_mesh,
                                test_mesh,
                                n_sample_points = 1000,
                                use_convex_hull = True,
                                verbose = False):
    """
    Purpose: Function that will determine the percentage of vertices
    of one mesh being inside of another
    
    Ex:
    tu.percentage_vertices_inside(
                    main_mesh = soma_meshes[3],
                    test_mesh = soma_meshes[1],
                    n_sample_points = 1000,
                    use_convex_hull = True,
                    verbose = True)

    """

    if use_convex_hull:
        main_mesh = main_mesh.convex_hull

    mesh = test_mesh

    if n_sample_points > len(mesh.vertices):
        n_sample_points = len(mesh.vertices)

    #gets the number of samples on the mesh to test (only the indexes)
    idx = np.random.choice(len(mesh.vertices),n_sample_points , replace=False)
    #gets the sample's vertices
    points = mesh.vertices[idx,:]

    #find the signed distance from the sampled vertices to the main mesh
    # Points outside the mesh will be negative
    # Points inside the mesh will be positive
    signed_distance = trimesh.proximity.signed_distance(main_mesh,points)

    #gets the 
    inside_percentage = sum(signed_distance >= 0)/n_sample_points

    if verbose:
        print(f"Inside Percentage = {inside_percentage}")
        
    return inside_percentage


def test_inside_meshes(main_mesh,
                        test_meshes,
                        n_sample_points = 10,
                        use_convex_hull = True,
                       inside_percentage_threshold = 0.9,
                        return_outside=False,
                        return_meshes=False,
                       verbose=False,
    ):
    """
    To determine which of the test meshes
    are inside the main mesh
    
    Ex:
    tu.test_inside_meshes(
                    main_mesh = soma_meshes[3],
                    test_meshes = soma_meshes[1],
                    n_sample_points = 1000,
                    use_convex_hull = True,
                    inside_percentage_threshold=0.9,
                    verbose = True)

    
    """
    if not nu.is_array_like(test_meshes):
        test_meshes = [test_meshes]
    
    test_meshes = np.array(test_meshes)
    inside_indices = []
    
    for i,t_mesh in enumerate(test_meshes):
        perc_inside = percentage_vertices_inside(
                                main_mesh,
                                t_mesh,
                                n_sample_points = n_sample_points,
                                use_convex_hull = use_convex_hull,
                                verbose = False)
        
        if perc_inside > inside_percentage_threshold:
            if verbose:
                print(f"Mesh {i} was inside because inside percentage was {perc_inside}")
            inside_indices.append(i)
        else:
            if verbose:
                print(f"Mesh {i} was OUTSIDE because inside percentage was {perc_inside}")
    
    inside_indices = np.array(inside_indices)
    
    if return_outside:
        return_indices = np.delete(np.arange(len(test_meshes)),inside_indices)
    else:
        return_indices = inside_indices
    
    
    if return_meshes:
        return test_meshes[return_indices]
    else:
        return return_indices
        
        
def meshes_distance_matrix(mesh_list,
                                distance_type="shortest_vertex_distance",
                          verbose=False):
    """
    Purpose: To determine the pairwise distance between meshes
    """
    
    if verbose:
        print(f"Using distance_type = {distance_type}")
        
    if distance_type == "shortest_vertex_distance":
       
        dist_matrix_adj = []
        for i,m1 in enumerate(mesh_list):
            
            local_distance = []
            m1_kd = KDTree(m1.vertices)
            
            for j,m2 in enumerate(mesh_list):
                
                if j == i:
                    local_distance.append(np.inf)
                    continue
                    
                dist, _ = m1_kd.query(m2.vertices)
                local_distance.append(np.min(dist))
                
            dist_matrix_adj.append(local_distance)
            
        dist_matrix_adj = np.array(dist_matrix_adj)
        
    elif distance_type == "mesh_center":
        dist_matrix = nu.get_coordinate_distance_matrix([tu.mesh_center_vertex_average(k)
                                                         for k in mesh_list])
        dist_matrix_adj = dist_matrix + np.diag([np.inf]*len(dist_matrix))
        
    else:
        raise Exception("Unimplemented distance_type")
    
    return dist_matrix_adj
    
def meshes_within_close_proximity(mesh_list,
                                distance_type="shortest_vertex_distance",
                                  distance_threshold = 20000,
                                  return_distances = True,
                                verbose=False):
    """
    Purpose: To Get the meshes that are within close proximity of each other
    as defined by mesh centers or the absolute shortest vertex distance
    
    """
    if len(mesh_list)<2:
        return [],[]
    
    dist_matrix_adj = meshes_distance_matrix(mesh_list,
                                distance_type=distance_type,
                          verbose=verbose)
    if verbose:
        print(f"dist_matrix_adj = {dist_matrix_adj}")
        
    soma_pairings_to_check = np.array(nu.unique_non_self_pairings(np.array(np.where(dist_matrix_adj<=distance_threshold)).T))
    
    if len(soma_pairings_to_check) > 0 and (0 not in soma_pairings_to_check.shape):
        distances_per_pair = np.array([dist_matrix_adj[k1][k2] for k1,k2 in soma_pairings_to_check])
    else:
        distances_per_pair = []
    
    if return_distances:
        return soma_pairings_to_check,distances_per_pair
    else:
        return soma_pairings_to_check
    
    
    
def filter_away_inside_meshes(mesh_list,
                                distance_type="shortest_vertex_distance",
                                distance_threshold = 2000,
                                inside_percentage_threshold = 0.15,
                                verbose = False,
                                return_meshes = False,
                                max_mesh_sized_filtered_away=np.inf,
                                ):

    """
    Purpose: To filter out any meshes
    that are inside of another mesh in the list

    1) Get all the pairings of meshes to check and the distances between
    2) Find the order of pairs to check 1st by distance
    3) Create a viable meshes index list with initially all indexes present

    For each pair (in the order pre-determined in 2)
    a) If both indexes are not in the viable meshes list --> continue
    b) check the percentage that each is inside of the other
    c) Get the ones that are above a threshold
    d1) If none are above threshold then continue
    d2) If one is above threshold then that is the losing index
    d3) If both are above the threshold then pick the smallest one as the losing index
    e) remove the losing index from the viable meshes index list

    4) Return either the viables meshes indexes or the meshes themselves


    """
    

    if len(mesh_list) < 2:
        if verbose:
            print("Mesh list was less than 2 object so returning")
            
        if return_meshes:
            return mesh_list
        else:
            return np.arange(len(mesh_list))
    
    # May want to put a size threshold on what we filter away
    mesh_list_len = np.array([len(k.faces) for k in mesh_list])
    must_keep_indexes = np.where(mesh_list_len>max_mesh_sized_filtered_away)
    
    if verbose:
        print(f"must_keep_indexes = {must_keep_indexes}")
    
    if len(must_keep_indexes) == len(mesh_list):
        if verbose:
            print(f"All meshes were above the max_mesh_sized_filtered_away threshold: {max_mesh_sized_filtered_away}")
        if return_meshes:
            return mesh_list
        else:
            return np.arange(len(mesh_list))   
            

    mesh_list = np.array(mesh_list)

    #1) Get all the pairings of meshes to check and the distances between
    return_pairings,pair_distances = tu.meshes_within_close_proximity(mesh_list,
                                    distance_type=distance_type,
                                    distance_threshold = distance_threshold,
                                                      verbose=verbose)

    if verbose:
        print(f"return_pairings = {return_pairings}")
        print(f"pair_distances = {pair_distances}")

    #2) Find the order of pairs to check 1st by distance
    pair_to_check_order = np.argsort(pair_distances)


    #3) Create a viable meshes index list with initially all indexes present
    viable_meshes = np.arange(len(mesh_list))

    #For each pair (in the order pre-determined in 2)
    for pair_idx in pair_to_check_order:

        curr_pair = return_pairings[pair_idx]
        mesh_1_idx,mesh_2_idx = curr_pair

        if verbose:
            print(f"\n-- working on pair: {curr_pair} --")

        #a) If both indexes are not in the viable meshes list --> continue
        if (mesh_1_idx not in viable_meshes) or (mesh_2_idx not in viable_meshes):
            print("Continuing because both meshes not in viable mesh list")
            continue

        #b) check the percentage that each is inside of the other
        
        
        inside_percentages = np.array([percentage_vertices_inside(
                                    main_mesh = mesh_list[curr_pair[1-i]],
                                    test_mesh = mesh_list[curr_pair[i]]) for i in range(0,2)])

        if verbose:
            print(f"inside_percentages = {inside_percentages}")

        #c) Get the ones that are above a threshold
        above_inside_threshold_meshes = np.where(inside_percentages > inside_percentage_threshold)[0]

        #d1) If none are above threshold then continue
        if len(above_inside_threshold_meshes) == 0:
            if verbose:
                print(f"None above the threshold {inside_percentage_threshold} so continuing")
            continue

        elif len(above_inside_threshold_meshes) == 1:
            losing_index = curr_pair[above_inside_threshold_meshes[0]]

            if verbose:
                print(f"Only 1 above the threshold: mesh {losing_index}")

        elif len(above_inside_threshold_meshes) == 2:
            sizes = np.array([len(mesh_list[k].faces) for k in curr_pair])
            losing_index = curr_pair[np.argmin(sizes)]

            if verbose:
                print(f"2 above the threshold so picking the smallest of the size {sizes}: mesh {losing_index}")
        else:
            raise Exception("More than 2 above threshold")

        #e) remove the losing index from the viable meshes index list
        viable_meshes = viable_meshes[viable_meshes!=losing_index]

    # making sure to add back in the meshes that shouldn't be filtered away
    viable_meshes = np.union1d(viable_meshes,must_keep_indexes)
        
    if return_meshes:
        return [k for i,k in enumerate(mesh_list) if i in viable_meshes]
    else:
        return viable_meshes

    
def is_mesh(obj):
    if type(obj) == type(trimesh.Trimesh()):
        return True
    else:
        return False

def turn_off_logging():
    logging.getLogger("trimesh").setLevel(logging.ERROR)

turn_off_logging()


# -------------- 1/21 ------------- #
def bounding_box_oriented_side_lengths(mesh,return_largest_side=False):
    
    bbox = mesh.bounding_box_oriented.vertices
    x_axis_unique = np.unique(bbox[:,0])
    y_axis_unique = np.unique(bbox[:,1])
    z_axis_unique = np.unique(bbox[:,2])
    x_length = (np.max(x_axis_unique) - np.min(x_axis_unique)).astype("float")
    y_length = (np.max(y_axis_unique) - np.min(y_axis_unique)).astype("float")
    z_length = (np.max(z_axis_unique) - np.min(z_axis_unique)).astype("float")
    
    if return_largest_side:
        return np.max([x_length,y_length,z_length])
    else:
        return x_length,y_length,z_length
    
def bounding_box_longest_side(mesh):
    return bounding_box_oriented_side_lengths(mesh,return_largest_side=True)

def filter_meshes_by_bounding_box_longest_side(meshes,
                                              side_length_threshold):
    longest_side_lengths = np.array([tu.bounding_box_longest_side(k) for k in meshes])
    keep_indexes = np.where(longest_side_lengths<=side_length_threshold)[0]
    return [meshes[i] for i in keep_indexes]

def box_mesh(center,
             radius=100):
    box_obj = trimesh.creation.box(extents=[radius,radius,radius])
    box_obj.vertices = box_obj.vertices + center
    return box_obj

def center_mesh(mesh,new_center):
    new_mesh = mesh
    added_offset =  np.array(new_center) - new_mesh.center_mass
    new_mesh.vertices = new_mesh.vertices + added_offset
    return new_mesh

def center_mesh_at_point(mesh,new_center):
    mesh.vertices = mesh.vertices - new_center
    return mesh

def sphere_mesh(center=[0,0,0],radius=100):
    sph = trimesh.creation.icosphere(subdivisions = 1,radius=radius)
    return center_mesh(sph,center)


def kdtree_length(kdtree_obj):
    return len(kdtree_obj.data_pts)

def largest_border_to_coordinate(
    mesh,
    coordinate,
    distance_threshold = 10000,
    plot_border_vertices = False,
    error_on_no_border = True,
    plot_winning_border = False,
    verbose = False):

    """
    Purpose: To find the biggest border within a certain radius

    Pseudocode: 
    1) Find all of the vertex border groups
    2) Find the average vertex of these groups
    3) Find the distance of these groups from the coordinate
    4) Filter for those within a certain radius
    5) Return the border group that is the largest
    """

    #1) Find all of the vertex border groups
    border_vertex_groups = tu.find_border_vertex_groups(mesh,
                                                       return_coordinates=True)

    if len(border_vertex_groups) == 0:
        if error_on_no_border:
            raise Exception("No borders detected")
        else:
            if verbose:
                print(f"No borders detected")
        winning_border =  None
    else:
        if verbose:
            print(f"# of border_vertex_groups = {len(border_vertex_groups)}")

        if plot_border_vertices:
            ipvu.plot_objects(mesh,
                             scatters=[np.array(k).reshape(-1,3) for k in border_vertex_groups],
                             scatters_colors="red")

        #2) Find the average vertex of these groups
        border_vertex_averages = np.array([np.mean(k,axis=0) for k in border_vertex_groups])


        #3) Find the distance of these groups from the coordinate
        border_distances = np.linalg.norm(border_vertex_averages - coordinate,axis=1)

        #4) Filter for those within a certain radius
        viable_borders_idx = np.where(border_distances<distance_threshold)[0]

        if len(viable_borders_idx)>0:
            viable_borders = [border_vertex_groups[k] for k in viable_borders_idx]
            viable_borders_len = np.array([len(k) for k in viable_borders])
            winning_border_idx = np.argmax(viable_borders_len)
            winning_border = viable_borders[winning_border_idx]
        else:
            if verbose:
                print(f"No borders within the threshold of distance_threshold so returning just the closest border")
            winning_border = border_vertex_groups[np.argmin(border_distances)]

        if plot_winning_border:
            ipvu.plot_objects(mesh,
                             scatters=[winning_border,coordinate],
                             scatters_colors=["red","green"],
                             scatter_size=0.3)


    return winning_border

def closest_mesh_to_coordinates(mesh_list,coordinates,
                               return_mesh=True,
                                  return_distance=False,
                                distance_method = "min_distance", 
                              verbose=False,):
    return closest_mesh_to_coordinate(mesh_list,coordinates,
                                  return_mesh=return_mesh,
                                  return_distance=return_distance,
                                      distance_method=distance_method,
                              verbose=verbose,)

def closest_mesh_to_coordinate(mesh_list,coordinate,
                                  return_mesh=True,
                                  return_distance=False,
                               distance_method = "min_distance", 
                               return_idx = False,
                              verbose=False,):
    """
    Purpose: Will pick the closest mesh from a list of meshes
    to a certain coordinate point

    Pseudocode: 
    Iterate through all of the meshes
    1) Build KDTree of mesh
    2) query the coordinate against mesh
    3) Save the distance in array

    4) Find the index with the smallest distance
    5) Return the mesh or the index
    
    ** distance methods are:
    1) min_distance: will measure the distance between the closest mesh face and coordinate
    2) bbox_center: measure the bounding box center of the mesh to the coordinate
    
    Ex: 
    closest_mesh_to_coordinate(mesh_list = new_meshes,
    coordinate = current_endpoints[1],
    verbose=False,
    return_mesh=False,
    return_distance=True)
    """

    from pykdtree.kdtree import KDTree
    

    mesh_to_dist = []
    coordinate = np.array(coordinate).reshape(-1,3)
    for j,m in enumerate(mesh_list):
        if distance_method == "min_distance":
            curr_kd = KDTree(m.triangles_center)
            dist,_ = curr_kd.query(coordinate)
            total_dist = np.sum(dist)
            
        elif distance_method == "bbox_center":
            bbox_center = tu.bounding_box_center(m)
            dist = np.linalg.norm(coordinate - bbox_center,axis=1)
            total_dist = np.sum(dist)
        elif distance_method == "mesh_center":
            center = tu.mesh_center_vertex_average(m)
            dist = np.linalg.norm(coordinate - center,axis=1)
            total_dist = np.sum(dist)
        elif distance_method == "min_distance_and_bbox_center":
            curr_kd = KDTree(m.triangles_center)
            dist_min,_ = curr_kd.query(coordinate)
            
            bbox_center = tu.bounding_box_center(m)
            dist_bbox = np.linalg.norm(coordinate - bbox_center,axis=1)
            
            total_dist = np.sum(dist_min) + np.sum(dist_bbox)
        else:
            raise Exception(f"Unimplemented Type distance_method ({distance_method})")

        if verbose:
                print(f"Mesh {j}: {total_dist} nm away")
        mesh_to_dist.append(total_dist)

    closest_idx = np.argmin(mesh_to_dist)

    if verbose:
        print(f"\nClosest mesh: {closest_idx}")

    if return_mesh:
        return_value = mesh_list[closest_idx]
    else:
        return_value = closest_idx
        
    if return_idx:
        return_value = closest_idx

    if return_distance:
        return_value = [return_value,mesh_to_dist[closest_idx]]
        
#     if return_idx:
#         return_value = closest_idx

    return return_value




    

def bbox_volume_oriented(mesh):
    return mesh.bounding_box_oriented.volume
def bbox_volume(mesh):
    return mesh.bounding_box.volume

def mesh_size(mesh,size_type="faces",
             percentile=70,
             replace_zero_values_with_center_distance=True):
    """
    Purpose: Will return the size of a mesh based on the 
    size type (vertices or faces)
    """
    if size_type == "faces":
        return len(mesh.faces)
    elif size_type == "vertices":
        return len(mesh.vertices)
    elif size_type == "ray_trace_mean":
        return np.mean(tu.ray_trace_distance(mesh,
                                            replace_zero_values_with_center_distance=replace_zero_values_with_center_distance))
    elif size_type == "ray_trace_median":
        return np.median(tu.ray_trace_distance(mesh,
                                              replace_zero_values_with_center_distance=replace_zero_values_with_center_distance))
    elif size_type == "ray_trace_percentile":
        return np.percentile(tu.ray_trace_distance(mesh,
                                                  replace_zero_values_with_center_distance=replace_zero_values_with_center_distance),percentile)
    elif size_type == "skeleton":
        try:
            return sk.calculate_skeleton_distance(sk.surface_skeleton(mesh))
        except:
            return 0
    elif size_type == "volume":
        try:
            return tu.mesh_volume(mesh)
        except:
            return 0
    elif size_type == "bbox_volume":
        return bbox_volume_oriented(mesh)
    elif size_type == "bbox_volume_oriented":
        return bbox_volume_oriented(mesh)
    else:
        raise Exception(f"Unimplemented type {size_type}")
    
    
    
def filter_meshes_by_size(mesh_list,
                          size_threshold,
                         size_type="faces",
                          above_threshold=True,
                          return_indices = False,
                         verbose=False,
                         **kwargs):
    """
    Purpose: Will return the meshes
    or indices of the meshes that are above (or below if abvoe threshold
    set to False) the vertices or faces threshold
    
    Pseudocode:
    1) Calculate the sizes of the meshes based on the threshold set
    2) Find the indices of the meshes that are above or below the threshold
    3) Return either the meshes or indices
    
    Ex:
    tu.filter_meshes_by_size(mesh_list=new_meshes,
                          faces_threshold = None,
                         vertices_threshold=100,
                          above_threshold=True,
                          return_indices = True,
                         verbose=False)
    """
    
#     if faces_threshold is not None:
#         size_threshold = faces_threshold
#         size_type = "faces"
#     elif vertices_threshold is not None:
#         size_threshold = vertices_threshold
#         size_type = "vertices"
#     else:
#         raise Exception("Neither faces nor vertices threshold set")
    
    if size_threshold is None:
        if return_indices:
            return np.arange(len(mesh_list))
        else:
            return mesh_list
    
    mesh_sizes = np.array([mesh_size(k,size_type,**kwargs) for k in mesh_list])
    
    if verbose:
        print(f"mesh_sizes with type ({size_type}): {mesh_sizes}")
    
    #2) Find the indices of the meshes that are above or below the threshold
    if above_threshold:
        keep_indices = np.where(mesh_sizes>size_threshold)[0]
    else:
        keep_indices = np.where(mesh_sizes<=size_threshold)[0]
    
    if verbose:
        print(f"keep_indices = {keep_indices}")
        
    if return_indices:
        return keep_indices
    else:
        keep_meshes = np.array(mesh_list)[keep_indices]
        if type(mesh_list) == list:
            keep_meshes = list(keep_meshes)
        return keep_meshes
    
def filter_meshes_by_size_min_max(mesh_list,
                          min_size_threshold,
                          max_size_threshold,
                         size_type="faces",
                          return_indices = False,
                         verbose=False):
    
    min_indices = filter_meshes_by_size(mesh_list,
                          size_threshold=min_size_threshold,
                         size_type=size_type,
                          above_threshold=True,
                          return_indices = True,
                         verbose=verbose)
    
    
    max_indices = filter_meshes_by_size(mesh_list,
                          size_threshold=max_size_threshold,
                         size_type=size_type,
                          above_threshold=False,
                          return_indices = True,
                         verbose=verbose)
    
    min_max_indices = np.intersect1d(min_indices,max_indices)
    
    if verbose:
        print(f"min_indices = {min_indices}")
        print(f"max_indices = {max_indices}")
        print(f"min_max_indices = {min_max_indices}")
        
    if return_indices:
        return min_max_indices
    else:
        keep_meshes = np.array(mesh_list)[min_max_indices]
        if type(mesh_list) == list:
            keep_meshes = list(keep_meshes)
        return keep_meshes
    
def bbox_side_length_ratios(current_mesh):
    """
    Will compute the ratios of the bounding box sides
    To be later used to see if there is skewness
    """

    # bbox = current_mesh.bounding_box_oriented.vertices
    bbox = current_mesh.bounding_box_oriented.vertices
    x_axis_unique = np.unique(bbox[:,0])
    y_axis_unique = np.unique(bbox[:,1])
    z_axis_unique = np.unique(bbox[:,2])
    x_length = (np.max(x_axis_unique) - np.min(x_axis_unique)).astype("float")
    y_length = (np.max(y_axis_unique) - np.min(y_axis_unique)).astype("float")
    z_length = (np.max(z_axis_unique) - np.min(z_axis_unique)).astype("float")
    #print(x_length,y_length,z_length)
    #compute the ratios:
    xy_ratio = float(x_length/y_length)
    xz_ratio = float(x_length/z_length)
    yz_ratio = float(y_length/z_length)
    side_ratios = [xy_ratio,xz_ratio,yz_ratio]
    flipped_side_ratios = []
    for z in side_ratios:
        if z < 1:
            flipped_side_ratios.append(1/z)
        else:
            flipped_side_ratios.append(z)
    return flipped_side_ratios
    
    

def mesh_volume_ratio(mesh,
                     bbox_type="bounding_box_oriented",
                      verbose=False,
                     ):
    """
    compute the ratio of the bounding box to the 
    volume of the mesh
    """
    
    if bbox_type == "bounding_box_oriented":
        bbox_volume = mesh.bounding_box_oriented.volume
    elif bbox_type == "bounding_box":
        bbox_volume = mesh.bounding_box.volume
    else:
        raise Exception(f"Unimplementing bbox_type: {bounding_box}")
        
    curr_mesh_volume = tu.mesh_volume(mesh) 
    
    
    ratio_val = bbox_volume/curr_mesh_volume
    
    if verbose:
        print(f"bbox_volume (using type {bounding_box}) = {bbox_volume}")
        print(f"curr_mesh_volume = {curr_mesh_volume}")
        print(f"volume ratio = {bounding_box}")
    
    return ratio_val

def plot_segmentation(meshes,cgal_info,
                     mesh_alpha = 1):
    print(f"Segmentation Info:")
    for j,(m,c) in enumerate(zip(meshes,cgal_info)):
        print(f"Mesh {j}: {m} ({c})")
    ipvu.plot_objects(meshes=meshes,
                     meshes_colors="random",
                     mesh_alpha=mesh_alpha)
    
def closest_split_to_coordinate(mesh,
                               coordinate,
                               plot_split=False,
                                plot_closest_mesh=False,
                               significance_threshold=0,
                               connectivity="faces",
                               verbose=False,):
    """
    To run the mesh splitting and then to 
    choose the closest split mesh to a coordinate
    
    """
    sig_pieces = tu.split_significant_pieces(mesh,
                            significance_threshold=20,
                            connectivity="faces",
                           )
    if plot_split:
        print(f"Mesh Split with significance_threshold = {significance_threshold}")
        ipvu.plot_objects(meshes=sig_pieces,
                         meshes_colors="random",
                         mesh_alpha=1)
    
    closest_mesh_idx = tu.closest_mesh_to_coordinate(sig_pieces,
                             coordinate=coordinate,
                             return_mesh=False)
    
    potential_webbing_mesh = sig_pieces[closest_mesh_idx]
    non_webbing_meshes = np.array(sig_pieces)[np.arange(len(sig_pieces)) != closest_mesh_idx]

    if verbose:
        print(f"Winning Mesh idx = {closest_mesh_idx}, mesh = {potential_webbing_mesh}")

    if plot_closest_mesh:
        ipvu.plot_objects(main_mesh=potential_webbing_mesh,
                      main_mesh_alpha=1,
                        meshes=non_webbing_meshes,
                     meshes_colors="red",
                     mesh_alpha=1)
    
    return potential_webbing_mesh
    
def closest_segmentation_to_coordinate(mesh,
                                       coordinate,
                                       clusters=3,
                                       smoothness=0.2,
                                       plot_segmentation=False,
                                       plot_closest_mesh=False,
                                       return_cgal=False,
                                      verbose=False,
                                      mesh_segmentation=None,
                                      mesh_segmentation_cdfs=None):
    """
    Purpose: To run a mesh segmentation and then 
    choose the segment that is closest to a coordinate
    
    """

    # Run segmentation algorithm to find the webbing
    if mesh_segmentation is None or mesh_segmentation_cdfs is None:
        mesh_segs, cgal_info = tu.mesh_segmentation(mesh,
                            clusters=clusters,
                            smoothness=smoothness)
    else:
        mesh_segs = mesh_segmentation
        cgal_info = mesh_segmentation_cdfs
        
        
    if plot_segmentation:
        tu.plot_segmentation(mesh_segs,cgal_info)

    closest_web_idx = tu.closest_mesh_to_coordinate(mesh_segs,
                                 coordinate=coordinate,
                                distance_method = "min_distance_and_bbox_center",
                                 return_mesh=False)

    web_mesh = mesh_segs[closest_web_idx]
    web_mesh_cdf = cgal_info[closest_web_idx]
    
    if verbose:
        print(f"Winning mesh idx = {closest_web_idx}, mesh = {web_mesh}, cdf = {web_mesh_cdf}")

    webless_meshes = np.array(mesh_segs)[np.arange(len(mesh_segs)) != closest_web_idx]

    #find the mesh that is closest to the connecting point

    if plot_closest_mesh:
        ipvu.plot_objects(main_mesh=web_mesh,
                          main_mesh_alpha=1,
                            meshes=webless_meshes,
                         meshes_colors="red",
                         mesh_alpha=1)
        
    if return_cgal:
        return web_mesh,web_mesh_cdf
    else:
        return web_mesh
    
def mesh_overlap_with_restriction_mesh(mesh,
                               restriction_mesh,
                               size_measure = "faces",
                            match_threshold = 0.001,
                                verbose=False,
                                      restriction_mesh_kd = None):
    """
    Purpose: To find what percentage of a mesh matches a mesh
    that is restriction mesh
    
    """
    m= mesh

    original_size = tu.mesh_size(m,size_measure)

    m_with_restrict = tu.original_mesh_faces_map(restriction_mesh,
                                                 m,
                               original_mesh_kdtree=restriction_mesh_kd,
                               match_threshold = match_threshold,
                               exact_match = False,
                               return_mesh = True
                              )

    if nu.is_array_like(m_with_restrict):
        if verbose:
            print(f"Mesh {m}: No Overlap with restriction")
        ratio = 0
    else:
        restricted_size = tu.mesh_size(m_with_restrict,size_measure)
        ratio = restricted_size/original_size
        if verbose:
            print(f"Mesh {m}: original_size = {original_size}, restricted_size = {restricted_size}, ratio = {ratio}")

    return ratio

def restrict_mesh_list_by_mesh(mesh_list,
                              restriction_mesh,
                               percentage_threshold=0.6,
                               size_measure = "faces",
                            match_threshold = 0.001,
                                verbose=False,
                              return_meshes=False,
                              return_under_threshold=False):
    """
    Purplse: To restrict a mesh list by the 
    """
    restr_kd = KDTree(restriction_mesh.triangles_center)

    mesh_match_list = []
    mesh_list_ratios = np.array([tu.mesh_overlap_with_restriction_mesh(m,
                                      restriction_mesh,
                                    size_measure=size_measure,
                                    match_threshold=match_threshold,
                                    restriction_mesh_kd = restr_kd) for m in mesh_list])
    mesh_match_idx = np.where(mesh_list_ratios>percentage_threshold)[0]
    
    if return_under_threshold:
        mesh_match_idx = np.delete(np.arange(len(mesh_list)),mesh_match_idx)

    if return_meshes:
        return np.array(mesh_list)[mesh_match_idx]
    else:
        return mesh_match_idx
    
    
    
def min_cut_to_partition_mesh_vertices(mesh,
                                      source_coordinates,
                                      sink_coordinates,
                                       plot_source_sink_vertices= False,
                                       plot_cut_vertices = False,
                                      verbose = False,
                                      return_edge_midpoint = True):
    """
    Purpose: To find the vertex points would need to cut to 
    seperate groups of points on a mesh (or close to a mesh)
    so that the groups are on seperate component meshes

    Pseudocode: 
    1) Create a kdtree of the mesh vertices
    and map the source/sink coordinate to vertex indices on the mesh
    2) Get the mesh adjacency graph
    3) Find the min_cut edges on the adajacency graph, if None return None
    4) Return either vertex coordinates of all nodes in edges,
    or the coordinate of the middle point in the edges
    
    Ex: 
    source_coordinates = [skeleton_offset_points[0],skeleton_offset_points[1]]
    sink_coordinates = [skeleton_offset_points[2],skeleton_offset_points[3]]
    curr_output = tu.min_cut_to_partition_mesh_vertices(mesh_inter,
                                          source_coordinates,
                                          sink_coordinates,
                                           plot_source_sink_vertices= True,
                                          verbose = True,
                                          return_edge_midpoint = True,
                                                    plot_cut_vertices = True)
    curr_output
    """
    mesh
    source_coordinates
    sink_coordinates 

    
    
    mesh_inter_kd = KDTree(mesh.vertices)
    dist,source_vert_idx = mesh_inter_kd.query(np.array(source_coordinates).reshape(-1,3))
    dist,sink_vert_idx = mesh_inter_kd.query(np.array(sink_coordinates).reshape(-1,3))
    
    if len(np.unique(source_vert_idx)) < 2 or len(np.unique(sink_vert_idx)) < 2:
        if verbose: 
            print(f"Could not find a cut edge because not different nodes")
        return None

    if plot_source_sink_vertices:
        sink_vertices = mesh.vertices[np.array(sink_vert_idx)]
        source_vertices = mesh.vertices[np.array(source_vert_idx)]
        
        sink_color = "red"
        source_color = "aqua"
        print(f"Source = {source_color}, Sink = {sink_color}")
        colors = [sink_color,source_color]
        ipvu.plot_objects(main_mesh=mesh,
                         scatters=[sink_vertices,source_vertices],
                         scatters_colors=colors)

    G = mesh.vertex_adjacency_graph

    G_cut_edges = xu.min_cut_to_partition_node_groups(G,source_nodes=source_vert_idx,
                                                      sink_nodes = sink_vert_idx,
                                                      verbose=verbose)
    
    #print(f"G_cut_edges = {G_cut_edges}")
    if G_cut_edges is None:
        if verbose: 
            print(f"Could not find a cut edge for the source and sink coordinates: returning None")
        return None
    
    G_cut_edges_vertices = mesh.vertices[G_cut_edges]
    
    if return_edge_midpoint:
        return_value = np.mean(G_cut_edges_vertices,axis=1)
    else:
        return_value = G_cut_edges_vertices.reshape(-1,3)
        
    if verbose:
        print(f"# 0f cut points = {len(return_value)}")
        
    if plot_cut_vertices:
        ipvu.plot_objects(mesh,
                          scatters=[return_value],
                          scatters_colors="red")
        
    return return_value

def coordinates_to_enclosing_sphere_center_radius(coordinates,
                    verbose = False):
    """
    Purpose: to get the volume that would be needed to 
    encapsulate a set of points
    
    Pseudocode: 
    1) get the mean of the points
    2) Return the max distance from the mean

    """
    center = np.mean(coordinates,axis=0)
    radius = np.max(np.linalg.norm(coordinates-center))
    
    if verbose: 
        print(f"center = {center}, radius = {radius}")
    
    return (center,radius)

def coordinates_to_enclosing_sphere(coordinates,
                    verbose = False):
    return tu.sphere_mesh(*coordinates_to_enclosing_sphere_center_radius(coordinates))

def coordinates_to_enclosing_sphere_volume(coordinates,
                    verbose = False):
    """
    Purpose: to get the volume that would be needed to 
    encapsulate a set of points
    
    Pseudocode: 
    1) get the mean of the points
    2) Return the max distance from the mean

    """
    center,radius = coordinates_to_enclosing_sphere_center_radius(coordinates)
    return 4/3*np.pi*(radius)**3

def coordinates_to_bounding_box(coordinates,oriented=True):
    t = trimesh.Trimesh()
    t.vertices = coordinates
    return tu.bounding_box(t,oriented=oriented)

# -------- 6/7: Used for the synapse filtering ----------#

def mesh_to_kdtree(mesh):
    return KDTree(mesh.triangles_center)

def valid_coordiantes_mapped_to_mesh(mesh,
                                     coordinates,
                                     mesh_kd = None,
                                     mapping_threshold = 500,
                                    original_mesh = None,
                                    original_mesh_kdtree = None,
                                    original_mesh_faces = None,
                                     return_idx = True,
                                     return_errors = False,
                                    verbose = False):
    """
    Purpose: To determine if coordinates are within
    a certain mapping distance to a mesh (to determine if they are valid or not)

    If an original mesh is specified then if a coordinate maps
    to the original mesh but not the main mesh specified then it is not valid
    """

    mesh_errored_syn_idx = []
    if original_mesh is None:
        if verbose:
            print(f"Not using original mesh for invalidaiton of coordinates")
            
        if mesh_kd is None:
            mesh_kd = tu.mesh_to_kdtree(mesh)
        dist,closest_face = mesh_kd.query(coordinates)
        mesh_errored_syn_idx = np.where(dist>mapping_threshold)[0]
        
    else:
        if verbose:
            print(f"Using original mesh for invalidaiton of coordinates")
            
        if original_mesh_kdtree is None:
            if verbose:
                print(f"calculating original_mesh_kd because None")
            original_mesh_kdtree = tu.mesh_to_kdtree(original_mesh)

        if original_mesh_faces is None:
            if verbose:
                print(f"calculating original_mesh_faces because None")
            original_mesh_faces = tu.original_mesh_faces_map(original_mesh,
                                                                mesh,
                                                                exact_match=True,
                                                                original_mesh_kdtree=original_mesh_kdtree)
            
        neuron_mesh_labels = np.zeros(len(original_mesh.triangles_center))
        neuron_mesh_labels[original_mesh_faces] = 1
        
        dist,closest_face = original_mesh_kdtree.query(coordinates)

        closest_face_labels = neuron_mesh_labels[closest_face]

        mesh_errored_syn_idx = np.where((dist>mapping_threshold) | ((closest_face_labels==0)))[0]
        
    if not return_errors:
        mesh_errored_syn_idx
        mesh_errored_syn_idx = np.delete(np.arange(len(coordinates)),mesh_errored_syn_idx)
    
    if not return_idx:
        mesh_errored_syn_idx = coordinaes[mesh_errored_syn_idx]
        
    return mesh_errored_syn_idx

def mesh_kdtree_face(mesh):
    original_mesh_midpoints = mesh.triangles_center
    original_mesh_kdtree = KDTree(original_mesh_midpoints)
    return original_mesh_kdtree

def mesh_kdtree_vertices(mesh):
    original_mesh_midpoints = mesh.vertices
    original_mesh_kdtree = KDTree(original_mesh_midpoints)
    return original_mesh_kdtree


# -------- trying to help differentiating soma mergers from non soma mergers ------
default_percentile = 70
default_percentage = 70

def surface_area(mesh):
    return mesh.area

def default_mesh_stats_to_run(ray_trace_perc_options = (30,50,70,90,95),
                             interpercentile_options = (30,50,70,90,95),
                             center_to_width_ratio_options = (30,50,70,90,95)):
    basic_funcs = [dict(name="surface_area",
        function=tu.surface_area,
                        divisor = 1_000_000,
                       ),
     dict(name="volume",
         function=tu.mesh_volume,
          divisor = 1_000_000_000
         ),
    dict(name="sa_to_volume",
        function = tu.surface_area_to_volume),
     dict(name="n_faces",
         function = tu.mesh_size,
         kwargs = dict(size_type="faces"),
        ),]

    ray_trace_funcs = [dict(name=f"ray_trace_percentile_{k}",
                           function = tu.mesh_size,
                           kwargs = dict(size_type = "ray_trace_percentile",
                                        percentile = k)) for k in ray_trace_perc_options]
    
    interpercentile_funcs = gu.combine_list_of_lists([[
                                dict(name=f"ipr_x_{k}",
                                   function=tu.interpercentile_range_face_midpoints_x,
                                   kwargs = dict(percentage = k)),
                                dict(name=f"ipr_y_{k}",
                                   function=tu.interpercentile_range_face_midpoints_y,
                                   kwargs = dict(percentage = k)),
                            dict(name=f"ipr_z_{k}",
                                               function=tu.interpercentile_range_face_midpoints_z,
                                               kwargs = dict(percentage = k)),
                            dict(name=f"ipr_volume_{k}",
                                   function=tu.interpercentile_range_face_midpoints_volume,
                                   kwargs = dict(percentage = k))] for k in interpercentile_options ])
    eigenvector_sizes = [dict(name=f"ipr_eig_xz_max_{k}",
                             function = tu.ipr_largest_eigenvector_xz,
                             kwargs = dict(percentage = k)) for k in interpercentile_options]
    mesh_center_funcs = gu.combine_list_of_lists([[dict(name=f"center_to_width_{k}",
                             function = tu.ratio_closest_face_to_mesh_center_distance_to_width,
                             kwargs = dict(percentile = k)),
                            
                                                   dict(name=f"center_to_width_farthest_{k}",
                             function = tu.ratio_farthest_face_to_mesh_center_distance_to_width,
                             kwargs = dict(percentile = k)),
                                                   
                             dict(name=f"ipr_eig_xz_to_width_{k}",
                                 function = tu.ratio_ipr_eigenvector_xz_diff_to_width,
                                 kwargs=dict(percentage=k))] for k in center_to_width_ratio_options])

    
    return gu.combine_list_of_lists([basic_funcs, ray_trace_funcs,interpercentile_funcs,mesh_center_funcs,eigenvector_sizes])
def mesh_stats(mesh,stats_dicts=None,**kwargs):
    if stats_dicts is None:
        stats_dicts = tu.default_mesh_stats_to_run(**kwargs)
    output_dict = dict()
    for st_dict in stats_dicts:
        kwargs = st_dict.get("kwargs",dict())
        divisor = st_dict.get("divisor",1)
        output_dict[st_dict["name"]] = st_dict["function"](mesh,**kwargs)/divisor
    return output_dict

default_percentage = 70
def interpercentile_range_face_midpoints(mesh,
                                         percentage=default_percentage,
                                         verbose = False):
    """
    purpose: To find the 
    
    
    """
    return nu.interpercentile_range(mesh.triangles_center,
                             percentage,
                             verbose = verbose,
                                axis = 0)

def interpercentile_range_face_midpoints_x(mesh,
                                           percentage=default_percentage,
                                           verbose = False):
    return interpercentile_range_face_midpoints(mesh,percentage,verbose = verbose)[0]

def interpercentile_range_face_midpoints_y(mesh,
                                           percentage=default_percentage,
                                           verbose = False):
    return interpercentile_range_face_midpoints(mesh,percentage,verbose = verbose)[1]

def interpercentile_range_face_midpoints_z(mesh,
                                           percentage=default_percentage,
                                           verbose = False):
    return interpercentile_range_face_midpoints(mesh,percentage,verbose = verbose)[2]

def interpercentile_range_face_midpoints_volume(mesh,
                                           percentage=default_percentage,
                                           verbose = False):
    return np.prod(interpercentile_range_face_midpoints(mesh,percentage,verbose = verbose))

def closest_face_to_mesh_center_distance(mesh):
    curr_mesh_center = tu.mesh_center_weighted_face_midpoints(mesh).reshape(-1,3)
    return tu.closest_face_to_coordinate_distance(mesh,
                                      curr_mesh_center)

def farthest_face_to_mesh_center_distance(mesh):
    curr_mesh_center = tu.mesh_center_weighted_face_midpoints(mesh).reshape(-1,3)
    return tu.farthest_face_to_coordinate_distance(mesh,
                                      curr_mesh_center)

def ratio_mesh_stat_to_width(mesh,
                             mesh_stat_function,
                            width_func_name = "ray_trace_percentile",
                            percentile = default_percentile,
                           verbose = False,
                             closest_distance = None,
                            **kwargs):
    """
    Purpose: To measure the ratio between the distance between the closest
    mesh face and the overall width of the mesh
    """
    if closest_distance is None:
        closest_distance = mesh_stat_function(mesh,**kwargs)
        
    curr_ray_trace = tu.mesh_size(mesh,
             size_type=width_func_name,
            percentile = percentile)
    ratio = closest_distance/curr_ray_trace
    if verbose:
        print(f"{mesh_stat_function.__name__} = {closest_distance}")
        print(f"curr_ray_trace = {curr_ray_trace}")
        print(f"ratio = {ratio}")
    return ratio


def ratio_closest_face_to_mesh_center_distance_to_width(mesh,
                            width_func_name = "ray_trace_percentile",
                            percentile = default_percentile,
                           verbose = False):
    return tu.ratio_mesh_stat_to_width(mesh,
                             mesh_stat_function=tu.closest_face_to_mesh_center_distance,
                            width_func_name = width_func_name,
                            percentile = percentile,
                           verbose = verbose)

def ratio_farthest_face_to_mesh_center_distance_to_width(mesh,
                            width_func_name = "ray_trace_percentile",
                            percentile = default_percentile,
                           verbose = False):
    return tu.ratio_mesh_stat_to_width(mesh,
                             mesh_stat_function=tu.farthest_face_to_mesh_center_distance,
                            width_func_name = width_func_name,
                            percentile = percentile,
                           verbose = verbose)


    
    
def face_midpoints_x_z(mesh):
    """
    Application: Can just analyze the data that is not going top to bottom in microns volume
    
    """
    return mesh.triangles_center[:,[0,2]]


def ipr_largest_eigenvector_xy(mesh,
                              percentage = default_percentage,
                              verbose = False):
    new_data = dru.largest_eigenvector_proj(tu.face_midpoints_x_z(mesh))
    return nu.interpercentile_range(new_data,percentage,verbose = verbose)


def ipr_largest_eigenvector_xz(mesh,
                              percentage = default_percentage,
                              verbose = False):
    new_data = dru.largest_eigenvector_proj(tu.face_midpoints_x_z(mesh))
    return nu.interpercentile_range(new_data,percentage,verbose = verbose)

def ipr_second_largest_eigenvector_xz(mesh,
                              percentage = default_percentage,
                              verbose = False):
    new_data = dru.second_largest_eigenvector_proj(tu.face_midpoints_x_z(mesh))
    return nu.interpercentile_range(new_data,percentage,verbose = verbose)
    
def ipr_first_second_largest_eigenvector_xz_diff(mesh,
                              percentage = default_percentage,
                              verbose = False
                                                ):
    return (tu.ipr_largest_eigenvector_xz(mesh,
                              percentage = percentage,
                              verbose = verbose) - 
            tu.ipr_second_largest_eigenvector_xz(mesh,
                              percentage = percentage,
                              verbose = verbose))

def ratio_ipr_eigenvector_xz_diff_to_width(mesh,
                            width_func_name = "ray_trace_percentile",
                            percentile = default_percentile,
                           verbose = False,
                                          **kwargs):
    return tu.ratio_mesh_stat_to_width(mesh,
                             mesh_stat_function=tu.ipr_first_second_largest_eigenvector_xz_diff,
                            width_func_name = width_func_name,
                            percentile = percentile,
                           verbose = verbose,
                                      **kwargs)

def surface_area_to_volume(mesh,
                          surface_area_divisor = 1_000_000,
                          volume_divisor = 1_000_000_000,
                           verbose = False,
                          ):
    curr_area = tu.surface_area(mesh)/surface_area_divisor
    curr_vol = tu.mesh_volume(mesh)/volume_divisor
    if curr_vol <= 0:
        if verbose:
            print(f"Volume 0")
        return 0
    
    surface_area_to_vol_ratio = curr_area/ curr_vol
    
    if verbose:
        print(f"surface area = {curr_area}, voluem = {curr_vol}, surface_area_to_vol_ratio = {surface_area_to_vol_ratio}")
    return surface_area_to_vol_ratio


def mesh_centered_at_origin(mesh):
    """
    To move a mesh to where the mesh 
    center is at the origina
    """
    mesh.vertices = mesh.vertices - tu.mesh_center_vertex_average(mesh)
    return mesh

centered_at_origin = mesh_centered_at_origin
center_at_origin = mesh_centered_at_origin

def rotate_mesh_from_matrix(mesh,matrix):
    new_mesh = mesh.copy()
    new_mesh.vertices = new_mesh.vertices @ matrix
    return new_mesh


def faces_closer_than_distance_of_coordinates(
    mesh,
    coordinate,
    distance_threshold,
    verbose = False,
    closer_than = True,
    return_mesh = False,
    plot = False,
    ):
    """
    Purpose: To get the faces of a 
    mesh that are within a certain distance
    of a coordinate/coordinates

    Pseudocode: 
    1) Build KDTree from coordinates
    2) Query the mesh triangles against KDTree
    3) Find the meshes that are closer/farther than threshold
    4) Return face ids (or submesh)
    
    Ex: 
    
    rest_mesh = tu.faces_closer_than_distance_of_coordinates(
    mesh = branch.mesh,
    coordinate = curr_point,
    distance_threshold = 5_000,
    verbose = True,
    return_mesh=True
    )

    ipvu.plot_objects(rest_mesh)
    """
    

    from pykdtree.kdtree import KDTree

    coordinate = np.array(coordinate).reshape(-1,3)
    kd = KDTree(coordinate)
    dist, _ = kd.query(mesh.triangles_center)

    if closer_than:
        face_id = np.where(dist < distance_threshold )[0]
    else:
        face_id = np.where(dist > distance_threshold )[0]

    if verbose:
        print(f"# of faces within {distance_threshold} = {len(face_id)}/{len(mesh.faces)}")

        
    if plot:
        submesh = mesh.submesh([face_id],append=True)
        ipvu.plot_objects(
            mesh,
            meshes = [submesh],
            meshes_colors = ["red"],
            scatters = [coordinate],
        )
    if return_mesh:
        return mesh.submesh([face_id],append=True)
    else:
        return face_id

def faces_farther_than_distance_of_coordinates(
    mesh,
    coordinate,
    distance_threshold,
    verbose = False,
    return_mesh = False,
    plot=False,
    ):
    
    return faces_closer_than_distance_of_coordinates(
    mesh,
    coordinate,
    distance_threshold,
    verbose = verbose,
    closer_than = False,
    return_mesh = return_mesh,
        plot=plot,
    )

def n_faces(mesh):
    try: 
        return len(mesh.faces)
    except:
        return len(mesh)
    
    
def overlapping_attribute(
    mesh1,
    mesh2,
    attribute_name,
    verbose=False,
    return_idx = False):
    """
    Will return the attributes
    that are overlapping for 2  meshes
    
    if return idx will return idx of first
    """
    global_start = time.time()
    original_mesh_midpoints = getattr(mesh1,attribute_name)
    submesh_midpoints = getattr(mesh2,attribute_name)
    
    #1) Put the submesh face midpoints into a KDTree
    submesh_mesh_kdtree = KDTree(submesh_midpoints)
    #2) Query the fae midpoints of submesh against KDTree
    distances,closest_node = submesh_mesh_kdtree.query(original_mesh_midpoints)
    
    match_idx = np.where(distances==0)[0]
    
    if return_idx:
        return match_idx
    else:
        return original_mesh_midpoints[match_idx]
    
def overlapping_vertices(
    mesh1,
    mesh2,
    verbose=False,
    return_idx = False):
    
    return overlapping_attribute(
    mesh1,
    mesh2,
    attribute_name = "vertices",
    verbose=verbose,
    return_idx = return_idx)


def overlapping_faces(
    mesh1,
    mesh2,
    verbose=False,
    return_idx = False):
    
    return overlapping_attribute(
    mesh1,
    mesh2,
    attribute_name = "triangles_center",
    verbose=verbose,
    return_idx = return_idx)


def radius_sphere_from_volume(volume):
    """
    Purpose: To calculate the radius
    from the volume assuming spherical
    
    V = (4/3)*np.pi*(r**3)
    (V*(3/(4*np.pi)))**(1/3)
    """
    return (volume*(3/(4*np.pi)))**(1/3)


def mesh_list_distance_connectivity(
    meshes,
    pairs_to_test=None,
    max_distance = np.inf,
    return_G = False,
    verbose = False
    ):
    """
    Purpose: To find the distances
    between all meshes in a list and represent
    them as edges (can passback as a graph if need be)
    
    Pseudocode: 
    For all possile combinations of meshes (or those prescribed)
    1) Find the distance between the meshes
    2) 
    """
    if pairs_to_test is None:
        pairs_to_test = nu.all_unique_choose_2_combinations(np.arange(len(meshes)))
        
    total_edges = []
    for i,j in pairs_to_test:
        if i > j:
            continue
        dist = tu.closest_distance_between_meshes(
            meshes[i],
            meshes[j],
            )
        if verbose:
            print(f"Mesh {i},{j} dist = {dist}")
            
        if dist < max_distance:
            if verbose:
                print(f"(above threshold: {dist < max_distance})")
            total_edges.append([i,j,dist])
            
    
    if verbose:
        print(f"Total edges = {len(total_edges)}")
        
    if return_G:
        G = nx.Graph()
        G.add_weighted_edges_from(total_edges)
        return G
    else:
        return np.array(total_edges)
    
    
def closest_mesh_to_mesh(
    mesh,
    meshes,
    return_closest_distance = False,
    return_closest_vertex_on_mesh = False,
    verbose = True,
    ):
    
    """
    Purpose: To find the index of the closest mesh 
    out of a list of meshes and the closest
    distance and vertex
    """
    piece = mesh

    piece_kd = KDTree(piece.vertices)

    nst_closest_dists = []
    nst_closest_vertex = []

    for s in meshes:
        dists,closest_vert = piece_kd.query(s.vertices)
        min_idx = np.argmin(dists)
        nst_closest_dists.append(dists[min_idx])
        nst_closest_vertex.append(piece.vertices[closest_vert[min_idx]])

    closest_soma_idx = np.argmin(nst_closest_dists)
    
    if verbose:
        print(f"closest_dists = {nst_closest_dists}")
        print(f"closest_vertex = {nst_closest_vertex}")
        print(f"closest_soma_idx = {closest_soma_idx}")

    
    if not return_closest_vertex_on_mesh and not return_closest_distance:
        return closest_soma_idx
    return_value = [closest_soma_idx]
    
    if return_closest_distance:
        closest_dist = nst_closest_dists[closest_soma_idx]
        return_value.append(closest_dist)
    
    if return_closest_vertex_on_mesh:
        closest_vertex = nst_closest_vertex[closest_soma_idx]
        return_value.append(closest_vertex)
        
    return return_value

width_ray_trace_default = 0
def width_ray_trace_perc(
    mesh,
    percentile = 50,
    verbose = False,
    default_value_if_empty = width_ray_trace_default,
    ray_inter = None,):
    return tu.width_ray_trace_perc_of_submesh(
          mesh,
        percentile = percentile,
        verbose = verbose,
        default_value_if_empty = default_value_if_empty,
        ray_inter = ray_inter,
    )

width_ray_trace_median = width_ray_trace_perc

def width_ray_trace_perc_of_submesh(
    mesh,
    submesh=None,
    submesh_face_idx = None,
    percentile = 50,
    verbose = False,
    default_value_if_empty = width_ray_trace_default,
    ray_inter = None,
    ):
    
    if ray_inter is None:
        ray_inter = ray_pyembree.RayMeshIntersector(mesh)
        
    if submesh_face_idx is None:
        if submesh is None:
            submesh_face_idx = np.arange(len(mesh.faces))
        else:
            submesh_face_idx = tu.original_mesh_faces_map(mesh,subemsh)
        
    curr_width_distances = tu.ray_trace_distance(
        mesh=mesh,
        face_inds=submesh_face_idx,
        ray_inter=ray_inter
    )
    
    filtered_widths = curr_width_distances[curr_width_distances>0]
    if len(filtered_widths)>0:
        final_width = np.percentile(filtered_widths,percentile)
    else:
        final_width = default_value_if_empty
        
    if verbose:
        print(f"final width (# of submesh faces = {len(submesh_face_idx)} ) = {final_width}")
    return final_width

width_of_submesh = width_ray_trace_perc_of_submesh
    
def widths_of_submeshes(
    mesh,
    submeshes=None,
    default_value_if_empty = 100000,
    ray_inter = None,
    submeshes_face_idx = None,
    percentile = 50,
    verbose = False,
    ):
    """
    Purpose: To find the widths of submeshes in relation 
    to a larger mesh
    """

    if ray_inter is None:
        ray_inter = ray_pyembree.RayMeshIntersector(mesh)

    submeshes_widths = []
    for lm in submeshes:
        face_indices_leftover_0 = tu.original_mesh_faces_map(mesh,lm)
        curr_width_distances = tu.ray_trace_distance(mesh=mesh,
                                  face_inds=face_indices_leftover_0,
                                                         ray_inter=ray_inter
                                )
        filtered_widths = curr_width_distances[curr_width_distances>0]
        if len(filtered_widths)>0:
            submeshes_widths.append(np.percentile(filtered_widths,percentile))
        else:
            submeshes_widths.append(default_value_if_empty)

    submeshes_widths = np.array(submeshes_widths)
    if verbose:
        print(f"submeshes_widths = {submeshes_widths}")
        
    return submeshes_widths
    
def mesh_bbox_contains_skeleton(mesh,
                                skeleton,
                               perc_contained_threshold=1,
                               verbose = False):
    """
    Purpose: To determine if a mesh bounding box contains all
    the points of a skeleton
    
    """
    sk_points = np.unique(skeleton.reshape(-1,3),axis=0)
    contains_map = mesh.bounding_box.contains(sk_points.reshape(-1,3))

    n_contained = np.sum(contains_map)
    total_points = len(contains_map)
    perc_contained = n_contained/total_points
    if verbose:
        print(f"{n_contained} out of {total_points} total (perc = {perc_contained})")
    
    return perc_contained >= perc_contained_threshold
    
    
def submesh(mesh,face_idx,always_return_mesh=True):
    new_submesh = mesh.submesh([list(face_idx)],only_watertight=False,append=True)
    if not tu.is_mesh(new_submesh) and always_return_mesh:
        return tu.empty_mesh()
    else:
        return new_submesh
    
def bbox_intersect_test(
    mesh_1,
    mesh_2,
    plot = True,
    mesh_1_color = "green",
    mesh_2_color = "black",
    mesh_alpha = 1,
    mesh_1_bbox_color = "red",
    mesh_2_bbox_color = "blue",
    bbox_alpha = 0.2,
    verbose = False,
    ):

    """
    Purpose: To determine if a mesh bounding box intersects
    another bounding box

    Pseducode: 
    1) Plot the bounding boxes of the corner and the mesh
    2) Determine whether the two bounding boxes intersect
    
    Ex: 
    bbox_intersect_test(
    mesh_1=segment_mesh,
    mesh_2=ease_bounding_box,
    plot = True,
    verbose = True,
    )

    """

    meshes_to_plot = []
    meshes_to_plot_colors = []
    meshes_alpha = []
    for j,(m,m_color) in enumerate(zip([mesh_1,mesh_2],[mesh_1_color,mesh_2_color])):
        if tu.is_mesh(m):
            meshes_to_plot.append(m)
            meshes_to_plot_colors.append(m_color)
            meshes_alpha.append(mesh_alpha)
        else:
            if verbose:
                print(f"mesh_{j+1} is not mesh")

    mesh1_bbox = tu.bounding_box(mesh_1)
    mesh2_bbox = tu.bounding_box(mesh_2)

    if plot:
        ipvu.plot_objects(
            meshes=meshes_to_plot + [mesh1_bbox,mesh2_bbox],
            meshes_colors=meshes_to_plot_colors + [mesh_1_bbox_color,mesh_2_bbox_color],
            mesh_alpha=meshes_alpha + [bbox_alpha]*2,
            axis_box_off=False)


    intersect_result = nu.bbox_intersect_test_from_corners(
        tu.bounding_box_corners(mesh1_bbox),
        tu.bounding_box_corners(mesh2_bbox),
        verbose = verbose
    )

    if verbose:
        print(f"intersect_result = {intersect_result}")
        
    return intersect_result


# --------------- 2/9 -----------------
def farthest_coordinate_to_faces(
    mesh,
    coordinates,
    return_distance = False,
    verbose = False,
    plot = False,
    ):
    """
    To find the coordinate who has the 
    farthest closest distance to a mesh
    
    Ex: 
    import neurd
    
    branch_obj = neuron_obj[0][0]
    neurd.plot_branch_spines(branch_obj)
    farthest_coord = farthest_coordinate_to_faces(
        branch_obj.mesh,
        branch_obj.skeleton,
        return_distance = False,
        verbose = True
    )

    ipvu.plot_objects(
    branch_obj.mesh,
        branch_obj.skeleton,
        scatters=[farthest_coord],
        scatter_size=2
    )
    """
    coordinates = np.array(coordinates).reshape(-1,3)
    m_tree = tu.mesh_kdtree_face(mesh)
    dist,closest_face = m_tree.query(coordinates)
    farthest_idx = np.argmax(dist)
    farthest_coordinate = coordinates[farthest_idx]
    farthest_dist = dist[farthest_idx]
    
    if verbose:
        print(f"Farthest coordinate = {farthest_coordinate},"
              f" closest face to farthest coordinate = {farthest_idx}")
        print(f"Farthest distance = {farthest_dist}")
    
    if plot:
        farthest_color = "blue"
        print(f"Closest coordinate color = {farthest_color}")
        ipvu.plot_objects(
            mesh,
            scatters=[coordinates,farthest_coordinate],
            scatter_size=[0.2,2],
            scatters_colors = ["red","blue"]
        )
    
    if return_distance:
        return farthest_dist
    else:
        return farthest_coordinate

def largest_conn_comp(
    mesh,
    connectivity='vertices',
    return_face_indices = False,):
    """
    Purpose: To return the largest connected 
    component of the mesh
    """
    return_meshes,return_idx = tu.split_significant_pieces(
    mesh,
    significance_threshold=-1,
    connectivity=connectivity,
    return_face_indices=True,
    )
    
    if return_face_indices:
        return return_meshes[0],return_idx[0]
    else:
        return return_meshes[0]


def closest_n_attributes_to_coordinate(
    mesh,
    coordinate,
    n,
    attribute,
    return_idx = False,
    plot = False):
    
    """
    Ex: 
    tu.closest_n_attributes_to_coordinate(
        branch_obj.mesh,
        coordinate = branch_obj.mesh_center,
        attribute = "vertices",
        n = 5,
        plot = True)
    
    """
    
    coordinate = np.array(coordinate)
    
    kd = KDTree(coordinate.reshape(-1,3))
    curr_attrs = getattr(mesh,attribute)
    dist,_ = kd.query(curr_attrs.reshape(-1,3))
    closest_idx = np.argsort(dist)[:n]
    closest_attrs = curr_attrs[closest_idx]
    
    if plot: 
        ipvu.plot_objects(
            mesh,
        scatters = [coordinate,closest_attrs],
        scatters_colors=["red","blue"],
        scatter_size = 1)
    
    if return_idx:
        return closest_idx
    else:
        return closest_attrs
    
def translate_mesh(
    mesh,
    translation=None,
    new_center = None,
    inplace = False):
    
    if not inplace:
        mesh = copy.deepcopy(mesh)
    
    if translation is None:
        translation = new_center - tu.mesh_center_vertex_average(mesh)
        
    mesh.vertices += translation
    return mesh

def scatter_mesh_with_radius(array,radius):
    """
    Purpose: to generate a mesh of spheres at certain coordinates with certain size

    """
    if not nu.is_array_like(radius):
        radius = [radius]*len(array)
    
    total_mesh = tu.combine_meshes([tu.sphere_mesh(k,r)
                    for k,r in zip(array,radius)])
    return total_mesh

def connected_components_from_mesh(
    mesh,
    return_face_idx_map = False,
    plot=False,
    **kwargs
    ):
    
    return_value= tu.split(
        mesh,
        return_components = False,
        return_face_idx_map=return_face_idx_map,
        **kwargs
    )
    
    if return_face_idx_map:
        meshes,face_idx = return_value
    else:
        meshes = return_value
    
    if plot:
        ipvu.plot_objects(
            meshes=meshes,
            meshes_colors = "random"
        )
        
    return return_value

def area(mesh):
    return mesh.area

def skeleton_non_branching_from_mesh(mesh,plot=False):
    """
    Purpose: To generate surface skeletons
    """
    meshes_conn_comp = connected_components_from_mesh(mesh)
    skeleton = sk.stack_skeletons([sk.surface_skeleton(k,plot=plot) for k in meshes_conn_comp])
    return skeleton

def skeletal_length_from_mesh(mesh,plot=False):
    return sk.calculate_skeletal_length(skeleton_non_branching_from_mesh(mesh,plot=plot))

def width_ray(mesh,percentile=50):
    return width_ray_trace_perc(mesh,percentile=percentile)

def overlapping_vertices_from_face_lists(
    mesh,
    face_lists,
    return_idx = False,
    ):
    """
    Purpose: To find the vertices shared
    between two arrays of face_idxs

    Pseudocode: 
    1) Find the intersection of the vertices (after indexing faces into vertices array)
    2) Index the vertices intersection into verteices
    """
    vertices_list = [mesh.faces[k].ravel() for k in face_lists]
    intersect_verts = nu.intersect1d_multi_list(vertices_list)
    if return_idx: 
        return intersect_verts
    else:
        return mesh.vertices[intersect_verts]

def coordinate_on_mesh_mesh_border(
    mesh,
    mesh_border=None,
    overlapping_vertices = None,
    mesh_border_minus_meshes = None,
    meshes_to_minus = None,
    coordinate_method = 'first_coordinate',#"mean",
    verbose = False,
    verbose_time = False,
    return_winning_coordinate_group = False,
    plot= True,
    ):
    """
    Purpose: to find a coordinate on the border between 2 meshes
    
    New method: 
    1) Finds the overlapping vertices, if none then find the closest one
    2) Find the border vertices groups
    3a) If no border vertices then just picks overlapping vertices as winning group
    3b) Picks the border vertices groups which has a closest overall distance to any of border vertices
    """
    if verbose_time:
        st = time.time()
    
    if mesh_border_minus_meshes is None and overlapping_vertices is None:
        if meshes_to_minus is None:
            meshes_to_minus = []
        meshes_to_minus = nu.to_list(meshes_to_minus)
        meshes_to_minus.append(mesh)

        
        mesh_border_minus_meshes = tu.subtract_mesh(
            mesh_border,
            meshes_to_minus,
        )
        
        if verbose_time:
            print(f"Time for subtract_mesh = {time.time() - st}")
            st = time.time()

    #1) Finds the overlapping vertices, if none then find the closest one
#     overlap_verts = nu.intersect2d(
#         mesh.vertices,
#         mesh_border_minus_meshes.vertices,
#     )
    
#     if verbose_time:
#         print(f"Time for overlap_verts option 1 = {time.time() - st}")
#         st = time.time()
    
    if overlapping_vertices is None:
        overlap_verts = tu.overlapping_vertices(
            mesh,
            mesh_border_minus_meshes
        )

        if verbose_time:
            print(f"Time for overlap_verts option 2 = {time.time() - st}")
            st = time.time()

        if len(overlap_verts) == 0:
            if verbose:
                print(f"Using closest mesh vertex for overlap verts")
            overlap_verts = tu.closest_mesh_vertex_to_other_mesh(
                mesh,
                mesh_border_minus_meshes,
                plot = False,
                verbose = False,
            ).reshape(-1,3)

            if verbose_time:
                print(f"Time for closest_mesh_vertex_to_other_mesh = {time.time() - st}")
                st = time.time()
    else:
        
        overlap_verts = overlapping_vertices
        
    if verbose:
        print(f"# of overlap verts = {len(overlap_verts)}")
    #2) Find the border vertices groups
    try:
        vertex_groups = tu.find_border_vertex_groups(
            mesh,
            return_coordinates=True,
        )
        
        if verbose_time:
            print(f"Time for find_border_vertex_groups = {time.time() - st}")
            st = time.time()
    except:
        vertex_groups = []
    
    if len(vertex_groups) == 0:
        coordinate_group = overlap_verts
        winning_coordinate_group = None
    else:
        dist_to_overlap_group = []
        for v_g in vertex_groups:
            dist_to_overlap_group.append(
                np.min([np.linalg.norm(overlap_verts - k) for k in v_g])
            )
            
        winning_idx = np.argmin(dist_to_overlap_group)
        if verbose:
            print(f"dist_to_overlap_group = {dist_to_overlap_group}")
            print(f"Vertex group {winning_idx} was closest at {dist_to_overlap_group[winning_idx]}")
            
        coordinate_group =  vertex_groups[winning_idx]
        winning_coordinate_group = coordinate_group
        
        if verbose_time:
            print(f"Time for winning coordinate = {time.time() - st}")
            st = time.time()
        
    #3) Find average vertices that make up the coordinate    
    if coordinate_method == "mean":
        coordinate = np.mean(coordinate_group,axis = 0)
    else:
        coordinate = coordinate_group[0]
    if verbose:
        print(f"coordinate = {coordinate}")
    
    if plot:
        ipvu.plot_objects(
            mesh_border_minus_meshes,
            meshes = [mesh],
            meshes_colors = ["red"],
            scatters=[coordinate.reshape(-1,3),overlap_verts],
            scatters_colors=["red","blue"],
        )
        
    if return_winning_coordinate_group:
        return coordinate,winning_coordinate_group
    else:
        return coordinate
    
def connected_components_from_face_idx(
    mesh,
    face_idx,
    return_meshes= True,
    verbose = False,
    ):
    
    """
    Purpose: To return the face_idxs
    of the connected components for mesh defined
    by face idx
    """
    submesh = mesh.submesh([face_idx],append=True)
    
    ret_meshes,return_idxs = tu.split(
        mesh = submesh,
        return_components = True
    )
    
    return_idxs = [face_idx[k] for k in return_idxs]
    
    if verbose:
        print(f"# of components = {len(ret_meshes)}")
    
    if return_meshes:
        return return_idxs,ret_meshes
    else:
        return return_idxs
    
def filter_away_connected_comp_in_face_idx_with_minimum_vertex_distance_to_coordinates(
    mesh,
    face_idx,
    coordinates,
    min_distance_threshold = 0.0001,
    verbose = False,
    return_meshes = False,
    plot = True,
    ):
    """
    Purpose: To filter away any connected components
    that have a vertices touching at least one coordinate
    """

    coordinates = np.array(coordinates).reshape(-1,3)

    conn_comp_idx,conn_comp_meshes = connected_components_from_face_idx(
        mesh,
        face_idx = face_idx,
        verbose = verbose,
    )

    final_comp_idx = []
    final_comp_meshes = []
    filtered_away_meshes = []
    for j,(c_idx,c_mesh) in enumerate(zip(conn_comp_idx,conn_comp_meshes)):
        for i,coord in enumerate(coordinates):
            min_dist = np.min(np.linalg.norm(c_mesh.vertices - coord,axis = 1))
            if min_dist > min_distance_threshold:
                final_comp_idx.append(c_idx)
                final_comp_meshes.append(c_mesh)
            else:
                if verbose:
                    print(f"Filtering away connected comonent {j} because too close to coordinate {i} (min_dist = {min_dist})")
                filtered_away_meshes.append(c_mesh)
                    
    if plot:
        ipvu.plot_objects(
            meshes = final_comp_meshes + filtered_away_meshes,
            meshes_colors = ["green"]*len(final_comp_meshes) + ["red"]*len(filtered_away_meshes)
        )

    if return_meshes:
        return final_comp_idx,final_comp_meshes
    else:
        return final_comp_idx

def closest_mesh_attribute_to_coordinates_fast(
    mesh,
    coordinates,
    attribute = "vertices",
    return_distance = False,
    return_attribute = False,
    stop_after_0_dist = True,
    verbose = False,
    verbose_time = False
    ):
    if verbose_time:
        st = time.time()
    coordinates =np.array(coordinates).reshape(-1,3)
    if "vertic" in attribute:
        attr = mesh.vertices
    if "face" in attribute:
        attr = mesh.triangles_center
    else:
        attr = getattr(mesh,attribute)
        
    if not stop_after_0_dist:
        dist = np.array([np.min(np.linalg.norm(coordinates-c,axis=1)) for c in attr])
    else:
        dist = []
        for c in attr:
            curr_dist = np.min(np.linalg.norm(coordinates-c,axis=1))
            if curr_dist < 0.000001:
                curr_dist = 0
                dist.append(curr_dist)
                break
            else:
                dist.append(curr_dist)
                
    win_idx = np.argmin(dist)
    win_dist = dist[win_idx]
    
    if verbose:
        print(f"winning_idx = {win_idx} with closest distance = {win_dist}")
    if verbose_time:
        print(f"time = {time.time() - st}")
    if return_attribute:
        win_idx = attr[win_idx]
    if return_distance:
        return win_idx,win_dist
    else:
        return win_idx

def closest_mesh_distance_to_coordinates_fast(
    mesh,
    coordinates,
    attribute = "vertices",
    return_attribute = False,
    stop_after_0_dist = True,
    verbose = False,
    verbose_time = False
    ):
    
    _,dist  = closest_mesh_attribute_to_coordinates_fast(
    mesh,
    coordinates,
    attribute = attribute,
    return_attribute = return_attribute,
    stop_after_0_dist = stop_after_0_dist,
    verbose = verbose,
    verbose_time = verbose_time,
    return_distance = True,
    )
    
    return dist

def connected_component_meshes(
    mesh,
    verbose = False,
    plot = False,):
    return_meshes =  tu.split(mesh,return_components=False)
    if verbose or plot:
        print(f"# of conn comp meshes = {len(return_meshes)}")
    if plot:
        ipvu.plot_objects(
            meshes = return_meshes,
            meshes_colors = "random",
        )
        
    return return_meshes

def closest_connected_component_mesh_to_coordinates(
    mesh,
    coordinates,
    plot = False,
    ):

    closest_mesh = tu.closest_mesh_to_coordinates(
        tu.connected_component_meshes(new_mesh),
        coordinates,
    )
    if plot:
        ipvu.plot_objects(closest_mesh,scatters=[coordinates])
        
def faces_defined_by_vertices_idx_list_to_mesh(
    mesh,
    faces,#(n,3) array of vertices idx
    vertices = None,
    plot = False,
    verbose = False,
    ):
    """
    Purpose: To create a mesh from 
    an (n,3) array representing new faces
    """
    if vertices is not None:
        vertices = np.vstack([mesh.vertices,vertices])
    else:
        vertices = mesh.vertices
    unique_verts,inv_faces =np.unique(faces,return_inverse=True)
    new_mesh = trimesh.Trimesh(
        vertices=vertices[unique_verts],
        faces = inv_faces.reshape(-1,3)
    )
    
    if plot:
        ipvu.plot_objects(new_mesh)
    if verbose:
        print(f"new_mesh = {new_mesh}")
        
    return new_mesh


def stitch(
    mesh,
    faces=None, 
    insert_vertices=False,
    calculate_normals = False,
    vertices_to_stitch = None,
    return_mesh = True,
    return_mesh_with_holes_stitched = False,
    plot = False,
    verbose = False,):
    """
    Create a fan stitch over the boundary of the specified
    faces. If the boundary is non-convex a triangle fan
    is going to be extremely wonky.
    Parameters
    -----------
    vertices : (n, 3) float
      Vertices in space.
    faces : (n,) int
      Face indexes to stitch with triangle fans.
    insert_vertices : bool
      Allow stitching to insert new vertices?
    Returns
    ----------
    fan : (m, 3) int
      New triangles referencing mesh.vertices.
    vertices : (p, 3) float
      Inserted vertices (only returned `if insert_vertices`)
    """
    if faces is None:
        faces = np.arange(len(mesh.faces))

    # get a sequence of vertex indices representing the
    # boundary of the specified faces
    # will be referencing the same indexes of `mesh.vertices`
    points = [e.points for e in
              faces_to_path(mesh, faces)['entities']
              if len(e.points) > 3 and
              e.points[0] == e.points[-1]]
    
    if verbose:
        points_unfiltered = [e.points for e in
              faces_to_path(mesh, faces)['entities']
              if len(e.points) > 3]
        print(f"points_unfiltered = {points_unfiltered}")
        print(f"points = {points}")
    if vertices_to_stitch is not None:
        points = [p for p in points if
                 nu.closest_dist_between_coordinates(
                        mesh.vertices[p],
                        vertices_to_stitch
                 ) < 0.000001]

    
    # get properties to avoid querying in loop
    vertices = mesh.vertices
    normals = mesh.face_normals

    # find which faces are associated with an edge
    edges_face = mesh.edges_face
    tree_edge = mesh.edges_sorted_tree

    if insert_vertices:
        # create one new vertex per curve at the centroid
        centroids = np.array([vertices[p].mean(axis=0)
                              for p in points])
        # save the original length of the vertices
        count = len(vertices)
        # for the normal check stack our local vertices
        vertices = np.vstack((vertices, centroids))
        # create a triangle between our new centroid vertex
        # and each one of the boundary curves
        fan = [np.column_stack((
            np.ones(len(p) - 1, dtype=int) * (count + i),
            p[:-1],
            p[1:]))
            for i, p in enumerate(points)]
    else:
        # since we're not allowed to insert new vertices
        # create a triangle fan for each boundary curve
        fan = [np.column_stack((
            np.ones(len(p) - 3, dtype=int) * p[0],
            p[1:-2],
            p[2:-1]))
            for p in points]
        centroids = None

    if verbose:
        print(f"fan = {fan}")
    if calculate_normals:
        # now we do a normal check against an adjacent face
        # to see if each region needs to be flipped
        for i, p, t in zip(range(len(fan)), points, fan):
            # get the edges from the original mesh
            # for the first `n` new triangles
            e = t[:10, 1:].copy()
            e.sort(axis=1)

            # find which indexes of `mesh.edges` these
            # new edges correspond with by finding edges
            # that exactly correspond with the tree
            query = tree_edge.query_ball_point(e, r=1e-10)
            if len(query) == 0:
                continue
            # stack all the indices that exist
            edge_index = np.concatenate(query)

            # get the normals from the original mesh
            original = normals[edges_face[edge_index]]

            # calculate the normals for a few new faces
            #print(f"vertices = {vertices}")
            #print(f"vertices[t[:3]] = {vertices[t[:3]]}")
            check, valid = triangles.normals(vertices[t[:3]])
            if not valid.any():
                continue
            # take the first valid normal from our new faces
            #print(f"check = {check}")
            #print(f"valid = {valid}")
            #check = check[valid][0]
            check = check[0]

            # if our new faces are reversed from the original
            # Adjacent face flip them along their axis
            sign = np.dot(original, check)
            if sign.mean() < 0:
                fan[i] = np.fliplr(t)

    if len(fan) > 0:
        fan = np.vstack(fan)
    else:
        fan = np.array([]).reshape(-1,3)

    if return_mesh or plot or return_mesh_with_holes_stitched:
        if len(fan) > 0:
            new_mesh = faces_defined_by_vertices_idx_list_to_mesh(
                mesh,
                fan,
                vertices = centroids
            )
        else:
            new_mesh = empty_mesh()
            
    if plot:
        ipvu.plot_objects(mesh,meshes = [new_mesh],meshes_colors = "red")
        
    if return_mesh_with_holes_stitched:
        new_mesh = tu.combine_meshes([mesh,new_mesh])
        new_mesh.fix_normals()
        return new_mesh
        
    if return_mesh:
        fan = new_mesh 
    
    if insert_vertices:
        return fan, centroids
    return fan

close_mesh_holes = stitch

def stitch_mesh_at_vertices(
    mesh,
    vertices,
    verbose = False,
    plot = False,
    ):
    """
    Purpose: To get the mesh that would
    stitch up a certain boundary defined
    by vertices
    
    Process: 
    1) Run stitching with certain vertices
    """
    m =  tu.stitch(
        mesh,
        vertices_to_stitch = vertices,
        return_mesh = True,
    )
    
    if verbose:
        print(f"new mesh = {m}")
    
    if plot:
        ipvu.plot_objects(m,scatters=[vertices])
        
    return m

close_mesh_holes_at_vertices = stitch_mesh_at_vertices
divide_mesh_into_connected_components = connected_components_from_mesh

def area_of_vertex_boundary(
    mesh,
    vertices,
    plot = False,):
    """
    Purpose: To find the area of a
    vertex boundary
    """
    stitch_m = tu.close_mesh_holes_at_vertices(
        mesh,
        vertices,
        plot = plot,
    )
    
    return stitch_m.area

def fill_single_triangle_holes(
    mesh,
    in_place = False,
    ):
    if not in_place:
        mesh = mesh.copy()
    trimesh.repair.fill_holes(mesh)
    return mesh

def fill_mesh_holes_with_fan(
    mesh,
    plot = False,
    verbose = False,
    in_place = False,
    try_with_fill_single_hole_fill_for_backup = True,
    **kwargs
    ):
        
    return_mesh = stitch(
        mesh,
        return_mesh_with_holes_stitched = True,
        plot = plot,
        **kwargs
    )
    
    if not tu.is_watertight(return_mesh) and try_with_fill_single_hole_fill_for_backup:
        if not in_place:
            mesh = mesh.copy()
        
        trimesh.repair.fill_holes(mesh)
        return_mesh = stitch(
            mesh,
            return_mesh_with_holes_stitched = True,
            plot = plot,
            **kwargs
        )
    
    if verbose:
        print(f"Original mesh = {mesh}")
        print(f"Filled Holes mesh = {return_mesh}")
        
    return return_mesh

fill_mesh_holes = fill_mesh_holes_with_fan
stitch_mesh_holes = fill_mesh_holes_with_fan

def close_hole_areas(
    mesh,
    return_meshes = False,
    sort_type = "max",
    verbose = False,
    ):
    """
    Purpose: To find the aresa of all
    conn components needed for closing holes
    """
    mesh_to_stitch = tu.close_mesh_holes(mesh)

    if len(mesh_to_stitch.faces) > 0:
        mesh_to_stitch_conn_comp = tu.connected_components_from_mesh(mesh_to_stitch)
    else:
        mesh_to_stitch_conn_comp = []

    if len(mesh_to_stitch_conn_comp) == 0:
        stitch_meshes_area= []
        if verbose:
            print(f"No meshes needed for closing hole")
    else:
        stitch_meshes_area = np.array([k.area for k in mesh_to_stitch_conn_comp])

        sort_idx = np.argsort(stitch_meshes_area)
        if sort_type == "max":
            sort_idx = np.flip(sort_idx)
            
        stitch_meshes_area = stitch_meshes_area[sort_idx]
        mesh_to_stitch_conn_comp = np.array(mesh_to_stitch_conn_comp)[sort_idx]
    
        
    if verbose:
        print(f"stitch_meshes_area= {stitch_meshes_area}")
    if return_meshes:
        return stitch_meshes_area,mesh_to_stitch_conn_comp
    else:
        return stitch_meshes_area

def close_hole_area_top_k_extrema(
    mesh,
    k,
    extrema="max",
    aggr_func = None,
    verbose = False,
    return_mesh = False,
    plot = False,
    default_value = 0,
    fill_to_meet_length = True,
    ):
    def dummy_func(x):
        return x
    if type(aggr_func) == str:
        aggr_func = getattr(np,aggr_func)
    elif aggr_func is None:
        aggr_func = dummy_func
    
    stitch_meshes_area,mesh_to_stitch_conn_comp = close_hole_areas(
        mesh,
        return_meshes=True,
        verbose=verbose,
        sort_type=extrema
    )

    if len(stitch_meshes_area) == 0:
        stitch_extrema_meshes = np.array([tu.empty_mesh()]*k)
        stitch_extrema = np.array([default_value]*k)
    else:  
        stitch_extrema = stitch_meshes_area[:k]
        stitch_extrema_meshes = mesh_to_stitch_conn_comp[:k]
        if verbose:
            print(f"{extrema} = {stitch_extrema}")

        if plot:
            ipvu.plot_objects(
                mesh,
                meshes = list(stitch_extrema_meshes),
                meshes_colors = "red",
            )
            
    if len(stitch_extrema) < k and fill_to_meet_length:
        stitch_extrema = np.hstack([stitch_meshes_area,[default_value]*k])[:k]
        stitch_extrema_meshes = np.hstack([stitch_extrema_meshes,[tu.empty_mesh()]*k])[:k]
        
    if verbose:
        print(f"stitch_extrema {extrema} {k} = {stitch_extrema}")
        print(f"mesh_to_stitch_conn_comp {extrema} {k} = {stitch_extrema_meshes}")
        
    stitch_extrema = aggr_func(stitch_extrema)
    
    
    if return_mesh:
        return stitch_extrema,stitch_extrema_meshes
    else:
        return stitch_extrema
    
def close_hole_area_max(
    mesh,
    verbose = False,
    return_mesh = False,
    plot = False,
    **kwargs
    ):
    
    val,meshes = close_hole_area_top_k_extrema(
        mesh,
        k=1,
        extrema="max",
        verbose = verbose,
        return_mesh = True,
        plot = plot,
        **kwargs
    )
    
    if return_mesh:
        return val[0],meshes[0]
    else:
        return val[0]
    
def close_hole_area_top_2_mean(
    mesh,
    verbose = False,
    return_mesh = False,
    plot = False,
    **kwargs
    ):
    
    val,meshes = close_hole_area_top_k_extrema(
        mesh,
        k=2,
        extrema="max",
        verbose = verbose,
        return_mesh = True,
        plot = plot,
        **kwargs
    )
    aggr_func = np.mean
    if return_mesh:
        return aggr_func(val),tu.combine_meshes(meshes)
    else:
        return aggr_func(val)

def stats_df(
    meshes,
    functions,
    suppress_errors = True,
    default_value = 0,
    plot = False,
    labels = None,
    ):
    """
    Purpose: Given a list of meshes
    and a list of functions that can be applied
    to mesh, want to compute a dataframe
    that stores the output of all the functions for
    each mesh in rows of a dataframe

    Pseudocode: 
    For each mesh:
        For each function:
            Compute the value
            Store in a dictionary

    Create pandas dataframe
    """
    if type(functions) == str:
        functions = [functions]
    elif type(functions) == dict:
        pass
    else:
        functions_dict = dict()
        for f in functions:
            if type(f) == str:
                functions_dict[f] = getattr(tu,f)
            else:
                functions_dict[f.__name__] = f
    
    if len(meshes) == 0:
        df = pu.empty_df(columns=list(functions_dict.keys()))
        return df
    
    total_dicts = []
    for m in meshes:
        curr_dict = dict()
        for f,f_func in functions_dict.items():
            try:
                val = f_func(m)
            except:
                if suppress_errors:
                    val = default_value
                else:
                    raise Exception("")
            curr_dict[f] = val
        total_dicts.append(curr_dict)
        
    df = pd.DataFrame.from_records(total_dicts)
    
    if plot:
        if labels == None:
            labels = np.ones(len(df))
            
        df_cp = df.copy()
        df_cp["label"] = labels
        for c in df.columns:
            mu.histograms_overlayed(df_cp,column=c,hue = "label")
            plt.show()
    return df

def query_meshes_from_restrictions(
    meshes,
    query,
    stats_df = None,
    print_stats_df = False,
    functions = None,
    return_idx = False,
    verbose = False,
    plot = False,
    ):
    """
    Purposse: To query a list of meshes using 
    a query string or list of conditions and
    functions computed on the meshes
    """
    if stats_df is None:
        stats_df = tu.stats_df(meshes,functions)
    
    if print_stats_df:
        try:
            display(stats_df)
        except:
            print(stats_df)
    
    query = pu.query_str(query,table_type="pandas")
    if verbose:
        print(f"query = {query}")
    restr_df = stats_df.query(query)
    
    if verbose:
        print(f"# of entries after query = {len(restr_df)}/{len(stats_df)}")
        
    idx = list(restr_df.index)
    return_meshes = [meshes[i] for i in idx]
    
    if plot:
        ipvu.plot_objects(
            tu.combine_meshes(meshes),
            meshes = return_meshes,
            meshes_colors = "red"
        )
    if return_idx:
        return idx
    else:
        return return_meshes
    
    
from datasci_tools import mesh_utils as meshu
clear_mesh_cache = meshu.clear_mesh_cache
clear_all_mesh_cache_in_nested_data_struct = meshu.clear_all_mesh_cache_in_nested_data_struct
        
        
query_meshes_from_stats = query_meshes_from_restrictions
restrict_meshes_from_stats = query_meshes_from_restrictions


#--- from mesh_tools ---
from . import compartment_utils as cu
from . import skeleton_utils as sk

#--- from machine_learning_tools ---
try:
    from machine_learning_tools import dimensionality_reduction_utils as dru
except:
    dru = None

#--- from datasci_tools ---
from datasci_tools import general_utils as gu
from datasci_tools import ipyvolume_utils as ipvu
from datasci_tools import matplotlib_utils as mu
from datasci_tools import networkx_utils as xu
from datasci_tools import numpy_utils as nu
from datasci_tools import pandas_utils as pu
from datasci_tools import system_utils as su
from datasci_tools.tqdm_utils import tqdm

from . import trimesh_utils as tu
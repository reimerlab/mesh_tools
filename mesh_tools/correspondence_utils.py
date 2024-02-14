import numpy as np
import time
import itertools
from copy import deepcopy


def plot_correspondence(
    mesh,
    correspondence,
    idx_to_show = None,
    submesh_from_face_idx = True,
    verbose = True,
    ):
    """
    Purpose: Want to plot mesh correspondence first pass

    Pseudocode: 
    For each entry:
    1) Plot mesh (from idx)
    2) plot skeleton
    """

    if idx_to_show is None:
        idx_to_show = list(correspondence.keys())

    idx_to_show = nu.array_like(idx_to_show)

    for k,v in correspondence.items():
        if k not in idx_to_show:
            continue

        try:
            submesh_idx = v["correspondence_face_idx"]
        except:
            submesh_idx = v["branch_face_idx"]
        subskeleton = v["branch_skeleton"]
        if verbose:
            print(f"For branch {k}: # of mesh faces = {len(submesh_idx)}, skeleton length = {sk.calculate_skeleton_distance(subskeleton)}")
        if submesh_from_face_idx:
            submesh = mesh.submesh([submesh_idx],append=True)
        else:
            try:
                submesh = v["correspondence_mesh"]
            except:
                submesh = v["branch_mesh"]

        ipvu.plot_objects(
            mesh,
            subskeleton,
            meshes = [submesh],
            meshes_colors = ["red"],
            buffer = 0,
        )
def mesh_correspondence_first_pass(
    mesh,
    skeleton=None,
    skeleton_branches=None,
    distance_by_mesh_center=True,
    remove_inside_pieces_threshold = 0,
    skeleton_segment_width = 1000,
    initial_distance_threshold = 3000,
    skeletal_buffer = 100,
    backup_distance_threshold = 6000,
    backup_skeletal_buffer = 300,
    connectivity="edges",
    plot = False,
    ):
    """
    Will come up with the mesh correspondences for all of the skeleton
    branches: where there can be overlaps and empty faces
    
    """
    curr_limb_mesh = mesh
    curr_limb_sk = skeleton
    
    if remove_inside_pieces_threshold > 0:
        curr_limb_mesh_indices = tu.remove_mesh_interior(curr_limb_mesh,
                                                 size_threshold_to_remove=remove_inside_pieces_threshold,
                                                 try_hole_close=False,
                                                 return_face_indices=True,
                                                )
        curr_limb_mesh = curr_limb_mesh.submesh([curr_limb_mesh_indices],append=True,repair=False)
    else:
        curr_limb_mesh_indices = np.arange(len(curr_limb_mesh.faces))
    
    if skeleton_branches is None:
        if skeleton is None:
            raise Exception("Both skeleton and skeleton_branches is None")
        curr_limb_branches_sk_uneven = sk.decompose_skeleton_to_branches(curr_limb_sk) #the line that is decomposing to branches
    else:
        curr_limb_branches_sk_uneven = skeleton_branches 

    #Doing the limb correspondence for all of the branches of the skeleton
    local_correspondence = dict()
    for j,curr_branch_sk in tqdm(enumerate(curr_limb_branches_sk_uneven)):
        local_correspondence[j] = dict()

        
        returned_data = cu.mesh_correspondence_adaptive_distance(curr_branch_sk,
                                      curr_limb_mesh,
                                     skeleton_segment_width = skeleton_segment_width,
                                     distance_by_mesh_center=distance_by_mesh_center,
                                    distance_threshold = initial_distance_threshold,
                                    buffer = skeletal_buffer,
                                                                connectivity=connectivity)
        if len(returned_data) == 0:
            print("Got nothing from first pass so expanding the mesh correspondnece parameters ")
            returned_data = cu.mesh_correspondence_adaptive_distance(curr_branch_sk,
                                      curr_limb_mesh,
                                     skeleton_segment_width = skeleton_segment_width,
                                     distance_by_mesh_center=distance_by_mesh_center,
                                    buffer=backup_skeletal_buffer,
                                     distance_threshold=backup_distance_threshold,
                                    return_closest_face_on_empty=True,
                                        connectivity=connectivity)
            
        # Need to just pick the closest face is still didn't get anything
        
        # ------ 12/3 Addition: Account for correspondence that does not work so just picking the closest face
        curr_branch_face_correspondence, width_from_skeleton = returned_data
        
            
#             print(f"curr_branch_sk.shape = {curr_branch_sk.shape}")
#             np.savez("saved_skeleton_branch.npz",curr_branch_sk=curr_branch_sk)
#             tu.write_neuron_off(curr_limb_mesh,"curr_limb_mesh.off")
#             #print(f"returned_data = {returned_data}")
#             raise Exception(f"The output from mesh_correspondence_adaptive_distance was nothing: curr_branch_face_correspondence")


        if len(curr_branch_face_correspondence) > 0:
            curr_submesh = curr_limb_mesh.submesh([list(curr_branch_face_correspondence)],append=True,repair=False)
        else:
            curr_submesh = trimesh.Trimesh(vertices=np.array([]),faces=np.array([]))


        local_correspondence[j]["branch_skeleton"] = curr_branch_sk
        local_correspondence[j]["correspondence_mesh"] = curr_submesh
        local_correspondence[j]["correspondence_face_idx"] = curr_limb_mesh_indices[curr_branch_face_correspondence]
        local_correspondence[j]["width_from_skeleton"] = width_from_skeleton
        
        
    if plot:
        plot_correspondence(mesh,local_correspondence)
    return local_correspondence



        
def correspondence_1_to_1(
    mesh,
    local_correspondence,
    curr_limb_endpoints_must_keep=None,
    must_keep_labels=dict(),
    plot = False,
                    ):
    """
    Will Fix the 1-to-1 Correspondence of the mesh
    correspondence for the limbs and make sure that the
    endpoints that are designated as touching the soma then 
    make sure the mesh correspondnece reaches the soma limb border
    
    has an optional argument must_keep_labels that will allow you to specify some labels that are a must keep
    
    """
    
    if len(tu.split(mesh)[0])>1:
        su.compressed_pickle(mesh,"mesh")
        raise Exception("Mesh passed to correspondence_1_to_1 is not just one mesh")
    
    mesh_start_time = time.time()
    print(f"\n\n--- Working on 1-to-1 correspondence-----")

    #geting the current limb mesh

    no_missing_labels = list(local_correspondence.keys()) #counts the number of divided branches which should be the total number of labels
    curr_limb_mesh = mesh

    #set up the face dictionary
    face_lookup = dict([(j,[]) for j in range(0,len(curr_limb_mesh.faces))])

    for j,branch_piece in local_correspondence.items():
        curr_faces_corresponded = branch_piece["correspondence_face_idx"]

        for c in curr_faces_corresponded:
            face_lookup[c].append(j)

    original_labels = set(list(itertools.chain.from_iterable(list(face_lookup.values()))))
    print(f"max(original_labels),len(original_labels) = {(max(original_labels),len(original_labels))}")

    if len(original_labels) != len(no_missing_labels):
        raise Exception(f"len(original_labels) != len(no_missing_labels) for original_labels = {len(original_labels)},no_missing_labels = {len(no_missing_labels)}")

    if max(original_labels) + 1 > len(original_labels):
        raise Exception("There are some missing labels in the initial labeling")



    #here is where can call the function that resolves the face labels
    face_coloring_copy = cu.resolve_empty_conflicting_face_labels(
                     curr_limb_mesh = curr_limb_mesh,
                     face_lookup=face_lookup,
                     no_missing_labels = list(original_labels),
                    must_keep_labels=must_keep_labels,
                    branch_skeletons = [local_correspondence[k]["branch_skeleton"] for k in local_correspondence.keys()],
    )

    """  9/17 Addition: Will make sure that the desired starting node is touching the soma border """
    """
    Pseudocode:
    For each soma it is touching
    0) Get the soma border
    1) Find the label_to_expand based on the starting coordinate
    a. Get the starting coordinate

    soma_to_piece_touching_vertices=None
    endpoints_must_keep

    """

    #curr_limb_endpoints_must_keep --> stores the endpoints that should be connected to the soma
    #curr_soma_to_piece_touching_vertices --> maps soma to  a list of grouped touching vertices


    # -- splitting the mesh pieces into individual pieces
    divided_submeshes,divided_submeshes_idx = tu.split_mesh_into_face_groups(curr_limb_mesh,face_coloring_copy)

    #-- check that all the split mesh pieces are one component --#
    local_correspondence_revised = deepcopy(local_correspondence)
    #save off the new data as branch mesh
    for k in local_correspondence_revised.keys():
        local_correspondence_revised[k]["branch_mesh"] = divided_submeshes[k]
        local_correspondence_revised[k]["branch_face_idx"] = divided_submeshes_idx[k]

        #clean the limb correspondence that we do not need
        del local_correspondence_revised[k]["correspondence_mesh"]
        del local_correspondence_revised[k]["correspondence_face_idx"]
    
    if plot:
        plot_correspondence_on_single_mesh(mesh,local_correspondence_revised)
    
    return local_correspondence_revised


def plot_correspondence_on_single_mesh(
    mesh,
    correspondence,
    ):
    """
    Purpose: To plot the correspondence dict
    once a 1 to 1 was generated

    """

    try:
        meshes = [k["branch_mesh"] for k in correspondence.values()]
    except:
        meshes = [k["correspondence_mesh"] for k in correspondence.values()]
        
    skeletons = [k["branch_skeleton"] for k in correspondence.values()]
    colors = mu.generate_non_randon_named_color_list(len(meshes))

    ipvu.plot_objects(
        meshes = meshes,
        meshes_colors=colors,
        skeletons=skeletons,
        skeletons_colors=colors,
        buffer = 0,
    )


from datasci_tools import ipyvolume_utils as ipvu
from datasci_tools import numpy_utils as nu
from datasci_tools import matplotlib_utils as mu

from datasci_tools.tqdm_utils import tqdm


from . import compartment_utils as cu
from . import skeleton_utils as sk
from . import trimesh_utils as tu
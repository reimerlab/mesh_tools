"""
Purpose: 

"""
import unittest
import trimesh
import numpy as np


class TestEnvironment(unittest.TestCase):
  
    def setUp(self):
        def load_mesh_no_processing(filepath):
            return trimesh.load_mesh(filepath,process=False)
        
        self.mesh = load_mesh_no_processing("../fixtures/elephant.off")
        self.branch_mesh = load_mesh_no_processing("../fixtures/neuron_branch.off")
        
    def test_kdtree_import(self):
        try:
            from pykdtree.kdtree import KDTree
        except:
            KDTree = None
            
        self.assertIsNotNone(KDTree)
            
    def test_meshparty_import(self):
        try:
            import meshparty
        except:
            meshparty = None
            
        self.assertIsNotNone(meshparty)
            
    def test_cgal_segmentation_import(self):
        try:
            import cgal_Segmentation_Module as csm
        except:
            csm = None
            
        self.assertIsNotNone(csm)
        
    def test_cgal_skeletonization_import(self):
        
        try:
            import calcification_Module as cm
        except:
            cm = None
            
        self.assertIsNotNone(cm)
        
    def test_cgal_skeletonization_param_import(self):
        
        try:
            from calcification_param_Module import calcification_param
        except:
            calcification_param = None
            
        self.assertIsNotNone(calcification_param)
        
    def test_open3d_import(self):
        try:
            import open3d as o3d
        except:
            o3d = None
            
        self.assertIsNotNone(o3d)
        
    # ---- meshlab backend tests ---
    def test_meshlab_decimation(self):
        from mesh_tools import trimesh_utils as tu
        mesh_dec = tu.decimate(self.mesh,0.25)
        self.assertIsNotNone(mesh_dec)
        
    def test_meshlab_poisson_surface_reconstruction(self):
        from mesh_tools import trimesh_utils as tu
        branch_mesh_reconstructed = tu.poisson_surface_reconstruction(
            self.branch_mesh
        )

        self.assertIsNotNone(branch_mesh_reconstructed)
        
    def test_meshlab_fill_holes(self):
        from mesh_tools import trimesh_utils as tu
        branch_mesh_no_holes = tu.fill_holes(self.branch_mesh)
        self.assertIsNotNone(branch_mesh_no_holes)
        
    def mesh_interior(self):
        from mesh_tools import trimesh_utils as tu
        mesh_interior = tu.mesh_interior(self.branch_mesh)
        self.assertIsNotNone(mesh_interior)
        
    # ---- open3d backend tests ---
    def test_open3d_manifold(self):
        from mesh_tools import trimesh_utils as tu
        self.assertIsTrue(tu.is_manifold(self.mesh))
        
    def test_open3d_manifold(self):
        from mesh_tools import trimesh_utils as tu
        self.assertIsNotNone(tu.mesh_volume_o3d(self.mesh))
        
    # ---- CGAL backend tests ---
    def test_cgal_segmentation(self):
        from mesh_tools import trimesh_utils as tu
        clusters = 5
        smoothness = 0.08

        seg = tu.mesh_segmentation(
            self.branch_mesh,
            plot_segmentation=False,
            clusters=clusters,
            smoothness=smoothness,
        )
        
        self.assertIsNotNone(seg)
        
    def test_cgal_skeletonization(self): 
        from mesh_tools import skeleton_utils as sk
        skeleton = sk.skeleton_cgal(self.branch_mesh,plot=False)
        self.assertIsNotNone(skeleton)
        
    def test_kdtree_query(self):
        from pykdtree.kdtree import KDTree
        
        coordinate = np.array([[0,0,0]])
        mesh_coords = self.mesh.vertices

        kdtree_obj = KDTree(mesh_coords)
        dist, closest_idx = kdtree_obj.query(coordinate)
        
        self.assertIsNotNone(dist)
        
        
if __name__ == '__main__':
    unittest.main()
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
        
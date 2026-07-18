# components/visualization/simple_visualizer.py
import open3d as o3d
import numpy as np
import logging
from typing import Optional

from components.memory.memory_manager import MemoryManager
from components.memory.change_memory import ChangeMemory
from config import VISUALIZATION_ENABLE_3D_RECONSTRUCTION, VISUALIZATION_HEADLESS

logger = logging.getLogger(__name__)

class Open3DVisualizer:
    """
    Simplified visualizer for StateScribe.
    Only displays:
    1. Incremental point cloud from all frames
    2. 3D bounding boxes from change memory
    """
    
    def __init__(self, memory_manager: MemoryManager, change_memory: ChangeMemory):
        self.memory_manager = memory_manager
        self.change_memory = change_memory
        self.vis = o3d.visualization.Visualizer()
        self.is_initialized = False
        self.enable_reconstruction = bool(VISUALIZATION_ENABLE_3D_RECONSTRUCTION)
        
        # TSDF volume for point cloud
        self.global_volume = None
        self.integrated_frames = set()
        self._initialize_global_volume()
        
        # Visualization geometries
        self.mesh_geometry = None
        self.bbox_geometries = []
        
    def _initialize_global_volume(self):
        """Initialize TSDF volume"""
        if not self.enable_reconstruction:
            self.global_volume = None
            logger.debug("3D reconstruction disabled; skipping TSDF initialization.")
            return
        voxel_length = 0.03
        sdf_trunc = 0.09
        self.global_volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        logger.debug("TSDF volume initialized")
    
    def initialize(self):
        """Create visualization window"""
        if VISUALIZATION_HEADLESS:
            logger.debug("Headless mode, no window created")
            self.is_initialized = True
            return
        
        self.vis.create_window("StateScribe - Change Detection", width=1280, height=720)
        self.vis.get_render_option().background_color = np.asarray([1.0, 1.0, 1.0])
        self.is_initialized = True
        logger.debug("Visualizer window created")
    
    def update(self) -> bool:
        """
        Update visualization.
        Returns False if window closed.
        """
        if not self.is_initialized or VISUALIZATION_HEADLESS:
            return True
        
        # Sync point cloud - always check for new frames
        self._sync_point_cloud()
        
        # Update bounding boxes - always refresh from change memory
        self._update_bboxes()
        
        # Poll events and render
        if not self.vis.poll_events():
            return False
        self.vis.update_renderer()
        
        return True
    
    def _sync_point_cloud(self):
        """Incrementally integrate new frames and update mesh"""
        if not self.enable_reconstruction or self.global_volume is None:
            return
        frames = self.memory_manager.get_memory()
        
        # Find new frames
        missing = []
        for frame in reversed(frames):
            ts = frame.timestamp
            if ts not in self.integrated_frames:
                missing.append(frame)
            else:
                break
        
        # Integrate new frames
        for frame in reversed(missing):
            self._integrate_frame(frame)
            self.integrated_frames.add(frame.timestamp)
            logger.debug(f"Integrated frame {frame.timestamp}")
        
        # Always update mesh if there are new frames (real-time update)
        if missing:
            self._update_mesh()
            logger.debug(f"Updated mesh with {len(missing)} new frames")
    
    def _integrate_frame(self, frame):
        """Integrate a frame into TSDF"""
        if self.global_volume is None:
            return
        rgb_o3d = o3d.geometry.Image(frame.rgb_image)
        depth_o3d = o3d.geometry.Image(frame.depth_map.astype(np.float32))
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
        )
        
        h, w, _ = frame.rgb_image.shape
        intrinsics = o3d.camera.PinholeCameraIntrinsic(
            w, h, frame.intrinsics[0, 0], frame.intrinsics[1, 1],
            frame.intrinsics[0, 2], frame.intrinsics[1, 2]
        )
        
        extrinsic = np.linalg.inv(frame.pose_matrix)
        self.global_volume.integrate(rgbd, intrinsics, extrinsic)
    
    def _update_mesh(self):
        """Extract and display mesh from TSDF (real-time update)"""
        if self.global_volume is None:
            return
        mesh = self.global_volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        
        if self.mesh_geometry is None:
            # First time: add geometry
            self.mesh_geometry = mesh
            self.vis.add_geometry(mesh)
            logger.debug("Added initial mesh to visualizer")
        else:
            # Update existing geometry (real-time)
            self.mesh_geometry.vertices = mesh.vertices
            self.mesh_geometry.triangles = mesh.triangles
            self.mesh_geometry.vertex_normals = mesh.vertex_normals
            if mesh.has_vertex_colors():
                self.mesh_geometry.vertex_colors = mesh.vertex_colors
            self.vis.update_geometry(self.mesh_geometry)
        self.vis.reset_view_point(True)
    
    def _update_bboxes(self):
        """Update 3D bounding boxes from change memory (real-time update)"""
        # Clear old bboxes
        for bbox_geom in self.bbox_geometries:
            self.vis.remove_geometry(bbox_geom, reset_bounding_box=False)
        self.bbox_geometries.clear()
        
        active_color = (0.0, 1.0, 0.0)
        disappeared_color = (1.0, 0.2, 0.2)

        def add_bbox_for_object(obj, color):
            latest = obj.get_latest_snapshot()
            if not latest or latest.bbox_3d is None or not isinstance(latest.bbox_3d, np.ndarray):
                return
            bbox_3d = latest.bbox_3d

            if bbox_3d.shape[0] != 6:
                logger.debug(f"Invalid bbox shape for {obj.object_id}: {bbox_3d.shape}")
                return

            min_bound = bbox_3d[:3]
            max_bound = bbox_3d[3:]

            extent = max_bound - min_bound
            if np.any(extent < 0):
                logger.debug(f"Invalid bbox bounds for {obj.object_id}: extent={extent}")
                return

            if np.all(extent < 0.001):
                logger.debug(f"Skipping tiny bbox for {obj.object_id}: extent={extent}")
                return

            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            bbox.color = color
            self.vis.add_geometry(bbox, reset_bounding_box=False)
            self.bbox_geometries.append(bbox)
            # logger.debug(f"Added bbox for {obj.object_id}: {latest.description[:30]}")

        # Add new bboxes for active objects (real-time)
        for obj in self.change_memory.get_active_objects():
            add_bbox_for_object(obj, active_color)

        # Always add disappeared objects
        for obj in self.change_memory.get_disappeared_objects():
            add_bbox_for_object(obj, disappeared_color)
    
    def destroy(self):
        """Destroy visualization window"""
        if self.is_initialized and not VISUALIZATION_HEADLESS:
            self.vis.destroy_window()
            logger.debug("Visualizer destroyed")
    
    def clear(self):
        """Clear all geometries"""
        if self.is_initialized and not VISUALIZATION_HEADLESS:
            self.vis.clear_geometries()
        
        self._initialize_global_volume()
        self.integrated_frames.clear()
        self.bbox_geometries.clear()
        self.mesh_geometry = None
        logger.debug("Visualizer cleared")

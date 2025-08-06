import numpy as np
import cv2
import gtsam
import matplotlib.pyplot as plt
import torch
import open3d as o3d
import viser
import viser.transforms as viser_tf
from termcolor import colored

from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from gtsam import noiseModel
from vggt_slam.loop_closure import ImageRetrieval
from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.map import GraphMap
from vggt_slam.submap import Submap
from vggt_slam.h_solve import ransac_projective
from vggt_slam.gradio_viewer import TrimeshViewer

def color_point_cloud_by_confidence(pcd, confidence, cmap='viridis'):
    """
    Color a point cloud based on per-point confidence values.
    
    Parameters:
        pcd (o3d.geometry.PointCloud): The point cloud.
        confidence (np.ndarray): Confidence values, shape (N,).
        cmap (str): Matplotlib colormap name.
    """
    assert len(confidence) == len(pcd.points), "Confidence length must match number of points"

    # Normalize confidence to [0, 1]
    confidence_normalized = (confidence - np.min(confidence)) / (np.ptp(confidence) + 1e-8)
    
    # Map to colors using matplotlib colormap
    colormap = plt.get_cmap(cmap)
    colors = colormap(confidence_normalized)[:, :3]  # Drop alpha channel

    # Assign to point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

class Viewer:
    def __init__(self, port: int = 8080):
        print(f"Starting viser server on port {port}")

        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

        # Global toggle for all frames and frustums
        self.gui_show_frames = self.server.gui.add_checkbox(
            "Show Cameras",
            initial_value=True,
        )
        self.gui_show_frames.on_update(self._on_update_show_frames)

        # Store frames and frustums by submap
        self.submap_frames: Dict[int, List[viser.FrameHandle]] = {}
        self.submap_frustums: Dict[int, List[viser.CameraFrustumHandle]] = {}

        num_rand_colors = 250
        self.random_colors = np.random.randint(0, 256, size=(num_rand_colors, 3), dtype=np.uint8)

    def visualize_frames(self, extrinsics: np.ndarray, images_: np.ndarray, submap_id: int) -> None:
        """
        Add camera frames and frustums to the scene for a specific submap.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """

        if isinstance(images_, torch.Tensor):
            images_ = images_.cpu().numpy()

        if submap_id not in self.submap_frames:
            self.submap_frames[submap_id] = []
            self.submap_frustums[submap_id] = []

        S = extrinsics.shape[0]
        for img_id in range(S):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_name = f"submap_{submap_id}/frame_{img_id}"
            frustum_name = f"{frame_name}/frustum"

            # Add the coordinate frame
            frame_axis = self.server.scene.add_frame(
                frame_name,
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.015,
                axes_radius=0.0015,
                origin_radius=0.002,
            )
            frame_axis.visible = self.gui_show_frames.value
            self.submap_frames[submap_id].append(frame_axis)

            # Convert image and add frustum
            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            frustum = self.server.scene.add_camera_frustum(
                frustum_name,
                fov=fov,
                aspect=w / h,
                scale=0.0035,
                image=img,
                line_width=3.0,
                color=self.random_colors[submap_id]
            )
            frustum.visible = self.gui_show_frames.value
            self.submap_frustums[submap_id].append(frustum)

    def _on_update_show_frames(self, _) -> None:
        """Toggle visibility of all camera frames and frustums across all submaps."""
        visible = self.gui_show_frames.value
        for frames in self.submap_frames.values():
            for f in frames:
                f.visible = visible
        for frustums in self.submap_frustums.values():
            for fr in frustums:
                fr.visible = visible


from gnss_processer import GNSSprocesser
class Solver:
    def __init__(self,
        init_conf_threshold: float,  # represents percentage (e.g., 50 means filter lowest 50%)
        use_point_map: bool = False,
        visualize_global_map: bool = False,
        use_sim3: bool = False,
        gradio_mode: bool = False):
        
        self.init_conf_threshold = init_conf_threshold
        self.use_point_map = use_point_map
        self.gradio_mode = gradio_mode
        self.gnss_processer = GNSSprocesser()
        if self.gradio_mode:
            self.viewer = TrimeshViewer()
        else:
            self.viewer = Viewer()

        self.flow_tracker = FrameTracker()
        self.map = GraphMap()
        self.use_sim3 = use_sim3
        if self.use_sim3:
            from vggt_slam.graph_se3 import PoseGraph
        else:
            from vggt_slam.graph import PoseGraph
        self.graph = PoseGraph()

        self.image_retrieval = ImageRetrieval()
        self.current_working_submap = None

        self.first_edge = True

        self.T_w_kf_minus = None

        self.prior_pcd = None
        self.prior_conf = None


        self.scale_factor = 1.0
        self.scale_aligned = False
        self.visual_positions = []  # 存储视觉位置历史
        self.gnss_positions = []    # 存储GNSS位置历史
        self.last_visual_pos = None
        self.last_gnss_pos = None
        
        print("Starting viser server...")
    def align_scale(self):
        """Calculate scale factor between visual and GNSS trajectories"""
        if len(self.visual_positions) < 2 or len(self.gnss_positions) < 2:
            print("Warning: Not enough points for scale alignment")
            return False
            
        # Calculate scale factors between consecutive points
        scale_factors = []
        for i in range(1, len(self.visual_positions)):
            visual_delta = self.visual_positions[i] - self.visual_positions[0]
            gnss_delta = self.gnss_positions[i] - self.gnss_positions[0]
            
            visual_dist = np.linalg.norm(visual_delta)
            gnss_dist = np.linalg.norm(gnss_delta)
            
            if visual_dist > 0.01:  # Avoid division by small values
                scale_factors.append(gnss_dist / visual_dist)
        
        if scale_factors:
            self.scale_factor = np.median(scale_factors)
            print(f"Scale alignment: visual scale factor = {self.scale_factor:.4f}")
            self.scale_aligned = True
            return True
        return False

    def apply_scale_to_poses(self, world_points):
        """Apply scale factor to all visual poses in the graph"""
        if not self.scale_aligned:
            print("Warning: Scale not aligned, cannot apply scaling")
            return
            
        print(f"Applying scale factor {self.scale_factor:.4f} to all poses")
        for submap in self.map.get_submaps():
            # Get current homography
            H = submap.get_reference_homography()
            
            # Scale the translation component
            H_scaled = H.copy()
            H_scaled[0:3, 3] *= self.scale_factor
            
            # Update the submap with scaled homography
            submap.set_reference_homography(H_scaled)
            
            # Update all poses in the submap
            scaled_poses = []
            for pose in submap.poses:
                pose_scaled = pose.copy()
                pose_scaled[0:3, 3] *= self.scale_factor
                scaled_poses.append(pose_scaled)
            submap.poses = scaled_poses
            world_points *= self.scale_factor
            # # Update all point clouds in the submap
            # for i in range(len(submap.points)):
            #     if submap.points[i] is not None:
            #         submap.points[i] = submap.points[i] * self.scale_factor
    def add_gnss_constraints(self):
        """Add GNSS position constraints to the factor graph"""
        if not self.scale_aligned:
            print("Warning: Scale not aligned, cannot add GNSS constraints")
            return
            
        print("Adding GNSS constraints to factor graph...")
        
        # Create a noise model for GNSS measurements (1m standard deviation)
        gnss_noise = noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0, 1e6, 1e6, 1e6]))
        
        for i, submap in enumerate(self.map.get_submaps()):
            # Get the GNSS position for this submap's reference frame
            if i < len(self.gnss_positions):
                gnss_pos = self.gnss_positions[i]
                
                # Create a pose with GNSS position but visual rotation
                H_w_submap = submap.get_reference_homography()
                gnss_pose = np.eye(4)
                gnss_pose[0:3, 0:3] = H_w_submap[0:3, 0:3]  # Keep visual rotation
                gnss_pose[0:3, 3] = gnss_pos  # Use GNSS position
                
                # Add GNSS constraint to the graph
                self.graph.add_prior_factor(
                    submap.get_id(), 
                    gnss_pose, 
                    gnss_noise
                )
                print(f"Added GNSS constraint for submap {submap.get_id()} at {gnss_pos}")
    def set_point_cloud(self, points_in_world_frame, points_colors, name, point_size):
        if self.gradio_mode:
            self.viewer.add_point_cloud(points_in_world_frame, points_colors)
        else:
            self.viewer.server.scene.add_point_cloud(
                name="pcd_"+name,
                points=points_in_world_frame,
                colors=points_colors,
                point_size=point_size,
                point_shape="circle",
            )

    def set_submap_point_cloud(self, submap):
        # Add the point cloud to the visualization.
        points_in_world_frame = submap.get_points_in_world_frame()
        points_colors = submap.get_points_colors()
        name = str(submap.get_id())
        self.set_point_cloud(points_in_world_frame, points_colors, name, 0.001)

    def set_submap_poses(self, submap):
        # Add the camera poses to the visualization.
        extrinsics = submap.get_all_poses_world()
        if self.gradio_mode:
            for i in range(extrinsics.shape[0]):
                self.viewer.add_camera_pose(extrinsics[i])
        else:
            images = submap.get_all_frames()
            self.viewer.visualize_frames(extrinsics, images, submap.get_id())

    def export_3d_scene(self, output_path="output.glb"):
        return self.viewer.export(output_path)

    def update_all_submap_vis(self):
        for submap in self.map.get_submaps():
            self.set_submap_point_cloud(submap)
            self.set_submap_poses(submap)

    def update_latest_submap_vis(self):
        submap = self.map.get_latest_submap()
        self.set_submap_point_cloud(submap)
        self.set_submap_poses(submap)

    def add_points(self, pred_dict):
        """
        Args:
            pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        """
        # Unpack prediction dict
        images = pred_dict["images"]  # (S, 3, H, W)

        extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)
        # print(intrinsics_cam)

        detected_loops = pred_dict["detected_loops"]

        if self.use_point_map:
            world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
            conf = pred_dict["world_points_conf"]  # (S, H, W)
            world_points = world_points_map
        else:
            depth_map = pred_dict["depth"]  # (S, H, W, 1)
            conf = pred_dict["depth_conf"]  # (S, H, W)
            world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)

        # Convert images from (S, 3, H, W) to (S, H, W, 3)
        # Then flatten everything for the point cloud
        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # now (S, H, W, 3)

        # Flatten
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4)
        #print("cam_to_world: ", cam_to_world)

        # estimate focal length from points
        points_in_first_cam = world_points[0,...]
        h, w = points_in_first_cam.shape[0:2]

        new_pcd_num = self.current_working_submap.get_id()
        if self.first_edge:
            self.first_edge = False
            self.prior_pcd = world_points[-1,...].reshape(-1, 3)
            self.prior_conf = conf[-1,...].reshape(-1)
            # Add node to graph.
            H_w_submap = np.eye(4)
            self.graph.add_homography(new_pcd_num, H_w_submap)
            self.graph.add_prior_factor(new_pcd_num, H_w_submap, self.graph.anchor_noise)
        else:
            prior_pcd_num = self.map.get_largest_key()
            prior_submap = self.map.get_submap(prior_pcd_num)

            current_pts = world_points[0,...].reshape(-1, 3)
        
            # TODO conf should be using the threshold in its own submap
            good_mask = self.prior_conf > prior_submap.get_conf_threshold() * (conf[0,...,:].reshape(-1) > prior_submap.get_conf_threshold())
            
            if self.use_sim3:
                # Note we still use H and not T in variable names so we can share code with the Sim3 case, 
                # and SIM3 and SE3 are also subsets of the SL4 group
                R_temp = prior_submap.poses[prior_submap.get_last_non_loop_frame_index()][0:3,0:3]
                t_temp = prior_submap.poses[prior_submap.get_last_non_loop_frame_index()][0:3,3]
                T_temp = np.eye(4)
                T_temp[0:3,0:3] = R_temp
                T_temp[0:3,3] = t_temp
                T_temp = np.linalg.inv(T_temp)
                scale_factor = np.mean(np.linalg.norm((T_temp[0:3,0:3] @ self.prior_pcd[good_mask].T).T + T_temp[0:3,3], axis=1) / np.linalg.norm(current_pts[good_mask], axis=1))
                print(colored("scale factor", 'green'), scale_factor)
                H_relative = np.eye(4)
                H_relative[0:3,0:3] = R_temp
                H_relative[0:3,3] = t_temp

                # apply scale factor to points and poses
                world_points *= scale_factor
                cam_to_world[:, 0:3, 3] *= scale_factor
            else:
                H_relative = ransac_projective(current_pts[good_mask], self.prior_pcd[good_mask])
            
            H_w_submap = prior_submap.get_reference_homography() @ H_relative

            non_lc_frame = self.current_working_submap.get_last_non_loop_frame_index()
            pts_cam0_camn = world_points[non_lc_frame,...].reshape(-1, 3)
            self.prior_pcd = pts_cam0_camn
            self.prior_conf = conf[non_lc_frame,...].reshape(-1)

            # Add node to graph.
            self.graph.add_homography(new_pcd_num, H_w_submap)

            # Add between factor.
            self.graph.add_between_factor(prior_pcd_num, new_pcd_num, H_relative, self.graph.relative_noise)

            # print("added between factor", prior_pcd_num, new_pcd_num, H_relative)

        # Create and add submap.
        self.current_working_submap.set_reference_homography(H_w_submap)
        self.current_working_submap.add_all_poses(cam_to_world)
        self.current_working_submap.add_all_points(world_points, colors, conf, self.init_conf_threshold, intrinsics_cam)
        self.current_working_submap.set_conf_masks(conf) # TODO should make this work for point cloud conf as well

        # Add in loop closures if any were detected.
        for index, loop in enumerate(detected_loops):
            assert loop.query_submap_id == self.current_working_submap.get_id()

            loop_index = self.current_working_submap.get_last_non_loop_frame_index() + index + 1

            if self.use_sim3:
                pose_world_detected = self.map.get_submap(loop.detected_submap_id).get_pose_subframe(loop.detected_submap_frame)
                pose_world_query = self.current_working_submap.get_pose_subframe(loop_index)
                pose_world_detected = gtsam.Pose3(pose_world_detected)
                pose_world_query = gtsam.Pose3(pose_world_query)
                H_relative_lc = pose_world_detected.between(pose_world_query).matrix()
            else:
                points_world_detected = self.map.get_submap(loop.detected_submap_id).get_frame_pointcloud(loop.detected_submap_frame).reshape(-1, 3)
                points_world_query = self.current_working_submap.get_frame_pointcloud(loop_index).reshape(-1, 3)
                H_relative_lc = ransac_projective(points_world_query, points_world_detected)


            self.graph.add_between_factor(loop.detected_submap_id, loop.query_submap_id, H_relative_lc, self.graph.relative_noise)
            self.graph.increment_loop_closure() # Just for debugging and analysis, keep track of total number of loop closures
        self.map.add_submap(self.current_working_submap)

    def test_add_points(self, pred_dict, gnss_measurements=None):
        """
        Args:
            pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        """
        # Unpack prediction dict
        images = pred_dict["images"]  # (S, 3, H, W)

        extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
        intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)
        # print(intrinsics_cam)

        detected_loops = pred_dict["detected_loops"]

        if self.use_point_map:
            world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
            conf = pred_dict["world_points_conf"]  # (S, H, W)
            world_points = world_points_map
        else:
            depth_map = pred_dict["depth"]  # (S, H, W, 1)
            conf = pred_dict["depth_conf"]  # (S, H, W)
            world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)

        # Convert images from (S, 3, H, W) to (S, H, W, 3)
        # Then flatten everything for the point cloud
        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)  # now (S, H, W, 3)

        # Flatten
        cam_to_world = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4)
        #print("cam_to_world: ", cam_to_world)

        # estimate focal length from points
        points_in_first_cam = world_points[0,...]
        h, w = points_in_first_cam.shape[0:2]

        new_pcd_num = self.current_working_submap.get_id()
        if self.first_edge:
            self.first_edge = False
            self.prior_pcd = world_points[-1,...].reshape(-1, 3)
            self.prior_conf = conf[-1,...].reshape(-1)
            # Add node to graph.
            H_w_submap = np.eye(4)
            self.graph.add_homography(new_pcd_num, H_w_submap)
            self.graph.add_prior_factor(new_pcd_num, H_w_submap, self.graph.anchor_noise)
        else:
            prior_pcd_num = self.map.get_largest_key()
            prior_submap = self.map.get_submap(prior_pcd_num)

            current_pts = world_points[0,...].reshape(-1, 3)
        
            # TODO conf should be using the threshold in its own submap
            good_mask = self.prior_conf > prior_submap.get_conf_threshold() * (conf[0,...,:].reshape(-1) > prior_submap.get_conf_threshold())
            
            if self.use_sim3:
                # Note we still use H and not T in variable names so we can share code with the Sim3 case, 
                # and SIM3 and SE3 are also subsets of the SL4 group
                R_temp = prior_submap.poses[prior_submap.get_last_non_loop_frame_index()][0:3,0:3]
                t_temp = prior_submap.poses[prior_submap.get_last_non_loop_frame_index()][0:3,3]
                T_temp = np.eye(4)
                T_temp[0:3,0:3] = R_temp
                T_temp[0:3,3] = t_temp
                T_temp = np.linalg.inv(T_temp)
                scale_factor = np.mean(np.linalg.norm((T_temp[0:3,0:3] @ self.prior_pcd[good_mask].T).T + T_temp[0:3,3], axis=1) / np.linalg.norm(current_pts[good_mask], axis=1))
                print(colored("scale factor", 'green'), scale_factor)
                H_relative = np.eye(4)
                H_relative[0:3,0:3] = R_temp
                H_relative[0:3,3] = t_temp

                # apply scale factor to points and poses
                world_points *= scale_factor
                cam_to_world[:, 0:3, 3] *= scale_factor
            else:
                H_relative = ransac_projective(current_pts[good_mask], self.prior_pcd[good_mask])
            
            H_w_submap = prior_submap.get_reference_homography() @ H_relative

            non_lc_frame = self.current_working_submap.get_last_non_loop_frame_index()
            pts_cam0_camn = world_points[non_lc_frame,...].reshape(-1, 3)
            self.prior_pcd = pts_cam0_camn
            self.prior_conf = conf[non_lc_frame,...].reshape(-1)

            # Add node to graph.
            self.graph.add_homography(new_pcd_num, H_w_submap)

            # Add between factor.
            self.graph.add_between_factor(prior_pcd_num, new_pcd_num, H_relative, self.graph.relative_noise)

            # print("added between factor", prior_pcd_num, new_pcd_num, H_relative)
        if gnss_measurements is not None and len(gnss_measurements) > 0:
            # Convert to ENU coordinates relative to first frame
            enu_positions = []
            for i, gps_info in enumerate(gnss_measurements):
                if i == 0 and not self.gnss_processer.reference_set:
                    # Set first frame as reference
                    self.gnss_processer.setReference(gps_info)
                
                # Convert to ENU coordinates
                enu = self.gnss_processer.gps_to_enu(gps_info)
                enu_positions.append(enu)
            
            # Store positions for scale alignment
            if len(enu_positions) > 0:
                visual_pos = H_w_submap[0:3, 3]  # Current visual position
                self.visual_positions.append(visual_pos)
                self.gnss_positions.append(enu_positions[0])  # First frame in submap
        # Create and add submap.
        self.current_working_submap.set_reference_homography(H_w_submap)
        self.current_working_submap.add_all_poses(cam_to_world)
        self.current_working_submap.add_all_points(world_points, colors, conf, self.init_conf_threshold, intrinsics_cam)
        self.current_working_submap.set_conf_masks(conf) # TODO should make this work for point cloud conf as well

        # Add in loop closures if any were detected.
        for index, loop in enumerate(detected_loops):
            assert loop.query_submap_id == self.current_working_submap.get_id()

            loop_index = self.current_working_submap.get_last_non_loop_frame_index() + index + 1

            if self.use_sim3:
                pose_world_detected = self.map.get_submap(loop.detected_submap_id).get_pose_subframe(loop.detected_submap_frame)
                pose_world_query = self.current_working_submap.get_pose_subframe(loop_index)
                pose_world_detected = gtsam.Pose3(pose_world_detected)
                pose_world_query = gtsam.Pose3(pose_world_query)
                H_relative_lc = pose_world_detected.between(pose_world_query).matrix()
            else:
                points_world_detected = self.map.get_submap(loop.detected_submap_id).get_frame_pointcloud(loop.detected_submap_frame).reshape(-1, 3)
                points_world_query = self.current_working_submap.get_frame_pointcloud(loop_index).reshape(-1, 3)
                H_relative_lc = ransac_projective(points_world_query, points_world_detected)


            self.graph.add_between_factor(loop.detected_submap_id, loop.query_submap_id, H_relative_lc, self.graph.relative_noise)
            self.graph.increment_loop_closure() # Just for debugging and analysis, keep track of total number of loop closures
        self.map.add_submap(self.current_working_submap)
        if len(self.visual_positions) >= 2 and not self.scale_aligned:
            if self.align_scale():
                self.apply_scale_to_poses(world_points)
                self.add_gnss_constraints()
        # if gnss_measurements is not None and len(gnss_measurements) > 0:
        #     print(f"Adding {len(gnss_measurements)} GNSS measurements to graph")
        #     self.graph.add_gnss(gnss_measurements)

    def sample_pixel_coordinates(self, H, W, n):
        # Sample n random row indices (y-coordinates)
        y_coords = torch.randint(0, H, (n,), dtype=torch.float32)
        # Sample n random column indices (x-coordinates)
        x_coords = torch.randint(0, W, (n,), dtype=torch.float32)
        # Stack to create an (n,2) tensor
        pixel_coords = torch.stack((y_coords, x_coords), dim=1)
        return pixel_coords

    def run_predictions(self, image_names, model, max_loops):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = load_and_preprocess_images(image_names).to(device)
        #print(f"Preprocessed images shape: {images.shape}")

        # print("Running inference...")
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        # Check for loop closures
        new_pcd_num = self.map.get_largest_key() + 1
        new_submap = Submap(new_pcd_num)
        # new_submap.add_all_frames(images)
        new_submap.add_all_frames(images)
        new_submap.set_frame_ids(image_names)
        new_submap.set_all_retrieval_vectors(self.image_retrieval.get_all_submap_embeddings(new_submap))

        # TODO implement this
        detected_loops = self.image_retrieval.find_loop_closures(self.map, new_submap, max_loop_closures=max_loops)
        if len(detected_loops) > 0:
            # print(colored("detected_loops", "yellow"), detected_loops)
            pass
        retrieved_frames = self.map.get_frames_from_loops(detected_loops)

        num_loop_frames = len(retrieved_frames)
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)
        if num_loop_frames > 0:
            image_tensor = torch.stack(retrieved_frames)  # Shape (n, 3, w, h)
            images = torch.cat([images, image_tensor], dim=0) # Shape (s+n, 3, w, h)

            # TODO we don't really need to store the loop closure frame again, but this makes lookup easier for the visualizer.
            # We added the frame to the submap once before to get the retrieval vectors,
            new_submap.add_all_frames(images)

        self.current_working_submap = new_submap

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        predictions["detected_loops"] = detected_loops
        # print("extrinsic_after", extrinsic)
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

        return predictions
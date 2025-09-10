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

from vggt_slam.loop_closure import ImageRetrieval
from vggt_slam.frame_overlap import FrameTracker
from vggt_slam.map import GraphMap
from vggt_slam.submap import Submap
from vggt_slam.h_solve import ransac_projective, ransac_projective_improved, apply_homography
from vggt_slam.gradio_viewer import TrimeshViewer
from KalmanFilter import KalmanFilter
from pcltest5 import move, move2

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
        self.kf = KalmanFilter(dim_state=6, dim_measurement=3)
        self.image_retrieval = ImageRetrieval()
        self.current_working_submap = None

        self.first_edge = True

        self.T_w_kf_minus = None

        self.prior_pcd = None
        self.prior_conf = None

        self.scale_factor_mean = 1.0
        self.scale_factor = 1.0
        self.scale_aligned = False
        self.visual_positions = []  # 存储视觉位置历史
        self.gnss_positions = []    # 存储GNSS位置历史
        self.last_visual_pos = None
        self.last_gnss_pos = None
        
        print("Starting viser server...")
    def align_scale(self):
        """计算尺度因子但不直接缩放位姿 - 通过GNSS约束引导优化"""
        # if len(self.visual_positions) < 1:
        #     print("Warning: Not enough points for scale alignment")
        #     return False
        
        # 计算从起点到各点的距离比例
        scale_factors = []
        for i in range(1, len(self.visual_positions)):
            visual_delta_xy = self.visual_positions[i][:2] - self.visual_positions[i-1][:2]
            gnss_delta_xy = self.gnss_positions[i][:2] - self.gnss_positions[i-1][:2]
            visual_delta = self.visual_positions[i] - self.visual_positions[i-1]
            gnss_delta = self.gnss_positions[i] - self.gnss_positions[i-1]
            visual_dist = np.linalg.norm(visual_delta)
            gnss_dist = np.linalg.norm(gnss_delta)
            
            if visual_dist > 0.05:  # 避免除以小值
                factor = gnss_dist / visual_dist
                scale_factors.append(factor)
        
        if scale_factors:
            self.scale_factor_midian = np.median(scale_factors)
            self.scale_factor_mean = np.mean(scale_factors)
            self.scale_factor = self.scale_factor_mean
            print(f"Scale alignment: midian scale factor = {self.scale_factor_midian:.4f}")
            print(f"Scale alignment: mean scale factor = {self.scale_factor_mean:.4f}")
            self.scale_aligned = True
            print("Scale alignment complete. GNSS constraints will guide the optimizer.")
            return True
        else:
            print("Warning: Could not compute scale factor")
            return False


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
            # print(world_points)
            current_pts = world_points[0,...].reshape(-1, 3)
        
            # TODO conf should be using the threshold in its own submap
            # good_mask = self.prior_conf > prior_submap.get_conf_threshold() * (conf[0,...,:].reshape(-1) > prior_submap.get_conf_threshold())
            prior_thr   = prior_submap.get_conf_threshold()
            curr_thr    = prior_submap.get_conf_threshold()   # 如果当前子图也有自己的阈值，改用它
            good_mask = (self.prior_conf.reshape(-1) > prior_thr) & (conf[0,:].reshape(-1) > curr_thr)
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
                H_relative = ransac_projective(current_pts, self.prior_pcd)
                if prior_pcd_num >= 1:
                    H_relative = ransac_projective(current_pts, self.prior_pcd)
                    prev_pcd = prior_submap.get_all_points().reshape(-1, 3)
                    prev_colors = prior_submap.get_all_colors().reshape(-1, 3).astype(np.float32) / 255.0
                    pcd_prev = o3d.geometry.PointCloud()
                    pcd_prev.points = o3d.utility.Vector3dVector(prev_pcd)
                    color_prev = o3d.utility.Vector3dVector(prev_colors)
                    pcd_prev.colors = color_prev
                    curr_pcd = world_points.reshape(-1, 3)
                    curr_pcd = apply_homography(H_relative, curr_pcd)
                    curr_colors = colors.reshape(-1, 3).astype(np.float32) / 255.0
                    pcd_curr = o3d.geometry.PointCloud()
                    pcd_curr.points = o3d.utility.Vector3dVector(curr_pcd)
                    color_curr = o3d.utility.Vector3dVector(curr_colors)
                    pcd_curr.colors = color_curr
                    # pcd_curr.transform(H_relative)
                    # o3d.visualization.draw_geometries([pcd_prev, pcd_curr])
                    # o3d.visualization.draw_geometries([pcd_curr])
                    # moved, trans_matrix = move(down=-0.01, up=0.01, base_pcd=pcd_prev, move_pcd=pcd_curr, is_begin=False)
                    moved, trans_matrix, point_indices = move(
                        down=-0.01,
                        up=0.01,
                        base_pcd=pcd_prev,
                        move_pcd=pcd_curr,
                        is_begin=False,
                        return_indices=True
                    )
                    # moved = pcd_curr
                    # trans_matrix = np.eye(4)
                    moved = moved.transform(np.linalg.inv(trans_matrix))
                    moved = moved.transform(np.linalg.inv(H_relative))
                    moved_points = np.asarray(moved.points)
                    moved_colors = np.asarray(moved.colors)
                    colors = (moved_colors * 255).astype(np.uint8).reshape(colors.shape)
                    world_points = moved_points.reshape(world_points.shape)
                    H_relative = trans_matrix @ H_relative
                else:
                    H_relative = ransac_projective(current_pts, self.prior_pcd)
                # o3d.visualization.draw_geometries([pcd_prev, moved])
            H_w_submap = prior_submap.get_reference_homography() @ H_relative


            non_lc_frame = self.current_working_submap.get_last_non_loop_frame_index()
            pts_cam0_camn = world_points[non_lc_frame,...].reshape(-1, 3)

            self.prior_pcd = pts_cam0_camn
            self.prior_conf = conf[non_lc_frame,...].reshape(-1)
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pts_cam0_camn)
            # pcd.colors = o3d.utility.Vector3dVector(colors[non_lc_frame,...].reshape(-1, 3).astype(np.float32) / 255.0)
            # o3d.visualization.draw_geometries([pcd])
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
        # print("================", len(self.current_working_submap.frame_ids))
        # for index in range(len(self.current_working_submap.frame_ids)):
        #     if index >=1:
        #         points_world_detected = self.current_working_submap.get_frame_pointcloud(index-1).reshape(-1, 3)
        #         points_world_query    = self.current_working_submap.get_frame_pointcloud(index).reshape(-1, 3)
        #         colors_detected = self.current_working_submap.get_frame_color(index-1).reshape(-1, 3).astype(np.float32) / 255.0
        #         colors_query    = self.current_working_submap.get_frame_color(index).reshape(-1, 3).astype(np.float32) / 255.0
        #         H_relative_lc = ransac_projective(points_world_query, points_world_detected)
        #         points_query_in_detected = apply_homography(H_relative_lc, points_world_query)
        #         pcd_det  = o3d.geometry.PointCloud()
        #         pcd_det.points  = o3d.utility.Vector3dVector(points_world_detected)
        #         pcd_qry  = o3d.geometry.PointCloud()
        #         pcd_qry.points  = o3d.utility.Vector3dVector(points_query_in_detected)
        #         pcd_det.colors = o3d.utility.Vector3dVector(colors_detected)
        #         pcd_qry.colors = o3d.utility.Vector3dVector(colors_query)
        #         moved_lc, trans_lc = move(down=-0.05, up=0.05, base_pcd=pcd_det, move_pcd=pcd_qry, is_begin=False)
        #         # o3d.visualization.draw_geometries([moved_lc, pcd_qry, pcd_det])
        #         moved_lc = moved_lc.transform(np.linalg.inv(trans_lc))
        #         moved_lc = moved_lc.transform(np.linalg.inv(H_relative_lc))
        #         moved_points = np.asarray(moved_lc.points)
        #         moved_colors = np.asarray(moved_lc.colors)
        #         moved_colors = (moved_colors * 255).astype(np.uint8)
        #         world_points[index] = moved_points.reshape(world_points[index].shape)
        #         # o3d.visualization.draw_geometries([moved_lc, pcd_qry])
        #         self.current_working_submap.set_frame_pointcloud(index, moved_points.reshape(self.current_working_submap.get_frame_pointcloud(index).shape))
        #         self.current_working_submap.set_frame_color(index, moved_colors.reshape(self.current_working_submap.get_frame_color(index).shape))
        #         H_relative_lc = trans_lc @ H_relative_lc
        #         self.graph.add_between_factor(index-1, index, H_relative_lc, self.graph.relative_noise)
        # self.current_working_submap.add_all_points(world_points, colors, conf, self.init_conf_threshold, intrinsics_cam)
        for index, loop in enumerate(detected_loops):
            assert loop.query_submap_id == self.current_working_submap.get_id()
            #时间线:
            # 历史子图1 (detected_submap_id=1)
            # 历史子图2 (detected_submap_id=2)
            # 历史子图3 (detected_submap_id=3)
            # ↓
            # 当前子图 (query_submap_id=4)
            # │
            # ├── 帧0
            # ├── 帧1
            # ├── 帧2 → 检测到与历史子图2匹配 (loop_index=2, detected_submap_id=2)
            # ├── 帧3
            # └── 帧4
            loop_index = self.current_working_submap.get_last_non_loop_frame_index() + index + 1
            print("loop index", loop_index)
            print("query submap id", loop.query_submap_id)
            print("detected submap id", loop.detected_submap_id)
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
                # points_world_detected = self.map.get_submap(loop.detected_submap_id).get_frame_pointcloud(loop.detected_submap_frame).reshape(-1, 3)
                # points_world_query    = self.current_working_submap.get_frame_pointcloud(loop_index).reshape(-1, 3)
                # colors_detected = self.map.get_submap(loop.detected_submap_id).get_frame_color(loop.detected_submap_frame).reshape(-1, 3).astype(np.float32) / 255.0
                # colors_query    = self.current_working_submap.get_frame_color(loop_index).reshape(-1, 3).astype(np.float32) / 255.0
                # H_relative_lc = ransac_projective(points_world_query, points_world_detected)
                # points_query_in_detected = apply_homography(H_relative_lc, points_world_query)
                # pcd_det  = o3d.geometry.PointCloud()
                # pcd_det.points  = o3d.utility.Vector3dVector(points_world_detected)
                # pcd_qry  = o3d.geometry.PointCloud()
                # pcd_qry.points  = o3d.utility.Vector3dVector(points_query_in_detected)
                # pcd_det.colors = o3d.utility.Vector3dVector(colors_detected)
                # pcd_qry.colors = o3d.utility.Vector3dVector(colors_query)
                # moved_lc, trans_lc = move(down=-0.05, up=0.05, base_pcd=pcd_det, move_pcd=pcd_qry, is_begin=False)
                # # o3d.visualization.draw_geometries([moved_lc, pcd_qry, pcd_det])
                # moved_lc = moved_lc.transform(np.linalg.inv(trans_lc))
                # moved_lc = moved_lc.transform(np.linalg.inv(H_relative_lc))
                # moved_points = np.asarray(moved_lc.points)
                # moved_colors = np.asarray(moved_lc.colors)
                # moved_colors = (moved_colors * 255).astype(np.uint8)
                # world_points[loop_index] = moved_points.reshape(world_points[loop_index].shape)
                # # o3d.visualization.draw_geometries([moved_lc, pcd_qry])
                # self.current_working_submap.set_frame_pointcloud(loop_index, moved_points.reshape(self.current_working_submap.get_frame_pointcloud(loop_index).shape))
                # self.current_working_submap.set_frame_color(loop_index, moved_colors.reshape(self.current_working_submap.get_frame_color(loop_index).shape))
                # H_relative_lc = trans_lc @ H_relative_lc
                #
                # # H_relative_lc = H_relative_lc @ trans_lc
                # # o3d.io.write_point_cloud("detected.pcd", pcd_det)
                # # o3d.io.write_point_cloud("query.pcd", pcd_qry)
                # #
                # # o3d.io.write_point_cloud(f"./test_pcd/{num}.pcd", moved_lc)
                # # o3d.visualization.draw_geometries([moved_lc])

            self.graph.add_between_factor(loop.detected_submap_id, loop.query_submap_id, H_relative_lc, self.graph.relative_noise)
            self.graph.increment_loop_closure() # Just for debugging and analysis, keep track of total number of loop closures

            # print("added loop closure factor", loop.detected_submap_id, loop.query_submap_id, H_relative_lc)
            # print("homography between nodes estimated to be", np.linalg.inv(self.map.get_submap(loop.detected_submap_id).get_reference_homography()) @ H_w_submap)

            # print("relative_pose factor added", relative_pose)
        self.current_working_submap.add_all_points(world_points, colors, conf, self.init_conf_threshold, intrinsics_cam)
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
            gnss_measurements (list): 
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
            # good_mask = self.prior_conf > prior_submap.get_conf_threshold() * (conf[0,...,:].reshape(-1) > prior_submap.get_conf_threshold())
            prior_thr   = prior_submap.get_conf_threshold()
            curr_thr    = prior_submap.get_conf_threshold()   # 如果当前子图也有自己的阈值，改用它
            good_mask = (self.prior_conf.reshape(-1) > prior_thr) & (conf[0,:].reshape(-1) > curr_thr)
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
                # R_temp = prior_submap.poses[prior_submap.get_last_non_loop_frame_index()][0:3,0:3]
                # t_temp = prior_submap.poses[prior_submap.get_last_non_loop_frame_index()][0:3,3]
                # T_temp = np.eye(4)
                # T_temp[0:3,0:3] = R_temp
                # T_temp[0:3,3] = t_temp
                # T_temp = np.linalg.inv(T_temp)
                # scale_factor = np.mean(np.linalg.norm((T_temp[0:3,0:3] @ self.prior_pcd[good_mask].T).T + T_temp[0:3,3], axis=1) / np.linalg.norm(current_pts[good_mask], axis=1))
                # print(colored("scale factor", 'green'), scale_factor)
                # # apply scale factor to points and poses
                # world_points *= scale_factor
                # # cam_to_world[:, 0:3, 3] *= scale_factor
                # current_pts[good_mask] *= scale_factor
                H_relative = ransac_projective(current_pts[good_mask], self.prior_pcd[good_mask])
                # trans = turbo_reg(current_pts[good_mask], self.prior_pcd[good_mask])
                # print(f"=================================={H_relative}==================================")
                # print(f"=================================={trans}==================================")
                # H_relative = trans
                # R_temp = prior_submap.poses[prior_submap.get_last_non_loop_frame_index()][0:3,0:3]
                # t_temp = prior_submap.poses[prior_submap.get_last_non_loop_frame_index()][0:3,3]
                # T_temp = np.eye(4)
                # T_temp[0:3,0:3] = R_temp
                # T_temp[0:3,3] = t_temp
                # T_temp = np.linalg.inv(T_temp)
                # scale_factor = np.mean(np.linalg.norm((T_temp[0:3,0:3] @ self.prior_pcd[good_mask].T).T + T_temp[0:3,3], axis=1) / np.linalg.norm(current_pts[good_mask], axis=1))
                # print(colored("scale factor", 'green'), scale_factor)
                # # H_relative = np.eye(4)
                # H_relative[0:3,0:3] = R_temp
                # H_relative[0:3,3] = t_temp
                # print(f"=================================={H_relative}==================================")
                # # apply scale factor to points and poses
                # world_points *= scale_factor
                # cam_to_world[:, 0:3, 3] *= scale_factor
                # #R_temp, t_temp = icp(current_pts[good_mask], self.prior_pcd[good_mask])
                # # H_relative = np.eye(4)
                # # H_relative[0:3,0:3] = R_temp
                # # H_relative[0:3,3] = t_temp
            H_w_submap = prior_submap.get_reference_homography() @ H_relative

            # # Visualize the point clouds
            # pcd1 = o3d.geometry.PointCloud()
            # pcd1.points = o3d.utility.Vector3dVector(self.prior_pcd)
            # pcd1 = color_point_cloud_by_confidence(pcd1, self.prior_conf)
            # pcd2 = o3d.geometry.PointCloud()
            # # current_pts = world_points[0,...].reshape(-1, 3)
            # # points = apply_homography(H_relative, current_pts)
            # # pcd2.points = o3d.utility.Vector3dVector(points)
            # # # pcd2 = color_point_cloud_by_confidence(pcd2, conf_flat, cmap='jet')
            # o3d.visualization.draw_geometries([pcd1, pcd2])

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
        current_visual_pos = cam_to_world[-1, :3, 3]  # 第一个位姿的平移部分
        # 处理GNSS数据
        if gnss_measurements is not None and len(gnss_measurements) > 0:
            last_gnss = gnss_measurements[-1]
            lat, lon, alt = last_gnss
            current_gnss = np.array(self.graph.gnss_processor.lla_to_enu(lat, lon, alt))
            # 如果是第一个有GNSS的帧，初始化
            if self.last_gnss_pos is None:
                self.last_visual_pos = current_visual_pos
                self.last_gnss_pos = current_gnss
                self.visual_positions.append(current_visual_pos)
                self.gnss_positions.append(current_gnss)
                print(f"Initialized GNSS alignment at frame {len(self.visual_positions)}")
            else:
                visual_delta_xy = current_visual_pos[:2] - self.last_visual_pos[:2]
                gnss_delta_xy = current_gnss[:2] - self.last_gnss_pos[:2]
                # 计算相对运动
                visual_delta = current_visual_pos - self.last_visual_pos
                gnss_delta = current_gnss - self.last_gnss_pos
                self.visual_positions.append(current_visual_pos)
                self.gnss_positions.append(current_gnss)
                if True:
                    # 计算当前段的尺度因子
                    visual_dist = np.linalg.norm(visual_delta)
                    gnss_dist = np.linalg.norm(gnss_delta)
                    print(f"Visual distance: {visual_dist:.4f}, GNSS distance: {gnss_dist:.4f}")
                    if visual_dist > 0.05:
                        scale_estimate = gnss_dist / visual_dist
                        print(f"Scale estimate from frame {len(self.visual_positions)-1} to {len(self.visual_positions)}: {scale_estimate:.4f}")
                        self.scale_factor = scale_estimate
                        # 如果有足够的估计点，计算全局尺度
                        #if len(self.visual_positions) >= 1:
                        self.align_scale()
                        self.scale_aligned = True
            # 更新最后位置
            self.last_visual_pos = current_visual_pos
            self.last_gnss_pos = current_gnss
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
                # trans_lc = turbo_reg(points_world_query, points_world_detected)
                # H_relative_lc = trans_lc
            self.graph.add_between_factor(loop.detected_submap_id, loop.query_submap_id, H_relative_lc, self.graph.relative_noise)
            self.graph.increment_loop_closure() # Just for debugging and analysis, keep track of total number of loop closures

            # print("added loop closure factor", loop.detected_submap_id, loop.query_submap_id, H_relative_lc)
            # print("homography between nodes estimated to be", np.linalg.inv(self.map.get_submap(loop.detected_submap_id).get_reference_homography()) @ H_w_submap)
        self.map.add_submap(self.current_working_submap)
        # 如果已经对齐尺度，应用到GNSS约束
        if gnss_measurements is not None and len(gnss_measurements) > 0:
                # 获取当前子图的参考帧（通常是第一帧）
                ref_frame_idx = -1
                
                # 确保有参考帧的GNSS数据
                if ref_frame_idx < len(gnss_measurements):
                    lat, lon, alt = gnss_measurements[ref_frame_idx]
                    enu = self.graph.gnss_processor.lla_to_enu(lat, lon, alt)
                    
                    # 获取当前子图ID
                    submap_id = self.current_working_submap.get_id()
                    
                    # 添加GNSS约束
                    # self.graph.add_gnss_to_submap(
                    #     submap_id, 
                    #     enu, 
                    #     scale_factor=self.scale_factor if self.scale_aligned else 180
                    # )

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

        # FIXED_FOCAL = 300       
        # new_intrinsic = intrinsic.clone()
        
        # # original_cx = intrinsic[..., 0, 2]
        # # original_cy = intrinsic[..., 1, 2]
        
        # new_intrinsic[..., 0, 0] = FIXED_FOCAL  # fx
        # new_intrinsic[..., 1, 1] = FIXED_FOCAL  # fy
        
        # print("\n============ 焦距修正 ============")
        # print(f"原始形状: {intrinsic.shape}")
        # print(f"首帧原始值:\n{intrinsic[0,0].cpu().numpy()}")
        # print(f"首帧修正后:\n{new_intrinsic[0,0].cpu().numpy()}")
        # print("================================")
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        predictions["detected_loops"] = detected_loops

        # intrinsic = new_intrinsic[0]  # 形状变为[num_frames, 3, 3]
        # print("\n============ 焦距输出 ============")
        # for frame_idx in range(intrinsic.shape[0]):  # 遍历所有帧
        #     frame_matrix = intrinsic[frame_idx]
        #     fx = frame_matrix[0, 0].item()
        #     fy = frame_matrix[1, 1].item()
        #     cx = frame_matrix[0, 2].item()
        #     cy = frame_matrix[1, 2].item()
            
        #     print(f"帧 {frame_idx+1}/{intrinsic.shape[0]}:")
        #     print(f"  fx = {fx:.3f}")
        #     print(f"  fy = {fy:.3f}")
        #     print(f"  主点 (cx, cy) = ({cx:.1f}, {cy:.1f})")
        # print("================================\n")
        # print("extrinsic_after", extrinsic)
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy

        return predictions
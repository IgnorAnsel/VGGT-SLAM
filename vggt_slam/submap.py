import re
import os
import cv2
import torch
import numpy as np
import open3d as o3d

class Submap:
    def __init__(self, submap_id):
        self.submap_id = submap_id
        self.H_world_map = None
        self.R_world_map = None
        self.poses = None
        self.frames = None
        self.vggt_intrinscs = None
        self.retrieval_vectors = None
        self.colors = None # (S, H, W, 3)
        self.conf = None # (S, H, W)
        self.conf_masks = None # (S, H, W)
        self.conf_threshold = None
        self.pointclouds = None # (S, H, W, 3)
        self.voxelized_points = None
        self.last_non_loop_frame_index = None
        self.frame_ids = None
    def add_all_poses(self, poses):
        self.poses = poses
    def get_all_points(self):
        return self.pointclouds
    def get_all_colors(self):
        return self.colors
    def get_all_conf_masks(self):
        return self.conf_masks
    def get_all_confidences(self):
        return self.conf
    def transH_world_map(self, trans):
        self.H_world_map = trans @ self.H_world_map
    def trans(self, trans):
        self.poses = self.poses @ trans
        self.H_world_map = trans @ self.H_world_map
    def add_all_points(self, points, colors, conf, conf_threshold_percentile, intrinsics):
        self.pointclouds = points
        self.colors = colors
        self.conf = conf
        self.conf_threshold = np.percentile(self.conf, conf_threshold_percentile)
        self.vggt_intrinscs = intrinsics
    def update_all_points(self, points, colors):
        self.pointclouds = points
        self.colors = colors
    def add_all_frames(self, frames):
        self.frames = frames
    
    def add_all_retrieval_vectors(self, retrieval_vectors):
        self.retrieval_vectors = retrieval_vectors

    def get_id(self):
        return self.submap_id

    def get_conf_threshold(self):
        return self.conf_threshold
    
    def get_frame_at_index(self, index):
        return self.frames[index, ...]
    
    def get_last_non_loop_frame_index(self):
        return self.last_non_loop_frame_index

    def get_all_frames(self):
        return self.frames
    
    def get_all_retrieval_vectors(self):
        return self.retrieval_vectors

    def get_all_poses_world(self, ignore_loop_closure_frames=False):
        projection_mat_list = self.vggt_intrinscs @ np.linalg.inv(self.poses)[:,0:3,:] @ np.linalg.inv(self.H_world_map)
        poses = []
        for index, projection_mat in enumerate(projection_mat_list):
            cal, rot, trans = cv2.decomposeProjectionMatrix(projection_mat)[0:3]
            # print("cal", cal/cal[2,2])
            trans = trans/trans[3,0] # TODO see if we should normalize the rotation too with this.
            pose = np.eye(4)
            pose[0:3, 0:3] = np.linalg.inv(rot)
            pose[0:3,3] = trans[0:3,0]
            poses.append(pose)
            if ignore_loop_closure_frames and index == self.last_non_loop_frame_index:
                break
        return np.stack(poses, axis=0)
    
    def get_frame_pointcloud(self, pose_index):
        return self.pointclouds[pose_index]
    def set_frame_pointcloud(self, pose_index, pointcloud):
        self.pointclouds[pose_index] = pointcloud
    def set_frame_color(self, pose_index, color):
        self.colors[pose_index] = color
    def get_frame_color(self, pose_index):
        return self.colors[pose_index]
    def set_frame_ids(self, file_paths):
        """
        Extract the frame number (integer or decimal) from the file names, 
        removing any leading zeros, and add them all to a list.

        Note: This does not include any of the loop closure frames.
        """
        frame_ids = []
        for path in file_paths:
            filename = os.path.basename(path)
            match = re.search(r'\d+(?:\.\d+)?', filename)  # matches integers and decimals
            if match:
                frame_ids.append(float(match.group()))
            else:
                raise ValueError(f"No number found in image name: {filename}")
        self.frame_ids = frame_ids

    def set_last_non_loop_frame_index(self, last_non_loop_frame_index):
        self.last_non_loop_frame_index = last_non_loop_frame_index

    def set_reference_homography(self, H_world_map):
        self.H_world_map = H_world_map
    
    def set_all_retrieval_vectors(self, retrieval_vectors):
        self.retrieval_vectors = retrieval_vectors
    
    def set_conf_masks(self, conf_masks):
        self.conf_masks = conf_masks

    def get_reference_homography(self):
        return self.H_world_map

    def get_pose_subframe(self, pose_index):
        return np.linalg.inv(self.poses[pose_index])
    
    def get_frame_ids(self):
        # Note this does not include any of the loop closure frames
        return self.frame_ids

    def filter_data_by_confidence(self, data):
        init_conf_mask = self.conf >= self.conf_threshold
        return data[init_conf_mask]

    def get_points_list_in_world_frame(self, ignore_loop_closure_frames=False):
        point_list = []
        frame_id_list = []
        frame_conf_mask = []
        for index,points in enumerate(self.pointclouds):
            points_flat = points.reshape(-1, 3)
            points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
            points_transformed = (self.H_world_map @ points_homogeneous.T).T
            point_list.append((points_transformed[:, :3] / points_transformed[:, 3:]).reshape(points.shape))
            frame_id_list.append(self.frame_ids[index])
            conf_mask = self.conf_masks[index] >= self.conf_threshold
            frame_conf_mask.append(conf_mask)
            if ignore_loop_closure_frames and index == self.last_non_loop_frame_index:
                break
        return point_list, frame_id_list, frame_conf_mask
    def updata_world_points(self, world_points, input_colors):
        # temp = np.prod(world_points[:-1])
        homogeneous = np.hstack([world_points, np.ones((len(world_points), 1))])
        H_inv = np.linalg.inv(self.H_world_map)
        sensor_points = (H_inv @ homogeneous.T).T
        temp = sensor_points[:, :3] / sensor_points[:, 3:]

        self.pointclouds = temp.reshape(self.pointclouds.shape)
        self.colors = input_colors.reshape(self.colors.shape)
    def updata_world_points_with_id(self, world_points, input_colors, id):
        # temp = np.prod(world_points[:-1])
        homogeneous = np.hstack([world_points, np.ones((len(world_points), 1))])
        H_inv = np.linalg.inv(self.H_world_map)
        sensor_points = (H_inv @ homogeneous.T).T
        temp = sensor_points[:, :3] / sensor_points[:, 3:]
        self.pointclouds[id] = temp.reshape(self.pointclouds[id].shape)
        self.colors[id] = input_colors.reshape(self.colors[id].shape)
    def get_points_in_world_frame(self, is_fitter = True):
        if is_fitter:
            points = self.filter_data_by_confidence(self.pointclouds)
        else:
            points = self.pointclouds
        points_flat = points.reshape(-1, 3)
        points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
        # test = np.eye(4)
        # print("H_world_map", self.H_world_map)
        points_transformed = (self.H_world_map @ points_homogeneous.T).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_transformed[:, :3] / points_transformed[:, 3:])
        pcd.colors = o3d.utility.Vector3dVector(self.filter_data_by_confidence(self.colors) / 255.0)
        # o3d.visualization.draw_geometries([pcd])
        return points_transformed[:, :3] / points_transformed[:, 3:]
    def get_points_in_world_frame_with_id(self, is_fitter = True, id = 0):
        if is_fitter:
            points = self.filter_data_by_confidence(self.pointclouds)
        else:
            points = self.pointclouds[id]
        points_flat = points.reshape(-1, 3)
        points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
        points_transformed = (self.H_world_map @ points_homogeneous.T).T
        return points_transformed[:, :3] / points_transformed[:, 3:]
    def get_points_len(self):
        return self.pointclouds.shape[0]
    def get_voxel_points_in_world_frame(self, voxel_size, nb_points=8, factor_for_outlier_rejection=2.0):
        if self.voxelized_points is None:
            if voxel_size > 0.0:
                points = self.filter_data_by_confidence(self.pointclouds)
                points_flat = points.reshape(-1, 3)
                colors = self.filter_data_by_confidence(self.colors)
                colors_flat = colors.reshape(-1, 3) / 255.0

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_flat)
                pcd.colors = o3d.utility.Vector3dVector(colors_flat)
                self.voxelized_points = pcd.voxel_down_sample(voxel_size=voxel_size)
                if (nb_points > 0):
                    self.voxelized_points, _ = self.voxelized_points.remove_radius_outlier(nb_points=nb_points,
                                                                                           radius=voxel_size * factor_for_outlier_rejection)
            else:
                raise RuntimeError("`voxel_size` should be larger than 0.0.")

        points_flat = np.asarray(self.voxelized_points.points)
        points_homogeneous = np.hstack([points_flat, np.ones((points_flat.shape[0], 1))])
        points_transformed = (self.H_world_map @ points_homogeneous.T).T

        voxelized_points_in_world_frame = o3d.geometry.PointCloud()
        voxelized_points_in_world_frame.points = o3d.utility.Vector3dVector(points_transformed[:, :3] / points_transformed[:, 3:])
        voxelized_points_in_world_frame.colors = self.voxelized_points.colors
        return voxelized_points_in_world_frame
    def get_points_colors(self, is_fitter = True):
        if is_fitter:
            colors = self.filter_data_by_confidence(self.colors)
        else:
            colors = self.colors
        return colors.reshape(-1, 3)
    def get_points_colors_with_id(self, is_fitter = True, id = 0):
        if is_fitter:
            colors = self.filter_data_by_confidence(self.colors)
        else:
            colors = self.colors[id]
        return colors.reshape(-1, 3)
    def scale_points(self, scale_factor):
        """Scale all points in this submap by the given factor"""
        if hasattr(self, 'pointclouds') and self.pointclouds is not None:
            # 缩放点云
            self.pointclouds = self.pointclouds * scale_factor
            
            # 如果有其他与尺度相关的数据，也需要缩放
            if hasattr(self, 'conf'):
                self.conf = self.conf * scale_factor  # 如果conf与距离相关



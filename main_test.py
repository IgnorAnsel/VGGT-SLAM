import os

from pcltest5 import move

os.environ.update({"QT_QPA_PLATFORM_PLUGIN_PATH": "/home/ansel/anaconda3/envs/mast3r-slam/lib/python3.11/site-packages/cv2/qt/plugins/platforms"})
import glob
import argparse

import numpy as np
import torch
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver

from vggt.models.vggt import VGGT

from video_process import VideoProcess as VP
from enu import process_data
import open3d as o3d
parser = argparse.ArgumentParser(description="VGGT-SLAM demo")


parser.add_argument("--video_path", type=str, default="/home/ansel/works/datasets/DJI_20250725181205_0001_V.mp4", help="Path to video file")
parser.add_argument("--gnss_path", type=str, default="/home/ansel/works/datasets/DJI_20250725181205_0001_V.SRT", help="Path to GNSS file")
parser.add_argument("--image_folder", type=str, default="/home/ansel/works/vggt-slam/VGGT-SLAM/test", help="Path to folder containing images")
parser.add_argument("--downsample_factor_video", type=int, default=2, help="Factor to frame extracting")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
parser.add_argument("--use_sim3", action="store_true", help="Use Sim3 instead of SL(4)")
parser.add_argument("--plot_focal_lengths", action="store_true", help="Plot focal lengths for the submaps")
parser.add_argument("--submap_size", type=int, default=3, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--downsample_factor", type=int, default=2, help="Factor to reduce image size by 1/N")
parser.add_argument("--max_loops", type=int, default=3, help="Maximum number of loop closures per submap")
parser.add_argument("--min_disparity", type=float, default=5, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--conf_threshold", type=float, default=15.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--use_video", action="store_true", help="Use video instead of images")

# submap_size 3
# max_loops 1
# min_disparity 5
# parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
# parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
# parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
# parser.add_argument("--log_results", action="store_true", help="save txt file with results")
# parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
# parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
# parser.add_argument("--use_sim3", action="store_true", help="Use Sim3 instead of SL(4)")
# parser.add_argument("--plot_focal_lengths", action="store_true", help="Plot focal lengths for the submaps")
# parser.add_argument("--submap_size", type=int, default=16, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
# parser.add_argument("--overlapping_window_size", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
# parser.add_argument("--downsample_factor", type=int, default=1, help="Factor to reduce image size by 1/N")
# parser.add_argument("--max_loops", type=int, default=1, help="Maximum number of loop closures per submap")
# parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
# parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
# parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
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
def create_point_cloud(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
    args = parser.parse_args()
    use_optical_flow_downsample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    solver = Solver(
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        use_sim3=args.use_sim3,
        gradio_mode=False
    )

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(device)
    
    vp_ = VP()
    if args.use_video:
        vp_ = VP(args.video_path)
        if args.gnss_path:
            vp_.add_gnss_path(args.gnss_path)
        image_folder = vp_.extract_frames(args.downsample_factor_video)
    else:
        image_folder = args.image_folder
    # Use the provided image folder path
    print(f"Loading images from {image_folder}...")
    image_names = [f for f in glob.glob(os.path.join(image_folder, "*")) 
               if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower() 
               and "db" not in os.path.basename(f).lower()]
    
    image_names = utils.sort_images_by_number(image_names)
    image_names = utils.downsample_images(image_names, args.downsample_factor)
    print(f"Found {len(image_names)} images")
    gps_info_1 = vp_.read_exif_from_image(image_names[0])
    image_names_subset = []
    real_t_subset = []
    data = []
    total_images = len(image_names)
    half_point = total_images // 1
    solver.graph.gnss_processor.setReference(gps_info_1)
    solver.kf.gnss_processer.setReference(gps_info_1)
    for i, image_name in enumerate(tqdm(image_names)):
        gps_info_2 = vp_.read_exif_from_image(image_name)
        # print("gps_info_2", gps_info_2)
        if use_optical_flow_downsample:
            # print(image_name)
            img = cv2.imread(image_name)
            enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
            if i == half_point:
                solver.map.write_poses_to_file(args.log_path)
                # Log the full point cloud as one file, used for visualization.
                solver.map.write_points_to_file(args.log_path.replace(".txt", "_points.pcd"))
                break
            if enough_disparity:
                image_names_subset.append(image_name)
                real_t_subset.append(gps_info_2)
        else:
            image_names_subset.append(image_name)
            real_t_subset.append(gps_info_2)

        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            # print("real_t_subset",real_t_subset)
            predictions = solver.run_predictions(image_names_subset, model, args.max_loops)

            data.append(predictions["intrinsic"][:,0,0])
            # solver.test_add_points(predictions, real_t_subset)

            solver.add_points(predictions)
            # solver.graph.optimize()
            solver.map.update_submap_homographies(solver.graph)
            loop_closure_detected = len(predictions["detected_loops"]) > 0

            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()
            else:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()
            image_names_subset = image_names_subset[-args.overlapping_window_size:]
            real_t_subset = real_t_subset[-args.overlapping_window_size:]
    # solver.graph.visualize_gnss_constraints()
    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())

    if not args.vis_map:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path)
        # solver.map.test_write_poses_to_file(args.log_path, solver.scale_factor_mean)

        # Log the full point cloud as one file, used for visualization.
        # solver.map.test_write_points_to_file(args.log_path.replace(".txt", "_points.pcd"), solver.scale_factor_mean)
        solver.map.write_points_to_file(args.log_path.replace(".txt", "_points.pcd"))
        if not args.skip_dense_log:
            # Log the dense point cloud for each submap.
            solver.map.save_framewise_pointclouds(args.log_path.replace(".txt", "_logs"))

    if args.plot_focal_lengths:
        # Define a colormap
        colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
        # Create the scatter plot
        plt.figure(figsize=(8, 6))
        for i, values in enumerate(data):
            y = values  # Y-values from the list
            x = [i] * len(values)  # X-values (same for all points in the list)
            plt.scatter(x, y, color=colors[i], label=f'List {i+1}')

        plt.xlabel("poses")
        plt.ylabel("Focal lengths")
        plt.grid()
        plt.show()

if __name__ == "__main__":
    main()

import open3d as o3d
import numpy as np

# 读取点云数据
def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd

# 计算法向量并进行地面点提取
def ground_point_extraction(pcd):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 使用RANSAC算法提取地面点
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.05, ransac_n=3, num_iterations=1000)
    ground_pcd = pcd.select_by_index(inliers)
    non_ground_pcd = pcd.select_by_index(inliers, invert=True)

    return ground_pcd, non_ground_pcd
def extract_ground_ransac(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    """
    使用 RANSAC 平面拟合提取地面点
        :param pcd: 输入点云
        :param distance_threshold: 点与平面之间的最大距离（单位：米）
        :param ransac_n: 每次迭代随机采样的点数（3 个点确定一个平面）
        :param num_iterations: RANSAC 迭代次数
        :return: 地面点云 (ground_cloud), 非地面点云 (non_ground_cloud)
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    ground_cloud = pcd.select_by_index(inliers)
    non_ground_cloud = pcd.select_by_index(inliers, invert=True)
    print("ground_cloud size:", len(ground_cloud.points))
    print("non_ground_cloud size:", len(non_ground_cloud.points))

    return ground_cloud, non_ground_cloud
# 可视化结果
def visualize(ground_pcd, non_ground_pcd):
    ground_pcd.paint_uniform_color([0, 1, 0])  # 地面点着色为绿色
    non_ground_pcd.paint_uniform_color([1, 0, 0])  # 非地面点着色为红色
    o3d.visualization.draw_geometries([ground_pcd, non_ground_pcd])

if __name__ == "__main__":
    pcd = load_point_cloud("/home/ansel/works/vggt-slam/VGGT-SLAM/results/远test/poses_points.pcd11.pcd")
    ground_pcd, non_ground_pcd = extract_ground_ransac(pcd)
    visualize(ground_pcd, non_ground_pcd)
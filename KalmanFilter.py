import os
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/home/ansel/anaconda3/envs/mast3r-slam/lib/python3.11/site-packages/cv2/qt/plugins/platforms"
os.environ.update({"QT_QPA_PLATFORM_PLUGIN_PATH": "/home/ansel/anaconda3/envs/mast3r-slam/lib/python3.11/site-packages/cv2/qt/plugins/platforms"})
import numpy as np
import matplotlib.pyplot as plt
import glob

from video_process import VideoProcess
from gnss_processer import GNSSprocesser
import vggt_slam.slam_utils as utils

class KalmanFilter:
    def __init__(self, dim_state, dim_measurement):
        # 状态维度和观测维度
        self.dim_state = dim_state
        self.dim_measurement = dim_measurement
        
        # 初始化状态向量 x 和估计误差协方差矩阵 P
        self.x = np.zeros((dim_state, 1))  # 初始状态，默认为零
        self.P = np.eye(dim_state)  # 初始误差协方差矩阵，默认为单位矩阵
        
        # 初始化状态转移矩阵 F，假设状态每秒更新一次
        self.F = np.eye(dim_state)
        self.F = np.array([
            [1, 0, 0, 0.033, 0, 0],
            [0, 1, 0, 0, 0.033, 0],
            [0, 0, 1, 0, 0, 0.033],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        # 初始化观测矩阵 H
        self.H = np.zeros((dim_measurement, dim_state))
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        # 初始化过程噪声协方差矩阵 Q 和测量噪声协方差矩阵 R
        self.Q = np.eye(dim_state) * 0.1  # 假设过程噪声较小
        self.R = np.eye(dim_measurement) * 1.0  # 假设测量噪声较大

        self.vp_ = VideoProcess() # 相机处理，用于读取图片的GNSS
        self.gnss_processer = GNSSprocesser()
        self.gnss_data_enu = []
        self.gnss_data = []
        self.optimized_data = []
    def set(self, gnss_setment):
        self.gnss_data = gnss_setment
    def reduce(self):
        for gps_info in self.gnss_data:
            lat, lon, alt = gps_info
        enu = self.gnss_processer.lla_to_enu(lat, lon, alt)
        self.gnss_data_enu.append(enu)
        self.gnss_data_enu_array = np.array(self.gnss_data_enu)

        for z in self.gnss_data_enu_array:
            z = z.reshape(-1, 1)
            self.predict()
            self.update(z)
            self.optimized_data.append(self.x[:3].flatten())
        self.optimized_data_array = np.array(self.optimized_data)
        return self.optimized_data_array
    def predict(self):
        # 预测步骤
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
    
    def update(self, z):
        # 更新步骤
        y = z - np.dot(self.H, self.x)  # 观测残差
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # 残差协方差
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))  # 卡尔曼增益
        self.x = self.x + np.dot(K, y)  # 更新状态估计
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))  # 更新误差协方差

# # 使用示例
# kf = KalmanFilter(dim_state=6, dim_measurement=3)

# # 假设读取图片中的 GPS 信息
# image_folder = "/home/ansel/works/vggt-slam/VGGT-SLAM/temp_frames"
# image_names = [f for f in glob.glob(os.path.join(image_folder, "*")) 
#                if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower() 
#                and "db" not in os.path.basename(f).lower()]
# image_names = utils.sort_images_by_number(image_names)

# # 读取第一张图片的 GPS 信息
# gps_info = kf.vp_.read_exif_from_image(image_names[0])
# kf.gnss_processer.setReference(gps_info)

# gnss_data = np.array(kf.gnss_data)

# # 存储优化后的 GNSS 数据
# optimized_data = []

# # 通过卡尔曼滤波器逐步处理每一时刻的GNSS测量值
# for z in gnss_data:
#     z = z.reshape(-1, 1)  # 将观测数据从一维数组转换为列向量
#     # 预测步骤
#     kf.predict()

#     # 更新步骤
#     kf.update(z)

#     # 将每次更新后的状态（位置）存储到 optimized_data
#     optimized_data.append(kf.x[:3].flatten())  # 只保存位置 (E, N, U)

# # 转换为 NumPy 数组
# optimized_data = np.array(optimized_data)
# os.environ.update({"QT_QPA_PLATFORM_PLUGIN_PATH": "/home/ansel/anaconda3/envs/mast3r-slam/lib/python3.11/site-packages/cv2/qt/plugins/platforms"})

# # 绘制 2D 平面图 (E, N)
# plt.figure(figsize=(10, 6))

# # 绘制 GNSS 测量数据
# plt.plot(gnss_data[:, 0], gnss_data[:, 1], 'ro-', label="GNSS Data")

# # 绘制优化后的卡尔曼滤波数据
# plt.plot(optimized_data[:, 0], optimized_data[:, 1], 'b-', label="Kalman Filtered Data")

# # 设置图标
# plt.title("GNSS Data vs Kalman Filtered Data")
# plt.xlabel("East (E)")
# plt.ylabel("North (N)")
# plt.legend()
# plt.grid(True)

# # 显示图形
# plt.show()

import os
from tqdm import tqdm
import shutil
import cv2
import re
import piexif
from pprint import pprint

class VideoProcess:
    def __init__(self, video_path = None):
        """
        初始化 VideoProcess 类
        :param video_path: 视频文件路径
        """
        if video_path is not None:
            self.video_path = video_path
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"视频文件 {video_path} 不存在")
            
            # 使用 OpenCV 获取总帧数和 FPS
            self.video_cap = cv2.VideoCapture(video_path)
            self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            print(f"视频总帧数: {self.total_frames}, FPS: {self.video_fps}")

            # GNSS 数据存储
            self.gnss_data = {}

    def add_gnss_path(self, gnss_path):
        """加载并解析 GNSS 数据（SRT 格式）"""
        if not os.path.exists(gnss_path):
            raise FileNotFoundError(f"GNSS 文件 {gnss_path} 不存在")

        with open(gnss_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 解析 SRT 格式的 GNSS 数据
        pattern = re.compile(
            r"FrameCnt: (\d+).*?"
            r"latitude: ([\d\.\-]+).*?"
            r"longitude: ([\d\.\-]+).*?"
            r"rel_alt: ([\d\.\-]+).*?"
            r"abs_alt: ([\d\.\-]+).*?"
            r"gb_yaw: ([\d\.\-]+).*?"
            r"gb_pitch: ([\d\.\-]+).*?"
            r"gb_roll: ([\d\.\-]+)",
            re.DOTALL
        )

        for match in pattern.finditer(content):
            frame_num = int(match.group(1))
            self.gnss_data[frame_num] = {
                "latitude": float(match.group(2)),
                "longitude": float(match.group(3)),
                "rel_alt": float(match.group(4)),
                "abs_alt": float(match.group(5)),
                "yaw": float(match.group(6)),
                "pitch": float(match.group(7)),
                "roll": float(match.group(8)),
            }

    def _convert_to_degrees(self, value):
        """将十进制经纬度转换为 EXIF 支持的度分秒格式"""
        degrees = int(value)
        minutes = int((value - degrees) * 60)
        seconds = (value - degrees - minutes / 60) * 3600
        return (
            (degrees, 1),
            (minutes, 1),
            (int(seconds * 10000), 10000),  # 以分数形式存储秒
        )
    def _write_gnss_to_exif_in(self, frame, lat, lon, alt):
        """将指定的 lat, lon, alt 数据写入图片的 EXIF"""
        # 构造 EXIF 数据
        exif_dict = {
            "GPS": {
                piexif.GPSIFD.GPSLatitudeRef: "N" if lat >= 0 else "S",
                piexif.GPSIFD.GPSLatitude: self._convert_to_degrees(abs(lat)),
                piexif.GPSIFD.GPSLongitudeRef: "E" if lon >= 0 else "W",
                piexif.GPSIFD.GPSLongitude: self._convert_to_degrees(abs(lon)),
                piexif.GPSIFD.GPSAltitudeRef: 0 if alt >= 0 else 1,
                piexif.GPSIFD.GPSAltitude: (int(abs(alt) * 1000), 1000),
            },
            "0th": {},  # 主图像标签（必须存在）
            "1st": {},  # 缩略图标签（必须存在）
            "Exif": {},  # Exif 标签（可选）
        }

        # 将 EXIF 数据序列化为字节
        exif_bytes = piexif.dump(exif_dict)

        # 临时保存图片并注入 EXIF
        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, frame)
        piexif.insert(exif_bytes, temp_path)

        # 重新读取带 EXIF 的图片
        with open(temp_path, "rb") as f:
            img_data = f.read()
        os.remove(temp_path)

        return img_data
    def _write_gnss_to_exif(self, frame, frame_num):
        """将 GNSS 数据写入图片的 EXIF"""
        if frame_num not in self.gnss_data:
            return frame
        
        gnss = self.gnss_data[frame_num]
        yaw = gnss["yaw"] % 360
        # 构造 EXIF 数据
        exif_dict = {
            "GPS": {
                piexif.GPSIFD.GPSLatitudeRef: "N" if gnss["latitude"] >= 0 else "S",
                piexif.GPSIFD.GPSLatitude: self._convert_to_degrees(abs(gnss["latitude"])),
                piexif.GPSIFD.GPSLongitudeRef: "E" if gnss["longitude"] >= 0 else "W",
                piexif.GPSIFD.GPSLongitude: self._convert_to_degrees(abs(gnss["longitude"])),
                piexif.GPSIFD.GPSAltitudeRef: 0 if gnss["abs_alt"] >= 0 else 1,
                piexif.GPSIFD.GPSAltitude: (int(abs(gnss["abs_alt"]) * 1000), 1000),
                piexif.GPSIFD.GPSImgDirection: (int(yaw * 100), 100),
                piexif.GPSIFD.GPSImgDirectionRef: "T",  # 真北方向
            },
            "0th": {},  # 主图像标签（必须存在）
            "1st": {},  # 缩略图标签（必须存在）
            "Exif": {},  # Exif 标签（可选）
        }

        # 将 EXIF 数据序列化为字节
        exif_bytes = piexif.dump(exif_dict)

        # 临时保存图片并注入 EXIF
        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, frame)
        piexif.insert(exif_bytes, temp_path)

        # 重新读取带 EXIF 的图片
        with open(temp_path, "rb") as f:
            img_data = f.read()
        os.remove(temp_path)

        return img_data
    def inject_gnss_to_image(self, image_path, lat, lon, alt, output_image_path):
        """
        注入 GPS 数据到指定图片的 EXIF 中
        :param image_path: 输入图片路径
        :param lat: 纬度
        :param lon: 经度
        :param alt: 高度
        :param output_image_path: 输出的图片路径
        """
        # 读取图像
        frame = cv2.imread(image_path)
        
        # 注入 GPS 数据到 EXIF
        img_data = self._write_gnss_to_exif_in(frame, lat, lon, alt)
        
        # 将带有 EXIF 的图片保存到指定位置
        with open(output_image_path, "wb") as f:
            f.write(img_data)

        print(f"Successfully injected GPS data to image: {output_image_path}")
    def extract_frames(self, frame_interval, output_dir="./temp_frames_120"):
        """
        抽帧并注入 GNSS 数据到 EXIF
        :param frame_interval: 每秒抽取的帧数（如 5 表示每秒抽 5 帧）
        :param output_dir: 输出目录
        """
        shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)

        skip_frames = int(self.video_fps / frame_interval) if frame_interval < self.video_fps else 1
        pbar = tqdm(total=self.total_frames, desc="抽取帧进度", unit="帧")

        frame_count = 1
        saved_count = 1

        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:
                # 注入 GNSS 数据到 EXIF（如果存在）
                if hasattr(self, 'gnss_data'):
                    img_data = self._write_gnss_to_exif(frame, frame_count)
                    output_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                    with open(output_path, "wb") as f:
                        f.write(img_data)
                else:
                    output_path = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
                    cv2.imwrite(output_path, frame)

                saved_count += 1

            frame_count += 1
            pbar.update(1)

        pbar.close()
        self.video_cap.release()
        print(f"成功抽取 {saved_count} 帧到目录: {output_dir}")
        return output_dir

    def read_exif_from_image(self, image_path):
        """
        读取图片中的 EXIF 数据（重点解析 GPS 信息）
        :param image_path: 图片路径
        :return: 包含 GPS 和其他 EXIF 数据的字典
        """
        try:
            # 提取所有 EXIF 数据
            exif_dict = piexif.load(image_path)
            
            # 提取 GPS 信息
            gps_info = {}
            if "GPS" in exif_dict:
                gps = exif_dict["GPS"]
                gps_info = {
                    "latitude": self._convert_to_decimal(gps.get(piexif.GPSIFD.GPSLatitude)),
                    "longitude": self._convert_to_decimal(gps.get(piexif.GPSIFD.GPSLongitude)),
                    "altitude": gps.get(piexif.GPSIFD.GPSAltitude, (0, 1))[0] / gps.get(piexif.GPSIFD.GPSAltitude, (0, 1))[1],
                    "yaw": gps.get(piexif.GPSIFD.GPSImgDirection, (0, 1))[0] / gps.get(piexif.GPSIFD.GPSImgDirection, (0, 1))[1],
                }
            latitude = float(gps_info.get("latitude"))
            longitude = float(gps_info.get("longitude"))
            truncated_lat = float(f"{latitude:.5f}")  # 保留5位小数，因为精度是m，但数据精度却是6位
            truncated_lon = float(f"{longitude:.5f}")
            # pprint(gps_info)
            return (
                truncated_lat,
                truncated_lon,
                float(gps_info.get("altitude"))
            )
        
        except Exception as e:
            print(f"读取 EXIF 失败: {e}")
            return None

    def _convert_to_decimal(self, dms):
        """
        将度分秒格式（EXIF 存储）转换为十进制
        :param dms: 例如 ((30, 1), (24, 1), (3070, 100))
        :return: 十进制度数（如 30.408529）
        """
        if not dms:
            return 0.0
        degrees = dms[0][0] / dms[0][1]
        minutes = dms[1][0] / dms[1][1]
        seconds = dms[2][0] / dms[2][1]
        return "{:.6f}".format(degrees + minutes / 60 + seconds / 3600)
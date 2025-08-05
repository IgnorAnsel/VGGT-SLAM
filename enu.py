import numpy as np

def process_data(gps_info_1, gps_info_2):
    lat1 = gps_info_1["latitude"]
    lon1 = gps_info_1["longitude"]
    alt1 = gps_info_1["altitude"]
    lat2 = gps_info_2["latitude"]
    lon2 = gps_info_2["longitude"]
    alt2 = gps_info_2["altitude"]
    # 转换为ENU坐标
    enu_diff = latlonalt_to_enu(lat2, lon2, alt2, lat1, lon1, alt1)

    return enu_diff

def latlonalt_to_enu(lat, lon, alt, lat0, lon0, alt0):
    lat, lon, alt = float(lat), float(lon), float(alt)
    lat0, lon0, alt0 = float(lat0), float(lon0), float(alt0)
    # 简化的ENU转换（小范围近似）
    earth_radius = 6378137.0  # WGS84长半轴
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    
    east = earth_radius * dlon * np.cos(np.radians(lat0))
    north = earth_radius * dlat
    up = alt - alt0
    
    return np.array([east, north, up])
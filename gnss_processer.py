from pyproj import Proj
import numpy as np
class GNSSprocesser:
    def __init__(self):
        self.gnss_data = []
        self.ref_lat, self.ref_lon, self.ref_alt = None, None, None
        self.reference_lla = None
        self.reference_enu = None
        self.enu_history = []

    def setReference(self, reference_lla):
        self.reference_lla = reference_lla
        self.ref_lat, self.ref_lon, self.ref_alt = reference_lla[0], reference_lla[1], reference_lla[2]
        self.zone = int((self.ref_lon + 180) // 6) + 1
        self.utm_transformer = Proj(
            proj='utm', 
            zone=self.zone, 
            ellps='WGS84',
            south=False
        )
        self.enu_history = []
        self.ref_east, self.ref_north = self.utm_transformer(self.ref_lon, self.ref_lat)
        # print(self.ref_east, self.ref_north)
    def lla_to_enu(self, lat, lon, alt):
        """将WGS84坐标转换为局部ENU坐标系"""
        east, north = self.utm_transformer(lon, lat)
        enu = np.array([
            east - self.ref_east,
            north - self.ref_north,
            alt - self.ref_alt
        ])
        self.enu_history.append(enu)
        return enu
    
if __name__ == "__main__":
    gnss = GNSSprocesser()
    gnss.setReference([30.411221, 104.077182, 707])
    print(gnss.lla_to_enu(30.411280, 104.077185, 707))
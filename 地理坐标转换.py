import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from pyproj import Proj, Transformer
from typing import List
import cartopy.crs as ccrs

class GeoConverter:
    def __init__(self, camera_config):
        # 坐标系参数
        self.ref_lat = camera_config['latitude']
        self.ref_lon = camera_config['longitude']
        self.altitude = camera_config['rel_alt']  # 飞行高度
        
        # 相机参数
        self.sensor_w = camera_config['sensor_width']  # 传感器宽度 (m)
        self.sensor_h = camera_config['sensor_height'] # 传感器高度 (m)
        self.focal = camera_config['focal_len'] / 1000 # 焦距转米
        
        # 图像参数
        self.img_w = camera_config['image_width']
        self.img_h = camera_config['image_height']
        self.pitch = np.deg2rad(camera_config['pitch'])  # 俯仰角转弧度
        
        # 计算UTM投影
        self.utm_zone = self._get_utm_zone()
        self.proj_utm = Proj(proj='utm', zone=self.utm_zone, ellps='WGS84')
        self.transformer = Transformer.from_crs(4326, 32600 + self.utm_zone)
        
        # 基准点UTM坐标
        self.ref_easting, self.ref_northing = self.transformer.transform(
            self.ref_lat, self.ref_lon
        )
        
        # 计算地面分辨率
        self.gsd = (self.sensor_w * self.altitude) / (self.focal * self.img_w)
        
    def _get_utm_zone(self):
        return int((self.ref_lon + 180) // 6 + 1)
    
    def pixel_to_geo(self, x_pixel, y_pixel):
        """核心坐标转换算法"""
        # 转为以图像中心为原点
        dx = x_pixel - self.img_w/2
        dy = self.img_h/2 - y_pixel  # Y轴翻转
        
        # 计算视场角
        fov_x = 2 * np.arctan(self.sensor_w / (2 * self.focal))
        fov_y = 2 * np.arctan(self.sensor_h / (2 * self.focal))
        
        # 计算地面偏移量（考虑俯仰角）
        theta_x = dx * fov_x / self.img_w
        theta_y = dy * fov_y / self.img_h
        
        # 地面坐标计算
        ground_x = self.altitude * np.tan(theta_x + self.pitch)
        ground_y = self.altitude * np.tan(theta_y) / np.cos(self.pitch)
        
        # 转换为UTM坐标
        easting = self.ref_easting + ground_y  # 东方向对应Y轴
        northing = self.ref_northing + ground_x  # 北方向对应X轴
        
        # 转回经纬度
        return self.transformer.transform(easting, northing, direction='INVERSE')

def plot_simple_trajectories(df: pd.DataFrame, ids: List[int], output_dir: str):
    """简化版轨迹绘制（不依赖cartopy）"""
    plt.figure(figsize=(12, 8))
    
    # 绘制基准点（无人机位置）
    plt.scatter(df['longitude'].iloc[0], df['latitude'].iloc[0],
               color='red', s=200, marker='*', 
               label='Drone Position')
    
    # 绘制各目标轨迹
    colors = plt.cm.tab10(np.linspace(0, 1, len(ids)))
    for idx, target_id in enumerate(ids):
        id_data = df[df['ID'] == target_id].sort_values('Frame')
        
        # 绘制轨迹线
        plt.plot(id_data['longitude'], id_data['latitude'],
                color=colors[idx], linewidth=1.5,
                marker='o', markersize=6,
                linestyle='--', alpha=0.7,
                label=f'Target {target_id}')
    
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.title('Object Trajectories in Geographic Coordinates')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # 自动调整坐标范围
    lat_pad = (df['latitude'].max() - df['latitude'].min()) * 0.1
    lon_pad = (df['longitude'].max() - df['longitude'].min()) * 0.1
    plt.xlim(df['longitude'].min()-lon_pad, df['longitude'].max()+lon_pad)
    plt.ylim(df['latitude'].min()-lat_pad, df['latitude'].max()+lat_pad)
    
    plt.savefig(os.path.join(output_dir, 'geo_trajectory_simple.png'), dpi=300)
    plt.close()

def process_geo_coordinates(csv_path: str, camera_config: dict, ids: List[int]):
    output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(csv_path, converters={'XY': literal_eval})
        df[['x', 'y']] = pd.DataFrame(df['XY'].tolist(), index=df.index)
        
        # 初始化坐标转换器
        converter = GeoConverter(camera_config)
        
        # 转换地理坐标
        geo_coords = [converter.pixel_to_geo(row.x, row.y) for _, row in df.iterrows()]
        df['latitude'], df['longitude'] = zip(*geo_coords)
        
        # 保存结果
        output_csv = os.path.join(output_dir, 'geo_output.csv')
        df.to_csv(output_csv, index=False)
        
        # 绘制两种轨迹图
        plot_simple_trajectories(df, ids, output_dir)  # 简化版
        # plot_advanced_trajectories(df, ids, output_dir)  # 高级版（需要cartopy）
        
        print(f"处理完成，结果保存在: {output_dir}")

    except Exception as e:
        print(f"处理错误: {str(e)}")

if __name__ == "__main__":
    # 无人机参数配置
    CAMERA_CONFIG = {
        'latitude': 22.80961,
        'longitude': 108.34397,
        'rel_alt': 11.4,
        'abs_alt': -65.188,
        'pitch': 10.0,
        'sensor_width': 22.3e-3,
        'sensor_height': 14.9e-3,
        'focal_len': 46.80,
        'image_width': 3840,
        'image_height': 2160
    }
    
    process_geo_coordinates(
        csv_path="Output/1.12/DJI_0399/DJI_0399.csv",
        camera_config=CAMERA_CONFIG,
        ids=[1, 2]
    )
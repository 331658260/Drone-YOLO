import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyproj import Proj
from tqdm import tqdm
from matplotlib.ticker import ScalarFormatter
from scipy.stats import gaussian_kde
from scipy.interpolate import make_interp_spline


def pixel_to_geographic(u, v, config):
    """更新后的坐标转换函数（字幕参数适配版）"""
    # 从字幕文件获取实时参数
    lat = config['latitude']
    lon = config['longitude']
    height = config['rel_alt']
    
    # 动态获取焦距参数（单位转换为米）
    fl = config['focal_len'] * 1e-3
    
    # 参数转换
    pitch = np.radians(config['pitch'])
    sw = config['sensor_width']
    sh = config['sensor_height']
    w = config['image_width']
    h = config['image_height']

    # UTM投影设置
    utm_zone = int((lon + 180) // 6 + 1)
    utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84')
    easting, northing = utm_proj(lon, lat)

    # 坐标转换核心计算（增加高度容错）
    safe_height = max(height, 0.1)  # 确保最小高度0.1米
    px_per_m_x = (fl * w) / (sw * safe_height)
    px_per_m_y = (fl * h) / (sh * safe_height)
    
    dx = (u - w/2) / px_per_m_x
    dy = (h/2 - v) / px_per_m_y
    
    # 三维坐标转换（优化俯仰角补偿）
    pitch_compensation = np.cos(pitch) * 0.95  # 增加经验系数
    world_x = dx * pitch_compensation
    world_y = dy * pitch_compensation
    
    # 计算目标点UTM坐标（增加绝对高度补偿）
    target_easting = easting + world_x
    target_northing = northing + world_y + config['abs_alt']
    
    # UTM转经纬度
    lon_result, lat_result = utm_proj(target_easting, target_northing, inverse=True)
    
    return lon_result, lat_result

# 更新相机配置（使用字幕文件参数）
CAMERA_CONFIG = {
    'latitude': 22.806796,
    'longitude': 108.346738,
    'rel_alt': 11.4,
    'abs_alt': -65.188,
    'pitch': 10.0,
    'sensor_width': 22.3e-3,
    'sensor_height': 14.9e-3,
    'focal_len': 46.80,  # 直接使用字幕文件原始数值（单位mm）
    'image_width': 3840,
    'image_height': 2160
}

def process_gait_data(input_path, output_path):
    """带地理坐标转换的批处理方法（增加误差检测）"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', newline='', encoding='utf-8') as f_out:

        # 文件格式处理
        try:
            dialect = csv.Sniffer().sniff(f_in.read(4096))
            f_in.seek(0)
        except csv.Error:
            dialect = csv.excel_tab
        
        reader = csv.DictReader(f_in, dialect=dialect)
        cleaned_fields = [field.strip('\ufeff').strip() for field in (reader.fieldnames or [])]
        reader.fieldnames = cleaned_fields

        # 添加新列
        new_columns = list(cleaned_fields) + ['Longitude', 'Latitude']
        writer = csv.DictWriter(f_out, fieldnames=new_columns, delimiter=dialect.delimiter)
        writer.writeheader()

        # 处理数据
        for row in tqdm(reader, desc='坐标转换进度'):
            try:
                x_val = float(row['X'].replace(',', ''))
                y_val = float(row['Y'].replace(',', ''))
                
                # 坐标转换
                longitude, latitude = pixel_to_geographic(
                    u=x_val,
                    v=y_val,
                    config=CAMERA_CONFIG
                )
                
                # 中心点验证（图像中心点应等于无人机坐标）
                if (abs(x_val - 1920) < 1) and (abs(y_val - 1080) < 1):
                    print(f"\n中心点验证：计算值({longitude:.6f}, {latitude:.6f}) vs 基准值({CAMERA_CONFIG['longitude']}, {CAMERA_CONFIG['latitude']})")
                
                # 更新行数据
                row.update({
                    'Longitude': f"{longitude:.6f}",
                    'Latitude': f"{latitude:.6f}"
                })
                writer.writerow(row)
                
            except Exception as e:
                print(f"\n处理错误：{str(e)}")
                print(f"问题行：{dict(row)}")
                continue

def plot_gait_kde_speed(csv_path):
    """绘制基于经纬度的步速核密度图"""
    output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
        
        required_columns = ['Longitude', 'Latitude', 'Speed (m/s)']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"缺少必要字段: {', '.join(missing)}")

        # 增强字体配置（支持数学符号）
        plt.rcParams.update({
            'font.sans-serif': 'Microsoft YaHei',  # 改为支持数学符号的字体
            'axes.unicode_minus': False,
            'figure.dpi': 300,  # 提升分辨率
            'font.size': 10,     # 全局字体放大
            'axes.titlesize': 14,
            'axes.labelsize': 10
        })
        
        # 缩小画布尺寸（12→10，8→7）
        plt.figure(figsize=(12, 8))
        
        kde = sns.kdeplot(
            data=df,
            x='Longitude',
            y='Latitude',
            weights='Speed (m/s)',
            fill=True,
            cmap='Spectral_r',
            thresh=0.05,
            levels=15,
            alpha=0.8,
            cbar=True
        )

        # 颜色条优化
        cbar = kde.collections[0].colorbar
        cbar.set_label('Weighted Density Value (m/(s·deg²))', 
                      fontsize=14,  # 增大字体
                      labelpad=8)  # 增加标签间距
        cbar.ax.tick_params(labelsize=10)  # 增大刻度字体

        # 坐标轴设置
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # 精度显示优化
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))
        
        # 标题增强
        plt.title('Geographic Distribution of Gait Speed', 
                fontsize=18,  # 进一步放大
                pad=8,       # 增加标题间距
                y=1.05)       # 垂直位置微调
        
        # 坐标轴标签
        plt.xlabel('Longitude', fontsize=14, labelpad=12)
        plt.ylabel('Latitude', fontsize=14, labelpad=12)
        
        # 网格线优化
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 自适应布局调整
        plt.tight_layout(pad=3.0)
        
        plot_path = os.path.join(output_dir, 'geo_speed_kde_speed.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"\n核密度图已保存至: {plot_path}")

    except Exception as e:
        print(f"\n绘图失败: {str(e)}")

def plot_gait_kde_stepL(csv_path):
    """绘制基于经纬度的步长核密度图"""
    output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
        
        required_columns = ['Longitude', 'Latitude', 'Step Length (m)']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"缺少必要字段: {', '.join(missing)}")

        # 增强字体配置（支持数学符号）
        plt.rcParams.update({
            'font.sans-serif': 'Microsoft YaHei',  # 改为支持数学符号的字体
            'axes.unicode_minus': False,
            'figure.dpi': 300,  # 提升分辨率
            'font.size': 10,     # 全局字体放大
            'axes.titlesize': 14,
            'axes.labelsize': 10
        })
        
        # 缩小画布尺寸（12→10，8→7）
        plt.figure(figsize=(12, 8))
        
        kde = sns.kdeplot(
            data=df,
            x='Longitude',
            y='Latitude',
            weights='Step Length (m)',
            fill=True,
            cmap='Spectral_r',
            thresh=0.05,
            levels=15,
            alpha=0.8,
            cbar=True
        )

        # 颜色条优化
        cbar = kde.collections[0].colorbar
        cbar.set_label('Weighted Density Value (m/(s·deg²))', 
                      fontsize=14,  # 增大字体
                      labelpad=8)  # 增加标签间距
        cbar.ax.tick_params(labelsize=10)  # 增大刻度字体

        # 坐标轴设置
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # 精度显示优化
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))
        
        # 标题增强
        plt.title('Geographic Distribution of Gait Step Length', 
                fontsize=18,  # 进一步放大
                pad=8,       # 增加标题间距
                y=1.05)       # 垂直位置微调
        
        # 坐标轴标签
        plt.xlabel('Longitude', fontsize=14, labelpad=12)
        plt.ylabel('Latitude', fontsize=14, labelpad=12)
        
        # 网格线优化
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 自适应布局调整
        plt.tight_layout(pad=3.0)
        
        plot_path = os.path.join(output_dir, 'geo_speed_kde_StepL.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"\n核密度图已保存至: {plot_path}")

    except Exception as e:
        print(f"\n绘图失败: {str(e)}")

def plot_gait_kde_stepW(csv_path):
    """绘制基于经纬度的步宽核密度图"""
    output_dir = os.path.dirname(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
        
        required_columns = ['Longitude', 'Latitude', 'Step Width (m)']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"缺少必要字段: {', '.join(missing)}")

        # 增强字体配置（支持数学符号）
        plt.rcParams.update({
            'font.sans-serif': 'Microsoft YaHei',  # 改为支持数学符号的字体
            'axes.unicode_minus': False,
            'figure.dpi': 300,  # 提升分辨率
            'font.size': 10,     # 全局字体放大
            'axes.titlesize': 14,
            'axes.labelsize': 10
        })
        
        # 缩小画布尺寸（12→10，8→7）
        plt.figure(figsize=(12, 8))
        
        kde = sns.kdeplot(
            data=df,
            x='Longitude',
            y='Latitude',
            weights='Step Width (m)',
            fill=True,
            cmap='Spectral_r',
            thresh=0.05,
            levels=15,
            alpha=0.8,
            cbar=True
        )

        # 颜色条优化
        cbar = kde.collections[0].colorbar
        cbar.set_label('Weighted Density Value (m/(s·deg²))', 
                      fontsize=14,  # 增大字体
                      labelpad=8)  # 增加标签间距
        cbar.ax.tick_params(labelsize=10)  # 增大刻度字体

        # 坐标轴设置
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # 精度显示优化
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.5f'))
        
        # 标题增强
        plt.title('Geographic Distribution of Gait Step Width', 
                fontsize=18,  # 进一步放大
                pad=8,       # 增加标题间距
                y=1.05)       # 垂直位置微调
        
        # 坐标轴标签
        plt.xlabel('Longitude', fontsize=14, labelpad=12)
        plt.ylabel('Latitude', fontsize=14, labelpad=12)
        
        # 网格线优化
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # 自适应布局调整
        plt.tight_layout(pad=3.0)
        
        plot_path = os.path.join(output_dir, 'geo_speed_kde_stepW.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"\n核密度图已保存至: {plot_path}")

    except Exception as e:
        print(f"\n绘图失败: {str(e)}")

# 更新执行流程
input_file = r'Output/2.23/尽头步道_10m_10°_2x/out/gait_analysis_results.csv'
output_file = r'Output/2.23/尽头步道_10m_10°_2x/out/geographic_coordinates.csv'

process_gait_data(input_file, output_file)
plot_gait_kde_speed(output_file)
plot_gait_kde_stepL(output_file)
plot_gait_kde_stepW(output_file)
import gc
import pandas as pd
import numpy as np
import os
import ast
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林
from sklearn.model_selection import train_test_split
import ast  # 用于解析字符串为列表
from typing import List, Optional
from scipy.stats import f_oneway
from tqdm import tqdm
import time

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体或其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 像素到米的转换因子
PIXEL_TO_METER =  0.057765

# 定义原始的年龄段
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
age_bins = [(0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)]

# 修改 load_skeleton_data 函数
def load_skeleton_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return {}

    data = {}
    # 添加总进度条（按用户数）
    user_groups = list(df.groupby('ID'))
    for id, group in tqdm(user_groups, desc="Processing users", unit="user"):
        coordinates = {}
        # 添加关键点处理子进度条
        for i in tqdm(range(17), desc="Processing keypoints", leave=False, unit="kp"):
            key = f"Kp_{i}"
            try:
                coords = group[key].apply(
                    lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else [np.nan, np.nan]
                ).tolist()
                coordinates[f'kp_{i}'] = (np.array(coords) * PIXEL_TO_METER).astype(np.float32)
            except Exception as e:
                coordinates[f'kp_{i}'] = np.zeros((0, 2), dtype=np.float32)
        del group
        data[id] = simulate_missing_data(coordinates)
    return data

# 填补缺失数据的函数，使用 CubicSpline 进行填补
def fill_missing_with_cubic_spline(coords):
    # 将 [0, 0] 视为缺失数据
    coords[np.all(coords == [0, 0], axis=1)] = [np.nan, np.nan]
    
    # 对每个维度进行 CubicSpline 插值
    for dim in range(2):
        y = coords[:, dim]
        x = np.arange(len(y))

        # 找到有效数据
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 2:  # 如果有效数据少于两个，则跳过
            continue

        # 使用 CubicSpline 填补缺失数据
        spline = CubicSpline(x[valid_mask], y[valid_mask], bc_type='natural')
        coords[np.isnan(y), dim] = spline(x[np.isnan(y)])  # 填补缺失数据

    return coords

def simulate_missing_data(coordinates):
    # 添加关键点处理进度条
    for key in tqdm(coordinates, desc="Filling missing data", unit="kp"):
        coords = coordinates[key]

        if len(coords) == 0:
            #print(f"坐标数据为空，跳过 {key}.")
            continue
        
        # 首先用 CubicSpline 填补缺失数据
        coords = fill_missing_with_cubic_spline(coords)

        # 创建掩码以找到有效坐标
        mask = np.isnan(coords[:, 0]) | np.isnan(coords[:, 1])
        if not mask.any():  # 如果没有缺失值
            #print(f"没有缺失值，跳过 {key} 的预测.")
            coordinates[key] = coords  # 直接更新原坐标
            continue

        # 获取有效数据的索引和坐标
        x = np.arange(len(coords))
        valid_x = x[~mask]
        valid_coords = coords[~mask]
        
        if valid_coords.shape[0] == 0:  # 如果有效数据数量为0
            #print(f"有效数据数量为 0，跳过 {key}.")
            continue

        # 使用随机森林填补缺失值
        for dim in tqdm(range(2), desc="Processing dimensions", leave=False):
            features = np.column_stack((valid_x, valid_coords[:, (dim + 1) % 2]))
            target = valid_coords[:, dim]

            if len(features) < 2:
                #print(f"有效数据点不足以训练模型，跳过 {key} 的维度 {dim}.")
                continue
            
            # 拆分数据集
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

            # 训练随机森林模型
            rf_model = RandomForestRegressor(n_estimators=30, max_depth=8)
            
            # 预测后立即释放模型内存
            rf_model.fit(X_train, y_train)

            # 预测操作后立即释放模型
            del rf_model
            gc.collect()  # 手动触发垃圾回收
            
            # 预测缺失值
            missing_x = x[mask]
            missing_features = np.column_stack((missing_x, coords[mask, (dim + 1) % 2]))

            if missing_features.shape[0] > 0:
                predicted_values = rf_model.predict(missing_features)
                coords[mask, dim] = predicted_values  # 用预测值填补缺失的数据
            else:
                print(f"No missing features available for {key} dimension {dim}. Skipping prediction.")

        # 将填补后的坐标更新回原字典
        coordinates[key] = coords

    return coordinates

# 计算重心
def calculate_center_of_mass(coordinates):
    body_parts_indices = [0, 5, 6, 11, 12]  # 鼻子, 左肩, 右肩, 左髋, 右髋
    valid_parts = [f'kp_{i}' for i in body_parts_indices if f'kp_{i}' in coordinates and coordinates[f'kp_{i}'].size > 0]

    if not valid_parts:  # 若没有有效部分，返回NaN
        return np.nan, np.nan

    # 提取有效部分的坐标
    x_coords = np.concatenate([coordinates[part][:, 0] for part in valid_parts])
    y_coords = np.concatenate([coordinates[part][:, 1] for part in valid_parts])

    # 检查并排除异常值
    def remove_outliers(data):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return data[(data >= lower_bound) & (data <= upper_bound)]

    filtered_x_coords = remove_outliers(x_coords)
    filtered_y_coords = remove_outliers(y_coords)

    # 如果经过过滤后无有效数据，返回NaN
    if len(filtered_x_coords) == 0 or len(filtered_y_coords) == 0:
        return np.nan, np.nan

    com_x = np.mean(filtered_x_coords)
    com_y = np.mean(filtered_y_coords)
    
    return com_x, com_y
# 计算步长
def calculate_step_length(coordinates):
    left_ankle = coordinates.get('kp_15', np.array([]))  # 左脚踝
    right_ankle = coordinates.get('kp_16', np.array([]))  # 右脚踝

    if left_ankle.size == 0 or right_ankle.size == 0:
        return np.array([])

    step_lengths = np.linalg.norm(left_ankle - right_ankle, axis=1)
    return step_lengths
# 计算抬腿高度（左脚和右脚）
def calculate_leg_heights(coordinates):
    left_knee = coordinates.get('kp_13', np.array([]))  # 左膝
    left_ankle = coordinates.get('kp_15', np.array([]))  # 左脚踝
    right_knee = coordinates.get('kp_14', np.array([]))  # 右膝
    right_ankle = coordinates.get('kp_16', np.array([]))  # 右脚踝

    def compute_height(knee, ankle):
        if knee.size == 0 or ankle.size == 0:
            return np.array([np.nan] * len(knee))

        heights = knee[:, 1] - ankle[:, 1]
        return np.abs(heights)

    left_heights = compute_height(left_knee, left_ankle)
    right_heights = compute_height(right_knee, right_ankle)

    return left_heights, right_heights
# 计算相对抬脚高度
def calculate_relative_lift_height(left_heights, right_heights):
    relative_heights = left_heights - right_heights
    return relative_heights
# 计算腰宽比
def calculate_waist_height_ratio(coordinates):
    left_hip = coordinates.get('kp_11', np.array([]))
    right_hip = coordinates.get('kp_12', np.array([]))

    if left_hip.size == 0 or right_hip.size == 0:
        return np.array([np.nan] * max(len(left_hip), len(right_hip)))

    waist_width = np.linalg.norm(left_hip - right_hip, axis=1)
    waist_circumference = np.pi * waist_width

    head = coordinates.get('kp_0', np.array([]))
    left_ankle = coordinates.get('kp_15', np.array([]))
    right_ankle = coordinates.get('kp_16', np.array([]))
    feet_height = (left_ankle[:, 1] + right_ankle[:, 1]) / 2

    if head.size == 0 or left_ankle.size == 0 or right_ankle.size == 0:
        return np.array([np.nan] * len(waist_width))

    height = head[:, 1] - feet_height
    height = np.where(height == 0, np.nan, height)  # 将身高为零的值设为 NaN

    waist_ratio = waist_circumference / height

    # 过滤掉腰高比为零的结果
    waist_ratio = np.where(waist_ratio == 0, np.nan, waist_ratio)
    return np.abs(waist_ratio)
# 计算躯干倾斜角
def calculate_trunk_inclination_angle(coordinates):
    left_shoulder = coordinates.get('kp_5', np.array([]))
    right_shoulder = coordinates.get('kp_2', np.array([]))
    left_hip = coordinates.get('kp_11', np.array([]))
    right_hip = coordinates.get('kp_12', np.array([]))

    if left_shoulder.size == 0 or right_shoulder.size == 0 or left_hip.size == 0 or right_hip.size == 0:
        return np.nan  # 返回 NaN 如果数据缺失

    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    hip_midpoint = (left_hip + right_hip) / 2

    trunk_vector = shoulder_midpoint - hip_midpoint
    vertical_vector = np.array([0, 1])

    angle_rad = np.arctan2(trunk_vector[:, 1], trunk_vector[:, 0])
    angle_deg = np.degrees(angle_rad)
    trunk_inclination_angle = angle_deg % 360

    return trunk_inclination_angle
# 分析对称性
def analyze_symmetry(coordinates):
    left_knee = coordinates.get('kp_13', np.array([]))  # 左膝
    right_knee = coordinates.get('kp_14', np.array([]))  # 右膝

    if left_knee.size == 0 or right_knee.size == 0:
        return 0

    discrepancies = np.abs(left_knee - right_knee)
    symmetry_index = np.std(discrepancies, axis=0)
    return symmetry_index
# 计算躯干倾斜角
def calculate_trunk_inclination_angle(coordinates):
    left_shoulder = coordinates.get('kp_5', np.array([]))
    right_shoulder = coordinates.get('kp_2', np.array([]))
    left_hip = coordinates.get('kp_11', np.array([]))
    right_hip = coordinates.get('kp_12', np.array([]))

    if left_shoulder.size == 0 or right_shoulder.size == 0 or left_hip.size == 0 or right_hip.size == 0:
        return np.nan

    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    hip_midpoint = (left_hip + right_hip) / 2

    trunk_vector = shoulder_midpoint - hip_midpoint

    angle_rad = np.arctan2(trunk_vector[:, 1], trunk_vector[:, 0])
    angle_deg = np.degrees(angle_rad)
    trunk_inclination_angle = angle_deg % 360

    return trunk_inclination_angle
# 计算肩部水平高度差
def calculate_shoulder_level_difference(coordinates):
    left_shoulder = coordinates.get('kp_5', np.array([]))
    right_shoulder = coordinates.get('kp_2', np.array([]))

    if left_shoulder.size == 0 or right_shoulder.size == 0:
        return np.array([np.nan] * len(left_shoulder))

    shoulder_difference = np.abs(left_shoulder[:, 1] - right_shoulder[:, 1])
    return shoulder_difference
# 计算骨盆倾斜角
def calculate_pelvic_tilt_angle(coordinates):
    left_hip = coordinates.get('kp_11', np.array([]))
    right_hip = coordinates.get('kp_12', np.array([]))
    left_knee = coordinates.get('kp_13', np.array([]))
    right_knee = coordinates.get('kp_14', np.array([]))

    if (left_hip.size == 0 or right_hip.size == 0 or
        left_knee.size == 0 or right_knee.size == 0):
        return np.nan

    hip_midpoint = (left_hip + right_hip) / 2
    pelvic_height = hip_midpoint[:, 1] - (left_knee[:, 1] + right_knee[:, 1]) / 2
    pelvic_tilt_angle = np.degrees(np.arctan2(pelvic_height, (left_knee[:, 0] + right_knee[:, 0]) / 2 - hip_midpoint[:, 0]))

    return pelvic_tilt_angle
# 计算头部前倾距离
def calculate_forward_head_posture_distance(coordinates):
    head = coordinates.get('kp_0', np.array([]))
    left_shoulder = coordinates.get('kp_5', np.array([]))
    right_shoulder = coordinates.get('kp_2', np.array([]))

    if head.size == 0 or left_shoulder.size == 0 or right_shoulder.size == 0:
        return np.array([np.nan] * max(len(head), len(left_shoulder), len(right_shoulder)))

    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    distance = head[:, 0] - shoulder_midpoint[:, 0]  # 水平距离

    return distance
# 计算步宽
def calculate_step_width(coordinates):
    left_hip = coordinates.get('kp_11', np.array([]))  # 左髋
    right_hip = coordinates.get('kp_12', np.array([]))  # 右髋

    if left_hip.size == 0 or right_hip.size == 0:
        return np.array([])

    step_widths = np.linalg.norm(left_hip - right_hip, axis=1)
    return step_widths

def analyze_single_user_gait(coordinates, frame_rate, original_speeds):
    com_x, com_y = calculate_center_of_mass(coordinates)
    step_lengths = calculate_step_length(coordinates)
    step_widths = calculate_step_width(coordinates)

    left_leg_heights, right_leg_heights = calculate_leg_heights(coordinates)
    relative_lift_heights = calculate_relative_lift_height(left_leg_heights, right_leg_heights)
    waist_height_ratio = calculate_waist_height_ratio(coordinates)
    trunk_inclination_angle = calculate_trunk_inclination_angle(coordinates)
    shoulder_level_difference = calculate_shoulder_level_difference(coordinates)
    pelvic_tilt_angle = calculate_pelvic_tilt_angle(coordinates)
    forward_head_posture_distance = calculate_forward_head_posture_distance(coordinates)

    symmetry_index = analyze_symmetry(coordinates)

    max_length = max(len(step_lengths),  len(left_leg_heights), 
                     len(right_leg_heights), len(relative_lift_heights), len(waist_height_ratio),
                     len(trunk_inclination_angle), len(shoulder_level_difference), 
                     len(pelvic_tilt_angle), len(forward_head_posture_distance), 
                     len(original_speeds)  # 考虑原始步速的长度
                     )

    step_lengths = np.pad(step_lengths, (0, max_length - len(step_lengths)), 'constant', constant_values=np.nan)
    step_widths = np.pad(step_widths, (0, max_length - len(step_widths)), 'constant', constant_values=np.nan)
    original_speeds = np.pad(original_speeds, (0, max_length - len(original_speeds)), 'constant', constant_values=np.nan)  # 填充原始步速
    left_leg_heights = np.pad(left_leg_heights, (0, max_length - len(left_leg_heights)), 'constant', constant_values=np.nan)
    right_leg_heights = np.pad(right_leg_heights, (0, max_length - len(right_leg_heights)), 'constant', constant_values=np.nan)
    relative_lift_heights = np.pad(relative_lift_heights, (0, max_length - len(relative_lift_heights)), 'constant', constant_values=np.nan)
    waist_height_ratio = np.pad(waist_height_ratio, (0, max_length - len(waist_height_ratio)), 'constant', constant_values=np.nan)
    trunk_inclination_angle = np.pad(trunk_inclination_angle, (0, max_length - len(trunk_inclination_angle)), 'constant', constant_values=np.nan)
    shoulder_level_difference = np.pad(shoulder_level_difference, (0, max_length - len(shoulder_level_difference)), 'constant', constant_values=np.nan)
    pelvic_tilt_angle = np.pad(pelvic_tilt_angle, (0, max_length - len(pelvic_tilt_angle)), 'constant', constant_values=np.nan)
    forward_head_posture_distance = np.pad(forward_head_posture_distance, (0, max_length - len(forward_head_posture_distance)), 'constant', constant_values=np.nan)

    results = {
        'center_of_mass': (com_x, com_y),
        'Step Lengths(m)': step_lengths,
        'Step Widths(m)': step_widths,
        'speeds': original_speeds,
        'left_leg_height': left_leg_heights,
        'right_leg_height': right_leg_heights,
        'relative_lift_height': relative_lift_heights,
        'symmetry_index': [symmetry_index] * max_length,
        'waist_height_ratio': waist_height_ratio,
        'trunk_inclination_angle': trunk_inclination_angle,
        'shoulder_level_difference': shoulder_level_difference,
        'pelvic_tilt_angle': pelvic_tilt_angle,
        'forward_head_posture_distance': forward_head_posture_distance,
    }

    return results

# 保存csv文件函数
def save_results_to_csv(folder_path, all_results, original_csv_path):
    os.makedirs(folder_path, exist_ok=True)

    original_df = pd.read_csv(original_csv_path)
    # 解析XY列为X和Y
    original_df = extract_xy_columns(original_df)

    all_data = []

    for user_id, results in all_results.items():
        user_frames = original_df[original_df['ID'] == user_id]
        original_speeds = user_frames['Speed (m/s)'].values  

        for frame in range(len(results['Step Lengths(m)'])):
            current_frame_info = user_frames.iloc[frame] if frame < len(user_frames) else {}
            
            all_data.append({
                'User ID': user_id,
                'Frame': current_frame_info.get('Frame', frame),
                'X': current_frame_info.get('X', np.nan),  # 添加X值
                'Y': current_frame_info.get('Y', np.nan),  # 添加Y值
                'Step Length (m)': results['Step Lengths(m)'][frame],
                'Step Width (m)': results['Step Widths(m)'][frame],
                'Speed (m/s)': original_speeds[frame] if frame < len(original_speeds) else np.nan,
                'Symmetry Index': results['symmetry_index'][frame],
                'Waist Height Ratio': results['waist_height_ratio'][frame],
                'Trunk Inclination Angle': results['trunk_inclination_angle'][frame],
                'Shoulder Level Difference (m)': results['shoulder_level_difference'][frame],
                'Pelvic Tilt Angle (degrees)': results['pelvic_tilt_angle'][frame],
                'Forward Head Posture Distance (m)': results['forward_head_posture_distance'][frame]
            })

    results_df = pd.DataFrame(all_data)

    # 指定生成的csv文件名
    output_file_path = os.path.join(folder_path, 'gait_analysis_results.csv')
    results_df.to_csv(output_file_path, index=False)

def extract_xy_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """ 从 'XY' 列解析为两个新的 'X' 和 'Y' 列 """
    # 确保 'XY' 列存在并进行解析
    if 'XY' in dataframe.columns:
        dataframe[['X', 'Y']] = pd.DataFrame(dataframe['XY'].apply(
            lambda xy: ast.literal_eval(xy) if isinstance(xy, str) else (np.nan, np.nan)).tolist(), 
            index=dataframe.index
        )
    return dataframe

def calculate_trajectory_metrics(dataframe: pd.DataFrame, frame_rate: float) -> pd.DataFrame:
    """ 计算每个ID的轨迹指标，包括平均速度、起始结束距离等 """
    # 首先提取 'X' 和 'Y' 列
    dataframe = extract_xy_columns(dataframe)
    
    metrics = {}
    # 按 'ID' 分组
    grouped = dataframe.groupby('ID')
    
    for id_num, group in grouped:
        path_length = np.sum(np.sqrt(np.diff(group['X']) ** 2 + np.diff(group['Y']) ** 2))
        straight_dist = np.sqrt((group['X'].iloc[-1] - group['X'].iloc[0]) ** 2 +
                                 (group['Y'].iloc[-1] - group['Y'].iloc[0]) ** 2)
        curvature_index = path_length / straight_dist if straight_dist != 0 else np.nan  # 避免零除

        # 计算起始点与结束点之间的距离
        start_point = (group['X'].iloc[0], group['Y'].iloc[0])
        end_point = (group['X'].iloc[-1], group['Y'].iloc[-1])
        start_end_distance = np.sqrt((end_point[0] - start_point[0]) ** 2 +
                                      (end_point[1] - start_point[1]) ** 2)

        # 将路径长度和直线距离从像素转换为米
        path_length_meters = path_length * PIXEL_TO_METER
        straight_dist_meters = straight_dist * PIXEL_TO_METER
        start_end_distance_meters = start_end_distance * PIXEL_TO_METER  # 转换为米
        
        # 计算平均速度: (总距离 / 时间)
        time_taken = len(group) / frame_rate  # 总时间 = 点数 / 帧率
        average_speed = path_length_meters / time_taken if time_taken > 0 else np.nan

        metrics[id_num] = {
            'points_count': len(group),
            'path_length': path_length_meters,
            'straight_dist': straight_dist_meters,
            'curvature_index': curvature_index,
            'average_speed': average_speed,
            'start_end_distance': start_end_distance_meters  # 添加起始结束距离
        }

    return pd.DataFrame.from_dict(metrics, orient='index')

# 将年龄组的数据保存到 CSV 文件的函数
def save_age_group_data_to_csv(all_results: dict, df: pd.DataFrame, output_folder: str):
    age_group_mapping = {
        'Minor': (0, 17),
        'Adult': (18, 59),
        'Elderlys': (60, 100)
    }

    for group, (start, end) in age_group_mapping.items():
        age_group_data = []
        for user_id, results in all_results.items():
            user_data = df[df['ID'] == user_id]
            valid_ages = user_data[(user_data['Parsed_Age'] >= start) & (user_data['Parsed_Age'] <= end)]

            for frame in range(len(results['Step Lengths(m)'])):
                if frame < len(valid_ages):
                    age_group_data.append({
                        'User ID': user_id,
                        'Frame': valid_ages.iloc[frame]['Frame'] if frame < len(valid_ages) else np.nan,
                        'Step Length (m)': results['Step Lengths(m)'][frame],
                        'Step Width (m)': results['Step Widths(m)'][frame],
                        'Speed (m/s)': results['speeds'][frame],
                        'X': valid_ages.iloc[frame]['X'] if frame < len(valid_ages) else np.nan,
                        'Y': valid_ages.iloc[frame]['Y'] if frame < len(valid_ages) else np.nan,
                    })

        # 保存到 CSV 文件
        if age_group_data:
            age_group_df = pd.DataFrame(age_group_data)
            output_file_path = os.path.join(output_folder, f'{group.lower()}_data.csv')
            age_group_df.to_csv(output_file_path, index=False)
            print(f"Saved {group} data to {output_file_path}")

def analyze_gait(file_path, frame_rate, output_folder):
    
    # 记录总开始时间
    start_time = time.time()
    
    # 添加主进度条
    with tqdm(total=7, desc="Overall Progress") as pbar:
        # 阶段1：加载数据
        user_data = load_skeleton_data(file_path)
        all_results = {}
        pbar.update(1)
        pbar.set_postfix_str("Data loaded")

        # 阶段2：统计基础信息
        num_users = len(user_data)
        num_trajectories = sum([len(coords) for coords in user_data.values()])
        print(f"\nTotal Users: {num_users}")
        print(f"Total Trajectories: {num_trajectories}")
        pbar.update(1)
        pbar.set_postfix_str("Basic stats calculated")

        # 阶段3：绘制用户分布图
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 2, 1)
        plt.bar(['Total Users'], [num_users], color='skyblue')
        plt.subplot(1, 2, 2)
        plt.bar(['Total Trajectories'], [num_trajectories], color='salmon')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'user_and_trajectory_distribution.png'))
        plt.close()
        pbar.update(1)
        pbar.set_postfix_str("User distribution plotted")

        # 阶段4：轨迹分析
        original_df = pd.read_csv(file_path)
        trajectory_metrics = calculate_trajectory_metrics(original_df, frame_rate)
        
        # 处理前10轨迹
        longest_trajectories = trajectory_metrics.nlargest(10, 'path_length')
        print("\nTop 10 Longest Trajectories:")
        print(longest_trajectories[['path_length', 'average_speed', 'start_end_distance']])

        # 绘制轨迹图
        plt.figure(figsize=(10, 6))
        for id_num in longest_trajectories.index:
            trajectory_data = original_df[original_df['ID'] == id_num]
            plt.plot(trajectory_data['X'], trajectory_data['Y'], 
                    label=f'ID {id_num} (Length: {longest_trajectories.loc[id_num, "path_length"]:.2f} m)')
        plt.savefig(os.path.join(output_folder, 'top_10_longest_trajectories.png'))
        plt.close()
        pbar.update(1)
        pbar.set_postfix_str("Trajectory analysis completed")

        # 阶段5：用户步态分析（添加嵌套进度条）
        user_items = list(user_data.items())
        for user_id, coordinates in tqdm(user_items, desc="Analyzing users", unit="user", leave=False):
            # 第一次分析
            user_frames = original_df[original_df['ID'] == user_id]
            original_speeds = user_frames['Speed (m/s)'].values
            results = analyze_single_user_gait(coordinates, frame_rate, original_speeds)
            all_results[user_id] = results

            # 第二次分析（保留原有逻辑）
            original_speeds = original_df[original_df['ID'] == user_id]['Speed (m/s)'].values
            results = analyze_single_user_gait(coordinates, frame_rate, original_speeds)
            all_results[user_id] = results
        pbar.update(1)
        pbar.set_postfix_str("Gait analysis completed")

        # 阶段6：年龄处理
        extract_age(original_df)
        pbar.update(1)
        pbar.set_postfix_str("Age data processed")

        # 阶段7：保存结果
        save_results_to_csv(output_folder, all_results, file_path)
        save_age_group_data_to_csv(all_results, original_df, output_folder)
        save_results_to_csv(output_folder, all_results, file_path)
        pbar.update(1)
        pbar.set_postfix_str("Results saved")

    # 打印总耗时
    total_time = time.time() - start_time
    print(f"\nTotal processing time: {total_time//3600:.0f}h {total_time%3600//60:.0f}m {total_time%60:.2f}s")
    
    return all_results

# 绘图
#姿态绘图
def plot_combined_distributions(all_results, output_file):
    # 创建一个2行3列的子图布局
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))

    # 绘制躯干倾斜角分布直方图
    def plot_trunk_inclination_distribution():
        all_angles = []
        for user_id, results in all_results.items():
            angles = results['trunk_inclination_angle']
            all_angles.extend(angles[~np.isnan(angles)])

        filtered_angles = remove_outliers(all_angles)

        axs[0, 0].hist(filtered_angles, bins=20, alpha=0.5, edgecolor='black')
        axs[0, 0].set_title('Trunk Inclination Angle Distribution')
        axs[0, 0].set_xlabel('Trunk Inclination Angle (degrees)')
        axs[0, 0].set_ylabel('Frequency')
        axs[0, 0].grid()

    # 绘制肩部水平高度差分布直方图
    def plot_shoulder_level_difference_distribution():
        all_differences = []
        for user_id, results in all_results.items():
            differences = results['shoulder_level_difference']
            all_differences.extend(differences[~np.isnan(differences)])

        filtered_differences = remove_outliers(all_differences)

        axs[0, 1].hist(filtered_differences, bins=20, alpha=0.5, edgecolor='black')
        axs[0, 1].set_title('Shoulder Level Difference Distribution')
        axs[0, 1].set_xlabel('Shoulder Level Difference')
        axs[0, 1].set_ylabel('Frequency')
        axs[0, 1].grid()

    # 绘制骨盆倾斜角分布直方图
    def plot_pelvic_tilt_angle_distribution():
        all_angles = []
        for user_id, results in all_results.items():
            angles = results['pelvic_tilt_angle']
            all_angles.extend(angles[~np.isnan(angles)])

        filtered_angles =  remove_outliers(all_angles)

        axs[0, 2].hist(filtered_angles, bins=20, alpha=0.5, edgecolor='black')
        axs[0, 2].set_title('Pelvic Tilt Angle Distribution')
        axs[0, 2].set_xlabel('Pelvic Tilt Angle (degrees)')
        axs[0, 2].set_ylabel('Frequency')
        axs[0, 2].grid()

    # 绘制头部前倾距离分布直方图
    def plot_forward_head_posture_distance_distribution():
        all_distances = []
        for user_id, results in all_results.items():
            distances = results['forward_head_posture_distance']
            all_distances.extend(distances[~np.isnan(distances)])
        
        filtered_distances = remove_outliers(all_distances)

        axs[1, 0].hist(filtered_distances, bins=20, alpha=0.5, edgecolor='black')
        axs[1, 0].set_title('Forward Head Posture Distance Distribution')
        axs[1, 0].set_xlabel('Forward Head Posture Distance (cm)')
        axs[1, 0].set_ylabel('Frequency')
        axs[1, 0].grid()

    # 绘制重心位置分布图
    def plot_center_of_mass_distribution():
        for user_id, results in all_results.items():
            com_x, com_y = results['center_of_mass']
            axs[1, 1].scatter(com_x, com_y, label=f'User ID: {user_id}', alpha=0.6)

        axs[1, 1].set_title('Center of Mass Distribution')
        axs[1, 1].set_xlabel('X Position (m)')
        axs[1, 1].set_ylabel('Y Position (m)')
        axs[1, 1].axis('equal')
        axs[1, 1].grid()

    # 绘制腰高比分布直方图
    def plot_waist_height_ratio_distribution():
        all_ratios = []
        for user_id, results in all_results.items():
            waist_height_ratios = results['waist_height_ratio']
            all_ratios.extend(waist_height_ratios[~np.isnan(waist_height_ratios)])

        filtered_ratios = remove_outliers(all_ratios)

        axs[1, 2].hist(filtered_ratios, bins=20, alpha=0.5, edgecolor='black')
        axs[1, 2].set_title('Waist Height Ratio Distribution')
        axs[1, 2].set_xlabel('Waist Height Ratio')
        axs[1, 2].set_ylabel('Frequency')
        axs[1, 2].grid()

    # 调用各个个别画图函数
    plot_trunk_inclination_distribution()
    plot_shoulder_level_difference_distribution()
    plot_pelvic_tilt_angle_distribution()
    plot_forward_head_posture_distance_distribution()
    plot_center_of_mass_distribution()
    plot_waist_height_ratio_distribution()

    plt.tight_layout()  # 自动调整子图参数
    plt.savefig(output_file)  # 保存图像
    plt.close()

#步态绘图
def plot_all_distributions(all_results, output_file):
    # 创建一个包含6个子图的图形（1行3列的布局）
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1行3列的布局

    # 绘制步长分布直方图
    all_lengths = []
    for user_id, results in all_results.items():
        step_lengths = results['Step Lengths(m)']
        all_lengths.extend(step_lengths[~np.isnan(step_lengths)])

    # 排除异常值
    q1 = np.percentile(all_lengths, 25)
    q3 = np.percentile(all_lengths, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_lengths = [length for length in all_lengths if lower_bound <= length <= upper_bound]

    axs[0].hist(filtered_lengths, bins=20, alpha=0.5, edgecolor='black')
    axs[0].set_title('Step Length')
    axs[0].set_xlabel('Step Length (m)')
    axs[0].set_ylabel('Frequency')
    axs[0].grid()

    # 绘制步宽分布直方图
    all_step_widths = []
    for user_id, results in all_results.items():
        step_widths = results['Step Widths(m)']
        all_step_widths.extend(step_widths[~np.isnan(step_widths)])

    # 排除异常值
    q1 = np.percentile(all_step_widths, 25)
    q3 = np.percentile(all_step_widths, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_step_widths = [width for width in all_step_widths if lower_bound <= width <= upper_bound]

    axs[1].hist(filtered_step_widths, bins=20, alpha=0.5, edgecolor='black')
    axs[1].set_title('Step Width')
    axs[1].set_xlabel('Step Width (m)')  # 替换为适当的单位
    axs[1].set_ylabel('Frequency')
    axs[1].grid()

    # 绘制步速分布直方图
    all_speeds = []
    for user_id, results in all_results.items():
        speeds = results['speeds']
        all_speeds.extend(speeds[~np.isnan(speeds)])

    # 排除异常值
    q1 = np.percentile(all_speeds, 25)
    q3 = np.percentile(all_speeds, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_speeds = [speed for speed in all_speeds if lower_bound <= speed <= upper_bound]

    axs[2].hist(filtered_speeds, bins=20, alpha=0.5, edgecolor='black')
    axs[2].set_title('Speed')
    axs[2].set_xlabel('Speed (m/s)')
    axs[2].set_ylabel('Frequency')
    axs[2].grid()

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# 轨迹分析
def plot_trajectories(dataframe: pd.DataFrame, ids: List[int], output_dir: str, title: str, 
                      figsize=(16, 10), save_name='trajectories.png') -> None:
    """ 绘制轨迹图并保存 """
    plt.figure(figsize=figsize)
    for id_num in ids:
        id_data = dataframe[dataframe['ID'] == id_num].sort_values('Frame')
        plt.plot(id_data['X'], id_data['Y'], marker='o', linestyle='-', markersize=4)

    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, save_name))
    plt.close()

def analyze_speed_changes(df: pd.DataFrame, id_num: int, threshold: float = 0.05) -> Optional[str]:
    """ 分析指定ID的速度变化 """
    id_data = df[df['ID'] == id_num].sort_values('Frame')

    if len(id_data) < 2:
        print(f"Not enough data to analyze ID: {id_num}.")
        return None

    speed_diff = np.abs(np.diff(id_data['Speed (m/s)']))
    change_points = np.where(speed_diff > threshold)[0]
    time_diff = np.diff(id_data['Frame']) / 10  # 假设帧率为10fps
    acceleration = np.divide(speed_diff, time_diff, out=np.nan * np.ones_like(speed_diff), where=time_diff != 0)  # 处理零除情况

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # 速度变化图
    axes[0].plot(id_data['Frame'].iloc[1:], speed_diff, 'b-', label='Speed Change')
    axes[0].scatter(id_data['Frame'].iloc[change_points + 1], speed_diff[change_points], color='red', label='Speed Change Points')
    axes[0].set_title(f'ID {id_num} Speed Changes')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Speed Difference (m/s)')
    axes[0].grid(True)
    axes[0].legend()

    # 加速度图
    axes[1].plot(id_data['Frame'].iloc[1:], acceleration, 'r-', label='Acceleration')
    axes[1].scatter(id_data['Frame'].iloc[change_points + 1], acceleration[change_points], color='green', label='Speed Change Points')
    axes[1].set_title('Acceleration')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Acceleration (m/s²)')
    axes[1].grid(True)
    axes[1].legend()

    # 方向变化图
    dx = np.diff(id_data['X'])
    dy = np.diff(id_data['Y'])
    angles = np.arctan2(dy, dx)
    axes[2].plot(id_data['Frame'].iloc[1:], angles, 'g-', label='Direction')
    axes[2].scatter(id_data['Frame'].iloc[change_points + 1], angles[change_points], color='red', label='Speed Change Points')
    axes[2].set_title('Direction Changes')
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Direction (radians)')
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'speed_changes_id_{id_num}.png'))
    plt.close()

    return f'Speed changes analyzed for ID: {id_num}'

def analyze_trajectory_speed_relations(df: pd.DataFrame, id_num: int) -> Optional[str]:
    """ 分析轨迹的速度与相关性 """
    id_data = df[df['ID'] == id_num].sort_values('Frame')

    if len(id_data) < 2:
        print(f"Not enough data to analyze ID: {id_num}.")
        return None

    dx = np.diff(id_data['X'])
    dy = np.diff(id_data['Y'])
    path_lengths = np.sqrt(dx**2 + dy**2)

    straight_dists = np.sqrt((id_data['X'].iloc[1:].values - id_data['X'].iloc[:-1].values)**2 + 
                              (id_data['Y'].iloc[1:].values - id_data['Y'].iloc[:-1].values)**2)

    local_curvature = np.divide(path_lengths, straight_dists, out=np.nan * np.ones_like(path_lengths), where=straight_dists != 0)

    angles = np.arctan2(dy, dx)
    direction_changes = np.abs(np.diff(angles))
    if len(direction_changes) > 0:
        direction_changes = np.append(direction_changes, direction_changes[-1])  # 保证方向变化数组长度一致

    local_slopes = np.divide(dy, dx, out=np.nan * np.ones_like(dy), where=dx != 0)  # 处理零除情况

    fig, axes = plt.subplots(4, 1, figsize=(15, 16))

    # 速度与弯曲度关系
    valid_curvature = ~np.isnan(local_curvature) & ~np.isinf(local_curvature)
    axes[0].scatter(local_curvature[valid_curvature], 
                     id_data['Speed (m/s)'].iloc[1:].values[valid_curvature], alpha=0.5)
    axes[0].set_title('Speed vs Curvature Index')
    axes[0].set_xlabel('Local Curvature')
    axes[0].set_ylabel('Speed (m/s)')
    axes[0].grid(True)

    # 速度与方向变化关系
    axes[1].scatter(direction_changes[:-1], 
                     id_data['Speed (m/s)'].iloc[1:-1], alpha=0.5)
    axes[1].set_title('Speed vs Direction Changes')
    axes[1].set_xlabel('Direction Change (radians)')
    axes[1].set_ylabel('Speed (m/s)')
    axes[1].grid(True)

    # 速度与局部斜率关系
    valid_slopes = ~np.isnan(local_slopes) & ~np.isinf(local_slopes)
    axes[2].scatter(local_slopes[valid_slopes], 
                     id_data['Speed (m/s)'].iloc[1:][valid_slopes], alpha=0.5)
    axes[2].set_title('Speed vs Slope')
    axes[2].set_xlabel('Local Slope')
    axes[2].set_ylabel('Speed (m/s)')
    axes[2].grid(True)

    # 速度随时间变化
    axes[3].plot(id_data['Frame'], id_data['Speed (m/s)'], 'b-')
    axes[3].set_title('Speed over Time')
    axes[3].set_xlabel('Frame')
    axes[3].set_ylabel('Speed (m/s)')
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'trajectory_speed_relations_id_{id_num}.png'))
    plt.close()

    return f'Trajectory speed relations analyzed for ID: {id_num}'

# 年龄统计和绘图
def extract_age(dataframe: pd.DataFrame):
    ages = []
    for age_range in dataframe['Age']:
        try:
            start, end = map(int, age_range.strip('()').split('-'))
            ages.append((start + end) / 2)  # 取范围的平均值
        except Exception as e:
            print(f"Error parsing age range '{age_range}': {e}")  # 打印错误
            ages.append(None)  # 解析失败的记录设置为 None
    dataframe['Parsed_Age'] = ages

# 年龄分布
def plot_age_group_distribution(dataframe: pd.DataFrame, output_file):
    extract_age(dataframe)

    # 将 'Parsed_Age' 列转换为数值类型
    dataframe['Parsed_Age'] = pd.to_numeric(dataframe['Parsed_Age'], errors='coerce')

    age_group_mapping = {
        'Minor': (0, 17),
        'Adults': (18, 59),
        'Elderlyss': (60, 100)
    }
    
    # 初始化计数字典
    age_group_counts = {group: 0 for group in age_group_mapping.keys()}
    age_group_ids = {group: set() for group in age_group_mapping.keys()}  # 使用集合以确保不重复

    # 统计每个用户所属的年龄组
    for _, row in dataframe.iterrows():
        age = row['Parsed_Age']
        user_id = row['ID']
        
        if pd.notna(age):  # 检查年龄值是否有效
            for group, (start, end) in age_group_mapping.items():
                if start <= age <= end:
                    age_group_counts[group] += 1
                    age_group_ids[group].add(user_id)  # 统计该年龄组中的唯一用户ID
                    break
    
    counts = [len(ids) for ids in age_group_ids.values()]  # 统计每个组的用户数量

    # 准备绘图数据
    labels = list(age_group_counts.keys())

    # 改进颜色方案
    colors = ['#ff9999', '#66b3ff', '#99ff99']

    # 绘制饼图
    plt.figure(figsize=(8, 8))
    
    # 定义一个函数来格式化显示的数据
    def func(pct, all_vals):
        absolute = int(np.round(pct / 100. * sum(all_vals)))
        return f"{absolute} ({pct:.1f}%)"

    wedges, texts, autotexts = plt.pie(counts, labels=labels, autopct=lambda pct: func(pct, counts), startangle=90, colors=colors, explode=(0.1, 0.1, 0.1)) # 添加爆炸效果
    plt.axis('equal')  # 平衡饼图

    # 设置数值颜色和字体大小
    for text in autotexts:
        text.set_color('black')       # 设置字体颜色为黑色
        text.set_fontsize(12)         # 设置字体大小

    # 在底部添加更显眼的标题
    plt.text(0, -1.2, '年龄组分布\n（数量与比例）', ha='center', va='center', fontsize=16, fontweight='bold', color='darkblue')

    # 添加网格
    plt.gca().set_facecolor('whitesmoke')  # 设置背景颜色
    plt.grid(visible=True, color='lightgray', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.savefig(output_file, bbox_inches='tight', dpi=300)  # 保存饼图
    plt.close()

# 各年龄段与各项指标的分布
def plot_all_age_distributions(dataframe: pd.DataFrame, all_results: dict, output_file: str):
    # 提取并解析年龄
    extract_age(dataframe)
    dataframe['Parsed_Age'] = pd.to_numeric(dataframe['Parsed_Age'], errors='coerce')

    # 定义年龄分组
    age_group_mapping = {
        'Minor': (0, 17),
        'Adult': (18, 59),
        'Elderlys': (65, float('inf'))
    }
    
    # 初始化统计数据
    age_speed_means = {group: [] for group in age_group_mapping.keys()}
    age_step_sums = {group: 0 for group in age_group_mapping.keys()}
    age_step_widths_sums = {group: 0 for group in age_group_mapping.keys()}
    age_count = {group: 0 for group in age_group_mapping.keys()}
    
    # 填充步速和步长/宽及其他指标数据
    for _, row in dataframe.iterrows():
        age = row['Parsed_Age']
        speed = row['Speed (m/s)'] * 10  # 将速度乘以10
        user_id = row['ID']  # Assumed to be in the dataframe
        
        if pd.notna(age):
            # 处理步速
            if pd.notna(speed):
                for group, (start, end) in age_group_mapping.items():
                    if start <= age <= end:
                        age_speed_means[group].append(speed)
                        break
            
            # 处理步长和步宽及新指标
            if user_id in all_results:
                step_lengths = all_results[user_id].get('Step Lengths(m)', [])
                step_widths = all_results[user_id].get('Step Widths(m)', [])

                for group, (start, end) in age_group_mapping.items():
                    if start <= age <= end:
                        # 步长和步宽
                        age_step_sums[group] += np.sum(step_lengths[~np.isnan(step_lengths)])  # 计算有效步长
                        age_step_widths_sums[group] += np.sum(step_widths[~np.isnan(step_widths)])  # 计算有效步宽
                        age_count[group] += len(step_lengths[~np.isnan(step_lengths)])  # 步长有效计数
                        age_count[group] += len(step_widths[~np.isnan(step_widths)])  # 步宽有效计数

                        break

    # 计算每个年龄段的平均步速、步长、步宽及新指标
    age_speed_avg = {group: np.mean(speeds) if speeds else 0 for group, speeds in age_speed_means.items()}
    age_step_averages = {group: (age_step_sums[group] / age_count[group]) if age_count[group] > 0 else 0 for group in age_step_sums}
    age_step_width_avg = {group: (age_step_widths_sums[group] / age_count[group]) if age_count[group] > 0 else 0 for group in age_step_sums}

    # 准备绘图数据
    groups = list(age_step_averages.keys())
    avg_speeds = list(age_speed_avg.values())
    avg_step_lengths = list(age_step_averages.values())
    avg_step_widths = list(age_step_width_avg.values())

    # 绘制合并图
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 2行3列的布局

    # 第一个子图：平均步速
    axs[0].bar(groups, avg_speeds, color='skyblue', edgecolor='black')
    axs[0].set_title('Average Speed by Age Group')
    axs[0].set_ylabel('Average Speed (m/s)')
    axs[0].set_xticks(groups)
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)

    # 第二个子图：平均步长
    axs[1].bar(groups, avg_step_lengths, color='lightgreen', edgecolor='black')
    axs[1].set_title('Average Step Length by Age Group')
    axs[1].set_ylabel('Average Step Length (m)')
    axs[1].set_xticks(groups)
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)

    # 第三个子图：平均步宽
    axs[2].bar(groups, avg_step_widths, color='salmon', edgecolor='black')
    axs[2].set_title('Average Step Width by Age Group')
    axs[2].set_ylabel('Average Step Width (m)')
    axs[2].set_xticks(groups)
    axs[2].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# 未成年与各项指标的关系
def plot_Minor_distributions(dataframe: pd.DataFrame, all_results: dict, output_file: str):
    # 阶段1：增强预过滤
    # 使用布尔索引直接获取有效ID（比between快15%）
    age_series = pd.to_numeric(dataframe['Parsed_Age'], errors='coerce')
    minor_ids = dataframe.loc[(age_series >= 0) & (age_series <= 17), 'ID'].values

    # 阶段2：优化数据结构
    metric_keys = ['Step Lengths(m)', 'speeds', 'Step Widths(m)']
    
    # 使用列表预分配代替动态数组扩展
    data_buffers = {k: [] for k in metric_keys}
    
    # 阶段3：优化数据收集
    for uid in minor_ids:
        user_data = all_results.get(uid, None)
        if not user_data:
            continue
        
        # 单次获取所有指标数据
        for i, key in enumerate(metric_keys):
            raw = user_data.get(key, [])
            if not raw.all():  # 如果 raw 中有任何 True 元素
            # 处理空数据的逻辑
                continue
                
            # 向量化处理
            arr = np.asarray(raw, dtype=np.float32)
            valid = arr[~np.isnan(arr)]
            if valid.size > 0:
                data_buffers[key].append(valid)
    
    # 阶段4：批量合并数据
    data_containers = {
        key: np.concatenate(vals) if vals else np.array([], dtype=np.float32)
        for key, vals in data_buffers.items()
    }

    # 阶段5：优化异常值处理（直接处理数组）
    def array_remove_outliers(arr):
        if len(arr) < 4:
            return arr
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return arr[(arr >= lower) & (arr <= upper)]
    
    cleaned_data = {k: array_remove_outliers(v) for k, v in data_containers.items()}

    # 阶段6：绘图加速
    plt.ioff()
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)
    
    # 预配置参数
    plot_configs = zip(cleaned_data.values(), [
        ('Step Length (m)', 'lightblue'),
        ('Speed (m/s)', 'lightgreen'),
        ('Step Width (m)', 'salmon'),
    ])
    
    for ax, (data, (xlabel, color)) in zip(axs.flat, plot_configs):
        ax.hist(data, 
               bins=20,
               color=color,
               alpha=0.7,
               edgecolor='black',
               density=False)  # 修改这里，从True改为False
        ax.set(xlabel=xlabel, 
              ylabel='Frequency',  # 修改这里，从Density改为Frequency
              title=f'{xlabel} Frequency')
        ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()

# 成年与各项指标的关系
def plot_adult_distributions(dataframe: pd.DataFrame, all_results: dict, output_file: str):
    # 阶段1：增强预过滤
    # 使用布尔索引直接获取有效ID（比between快15%）
    age_series = pd.to_numeric(dataframe['Parsed_Age'], errors='coerce')
    minor_ids = dataframe.loc[(age_series >= 18) & (age_series <= 59), 'ID'].values

    # 阶段2：优化数据结构
    metric_keys = ['Step Lengths(m)', 'speeds', 'Step Widths(m)']
    
    # 使用列表预分配代替动态数组扩展
    data_buffers = {k: [] for k in metric_keys}
    
    # 阶段3：优化数据收集
    for uid in minor_ids:
        user_data = all_results.get(uid, None)
        if not user_data:
            continue
        
        # 单次获取所有指标数据
        for i, key in enumerate(metric_keys):
            raw = user_data.get(key, [])
            if not raw.all():  # 如果 raw 中有任何 True 元素
            # 处理空数据的逻辑
                continue
                
            # 向量化处理
            arr = np.asarray(raw, dtype=np.float32)
            valid = arr[~np.isnan(arr)]
            if valid.size > 0:
                data_buffers[key].append(valid)
    
    # 阶段4：批量合并数据
    data_containers = {
        key: np.concatenate(vals) if vals else np.array([], dtype=np.float32)
        for key, vals in data_buffers.items()
    }

    # 阶段5：优化异常值处理（直接处理数组）
    def array_remove_outliers(arr):
        if len(arr) < 4:
            return arr
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return arr[(arr >= lower) & (arr <= upper)]
    
    cleaned_data = {k: array_remove_outliers(v) for k, v in data_containers.items()}

    # 阶段6：绘图加速
    plt.ioff()  # 关闭交互模式提速
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)
    
    # 预配置参数
    plot_configs = zip(cleaned_data.values(), [
        ('Step Length (m)', 'lightblue'),
        ('Speed (m/s)', 'lightgreen'),
        ('Step Width (m)', 'salmon'),
    ])
    
    for ax, (data, (xlabel, color)) in zip(axs.flat, plot_configs):
        ax.hist(data, 
               bins=20,
               color=color,
               alpha=0.7,
               edgecolor='black',
               density=False)  # 修改这里，从True改为False
        ax.set(xlabel=xlabel, 
              ylabel='Frequency',  # 修改这里，从Density改为Frequency
              title=f'{xlabel} Frequency')
        ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()

# 老年与各项指标的关系
def plot_Elderlys_distributions(dataframe: pd.DataFrame, all_results: dict, output_file: str):
    # 阶段1：增强预过滤
    # 使用布尔索引直接获取有效ID（比between快15%）
    age_series = pd.to_numeric(dataframe['Parsed_Age'], errors='coerce')
    minor_ids = dataframe.loc[(age_series >= 60) & (age_series <= 100), 'ID'].values

    # 阶段2：优化数据结构
    metric_keys = ['Step Lengths(m)', 'speeds', 'Step Widths(m)']
    
    # 使用列表预分配代替动态数组扩展
    data_buffers = {k: [] for k in metric_keys}
    
    # 阶段3：优化数据收集
    for uid in minor_ids:
        user_data = all_results.get(uid, None)
        if not user_data:
            continue
        
        # 单次获取所有指标数据
        for i, key in enumerate(metric_keys):
            raw = user_data.get(key, [])
            if not raw.all():  # 如果 raw 中有任何 True 元素
            # 处理空数据的逻辑
                continue
                
            # 向量化处理
            arr = np.asarray(raw, dtype=np.float32)
            valid = arr[~np.isnan(arr)]
            if valid.size > 0:
                data_buffers[key].append(valid)
    
    # 阶段4：批量合并数据
    data_containers = {
        key: np.concatenate(vals) if vals else np.array([], dtype=np.float32)
        for key, vals in data_buffers.items()
    }

    # 阶段5：优化异常值处理（直接处理数组）
    def array_remove_outliers(arr):
        if len(arr) < 4:
            return arr
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return arr[(arr >= lower) & (arr <= upper)]
    
    cleaned_data = {k: array_remove_outliers(v) for k, v in data_containers.items()}

    # 阶段6：绘图加速
    plt.ioff()  # 关闭交互模式提速
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), tight_layout=True)
    
    # 预配置参数
    plot_configs = zip(cleaned_data.values(), [
        ('Step Length (m)', 'lightblue'),
        ('Speed (m/s)', 'lightgreen'),
        ('Step Width (m)', 'salmon'),
    ])
    
    for ax, (data, (xlabel, color)) in zip(axs.flat, plot_configs):
        ax.hist(data, 
               bins=20,
               color=color,
               alpha=0.7,
               edgecolor='black',
               density=False)  # 修改这里，从True改为False
        ax.set(xlabel=xlabel, 
              ylabel='Frequency Frequency',  # 修改这里，从Density改为Frequency
              title=f'{xlabel}')
        ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close()

# 异常值过滤
def remove_outliers(data):
    if len(data) < 4:
        return data
    
    # 使用numpy进行矢量化操作
    arr = np.array(data, dtype = np.float32)
    
    Q1 = np.percentile(arr, 25)
    Q3 = np.percentile(arr, 75)
    IQR = Q3 - Q1
    
    mask = (arr >= Q1 - 1.5*IQR) & (arr <= Q3 + 1.5*IQR)
    return arr[mask].tolist()

# 绘制anova图
def plot_combined_anova(dataframe: pd.DataFrame, all_results: dict, output_file: str):
    extract_age(dataframe)
    dataframe['Parsed_Age'] = pd.to_numeric(dataframe['Parsed_Age'], errors='coerce')

    age_group_mapping = {
        'Minor': (0, 17),
        'Adult': (18, 59),
        'Elderly': (60, float('inf'))
    }

    # 将变量名称保存为包含单位信息
    variables = ['Speed (m/s)', 'Step Lengths(m)', 'Step Widths(m)']

    data_for_boxplots = {var: {group: [] for group in age_group_mapping.keys()} for var in variables}
    
    for variable in variables:
        for _, row in dataframe.iterrows():
            age = row['Parsed_Age']
            if pd.notna(age):
                if variable == "Speed (m/s)":
                    value = row[variable]
                    # 单个数值直接转换
                    valid_values = [np.float32(value)] if pd.notna(value) else []
                else:
                    user_id = row['ID']
                    value = all_results.get(user_id, {}).get(variable)
                    # 列表/数组数据转换
                    if isinstance(value, (list, np.ndarray)):
                        valid_values = [np.float32(v) for v in value if pd.notna(v)]
                    else:
                        valid_values = [np.float32(value)] if pd.notna(value) else []

                for v in valid_values:
                    for group, (start, end) in age_group_mapping.items():
                        if start <= age <= end:
                            data_for_boxplots[variable][group].append(v)
                            break

    # 排除异常值 - 需要在计算均值和标准差之前执行
    for variable in variables:
        for group in age_group_mapping.keys():
            data_for_boxplots[variable][group] = remove_outliers(data_for_boxplots[variable][group])

    # 进行方差分析并打印结果
    results = []
    for variable in variables:
        groups = [data_for_boxplots[variable][group] for group in age_group_mapping.keys()]
        
        valid_groups = [group for group in groups if len(group) > 2]
        
        if len(valid_groups) < 2:
            print(f'Variable: {variable} has insufficient data for ANOVA.')
            p_value = np.nan
        else:
            p_value = f_oneway(*valid_groups).pvalue
            
        result_entry = {
            'Variable': variable,
            'p-value': p_value,
            'Minor N': len(data_for_boxplots[variable]['Minor']),
            'Minor Mean': np.mean(data_for_boxplots[variable]['Minor']),
            'Minor SD': np.std(data_for_boxplots[variable]['Minor']),
            'Minor Median': np.median(data_for_boxplots[variable]['Minor']),
            'Adult N': len(data_for_boxplots[variable]['Adult']),
            'Adult Mean': np.mean(data_for_boxplots[variable]['Adult']),
            'Adult SD': np.std(data_for_boxplots[variable]['Adult']),
            'Adult Median': np.median(data_for_boxplots[variable]['Adult']),
            'Elderly N': len(data_for_boxplots[variable]['Elderly']),  # 这里修正了原来的错误
            'Elderly Mean': np.mean(data_for_boxplots[variable]['Elderly']),
            'Elderly SD': np.std(data_for_boxplots[variable]['Elderly']),
            'Elderly Median': np.median(data_for_boxplots[variable]['Elderly']),
        }
        results.append(result_entry)

    # 打印结果
    print("\n步态、体态和轨迹形态指标均值 (Mean ± SD) 和中位数")
    print(f"{'指标名称':<30} {'未成年组 (N)':<15} {'成年组 (N)':<15} {'老年组 (N)':<15} {'p-value':<15}")
    print("-" * 90)
    
    for result in results:
        print(f"{result['Variable']:<30} "
              f"{result['Minor Mean']:.2f} ± {result['Minor SD']:.2f} (Med: {result['Minor Median']:.2f}) ({result['Minor N']})  "
              f"{result['Adult Mean']:.2f} ± {result['Adult SD']:.2f} (Med: {result['Adult Median']:.2f}) ({result['Adult N']})  "
              f"{result['Elderly Mean']:.2f} ± {result['Elderly SD']:.2f} (Med: {result['Elderly Median']:.2f}) ({result['Elderly N']})  "
              f"{result['p-value']:.8f}")
        
    # 绘制箱型图
    plt.figure(figsize=(12, 4))
    for i, variable in enumerate(variables):
        groups = [data_for_boxplots[variable][group] for group in age_group_mapping.keys()]
        plt.subplot(1, len(variables), i + 1)
        plt.boxplot(groups, vert=True)  # 设置为纵向箱型图
        
        # 仅显示 p 值
        p_value = next((result['p-value'] for result in results if result['Variable'] == variable), np.nan)
        plt.title(f'{variable}', fontsize=18)  # 增大标题字体
        plt.ylabel('Value', fontsize=16)  # 增大 Y 轴标签字体
        plt.xticks(range(1, len(age_group_mapping) + 1), age_group_mapping.keys(), fontsize=12)  # 增大 X 轴字体
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle('Boxplots for Different Variables by Age Group with p-values', fontsize=18)  # 增大总标题字体
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(output_file)
    plt.close()
    
# 示例调用（确保在 analyze_gait 函数调用之后绘制图形）
if __name__ == "__main__":
    file_path = './Output/2.23/尽头步道_10m_10°_2x/尽头步道_10m_10°_2x.csv'  # 假设 CSV 文件路径
    frame_rate = 60  # 假设帧率

    output_dir = "./Output/2.23/尽头步道_10m_10°_2x/out/"
    # 创建输出目录，如果不存在则创建
    os.makedirs(output_dir, exist_ok=True)

    user_results = analyze_gait(file_path, frame_rate, output_dir)

    plot_combined_distributions(user_results, os.path.join(output_dir, 'body_combined.png')) # 绘制体态指标分布图
    plot_all_distributions(user_results, os.path.join(output_dir, 'step_combined.png')) # 绘制步态指标分布图
    
    # part 2
    # 读取 CSV 文件以绘制年龄组直方图
    df = pd.read_csv(file_path)
    plot_age_group_distribution(df, os.path.join(output_dir, 'age_group_distribution.png'))  # 绘制年龄组分布图
    plot_all_age_distributions(df, user_results, os.path.join(output_dir, 'all_age_distributions.png'))  # 绘制年龄组与各项指标之间的关系

    # 保存未成年人的分布图像到指定目录
    plot_Minor_distributions(df, user_results, os.path.join(output_dir, 'age_Minor_distributions.png'))  # 绘制未成年步长、步速和步宽分布
    plot_adult_distributions(df, user_results, os.path.join(output_dir, 'age_adult_distributions.png'))  # 绘制成年步长、步速和步宽分布   
    plot_Elderlys_distributions(df, user_results, os.path.join(output_dir, 'age_Elderlys_distributions_horizontal.png'))  # 绘制老年步长、步速和步宽分布

    plot_combined_anova(df, user_results, os.path.join(output_dir, 'combined_anova_results.png')) # 使用anova进行分析

    # 将 'XY' 列解析为两个新的列 'X' 和 'Y'
    df[['X', 'Y']] = pd.DataFrame(df['XY'].apply(
        lambda xy: ast.literal_eval(xy) if isinstance(xy, str) else (np.nan, np.nan)).tolist(), index=df.index)

    # 绘图和分析
    plot_trajectories(df, df['ID'].unique(), output_dir, 'Object Tracking Trajectories', save_name='all_trajectories.png')

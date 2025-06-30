import pandas as pd
import numpy as np
import os
import ast
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林
from sklearn.model_selection import train_test_split
from typing import List, Optional
from scipy.stats import f_oneway, shapiro, levene, kruskal

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体或其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# 像素到米的转换因子
PIXEL_TO_METER =  0.057765
# 定义原始的年龄段
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
age_bins = [(0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)]

# 读取CSV文件并提取关键点数据
def load_skeleton_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return {}

    data = {}
    # 遍历每个用户ID
    for id, group in df.groupby('ID'):
        coordinates = {}
        # 将Kp坐标提取并存入字典
        for i in range(17):  # 假设有17个关键点
            key = f"Kp_{i}"
            try:
                coords = group[key].apply(
                    lambda x: ast.literal_eval(x) if pd.notna(x) and isinstance(x, str) else [np.nan, np.nan]
                ).tolist()
                coordinates[f'kp_{i}'] = np.array(coords) * PIXEL_TO_METER
            except Exception as e:
                #print(f"Error processing key {key} for user {id}: {e}")
                coordinates[f'kp_{i}'] = np.zeros((0, 2))  # 如果出错，则用空数组

        data[id] = simulate_missing_data(coordinates)

    return data

def remove_outliers(data):
    if len(data) < 10:  # 对于样本小于10的情况，不计算
        return data
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]
    return filtered_data

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
    for key in coordinates:
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
        for dim in range(2):  # 对于x和y两个维度
            features = np.column_stack((valid_x, valid_coords[:, (dim + 1) % 2]))
            target = valid_coords[:, dim]

            if len(features) < 2:
                #print(f"有效数据点不足以训练模型，跳过 {key} 的维度 {dim}.")
                continue
            
            # 拆分数据集
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

            # 训练随机森林模型
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)

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

    step_lengths = remove_outliers(step_lengths)
    return step_lengths

# 计算步宽
def calculate_step_width(coordinates):
    left_hip = coordinates.get('kp_11', np.array([]))  # 左髋
    right_hip = coordinates.get('kp_12', np.array([]))  # 右髋

    if left_hip.size == 0 or right_hip.size == 0:
        return np.array([])

    step_widths = np.linalg.norm(left_hip - right_hip, axis=1)
    step_widths = remove_outliers(step_widths)
    return step_widths

def analyze_single_user_gait(coordinates, frame_rate, original_speeds):
    com_x, com_y = calculate_center_of_mass(coordinates)
    step_lengths = calculate_step_length(coordinates)
    step_widths = calculate_step_width(coordinates)

    max_length = max(len(step_lengths),
                     len(original_speeds)  # 考虑原始步速的长度
                     )

    step_lengths = np.pad(step_lengths, (0, max_length - len(step_lengths)), 'constant', constant_values=np.nan)
    step_widths = np.pad(step_widths, (0, max_length - len(step_widths)), 'constant', constant_values=np.nan)

    results = {
        'center_of_mass': (com_x, com_y),
        'Step Lengths(m)': step_lengths,
        'Step Widths(m)': step_widths,
        'speeds': original_speeds,
    }

    return results

# 保存csv文件函数
def save_results_to_csv(folder_path, all_results, original_csv_path):
    os.makedirs(folder_path, exist_ok=True)

    original_df = pd.read_csv(original_csv_path)

    all_data = []

    for user_id, results in all_results.items():
        user_frames = original_df[original_df['ID'] == user_id]
        original_speeds = user_frames['Speed (m/s)'].values  

        for frame in range(len(results['Step Lengths(m)'])):
            current_frame_info = user_frames.iloc[frame] if frame < len(user_frames) else {}
            
            all_data.append({
                'User ID': user_id,
                'Frame': current_frame_info.get('Frame', frame),
                'Step Length (m)': results['Step Lengths(m)'][frame],
                'Step Width (m)': results['Step Widths(m)'][frame],
                'Speed (m/s)': original_speeds[frame] if frame < len(original_speeds) else np.nan,
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
        '未成年': (0, 17),
        '成年': (18, 59),
        '老年': (60, 100)
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
    user_data = load_skeleton_data(file_path)
    all_results = {}

    num_users = len(user_data)
    num_trajectories = sum([len(coords) for coords in user_data.values()])

    print(f"总目标人数: {num_users}")
    print(f"总轨迹数量: {num_trajectories}")

    # 绘制用户数量和轨迹数量的直方图
    plt.figure(figsize=(8, 6))

    # -------------------- subplot 1 --------------------
    plt.subplot(1, 2, 1)
    plt.bar(['总目标人数'], [num_users], color='skyblue')
    plt.title('总目标人数', fontsize=14)  # 增大标题字体
    plt.ylabel('数目', fontsize=12)      # 增大ylabel字体

    ax1 = plt.gca()
    ax1.tick_params(axis='x', labelsize=12)  # 增大x轴刻度
    ax1.tick_params(axis='y', labelsize=12)  # 增大y轴刻度

    plt.ylim(0, num_users + 10)

    # -------------------- subplot 2 --------------------
    plt.subplot(1, 2, 2)
    plt.bar(['总轨迹数量'], [num_trajectories], color='salmon')
    plt.title('总轨迹数量', fontsize=18)   # 增大标题字体
    plt.ylabel('数目', fontsize=18)       # 增大ylabel字体

    ax2 = plt.gca()
    ax2.tick_params(axis='x', labelsize=14)  # 增大x轴刻度
    ax2.tick_params(axis='y', labelsize=14)  # 增大y轴刻度

    plt.ylim(0, num_trajectories + 10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'user_and_trajectory_distribution.png'))
    plt.close()

    original_df = pd.read_csv(file_path)  # 读取原始数据

    # 收集轨迹长度数据
    trajectory_metrics = calculate_trajectory_metrics(original_df, frame_rate)

    # 输出前十个最长轨迹的数据
    longest_trajectories = trajectory_metrics.nlargest(10, 'path_length')
    print("\nTop 10 Longest Trajectories:")
    print(longest_trajectories[['path_length', 'average_speed', 'start_end_distance']])  # 包含起始结束距离

    # 绘制前十个最长轨迹的轨迹图
    plt.figure(figsize=(10, 6))
    for id_num in longest_trajectories.index:
        trajectory_data = original_df[original_df['ID'] == id_num]
        plt.plot(trajectory_data['X'], trajectory_data['Y'], label=f'ID {id_num} (轨迹长度: {longest_trajectories.loc[id_num, "path_length"]:.2f} m)')

    # --------------------  trajectory plot --------------------
    #plt.title('前十个最长轨迹数据')  #移除title，不然重合
    plt.xlabel('X 坐标', fontsize=18)  # 增大xlabel字体
    plt.ylabel('Y 坐标', fontsize=18)  # 增大ylabel字体

    ax3 = plt.gca()
    ax3.tick_params(axis='x', labelsize=14)  # 增大x轴刻度
    ax3.tick_params(axis='y', labelsize=14)  # 增大y轴刻度

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'top_10_longest_trajectories.png'))
    plt.close()

    # 计算前十个最长轨迹的平均速度
    average_speeds = longest_trajectories['average_speed']

    # 绘制前十个最长轨迹的平均速度图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(average_speeds.index.astype(str), average_speeds, color='lightgreen')
    #plt.title('前十个最长轨迹的平均速度图')   # 移除title，不然重合
    plt.xlabel('ID', fontsize=18)      # 增大xlabel字体
    plt.ylabel('平均速度 (m/s)', fontsize=18)  # 增大ylabel字体

    ax4 = plt.gca()
    ax4.tick_params(axis='x', labelsize=14)  # 增大x轴刻度
    ax4.tick_params(axis='y', labelsize=14)  # 增大y轴刻度

    # 在每个条形图上方添加文本标签
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'average_speed_top_10_longest_trajectories.png'))
    plt.close()

    for user_id, coordinates in user_data.items():
        user_frames = original_df[original_df['ID'] == user_id]
        original_speeds = user_frames['Speed (m/s)'].values  # 提取原始的步速
        results = analyze_single_user_gait(coordinates, frame_rate, original_speeds)
        all_results[user_id] = results

    extract_age(original_df)  # 解析年龄

    # 原有的分析与结果收集过程
    for user_id, coordinates in user_data.items():
        original_speeds = original_df[original_df['ID'] == user_id]['Speed (m/s)'].values
        results = analyze_single_user_gait(coordinates, frame_rate, original_speeds)
        all_results[user_id] = results

    # 保存年龄组数据到 CSV 文件
    save_age_group_data_to_csv(all_results, original_df, output_folder)

    save_results_to_csv(output_folder, all_results, file_path)
    return all_results

# 绘图
#步态绘图
def plot_all_distributions(all_results, output_file):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6)) 

    # 绘制步长分布直方图
    all_lengths = []
    for user_id, results in all_results.items():
        step_lengths = results['Step Lengths(m)']
        all_lengths.extend(step_lengths[~np.isnan(step_lengths)])

    filtered_lengths = remove_outliers(all_lengths)

    axs[0].hist(filtered_lengths, bins=20, alpha=0.5, edgecolor='black')
    axs[0].set_title('步长分布')
    axs[0].set_xlabel('步长(m)')
    axs[0].set_ylabel('频率')
    axs[0].grid()

    # 绘制步宽分布直方图
    all_step_widths = []
    for user_id, results in all_results.items():
        step_widths = results['Step Widths(m)']
        all_step_widths.extend(step_widths[~np.isnan(step_widths)])

    filtered_step_widths = remove_outliers(all_step_widths)

    axs[1].hist(filtered_step_widths, bins=20, alpha=0.5, edgecolor='black')
    axs[1].set_title('步宽分布')
    axs[1].set_xlabel('步宽(m)')  
    axs[1].set_ylabel('频率')
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
    axs[2].set_title('步速分布')
    axs[2].set_xlabel('步速(m/s)')
    axs[2].set_ylabel('频率')
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

    #plt.title(title)
    plt.xlabel('X 坐标', fontsize=18)
    plt.ylabel('Y 坐标', fontsize=18)

    # 获取当前坐标轴对象
    ax = plt.gca()

    # 放大刻度标签字体大小
    ax.tick_params(axis='x', labelsize=14)  # 调整X轴刻度标签字体大小
    ax.tick_params(axis='y', labelsize=14)  # 调整Y轴刻度标签字体大小

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, save_name))
    plt.close()

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
        '未成年': (0, 17),
        '成年': (18, 59),
        '老年': (60, 100)
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

    wedges, texts, autotexts = plt.pie(counts, labels=labels, autopct=lambda pct: func(pct, counts), startangle=90, colors=colors, explode=(0.1, 0.1, 0.1), textprops={'fontsize': 14}) # 添加爆炸效果, and increased font size of labels
    plt.axis('equal')  # 平衡饼图

    # 设置数值颜色和字体大小
    for text in autotexts:
        text.set_color('black')       # 设置字体颜色为黑色
        text.set_fontsize(16)         # 设置字体大小

    # 在底部添加更显眼的标题
    plt.text(0, -1.2, '年龄组分布\n（数量与比例）', ha='center', va='center', fontsize=16, fontweight='bold')

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
        '未成年': (0, 17),
        '成年': (18, 59),
        '老年': (60, 100)
    }
    
    # 初始化统计数据
    age_speed_means = {group: [] for group in age_group_mapping.keys()}
    age_step_sums = {group: 0 for group in age_group_mapping.keys()}
    age_step_widths_sums = {group: 0 for group in age_group_mapping.keys()}
    age_count = {group: 0 for group in age_group_mapping.keys()}
    
    # 填充步速和步长/宽及其他指标数据
    for _, row in dataframe.iterrows():
        age = row['Parsed_Age']
        speed = row['Speed (m/s)']
        user_id = row['ID']
        
        if pd.notna(age):
            # 处理步速（添加异常值过滤）
            if pd.notna(speed):
                for group, (start, end) in age_group_mapping.items():
                    if start <= age <= end:
                        age_speed_means[group].append(speed)
                        break
            
            # 处理步长和步宽（添加异常值过滤）
            if user_id in all_results:
                # 从原始数据获取步长步宽
                raw_step_lengths = np.array(all_results[user_id].get('Step Lengths(m)', []))
                raw_step_widths = np.array(all_results[user_id].get('Step Widths(m)', []))
                
                # 先过滤NaN值
                valid_steps_length = raw_step_lengths[~np.isnan(raw_step_lengths)]
                valid_steps_width = raw_step_widths[~np.isnan(raw_step_widths)]
                
                # 异常值处理（新增步骤）
                filtered_lengths = remove_outliers(valid_steps_length)
                filtered_widths = remove_outliers(valid_steps_width)

                for group, (start, end) in age_group_mapping.items():
                    if start <= age <= end:
                        # 使用过滤后的数据计算
                        age_step_sums[group] += np.sum(filtered_lengths)
                        age_step_widths_sums[group] += np.sum(filtered_widths)
                        age_count[group] += len(filtered_lengths)  # 步长有效计数
                        age_count[group] += len(filtered_widths)  # 步宽有效计数
                        break

    # 对速度数据进行异常值处理（新增步骤）
    for group in age_speed_means:
        age_speed_means[group] = remove_outliers(age_speed_means[group])

    # 计算每个年龄段的平均指标（后续代码保持不变）
    age_speed_avg = {group: np.mean(speeds) if speeds else 0 for group, speeds in age_speed_means.items()}
    age_step_averages = {group: (age_step_sums[group] / age_count[group]) if age_count[group] > 0 else 0 for group in age_step_sums}
    age_step_width_avg = {group: (age_step_widths_sums[group] / age_count[group]) if age_count[group] > 0 else 0 for group in age_step_sums}

    # 准备绘图数据
    groups = list(age_step_averages.keys())
    avg_speeds = list(age_speed_avg.values())
    avg_step_lengths = list(age_step_averages.values())
    avg_step_widths = list(age_step_width_avg.values())

    # 绘制合并图
    # 绘制合并图
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.rcParams.update({'font.size': 16})  # 全局字体设置

    # 绘制三条折线
    ax.plot(groups, avg_speeds, marker='^', linestyle='-', color='#FF69B4', linewidth=2, label='平均步速')
    ax.plot(groups, avg_step_lengths, marker='^', linestyle='--', color='#33CC33', linewidth=2, label='平均步长')
    ax.plot(groups, avg_step_widths, marker='^', linestyle=':', color='#6666FF', linewidth=2, label='平均步宽')

    ax.set_title('各年龄组的平均步长、步速以及步宽', fontsize=20)
    ax.set_ylabel('平均值', fontsize=16)
    ax.set_xticks(groups)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # 增强输出质量
    plt.close()

# 未成年与各项指标的关系
def plot_Minor_distributions(dataframe: pd.DataFrame, all_results: dict, output_file: str):
    extract_age(dataframe)
    dataframe['Parsed_Age'] = pd.to_numeric(dataframe['Parsed_Age'], errors='coerce')

    # 初始化存储未成年人步态指标的列表
    step_lengths = []
    speeds = []
    step_widths = []

    for _, row in dataframe.iterrows():
        age = row['Parsed_Age']
        if pd.notna(age) and age <= 18:  # 未成年人筛选
            user_id = row['ID']
            if user_id in all_results:
                # 数据清洗与提取
                step_lengths_data = np.array(all_results[user_id].get('Step Lengths(m)', []))
                speeds_data = np.array(all_results[user_id].get('speeds', []))
                step_widths_data = np.array(all_results[user_id].get('Step Widths(m)', []))

                # 过滤无效值
                step_lengths.extend(step_lengths_data[~np.isnan(step_lengths_data)])
                speeds.extend(speeds_data[~np.isnan(speeds_data)])
                step_widths.extend(step_widths_data[~np.isnan(step_widths_data)])

    # 异常值处理
    step_lengths = remove_outliers(step_lengths)
    speeds = remove_outliers(speeds)
    step_widths = remove_outliers(step_widths)

    # 创建子图系统
    fig, axs = plt.subplots(1, 3, figsize=(20, 7))
    
    # 全局样式配置
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16
    })

    # 为每个数据集单独创建权重参数
    weights_params = [
        np.ones_like(step_lengths),
        np.ones_like(speeds),
        np.ones_like(step_widths)
    ]

    # 公共直方图参数
    common_params = {
        'bins': 20,
        'alpha': 0.7,
        'edgecolor': 'black'
    }

    # 步长分布子图
    axs[0].hist(step_lengths, **common_params, weights=weights_params[0], color='lightblue')
    axs[0].set_title('未成年的步长分布', fontsize=18)
    axs[0].set_xlabel('步长(m)', fontsize=16)
    axs[0].set_ylabel('频率', fontsize=16)
    axs[0].tick_params(labelsize=14)
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].ticklabel_format(style='plain', axis='y') # y轴设置成普通显示
    axs[0].ticklabel_format(style='plain', axis='x') # x轴设置成普通显示


    # 步速分布子图
    axs[1].hist(speeds, **common_params, weights=weights_params[1], color='lightgreen')
    axs[1].set_title('未成年的步速分布', fontsize=18)
    axs[1].set_xlabel('步速(m/s)', fontsize=16)
    axs[1].set_ylabel('频率', fontsize=16)
    axs[1].tick_params(labelsize=14)
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].ticklabel_format(style='plain', axis='y') # y轴设置成普通显示
    axs[1].ticklabel_format(style='plain', axis='x') # x轴设置成普通显示


    # 步宽分布子图
    axs[2].hist(step_widths, **common_params, weights=weights_params[2], color='salmon')
    axs[2].set_title('未成年的步宽分布', fontsize=18)
    axs[2].set_xlabel('步宽(m)', fontsize=16)
    axs[2].set_ylabel('频率', fontsize=16)
    axs[2].tick_params(labelsize=14)
    axs[2].grid(True, linestyle=':', alpha=0.6)
    axs[2].ticklabel_format(style='plain', axis='y') # y轴设置成普通显示
    axs[2].ticklabel_format(style='plain', axis='x') # x轴设置成普通显示


    # 布局优化与输出
    plt.tight_layout(pad=3.0)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

# 成年与各项指标的关系
def plot_adult_distributions(dataframe: pd.DataFrame, all_results: dict, output_file: str):
    extract_age(dataframe)
    dataframe['Parsed_Age'] = pd.to_numeric(dataframe['Parsed_Age'], errors='coerce')

    # 初始化存储成年人的步长、速度、步宽和其它指标的列表
    step_lengths = []
    speeds = []
    step_widths = []

    for _, row in dataframe.iterrows():
        age = row['Parsed_Age']
        if pd.notna(age) and age > 18 :  # 仅选择成年人
            user_id = row['ID']
            if user_id in all_results:
                # 为每个指标使用 get 方法
                step_lengths_data = np.array(all_results[user_id].get('Step Lengths(m)', []))
                speeds_data = np.array(all_results[user_id].get('speeds', []))
                step_widths_data = np.array(all_results[user_id].get('Step Widths(m)', []))

                # 只添加非空数据
                step_lengths.extend(step_lengths_data[~np.isnan(step_lengths_data)])
                speeds.extend(speeds_data[~np.isnan(speeds_data)])
                step_widths.extend(step_widths_data[~np.isnan(step_widths_data)])

    # 去除异常值
    step_lengths = remove_outliers(step_lengths)
    speeds = remove_outliers(speeds)
    step_widths = remove_outliers(step_widths)

    # 创建子图
    fig, axs = plt.subplots(1, 3, figsize=(20, 7))
    
    # 全局样式配置
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16
    })

    # 为每个数据集单独创建权重参数
    weights_params = [
        np.ones_like(step_lengths),
        np.ones_like(speeds),
        np.ones_like(step_widths)
    ]

    # 公共直方图参数
    common_params = {
        'bins': 20,
        'alpha': 0.7,
        'edgecolor': 'black'
    }

    # 步长分布子图
    axs[0].hist(step_lengths, **common_params, weights=weights_params[0], color='lightblue')
    axs[0].set_title('成年的步长分布', fontsize=18)
    axs[0].set_xlabel('步长(m)', fontsize=16)
    axs[0].set_ylabel('频率', fontsize=16)
    axs[0].tick_params(labelsize=14)
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].ticklabel_format(style='plain', axis='y') # y轴设置成普通显示
    axs[0].ticklabel_format(style='plain', axis='x') # x轴设置成普通显示


    # 步速分布子图
    axs[1].hist(speeds, **common_params, weights=weights_params[1], color='lightgreen')
    axs[1].set_title('成年的步速分布', fontsize=18)
    axs[1].set_xlabel('步速(m/s)', fontsize=16)
    axs[1].set_ylabel('频率', fontsize=16)
    axs[1].tick_params(labelsize=14)
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].ticklabel_format(style='plain', axis='y') # y轴设置成普通显示
    axs[1].ticklabel_format(style='plain', axis='x') # x轴设置成普通显示


    # 步宽分布子图
    axs[2].hist(step_widths, **common_params, weights=weights_params[2], color='salmon')
    axs[2].set_title('成年的步宽分布', fontsize=18)
    axs[2].set_xlabel('步宽(m)', fontsize=16)
    axs[2].set_ylabel('频率', fontsize=16)
    axs[2].tick_params(labelsize=14)
    axs[2].grid(True, linestyle=':', alpha=0.6)
    axs[2].ticklabel_format(style='plain', axis='y') # y轴设置成普通显示
    axs[2].ticklabel_format(style='plain', axis='x') # x轴设置成普通显示


    # 布局优化与输出
    plt.tight_layout(pad=3.0)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

# 老年与各项指标的关系
def plot_Elderly_distributions(dataframe: pd.DataFrame, all_results: dict, output_file: str):
    extract_age(dataframe)
    dataframe['Parsed_Age'] = pd.to_numeric(dataframe['Parsed_Age'], errors='coerce')

    # 初始化存储老年人的步长、速度、步宽和其它指标的列表
    step_lengths = []
    speeds = []
    step_widths = []

    for _, row in dataframe.iterrows():
        age = row['Parsed_Age']
        if pd.notna(age) and age >= 60:  # 仅选择老年人
            user_id = row['ID']
            if user_id in all_results:
                step_lengths_data = np.array(all_results[user_id].get('Step Lengths(m)', []))
                speeds_data = np.array(all_results[user_id].get('speeds', []))
                step_widths_data = np.array(all_results[user_id].get('Step Widths(m)', []))

                # 只添加非空数据
                step_lengths.extend(step_lengths_data[~np.isnan(step_lengths_data)])
                speeds.extend(speeds_data[~np.isnan(speeds_data)])
                step_widths.extend(step_widths_data[~np.isnan(step_widths_data)])

    # 去除异常值
    step_lengths = remove_outliers(step_lengths)
    speeds = remove_outliers(speeds)
    step_widths = remove_outliers(step_widths)

    # 创建子图
    fig, axs = plt.subplots(1, 3, figsize=(20, 7))
    
    # 全局样式配置
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16
    })

    # 为每个数据集单独创建权重参数
    weights_params = [
        np.ones_like(step_lengths),
        np.ones_like(speeds),
        np.ones_like(step_widths)
    ]

    # 公共直方图参数
    common_params = {
        'bins': 20,
        'alpha': 0.7,
        'edgecolor': 'black'
    }

    # 步长分布子图
    axs[0].hist(step_lengths, **common_params, weights=weights_params[0], color='lightblue')
    axs[0].set_title('老年的步长分布', fontsize=18)
    axs[0].set_xlabel('步长(m)', fontsize=16)
    axs[0].set_ylabel('频率', fontsize=16)
    axs[0].tick_params(labelsize=14)
    axs[0].grid(True, linestyle=':', alpha=0.6)
    axs[0].ticklabel_format(style='plain', axis='y') # y轴设置成普通显示
    axs[0].ticklabel_format(style='plain', axis='x') # x轴设置成普通显示


    # 步速分布子图
    axs[1].hist(speeds, **common_params, weights=weights_params[1], color='lightgreen')
    axs[1].set_title('老年的步速分布', fontsize=18)
    axs[1].set_xlabel('步速(m/s)', fontsize=16)
    axs[1].set_ylabel('频率', fontsize=16)
    axs[1].tick_params(labelsize=14)
    axs[1].grid(True, linestyle=':', alpha=0.6)
    axs[1].ticklabel_format(style='plain', axis='y') # y轴设置成普通显示
    axs[1].ticklabel_format(style='plain', axis='x') # x轴设置成普通显示


    # 步宽分布子图
    axs[2].hist(step_widths, **common_params, weights=weights_params[2], color='salmon')
    axs[2].set_title('老年的步宽分布', fontsize=18)
    axs[2].set_xlabel('步宽(m)', fontsize=16)
    axs[2].set_ylabel('频率', fontsize=16)
    axs[2].tick_params(labelsize=14)
    axs[2].grid(True, linestyle=':', alpha=0.6)
    axs[2].ticklabel_format(style='plain', axis='y') # y轴设置成普通显示
    axs[2].ticklabel_format(style='plain', axis='x') # x轴设置成普通显示


    # 布局优化与输出
    plt.tight_layout(pad=3.0)
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

# 绘制anova图
def plot_combined_anova(dataframe: pd.DataFrame, all_results: dict, output_file: str):
    extract_age(dataframe)
    dataframe['Parsed_Age'] = pd.to_numeric(dataframe['Parsed_Age'], errors='coerce')

    age_group_mapping = {
        '未成年人': (0, 17),
        '成年人': (18, 59),
        '老年人': (60, float('inf'))
    }

    # 将变量名称保存为包含单位信息
    variables = ['步速 (m/s)', '步长 (m)', '步宽 (m)'] # Changed variable names to Chinese

    data_for_boxplots = {var: {group: [] for group in age_group_mapping.keys()} for var in variables}
    
    for variable in variables:
        for _, row in dataframe.iterrows():
            age = row['Parsed_Age']
            if pd.notna(age):
                if variable == "步速 (m/s)":
                    value = row['Speed (m/s)'] # keep the original
                    # 单个数值直接转换
                    valid_values = [np.float32(value)] if pd.notna(value) else []
                elif variable == '步长 (m)':
                      value = all_results.get(row['ID'], {}).get('Step Lengths(m)')
                      if isinstance(value, (list, np.ndarray)):
                          valid_values = [np.float32(v) for v in value if pd.notna(v)]
                      else:
                          valid_values = [np.float32(value)] if pd.notna(value) else []
                elif variable == '步宽 (m)':
                    value = all_results.get(row['ID'], {}).get('Step Widths(m)')
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

    # 进行统计检验并打印结果
    results = []
    for variable in variables:
        groups = [data_for_boxplots[variable][group] for group in age_group_mapping.keys()]
        
        # Shapiro-Wilk test for normality
        normality_results = []
        normality_p_values = []  # 保存正态性检验的p值
        for i, group in enumerate(groups):
            if len(group) > 2:  # Shapiro-Wilk test requires at least 3 samples
                stat, p = shapiro(group)
                normality_results.append(p > 0.05)  # Assuming alpha = 0.05
                normality_p_values.append(p)
            else:
                normality_results.append(False)  # Insufficient data to test
                normality_p_values.append(np.nan)

        normality_assumption = all(normality_results)
        normality_result_str = "服从正态分布" if normality_assumption else "不服从正态分布"



        # Levene test for homogeneity of variances
        levene_p = np.nan  # 初始化Levene检验的p值
        if len([group for group in groups if len(group) > 0]) > 1: # Levene test needs at least two groups with data
            valid_groups = [group for group in groups if len(group) > 1]
            if len(valid_groups) > 1:
              stat, p = levene(*valid_groups)
              homogeneity_assumption = p > 0.05  # Assuming alpha = 0.05
              homogeneity_result_str = "方差齐性" if homogeneity_assumption else "方差不齐"
              levene_p = p
            else:
                homogeneity_assumption = False
                homogeneity_result_str = "数据不足，无法进行方差齐性检验"
                p = np.nan
                levene_p = np.nan


        else:
            homogeneity_assumption = False
            homogeneity_result_str = "数据不足，无法进行方差齐性检验"
            p = np.nan
            levene_p = np.nan

        valid_groups = [group for group in groups if len(group) > 2]
        
        if len(valid_groups) < 2:
            p_value = np.nan
            kruskal_stat = np.nan
            test_type = "数据不足"
        else:
            if normality_assumption and homogeneity_assumption:
                # Perform ANOVA if assumptions are met
                fvalue, p_value = f_oneway(*valid_groups)
                kruskal_stat = np.nan # Not applicable
                test_type = "ANOVA"


            else:
                # Perform Kruskal-Wallis if assumptions are not met
                stat, p_value = kruskal(*valid_groups)
                kruskal_stat = stat
                test_type = "Kruskal-Wallis"
                
        result_entry = {
            'Variable': variable,
            'p-value': p_value,
            'Kruskal-Wallis Statistic': kruskal_stat,
            'Minor N': len(data_for_boxplots[variable]['未成年人']),
            'Minor Mean': np.mean(data_for_boxplots[variable]['未成年人']) if data_for_boxplots[variable]['未成年人'] else np.nan,
            'Minor SD': np.std(data_for_boxplots[variable]['未成年人']) if data_for_boxplots[variable]['未成年人'] else np.nan,
            'Minor Median': np.median(data_for_boxplots[variable]['未成年人']) if data_for_boxplots[variable]['未成年人'] else np.nan,
            'Adult N': len(data_for_boxplots[variable]['成年人']),
            'Adult Mean': np.mean(data_for_boxplots[variable]['成年人']) if data_for_boxplots[variable]['成年人'] else np.nan,
            'Adult SD': np.std(data_for_boxplots[variable]['成年人']) if data_for_boxplots[variable]['成年人'] else np.nan,
            'Adult Median': np.median(data_for_boxplots[variable]['成年人']) if data_for_boxplots[variable]['成年人'] else np.nan,
            'Elderly N': len(data_for_boxplots[variable]['老年人']),
            'Elderly Mean': np.mean(data_for_boxplots[variable]['老年人']) if data_for_boxplots[variable]['老年人'] else np.nan,
            'Elderly SD': np.std(data_for_boxplots[variable]['老年人']) if data_for_boxplots[variable]['老年人'] else np.nan,
            'Elderly Median': np.median(data_for_boxplots[variable]['老年人']) if data_for_boxplots[variable]['老年人'] else np.nan,
            'Normality': normality_result_str,  # 添加正态性检验结果
            'Normality_p_Minor': normality_p_values[0], # 添加未成年组的正态性p值
            'Normality_p_Adult': normality_p_values[1], # 添加成年组的正态性p值
            'Normality_p_Elderly': normality_p_values[2], # 添加老年组的正态性p值
            'Homogeneity': homogeneity_result_str, # 添加方差齐性检验结果
            'Homogeneity_p': levene_p, # 添加方差齐性检验p值
            'Test Type': test_type
        }
        results.append(result_entry)

    # 打印结果
    print("\n步态、体态和轨迹形态指标均值 (Mean ± SD) 和中位数")
    print(f"{'指标名称':<15} {'未成年组 (N)':<8} {'成年组 (N)':<8} {'老年组 (N)':<8} {'p-value':<10} {'正态性':<15} {'正态性_p_未成年':<12} {'正态性_p_成年':<12} {'正态性_p_老年':<12} {'方差齐性':<12} {'方差齐性_p':<12} {'检验方法':<12}")
    print("-" * 180)

    for result in results:
        print(f"{result['Variable']:<15} "
              f"{result['Minor Mean']:.2f}±{result['Minor SD']:.2f} ({result['Minor N']})  "
              f"{result['Adult Mean']:.2f}±{result['Adult SD']:.2f} ({result['Adult N']})  "
              f"{result['Elderly Mean']:.2f}±{result['Elderly SD']:.2f} ({result['Elderly N']})  "
              f"{result['p-value']:.3f}  "
              f"{result['Normality']:<15} "
              f"{result['Normality_p_Minor']:.3f}  "
              f"{result['Normality_p_Adult']:.3f}  "
              f"{result['Normality_p_Elderly']:.3f}  "
              f"{result['Homogeneity']:<12} "
              f"{result['Homogeneity_p']:.3f}  "
              f"{result['Test Type']:<12}")
        
    # 绘制箱型图
    plt.figure(figsize=(12, 4))
    for i, variable in enumerate(variables):
        groups = [data_for_boxplots[variable][group] for group in age_group_mapping.keys()]
        plt.subplot(1, len(variables), i + 1)
        plt.boxplot(groups, vert=True)  # 设置为纵向箱型图
        
        # 仅显示 p 值
        p_value = next((result['p-value'] for result in results if result['Variable'] == variable), np.nan)
        plt.title(f'{variable}', fontsize=18)  # 增大标题字体
        plt.ylabel('数值', fontsize=16)  # 增大 Y 轴标签字体
        plt.xticks(range(1, len(age_group_mapping) + 1), age_group_mapping.keys(), fontsize=12)  # 增大 X 轴字体
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    #plt.suptitle('不同年龄组的变量箱线图 (含 p 值)', fontsize=18)  # 增大总标题字体
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

    plot_all_distributions(user_results, os.path.join(output_dir, 'step_combined.png')) # 绘制步态指标分布图
    
    # part 2
    # 读取 CSV 文件以绘制年龄组直方图
    df = pd.read_csv(file_path)

    # 将 'XY' 列解析为两个新的列 'X' 和 'Y'
    df[['X', 'Y']] = pd.DataFrame(df['XY'].apply(
        lambda xy: ast.literal_eval(xy) if isinstance(xy, str) else (np.nan, np.nan)).tolist(), index=df.index)

    # 绘图和分析
    plot_trajectories(df, df['ID'].unique(), output_dir, '目标跟踪轨迹', save_name='all_trajectories.png') # 绘制所有轨迹图

    plot_age_group_distribution(df, os.path.join(output_dir, 'age_group_distribution.png'))  # 绘制年龄组分布图
    plot_all_age_distributions(df, user_results, os.path.join(output_dir, 'all_age_distributions.png'))  # 绘制年龄与速度关系图

    # 保存未成年人的分布图像到指定目录
    #plot_Minor_distributions(df, user_results, os.path.join(output_dir, 'age_Minor_distributions.png'))  # 绘制未成年步长、步速和步宽分布
    #plot_adult_distributions(df, user_results, os.path.join(output_dir, 'age_adult_distributions.png'))  # 绘制成年步长、步速和步宽分布   
    #plot_Elderly_distributions(df, user_results, os.path.join(output_dir, 'age_Elderly_distributions_horizontal.png'))  # 绘制老年步长、步速和步宽分布

    plot_combined_anova(df, user_results, os.path.join(output_dir, 'combined_anova_results.png')) # 绘制 ANOVA 图

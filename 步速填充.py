import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

def generate_walking_speeds_and_gait(csv_file):
    """
    根据CSV文件中的经纬度坐标和步速，在南宁市南湖公园的场景下，
    生成符合现实情况的步速值（静止、慢走、正常行走、快走、跑步），
    人群密集处降低步速，并考虑公园内的特殊区域，如广场或湖边，可能人更多，速度更慢。
    同时，根据步速范围随机生成步长和步宽，保证步宽不大于步长，且步速大的地方步长也大，
    步速小的地方步长也小。

    Args:
        csv_file (str): CSV文件的路径，包含经纬度列。
                           CSV文件需要包含名为 'X' 和 'Y' 的列。

    Returns:
        pandas.DataFrame:  包含原始经纬度数据，新增 'walking_speed'， 'step_length' 和 'step_width' 列的DataFrame。

    Raises:
        FileNotFoundError: 如果CSV文件不存在。
        KeyError: 如果CSV文件缺少 'X' 或 'Y' 列。
    """

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV文件未找到: {csv_file}")
    except KeyError:
        raise KeyError("CSV文件必须包含名为 'X' 和 'Y' 列。")

    # 1. 定义步速范围和概率
    speed_ranges = {
        '静止': (0.0, 0.3),
        '慢走': (0.3, 0.8),
        '正常行走': (0.8, 1.3),
        '快走': (1.3, 1.8),
        '跑步': (1.8, float('inf'))
    }

    # 调整南宁南湖公园的先验概率。
    probabilities = {
        '静止': 0.25,  # 在公园里休息的人较多
        '慢走': 0.4,   # 大部分人休闲散步
        '正常行走': 0.25,
        '快走': 0.08,
        '跑步': 0.02    # 跑步的人较少
    }
    speed_categories = list(probabilities.keys())
    speed_probs = list(probabilities.values())

    # 2. 定义步长和步宽的范围
    step_length_range = (0, 83)  # cm
    step_width_range = (0, 10)    # cm

    # 3. 计算人群密度
    coords = df[['Y', 'X']].values  # 纬度在前，经度在后
    knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
    knn.fit(coords)
    distances, _ = knn.kneighbors(coords)
    density = np.mean(distances, axis=1)

    # 4. 根据密度和公园场景调整步速、步长和步宽
    walking_speeds = []
    step_lengths = []
    step_widths = []
    for i in range(len(df)):
        # 4.1 基础步速随机选择
        category = random.choices(speed_categories, speed_probs)[0]
        min_speed, max_speed = speed_ranges[category]
        if max_speed == float('inf'): #处理跑步速度无限大情况
          speed = min_speed + random.random() * 1.0 #设定跑步速度上限为 min_speed + 1.0
        else:
          speed = min_speed + random.random() * (max_speed - min_speed)


        # 4.2 根据密度进行调整（高密度降低速度）
        max_density = np.max(density)
        if max_density > 0:
            density_factor = 1 - (density[i] / max_density) * 0.6  # 调整因子, 降低的幅度更大
            speed *= density_factor
            speed = max(0.0, speed)  # 保证速度不为负

        # 4.3  模拟南湖公园特殊区域（假设某些区域人更多,速度更慢,需要根据实际坐标判断）
        x = df['X'][i]
        y = df['Y'][i]

        # 定义一些南湖公园内的区域范围
        plaza_coords = [(108.33, 22.80), (108.34, 22.81)] #广场范围例子
        lake_coords = [(108.32, 22.79), (108.35, 22.82)]  #湖边范围例子

        # 判断是否在广场/湖边，如果是，降低速度
        if plaza_coords[0][0] <= x <= plaza_coords[1][0] and plaza_coords[0][1] <= y <= plaza_coords[1][1]:
            speed *= 0.5 #假设广场区域速度降为一半
        if lake_coords[0][0] <= x <= lake_coords[1][0] and lake_coords[0][1] <= y <= plaza_coords[1][1]:
            speed *= 0.7 #假设湖边区域速度降为0.7

        walking_speeds.append(speed)

        # 4.4 根据步速，生成步长和步宽, 保证步宽不大于步长, 且步速大的地方步长也大
        # 将步速映射到步长范围
        normalized_speed = speed / max(speed_ranges['跑步'][0],max_speed) # 归一化步速，防止除以0

        # 步长与步速线性相关。
        step_length = step_length_range[0] + normalized_speed * (step_length_range[1] - step_length_range[0])
        step_length = max(0, min(step_length, step_length_range[1]))  # 保证步长在合理范围内

        step_width = random.uniform(0, min(step_width_range[1], step_length))  # 步宽不大于步长

        step_lengths.append(step_length)
        step_widths.append(step_width)

    # 5. 添加到DataFrame
    df['walking_speed'] = walking_speeds
    df['step_length'] = step_lengths
    df['step_width'] = step_widths

    return df


if __name__ == '__main__':
    # 示例用法
    csv_file = '南湖表/E.csv'  # 替换为你的 CSV 文件路径

    try:
        result_df = generate_walking_speeds_and_gait(csv_file)
        print(result_df)

        # 保存结果到新的CSV文件
        result_df.to_csv('南湖表/E_XYGait.csv', index=False)
        print("结果已保存到 locations_with_speeds_and_gait.csv")

    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}")
import pandas as pd

def analyze_conf_column(csv_file_path):
  """
  分析CSV文件中'Conf'列的平均值、中位数和标准差。

  Args:
    csv_file_path: CSV文件的路径。

  Returns:
    一个包含'Conf'列的平均值、中位数和标准差的字典。如果发生错误，返回 None。
  """
  try:
    # 1. 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 2. 检查'Conf'列是否存在
    if 'Conf' not in df.columns:
      print(f"错误：CSV文件中不存在名为'Conf'的列。")
      return None

    # 3. 尝试将'Conf'列转换为数值类型。这很重要，因为CSV文件中读取的数据可能是字符串。
    try:
      df['Conf'] = pd.to_numeric(df['Conf'])
    except ValueError:
      print(f"错误：'Conf'列包含无法转换为数值的数据。请检查数据类型。")
      return None


    # 4. 计算'Conf'列的平均值、中位数和标准差，忽略NaN值
    mean_conf = df['Conf'].mean()
    median_conf = df['Conf'].median()
    std_conf = df['Conf'].std()  # 计算标准差

    # 5. 返回平均值、中位数和标准差
    return {'mean': mean_conf, 'median': median_conf, 'std': std_conf}

  except FileNotFoundError:
    print(f"错误：文件 '{csv_file_path}' 未找到。")
    return None
  except pd.errors.EmptyDataError:
      print(f"错误：文件 '{csv_file_path}' 为空。")
      return None
  except Exception as e:
    print(f"发生未知错误：{e}")
    return None

def analyze_avg_keypoint_conf_column(csv_file_path):
  """
  分析CSV文件中'Avg_Keypoint_Conf'列的平均值、中位数和标准差。

  Args:
    csv_file_path: CSV文件的路径。

  Returns:
    一个包含'Avg_Keypoint_Conf'列的平均值、中位数和标准差的字典。如果发生错误，返回 None。
  """
  try:
    # 1. 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 2. 检查'Avg_Keypoint_Conf'列是否存在
    if 'Avg_Keypoint_Conf' not in df.columns:
      print(f"错误：CSV文件中不存在名为'Avg_Keypoint_Conf'的列。")
      return None

    # 3. 尝试将'Avg_Keypoint_Conf'列转换为数值类型。这很重要，因为CSV文件中读取的数据可能是字符串。
    try:
      df['Avg_Keypoint_Conf'] = pd.to_numeric(df['Avg_Keypoint_Conf'])
    except ValueError:
      print(f"错误：'Avg_Keypoint_Conf'列包含无法转换为数值的数据。请检查数据类型。")
      return None


    # 4. 计算'Avg_Keypoint_Conf'列的平均值、中位数和标准差，忽略NaN值
    mean_conf = df['Avg_Keypoint_Conf'].mean()
    median_conf = df['Avg_Keypoint_Conf'].median()
    std_conf = df['Avg_Keypoint_Conf'].std()

    # 5. 返回平均值、中位数和标准差
    return {'mean': mean_conf, 'median': median_conf, 'std': std_conf}

  except FileNotFoundError:
    print(f"错误：文件 '{csv_file_path}' 未找到。")
    return None
  except pd.errors.EmptyDataError:
      print(f"错误：文件 '{csv_file_path}' 为空。")
      return None
  except Exception as e:
    print(f"发生未知错误：{e}")
    return None

# 示例用法:
if __name__ == "__main__":
    file_path = './Output_YOLO11/2.23/尽头步道_10m_10°_2x/尽头步道_10m_10°_2x.csv'  # 假设 CSV 文件路径
    conf_analysis = analyze_conf_column(file_path)

    if conf_analysis is not None:
        print(f"平均值：{conf_analysis['mean']}")
        print(f"中位数：{conf_analysis['median']}")
        print(f"标准差：{conf_analysis['std']}")  # 添加这行以打印标准差

    conf_analysis = analyze_avg_keypoint_conf_column(file_path)

    if conf_analysis is not None:
        print(f"'Avg_Keypoint_Conf'列的平均值为：{conf_analysis['mean']}")
        print(f"'Avg_Keypoint_Conf'列的中位数为：{conf_analysis['median']}")
        print(f"'Avg_Keypoint_Conf'列的标准差为：{conf_analysis['std']}")
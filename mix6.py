import cv2
import supervision as sv
from ultralytics import YOLO 
import os
import numpy as np
from tqdm import tqdm
import csv

# 视频路径配置
video_dir1 = './202410/'
video_dir2 = '1.12/'
video_dir3 = 'DJI_0399/'
video_name = 'DJI_0399'
video_path = os.path.join(video_dir1, video_dir2, video_name+".mp4")

# 加载 YOLO 模型
device = 'cuda' 
detection_model = YOLO("Drone-YOLO.pt").to(device)
pose_model = YOLO("yolo11l-pose.pt").to(device)

# 初始化追踪器和展示工具
tracker = sv.ByteTrack()
smoother = sv.DetectionsSmoother(length=10)
edge_annotator = sv.EdgeAnnotator()
vertex_annotator = sv.VertexAnnotator()
corner_annotator = sv.BoxCornerAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator(trace_length=3000, color=sv.Color(r=255, g=0, b=0))

# 视频处理设置
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Error: Could not open video: {video_path}")

# 获取视频参数
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 传感器参数
sensor_width = 17.3  # 4/3 CMOS 传感器宽度，单位 mm
focal_length = 24    # 原焦距，单位 mm
zoom_factor = 14      # 放大倍数
flight_height = 50  # 飞行高度，单位 m

# 计算放大后的焦距
focal_length_zoomed = focal_length * zoom_factor

# 计算 GSD（地面采样距离，单位 米/像素）
GSD = (sensor_width * flight_height) / (frame_width * focal_length_zoomed )  # 米/像素

# 用来保存目标位置和时间戳的字典 (每个目标一个历史位置)
history_positions = {}

# 创建一个空白图像（白色背景），其大小与视频帧相同
blank_image = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255  # 白色背景

# 初始化视频处理的起始时间
start_time = 0

min_trajectory_length = 10

trajectories = {}

# 用来保存轨迹的最终图像
final_trajectory_image = blank_image.copy()

# 设置输出视频路径
output_dir = './Output_or/'
output_path = os.path.join(output_dir, video_dir2, video_dir3)
# 创建输出目录，如果不存在则创建
os.makedirs(output_path, exist_ok=True)
output_file = os.path.join(output_path, video_name +'.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# 目标检测
def detect_objects(frame):
    detection_results = detection_model(frame, iou=0.7, conf=0.7, classes=0, verbose=False, device=device)[0]
    detections = sv.Detections.from_ultralytics(detection_results)
    return detections

slicer = sv.InferenceSlicer(callback = detect_objects,
                            slice_wh = (2160, 2160),
                            overlap_ratio_wh = (0.2, 0.2),
                            iou_threshold = 0.7,
                            thread_workers = 1
                            )

# 姿态检测
def detect_pose(frame, x1, y1, x2, y2, expand_ratio=0.1):
    # 计算目标框的宽度和高度
    width = x2 - x1
    height = y2 - y1

    # 计算扩展后的新的边界框坐标
    x1_expanded = int(max(0, x1 - width * expand_ratio))
    y1_expanded = int(max(0, y1 - height * expand_ratio))
    x2_expanded = int(min(frame.shape[1], x2 + width * expand_ratio))
    y2_expanded = int(min(frame.shape[0], y2 + height * expand_ratio))

    # 裁剪需要进行姿态检测的区域
    person_crop = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
    pose_results = pose_model(person_crop, verbose=False, device=device)[0]

    # 提取关键点
    keypoints = sv.KeyPoints.from_ultralytics(pose_results)
    
    # 返回扩展后的坐标和关键点
    return keypoints, x1_expanded, y1_expanded

# 计算实际速度（单位：米/秒）
def calculate_average_speed(tracker_id, current_position, current_time):
    """
    计算目标的实际速度（米/秒），使用 GSD 转换像素速度。
    
    参数:
        tracker_id: 目标的唯一ID。
        current_position: 当前帧中目标的中心点 (x, y)。
        current_time: 当前帧的时间戳。
    
    返回:
        actual_speed: 实际速度，单位米/秒。
    """
    if tracker_id not in history_positions:
        history_positions[tracker_id] = {"positions": [], "timestamps": []}

    # 添加当前位置和时间戳
    history_positions[tracker_id]["positions"].append(current_position)
    history_positions[tracker_id]["timestamps"].append(current_time)

    # 确保只保留最近的若干个数据点以限制内存使用
    max_history = 20  # 可调整的历史点数
    if len(history_positions[tracker_id]["positions"]) > max_history:
        history_positions[tracker_id]["positions"].pop(0)
        history_positions[tracker_id]["timestamps"].pop(0)

    # 计算总距离和总时间
    positions = history_positions[tracker_id]["positions"]
    timestamps = history_positions[tracker_id]["timestamps"]
    total_distance = 0
    total_time = 0

    for i in range(1, len(positions)):
        prev_position = positions[i-1]
        curr_position = positions[i]
        prev_time = timestamps[i-1]
        curr_time = timestamps[i]

        # 计算像素距离
        pixel_distance = np.linalg.norm(np.array(curr_position) - np.array(prev_position))

        # 转换为实际距离 (米)
        actual_distance = pixel_distance * GSD
        total_distance += actual_distance
        total_time += (curr_time - prev_time)

    # 计算平均速度 (米/秒)
    if total_time > 0:
        actual_speed = total_distance / total_time
        return actual_speed
    else:
        return 0.0

# 提取目标区域的主要颜色并判断亮色暗色
def get_color_name(r, g, b):
    # 定义一些预定义的颜色及其RGB值
    colors = {
        "Red": (255, 0, 0),
        "Green": (0, 255, 0),
        "Blue": (0, 0, 255),
        "Yellow": (255, 255, 0),
        "Cyan": (0, 255, 255),
        "Magenta": (255, 0, 255),
        "Black": (0, 0, 0),
        "White": (255, 255, 255),
        "Gray": (128, 128, 128),
        "Orange": (255, 165, 0),
        "Pink": (255, 192, 203),
        "Brown": (165, 42, 42),
        "Purple": (128, 0, 128),
        "Violet": (238, 130, 238),
    }

    # 计算与预定义颜色的距离
    def color_distance(c1, c2):
        return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2 + (c1[2] - c2[2]) ** 2) ** 0.5

    closest_color = "Unknown"
    min_distance = float('inf')

    for color_name, color_value in colors.items():
        distance = color_distance((r, g, b), color_value)
        if distance < min_distance:
            min_distance = distance
            closest_color = color_name

    return closest_color

def get_dominant_color_and_brightness(frame, x1, y1, x2, y2):
    # 截取目标区域
    cropped_img = frame[y1:y2, x1:x2].astype(np.float32)  # 确保使用浮点数进行计算
    # 转换为HSV颜色空间
    hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    # 获取V通道（亮度）
    v_channel = hsv_img[:, :, 2]
    # 计算V通道的平均值
    avg_v = np.mean(v_channel)
    # 判断亮色暗色
    if avg_v > 127:  # V通道值大于127则为亮色
        brightness = 'Bright'
    else:
        brightness = 'Dark'
    
    # 使用K-means算法提取主要颜色
    pixels = hsv_img.reshape((-1, 3))  # 将图像转换为一维像素列表
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # 获取主色调
    dominant_color_hue = np.median(centers[:, 0])  # 主色调
    dominant_color_bgr = cv2.cvtColor(np.uint8([[[dominant_color_hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    
    # 将RGB值分解
    r_mean, g_mean, b_mean = dominant_color_bgr.astype(int)  # 确保转换为整型

    # 获取具体颜色名称
    color_name = get_color_name(r_mean, g_mean, b_mean)

    return r_mean, g_mean, b_mean, brightness, color_name

# 计算目标是否肥胖
def is_obese_by_whr(x1, y1, x2, y2, whr_threshold=0.5):
    """
    判断目标是否肥胖，根据腰高比(Waist-to-Height Ratio, WHR)。

    参数:
        x1, y1, x2, y2: 目标框的左上角和右下角坐标
        whr_threshold: 肥胖的WHR阈值
    返回:
        is_obese: 是否肥胖(True/False)
        whr: 腰高比的值
    """
    # 计算目标框的宽度和高度
    width = x2 - x1
    height = y2 - y1
    # 避免高度为零的情况
    if height == 0:
        return False, 0
    # 计算腰高比
    whr = (width / (height * 1.6)) * 2
    # 判断肥胖
    is_obese = whr > whr_threshold
    return is_obese, whr

# 进行年龄性别检测
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Load network
age_net = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
gender_net = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

def detect_age_gender(face_crop):
    # 预处理面部图像
    blob = cv2.dnn.blobFromImage(face_crop, 1, (227, 227), (104, 177, 123))
    
    # 性别预测
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[np.argmax(gender_preds)]
    
    # 年龄预测
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[np.argmax(age_preds)]
    
    return age, gender

# 定义CSV文件路径
csv_file_path = os.path.join(output_path, video_name + ".csv")

# 创建并初始化CSV文件
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # 写入表头，添加年龄和性别信息
    csv_writer.writerow(["Frame", "ID", "Speed (m/s)", "X", "Y", "H", "S", "V", "R", "G", "B", "WHR", "Conf", "color_name", "Age", "Gender"])

# 修改视频处理循环，加入CSV记录功能
with tqdm(total=frame_count, desc="Processing video frames", unit="frame") as pbar:
    frame_number = 0
    data_to_write = []  # 将数据存储在内存中，批量写入减少文件I/O
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = start_time + (frame_number / fps)
        #detections = slicer(frame)
        detections = detect_objects(frame)
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)
        #print(detections)
        labels = []

        if frame_number % 10 == 0:
            with open(csv_file_path, mode='a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)

                for class_id, tracker_id, xyxy, confidence in zip(detections.class_id, detections.tracker_id, detections.xyxy, detections.confidence):
                    x1, y1, x2, y2 = map(int, xyxy)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    is_obese, whr = is_obese_by_whr(x1, y1, x2, y2)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    current_position = (center_x, center_y)

                    average_speed = calculate_average_speed(tracker_id, current_position, current_time)
                    average_speed = round(average_speed, 3)
                    if average_speed == 0.0:
                        continue

                    # 只对目标框内的区域进行年龄和性别检测
                    cropped_face = frame[y1:y2, x1:x2]
                    if cropped_face.size > 0:
                        # 提取主颜色和亮度信息
                        r_mean, g_mean, b_mean, brightness, color_name = get_dominant_color_and_brightness(frame, x1, y1, x2, y2)

                        # 获取年龄和性别
                        age, gender = detect_age_gender(cropped_face)  # 传递裁剪的面部图像进行检测

                        # 提取HSV值
                        hsv_img = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2HSV)
                        h_mean = int(np.mean(hsv_img[:, :, 0]))
                        s_mean = int(np.mean(hsv_img[:, :, 1]))
                        v_mean = int(np.mean(hsv_img[:, :, 2]))
                    else:
                        r_mean, g_mean, b_mean = 0, 0, 0
                        h_mean, s_mean, v_mean = 0, 0, 0
                        brightness = 'Unknown'
                        color_name = 'Unknown'
                        age, gender = 'Unknown', 'Unknown'

                    confidence = round(confidence, 2)
                    data_to_write.append([
                        frame_number, tracker_id, average_speed, center_x, center_y, 
                        r_mean, g_mean, b_mean, h_mean, s_mean, v_mean, whr, confidence, color_name, age, gender
                    ])

        for class_id, tracker_id, xyxy, confidence in zip(detections.class_id, detections.tracker_id, detections.xyxy, detections.confidence):
            x1, y1, x2, y2 = map(int, xyxy)
            if x2 <= x1 or y2 <= y1:  # 如果框的宽度或高度为负数，则跳过
                continue

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            current_position = (center_x, center_y)

            if tracker_id not in trajectories:
                trajectories[tracker_id] = []
            trajectories[tracker_id].append(current_position)

            is_obese, whr = is_obese_by_whr(x1, y1, x2, y2)
            obesity_status = "Obese" if is_obese else "Normal"

            average_speed = calculate_average_speed(tracker_id, current_position, current_time)

            r_mean, g_mean, b_mean, brightness, color_name = get_dominant_color_and_brightness(frame, x1, y1, x2, y2)

            # 进行年龄和性别检测
            cropped_face = frame[y1:y2, x1:x2]
            if cropped_face.size > 0:
                age, gender = detect_age_gender(cropped_face)  # 传递裁剪的面部图像进行检测
            else:
                age, gender = 'Unknown', 'Unknown'  # 如果没有裁剪图像，则设置为未知

            whr_text = f"({obesity_status})"
            speed_text = f"Speed: {average_speed:.3f} m/s"
            color_text = f"({brightness})"
            confidence_text = f"Conf: {confidence:.2f}"

            keypoints, x1_expanded, y1_expanded = detect_pose(frame, x1, y1, x2, y2, expand_ratio=0.2)

            if keypoints is not None and hasattr(keypoints, 'xy') and keypoints.xy.size > 0:
                new_xy = np.zeros_like(keypoints.xy)

                offset_x = x1_expanded
                offset_y = y1_expanded

                for i in range(keypoints.xy.shape[1]):
                    current_keypoint = keypoints.xy[0, i]

                    if np.array_equal(current_keypoint, np.array([0, 0], dtype=current_keypoint.dtype)):
                        new_xy[0, i] = current_keypoint
                        continue

                    new_xy[0, i] = current_keypoint + np.array([offset_x, offset_y])

                keypoints.xy = new_xy
            #print(keypoints)
            frame = edge_annotator.annotate(frame.copy(), key_points=keypoints)
            frame = vertex_annotator.annotate(frame, key_points=keypoints)

            # 将年龄和性别添加到标签
            label = f"ID:{tracker_id} {whr_text} {color_text} {speed_text} {confidence_text} Age: {age} Gender: {gender}"
            labels.append(label)

        trajectory_image = trace_annotator.annotate(final_trajectory_image.copy(), detections=detections)

        for tracker_id, trajectory in list(trajectories.items()):
            if len(trajectory) < min_trajectory_length:
                del trajectories[tracker_id]
            else:
                for i in range(1, len(trajectory)):
                    cv2.line(final_trajectory_image, trajectory[i - 1], trajectory[i], (255, 0, 0), 2)

        final_trajectory_image = trajectory_image

        frame = corner_annotator.annotate(frame.copy(), detections=detections)
        frame = label_annotator.annotate(frame, detections=detections, labels=labels)
        frame = trace_annotator.annotate(frame, detections=detections)

        out.write(frame)
        frame_number += 1
        pbar.update(1)

# 在处理结束时写入 CSV
with open(csv_file_path, mode='a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(data_to_write)

trajectory_image_path = os.path.join(output_path, video_name + "_trajectory.png")
cv2.imwrite(trajectory_image_path, final_trajectory_image)

cap.release()
out.release()
print(f"Processing complete. Video saved to {output_file}")
print(f"Final trajectory image saved to {trajectory_image_path}")
print(f"CSV data saved to {csv_file_path}")
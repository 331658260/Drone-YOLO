import cv2
import supervision as sv
from ultralytics import YOLO
import os
from tqdm import tqdm

# 视频路径配置
video_dir1 = './202410/'
video_dir2 = '1.12/'
video_dir3 = 'DJI_0402/'
video_name = 'DJI_0402'
video_path = os.path.join(video_dir1, video_dir2, video_name+".mp4")

# 加载 YOLO 模型
device = 'cuda'
detection_model = YOLO("Drone-YOLO.pt").to(device)

# 初始化追踪器和展示工具
corner_annotator = sv.BoxCornerAnnotator()
label_annotator = sv.LabelAnnotator()

# 视频处理设置
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Error: Could not open video: {video_path}")

# 获取视频参数
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 计算需要处理的帧数（前3秒）
duration = 2  # 处理的秒数
max_frames_to_process = duration * fps

# 设置输出视频路径
output_dir = './Output_view/'
output_path = os.path.join(output_dir, video_dir2, video_dir3)
os.makedirs(output_path, exist_ok=True)
output_file = os.path.join(output_path, video_name + '_view.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# 目标检测
def detect_objects(frame):
    detection_results = detection_model(frame, classes=0, verbose=False, device=device)[0]
    detections = sv.Detections.from_ultralytics(detection_results)
    return detections

# 视频处理循环
slow_factor = 10  # 放慢倍数
window_size = (1080, 1080)  # 定义滑窗的大小
step_size = 640  # 窗口滑动的步幅（可以根据需要调整）

with tqdm(total=min(frame_count, max_frames_to_process), desc="Processing video frames", unit="frame") as pbar:
    processed_frames = 0  # 记录已处理的帧数

    while processed_frames < max_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 复制当前帧用于绘制滑窗和检测框
        current_frame = frame.copy()

        # 逐行滑动
        for current_y in range(0, frame_height, step_size):
            for current_x in range(0, frame_width, step_size):
                # 计算滑窗的结束坐标
                end_x = current_x + window_size[0]
                end_y = current_y + window_size[1]

                # 处理位于帧内部的滑窗
                if end_x <= frame_width and end_y <= frame_height:
                    # 定义当前滑窗
                    window = frame[current_y:end_y, current_x:end_x]
                    
                    # 进行目标检测
                    detections = detect_objects(window)

                    # 确保检测结果存在
                    if detections is not None and len(detections) > 0:
                        # 获取每个检测的标签和置信度
                        labels = [f"Class {int(cls)}: {conf:.2f}" for cls, conf in zip(detections.class_id, detections.confidence)]

                        # 在当前窗口的检测结果上绘制
                        annotated_window = corner_annotator.annotate(window.copy(), detections=detections)
                        annotated_window = label_annotator.annotate(annotated_window, detections=detections, labels=labels)

                        # 将标注结果粘贴回当前帧中
                        current_frame[current_y:end_y, current_x:end_x] = annotated_window

                    # 绘制当前滑窗的边界框
                    cv2.rectangle(current_frame, (current_x, current_y), (end_x, end_y), (255, 0, 0), 2)

                # 处理超出帧边缘的滑窗
                elif end_y > frame_height:  # 如果滑窗超出底部
                    # 只处理软重叠部分
                    if current_y + window_size[1] > frame_height:
                        window = frame[current_y:frame_height, current_x:end_x]
                    else:
                        continue  # 如果窗口不在视线范围内，则跳过
                    
                    # 进行目标检测
                    detections = detect_objects(window)

                    # 确保检测结果存在
                    if detections is not None and len(detections) > 0:
                        # 获取每个检测的标签和置信度
                        labels = [f"Class {int(cls)}: {conf:.2f}" for cls, conf in zip(detections.class_id, detections.confidence)]

                        # 在当前窗口的检测结果上绘制
                        annotated_window = corner_annotator.annotate(window.copy(), detections=detections)
                        annotated_window = label_annotator.annotate(annotated_window, detections=detections, labels=labels)

                        # 将标注结果粘贴回当前帧中
                        current_frame[current_y:frame_height, current_x:end_x] = annotated_window

                    # 绘制当前滑窗的边界框
                    cv2.rectangle(current_frame, (current_x, current_y), (end_x, frame_height), (255, 0, 0), 2)

                # 保存当前的帧，重复写入以放慢视频
                for _ in range(slow_factor):
                    out.write(current_frame)

        processed_frames += 1
        pbar.update(1)

# 清理
cap.release()
out.release()
print(f"Processing complete. Video saved to {output_file}")
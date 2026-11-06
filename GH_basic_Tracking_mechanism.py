# -*- coding: utf-8 -*-
"""
Original author: Raúl Vizcarra Chirinos (2024)
  https://github.com/rvizcarra15/IceHockey_ComputerVision_PyTorch
Adapted by Ryan Li: comments translated to Chinese, relative paths,
GPU (CUDA) loading, and a Puck class added to the team classifier.
"""
# ***********************************库引用***********************************#
from ultralytics import YOLO  # YOLO目标检测模型
import supervision as sv       # 计算机视觉监督工具
import pickle                 # 对象序列化存储
import os                     # 操作系统接口
import cv2                    # OpenCV计算机视觉库


# 输入-视频文件路径
video_path = 'video_input.mp4'
# 输出-处理后的视频文件路径
output_video_path = 'output_video.mp4'

# PICKLE文件路径（如果文件存在则加载，否则创建并保存新的追踪数据）
pickle_path = 'stubs/track_stubs.pkl'

# ******************************球员追踪机制类******************************#

class HockeyAnalyzer:
    def __init__(self, model_path):
        # 初始化YOLO目标检测模型
        self.model = YOLO(model_path) 
        # 初始化ByteTrack多目标追踪器
        self.tracker = sv.ByteTrack()

    # 批量检测视频帧中的目标
    def detect_frames(self, frames):
        batch_size = 20  # 每次处理的帧批量大小
        detections = []  # 存储检测结果的列表
        # 分批处理视频帧
        for i in range(0, len(frames), batch_size):
            # 使用YOLO模型进行目标检测（置信度阈值0.1）
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch
        return detections


# ****从文件加载追踪数据或执行目标检测-并保存PICKLE文件****#

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # 如果已存在追踪数据文件则直接加载
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # 检测视频帧中的目标
        detections = self.detect_frames(frames)

        # 初始化追踪数据结构（目前仅追踪球员）
        tracks = {"person": []}

        # 遍历每帧的检测结果
        for frame_num, detection in enumerate(detections):
            # 获取类别名称映射
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # 将检测结果转换为supervision格式
            detection_supervision = sv.Detections.from_ultralytics(detection)
            # 使用ByteTrack进行目标追踪
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            # 初始化当前帧的球员追踪数据
            tracks["person"].append({})

            # 处理追踪到的每个目标
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()  # 边界框坐标
                cls_id = frame_detection[3]         # 类别ID
                track_id = frame_detection[4]        # 追踪ID

                # 只处理球员类别（'person'）
                if cls_id == cls_names_inv.get('person', None):
                    # 存储球员的边界框信息
                    tracks["person"][frame_num][track_id] = {"bbox": bbox}

            # 存储当前帧的所有检测结果（可选）
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()  # 边界框坐标
                cls_id = frame_detection[3]         # 类别ID
                        
        # 保存追踪数据到pickle文件
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
# ******冰球运动员追踪标记设计（椭圆代替矩形框）******#
    
    # 在帧上绘制椭圆标记球员
    def draw_ellipse(self, frame, bbox, color, track_id=None, team=None):
        # 提取边界框的底部y坐标
        y2 = int(bbox[3])
        # 计算椭圆中心x坐标（边界框水平中心）
        x_center = (int(bbox[0]) + int(bbox[2])) // 2
        # 计算边界框宽度
        width = int(bbox[2]) - int(bbox[0])
        # 设置标记颜色（蓝色）
        color = (255, 0, 0)
        # 设置文本颜色（白色）
        text_color = (255, 255, 255)
        
        # 绘制椭圆标记
        cv2.ellipse(
            frame,
            center=(x_center, y2),                # 中心点（边界框底部中心）
            axes=(int(width) // 2, int(0.35 * width)),  # 长轴为宽度的50%，短轴为35%
            angle=0.0,                            # 旋转角度为0
            startAngle=-45,                       # 起始角度（-45度）
            endAngle=235,                         # 结束角度（235度）
            color=color,                          # 椭圆颜色
            thickness=2,                         # 线宽
            lineType=cv2.LINE_4                  # 线型
        )
    
        # 如果存在追踪ID，添加ID标签
        if track_id is not None:
            # 设置ID标签框尺寸
            rectangle_width = 40
            rectangle_height = 20
            # 计算标签框坐标
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = (y2 - rectangle_height // 2) + 15
            y2_rect = (y2 + rectangle_height // 2) + 15
    
            # 绘制ID背景矩形
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)  # 填充矩形
    
            # 设置ID文本位置（根据ID位数调整）
            x1_text = x1_rect + 12
            if track_id > 99:  # 三位数ID向左调整
                x1_text -= 10
                
            font_scale = 0.4  # 字体缩放比例
            # 在矩形上绘制追踪ID
            cv2.putText(
                frame,
                f"{track_id}",  # 显示的ID
                (int(x1_text), int(y1_rect + 15)),  # 文本位置
                cv2.FONT_HERSHEY_SIMPLEX,  # 字体类型
                font_scale,  # 字体大小
                text_color,  # 文本颜色（白色）
                thickness=2  # 文本线宽
            )

        return frame
    
# *********************** 球员标记和ID标注 *********************#

    # 在视频帧上绘制所有球员的标记
    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []  # 存储标注后的帧
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()  # 创建帧的副本以避免修改原数据
            # 获取当前帧的球员追踪数据
            player_dict = tracks["person"][frame_num]

            # 为每个球员绘制椭圆标记和ID
            for track_id, player in player_dict.items():
                bbox = player["bbox"]  # 球员边界框
                
                # 绘制椭圆标记和ID标签
                self.draw_ellipse(frame, bbox, (0, 255, 0), track_id)
            
            # 将标注后的帧添加到输出列表
            output_video_frames.append(frame)

        return output_video_frames
    
# **************** 执行追踪机制并生成输出视频 ****************#

# 读取视频所有帧到内存
video_frames = []
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    video_frames.append(frame)
cap.release()

# 初始化冰球分析器并执行追踪
tracker = HockeyAnalyzer('yolov8x.pt')
# 获取目标追踪数据（优先从pickle加载）
tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=pickle_path)
# 在视频帧上绘制所有标注
annotated_frames = tracker.draw_annotations(video_frames, tracks)

# **************** 保存处理后的视频文件 ********************#
# 设置视频编码器（MP4V）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# 获取视频尺寸（假设所有帧尺寸相同）
height, width, _ = annotated_frames[0].shape
# 创建视频写入对象
out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

# 逐帧写入处理后的视频
for frame in annotated_frames:
    out.write(frame)
# 释放视频写入资源
out.release()
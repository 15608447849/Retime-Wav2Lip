# -*- coding: utf-8 -*-
"""
Wav2Lip 推理代码详细解读
======================

这是一个基于深度学习的唇形同步（Lip-sync）推理脚本，用于将音频与视频中的人脸进行唇形同步。
主要功能是根据输入的音频文件，生成与音频内容匹配的唇形动作，并应用到目标视频或图像上。

核心技术栈：
- PyTorch: 深度学习框架
- OpenCV: 计算机视觉处理
- RetinaFace: 人脸检测
- GFPGAN: 超分辨率增强
- Mel频谱图: 音频特征提取

作者：李世平
日期：2025年
"""

# ============================================================================
# 1. 导入库部分 - 显示加载进度
# ============================================================================

print("\r正在加载 torch       ", end="")
import torch  # PyTorch深度学习框架，用于模型推理

print("\r正在加载 numpy       ", end="")
import numpy as np  # 数值计算库，用于数组操作

print("\r正在加载 Image       ", end="")
from PIL import Image  # Python图像处理库，用于图像格式转换

print("\r正在加载 argparse    ", end="")
import argparse  # 命令行参数解析库

print("\r正在加载 configparser", end="")
import configparser  # 配置文件解析库

print("\r正在加载 math        ", end="")
import math  # 数学计算库

print("\r正在加载 os          ", end="")
import os  # 操作系统接口库

print("\r正在加载 subprocess  ", end="")
import subprocess  # 子进程管理库，用于调用ffmpeg

print("\r正在加载 pickle      ", end="")
import pickle  # 序列化库，用于加载预训练模型

print("\r正在加载 cv2         ", end="")
import cv2  # OpenCV计算机视觉库

print("\r正在加载 audio       ", end="")
import audio  # 自定义音频处理模块

print("\r正在加载 RetinaFace ", end="")
from batch_face import RetinaFace  # RetinaFace人脸检测器

print("\r正在加载 re          ", end="")
import re  # 正则表达式库

print("\r正在加载 partial     ", end="")
from functools import partial  # 函数工具库

print("\r正在加载 tqdm        ", end="")
from tqdm import tqdm  # 进度条显示库

print("\r正在加载 warnings    ", end="")
import warnings  # 警告处理库

# 忽略torchvision的用户警告
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.transforms.functional_tensor"
)

print("\r正在加载 upscale     ", end="")
from enhance import upscale  # 图像超分辨率增强函数

print("\r正在加载 load_sr     ", end="")
from enhance import load_sr  # 超分辨率模型加载函数

print("\r正在加载 load_model  ", end="")
from easy_functions import load_model, g_colab  # 模型加载和Colab环境检测函数

print("\r库加载完成!     ")

# ============================================================================
# 2. 设备配置 - 自动检测可用的计算设备
# ============================================================================

# 设备优先级：CUDA GPU > Apple MPS > CPU
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
gpu_id = 0 if torch.cuda.is_available() else -1

if device == 'cpu':
    print('警告：未检测到GPU，将使用CPU进行推理，这会非常慢！')

# ============================================================================
# 3. 命令行参数配置 - 定义所有可配置的参数
# ============================================================================

parser = argparse.ArgumentParser(
    description="使用Wav2Lip模型进行野外视频唇形同步的推理代码"
)

# 必需参数
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="checkpoints/Wav2Lip.pth",
    help="要加载权重的已保存检查点名称",
    required=False,
)

parser.add_argument(
    "--segmentation_path",
    type=str,
    default="checkpoints/face_segmentation.pth",
    help="面部分割网络的已保存检查点名称",
    required=False,
)

parser.add_argument(
    "--face",
    type=str,
    default="res/my.jpg",
    help="包含要使用的人脸的视频/图像文件路径",
    required=False,
)

parser.add_argument(
    "--audio",
    type=str,
    default="res/A.wav",
    help="用作原始音频源的视频/音频文件路径",
    required=False,
)

# 输出配置
parser.add_argument(
    "--outfile",
    type=str,
    help="保存结果的视频路径",
    default="res/out.mp4",
)

# 处理模式配置
parser.add_argument(
    "--static",
    type=bool,
    help="如果为True，则仅使用第一个视频帧进行推理",
    default=False,
)

parser.add_argument(
    "--fps",
    type=float,
    help="仅在输入为静态图像时可指定(默认:25)",
    default=25.0,
    required=False,
)

# 图像处理参数
parser.add_argument(
    "--pads",
    nargs="+",
    type=int,
    default=[0, 10, 0, 0],
    help="填充（上、下、左、右）。请调整以至少包含下巴",
)

parser.add_argument(
    "--wav2lip_batch_size", 
    type=int, 
    help="Wav2Lip模型的批处理大小", 
    default=1
)

parser.add_argument(
    "--out_height",
    default=480,
    type=int,
    help="输出视频高度。在480或720时获得最佳结果",
)

parser.add_argument(
    "--crop",
    nargs="+",
    type=int,
    default=[0, -1, 0, -1],
    help="将视频裁剪到较小区域（上、下、左、右）。在resize_factor和rotate参数之后应用。"
    "如果存在多个面部则很有用。-1表示该值将根据高度、宽度自动推断",
)

parser.add_argument(
    "--box",
    nargs="+",
    type=int,
    default=[-1, -1, -1, -1],
    help="为面部指定恒定的边界框。仅在未检测到面部时作为最后手段使用。"
    "此外，仅在面部移动不多时才可能有效。语法：（上、下、左、右）。",
)

# 视频处理选项
parser.add_argument(
    "--rotate",
    default=False,
    action="store_true",
    help="有时从手机拍摄的视频可能翻转90度。如果为true，将视频向右翻转90度。"
    "如果尽管输入了正常视频但得到翻转结果，请使用此选项",
)

parser.add_argument(
    "--nosmooth",
    type=str,
    default=False,
    help="防止在短时间窗口内平滑面部检测",
)

# 高级处理选项
parser.add_argument(
    "--no_seg",
    default=False,
    action="store_true",
    help="防止使用面部分割",
)

parser.add_argument(
    "--no_sr", 
    default=False, 
    action="store_true", 
    help="防止使用超分辨率"
)

parser.add_argument(
    "--sr_model",
    type=str,
    default="gfpgan",
    help="上采样器名称 - gfpgan或RestoreFormer",
    required=False,
)

parser.add_argument(
    "--fullres",
    default=3,
    type=int,
    help="仅用于确定是否使用全分辨率，以便在使用时无需调整大小",
)

# 调试和预览选项
parser.add_argument(
    "--debug_mask",
    type=str,
    default=False,
    help="使背景变为灰度以更好地查看遮罩",
)

parser.add_argument(
    "--preview_window",
    type=str, 
    default="Full", 
    help="预览窗口设置: Full: 显示完整视频预览, Face: 仅显示面部区域预览, Both: 同时显示完整和面部预览"
)

parser.add_argument(
    "--preview_settings",
    type=str, 
    default=False,
    help="仅处理一帧"
)


# 遮罩处理参数
parser.add_argument(
    "--mouth_tracking",
    type=str,
    default=False,
    help="在每一帧中跟踪嘴部以生成遮罩",
)

parser.add_argument(
    "--mask_dilation",
    default=300,
    type=float,
    help="嘴部周围遮罩的大小",
    required=False,
)

parser.add_argument(
    "--mask_feathering",
    default=200,
    type=int,
    help="嘴部周围遮罩的羽化量",
    required=False,
)

# 质量设置
parser.add_argument(
    "--quality",
    type=str,
    help="在Fast、Improved和Enhanced之间选择",
    # default="Fast",
    default="Fast",
)

# ============================================================================
# 4. 预训练模型加载 - 加载面部关键点检测器和嘴部检测器
# ============================================================================

# 加载面部关键点预测器（68个关键点）
with open(os.path.join("checkpoints", "predictor.pkl"), "rb") as f:
    predictor = pickle.load(f)

# 加载嘴部检测器
with open(os.path.join("checkpoints", "mouth_detector.pkl"), "rb") as f:
    mouth_detector = pickle.load(f)

# ============================================================================
# 5. 全局变量初始化
# ============================================================================

# 创建变量以防止在未检测到面部时失败
kernel = last_mask = x = y = w = h = None




# 存储所有嘴部关键点的列表
all_mouth_landmarks = []

# 模型相关的全局变量
model = detector = detector_model = None

# ============================================================================
# 6. 核心函数定义
# ============================================================================

def do_load(checkpoint_path):
    """
    加载Wav2Lip模型和RetinaFace检测器
    
    参数:
        checkpoint_path (str): Wav2Lip模型检查点路径
    """
    global model, detector, detector_model
    print(f'do_load checkpoint_path = {checkpoint_path}')
    # 加载Wav2Lip主模型
    model = load_model(checkpoint_path)
    
    # 初始化RetinaFace人脸检测器
    detector = RetinaFace(
        gpu_id=gpu_id, 
        model_path="checkpoints/mobilenet.pth", 
        network="mobilenet"
    )
    detector_model = detector.model


def face_rect(images):
    """
    批量检测图像中的人脸矩形框
    
    参数:
        images (list): 图像列表
        
    生成器:
        tuple: 人脸边界框坐标 (x1, y1, x2, y2)
    """
    face_batch_size = 8  # 批处理大小
    num_batches = math.ceil(len(images) / face_batch_size)
    prev_ret = None
    
    for i in range(num_batches):
        # 获取当前批次的图像
        batch = images[i * face_batch_size : (i + 1) * face_batch_size]
        all_faces = detector(batch)  # 检测所有图像中的人脸
        
        for faces in all_faces:
            if faces:
                box, landmarks, score = faces[0]  # 取第一个检测到的人脸
                prev_ret = tuple(map(int, box))
            yield prev_ret


def create_tracked_mask(img, original_img):
    """
    创建跟踪式遮罩（每帧都重新检测嘴部）
    
    参数:
        img (numpy.ndarray): 当前处理的图像
        original_img (numpy.ndarray): 原始图像
        
    返回:
        tuple: (处理后的图像, 遮罩)
    """
    # print(f"create_tracked_mask img={img.shape}, original_img={original_img.shape}")
    global kernel, last_mask, x, y, w, h
    
    # 颜色空间转换：BGR -> RGB 创建副本，避免修改原始输入
    img_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB, img)
    original_rgb  = cv2.cvtColor(original_img.copy(), cv2.COLOR_BGR2RGB, original_img)
    
    # 检测人脸 应该用原始图像
    faces = mouth_detector(img)

    if len(faces) == 0:
        # 如果没有检测到人脸，使用上一次的遮罩
        if last_mask is not None:
            mask_resized = cv2.resize(last_mask, (img.shape[1], img.shape[0]))
            mask = Image.fromarray(mask_resized)
        else:
            # 没有遮罩，直接返回推理结果
            return img, None
    else:
        face = faces[0]
        shape = predictor(original_rgb, face)  # 获取68个面部关键点 在原图上获取关键点
        
        # 提取嘴部关键点（第48-67个点）
        mouth_points = np.array(
            [[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]
        )
        
        # 计算嘴部边界框
        x, y, w, h = cv2.boundingRect(mouth_points)
        # print(f"计算嘴部边界框 ({x}, {y}),{w}x{h}")

        # 根据边界框大小设置核大小
        kernel_size = int(max(w, h) * args.mask_dilation / 1000.0) # 修复这个性能问题
        kernel_size = max(3, min(kernel_size, 51))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # print(f" kernel_size = {kernel_size}")

        # 创建嘴部二值遮罩
        mask = np.zeros(original_rgb.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, mouth_points, 255)

        # 膨胀遮罩
        dilated_mask = cv2.dilate(mask, kernel)
    
        # 计算距离变换
        dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)
    
        # 归一化距离变换
        cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)
    
        # 转换为二值遮罩
        _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
        masked_diff = masked_diff.astype(np.uint8)
        # print(f" masked_diff = {masked_diff.shape}")

        # 应用高斯模糊进行羽化
        if not args.mask_feathering == 0:
            base_blur = args.mask_feathering
            blur_ratio = base_blur / 1000.0  # 将参数视为千分比
            blur = int(max(w, h) * blur_ratio)
            blur = max(3, min(blur, 101))  # 限制在合理范围
            if blur % 2 == 0:
                blur += 1

            # print(f"GaussianBlur START blur={blur}")
            masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)
            # print("GaussianBlur END ")
        mask = Image.fromarray(masked_diff)
        last_mask = mask  # 更新最后一次的遮罩

    # 转换为PIL图像进行混合
    input1 = Image.fromarray(img_rgb) # 推理结果
    input2 = Image.fromarray(original_rgb) # 原图

    # 确保图像尺寸一致
    if input1.size != input2.size:
        input1 = input1.resize(input2.size)
    if mask.size != input2.size:
        mask = mask.resize(input2.size)
    
    # 使用遮罩将处理后的嘴部粘贴到原图上
    result = input2.copy()
    result.paste(input1, (0, 0), mask)

    # 转换回numpy数组和BGR格式
    result_array = np.array(result)
    result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)
    
    return result_bgr, mask


def create_mask(img, original_img):
    """
    创建静态遮罩（使用缓存的遮罩）
    
    参数:
        img (numpy.ndarray): 当前处理的图像
        original_img (numpy.ndarray): 原始图像
        
    返回:
        tuple: (处理后的图像, 遮罩)
    """
    # print(f"create_mask {img.shape}, {original_img.shape}")
    global kernel, last_mask, x, y, w, h

    # 颜色空间转换：BGR -> RGB 创建副本，避免修改原始输入
    img_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB, img)
    original_rgb = cv2.cvtColor(original_img.copy(), cv2.COLOR_BGR2RGB, original_img)

    if last_mask is not None:
        # 使用缓存的遮罩
        # 调整遮罩尺寸
        mask_resized = cv2.resize(np.array(last_mask), (img.shape[1], img.shape[0]))
        mask = Image.fromarray(mask_resized)

    else:
        # 重新创建遮罩 原图上检测人脸
        faces = mouth_detector(img)

        if len(faces) == 0:
            # 没有检测到人脸
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
            return img, None
        else:
            face = faces[0]
            shape = predictor(img, face)
            
            # 获取嘴部关键点 第48-67个点
            mouth_points = np.array(
                [[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]
            )
            
            # 计算边界框
            x, y, w, h = cv2.boundingRect(mouth_points)
            
            # 设置核大小
            kernel_size = int(max(w, h) * args.mask_dilation / 1000)  # 也修复这个
            kernel_size = max(3, min(kernel_size, 51))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            
            # 创建二值遮罩
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, mouth_points, 255)
            
            # 膨胀遮罩
            dilated_mask = cv2.dilate(mask, kernel)
            
            # 距离变换
            dist_transform = cv2.distanceTransform(dilated_mask, cv2.DIST_L2, 5)
            # 归一化
            cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)
            
            # 阈值处理 转换为二值遮罩
            _, masked_diff = cv2.threshold(dist_transform, 50, 255, cv2.THRESH_BINARY)
            masked_diff = masked_diff.astype(np.uint8)
            
            # 应用高斯模糊进行羽化
            if not args.mask_feathering == 0:
                base_blur = args.mask_feathering
                blur_ratio = base_blur / 1000.0  # 将参数视为千分比
                blur = int(max(w, h) * blur_ratio)
                blur = max(3, min(blur, 101))  # 限制在合理范围
                if blur % 2 == 0:
                    blur += 1
                masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)
            
            mask = Image.fromarray(masked_diff)
            last_mask = mask  # 缓存遮罩
    
    # 图像混合处理
    input1 = Image.fromarray(img_rgb) # 推理结果
    input2 = Image.fromarray(original_rgb) # 原图


    # 确保尺寸一致
    if input1.size != input2.size:
        input1 = input1.resize(input2.size)
    if mask.size != input2.size:
        mask = mask.resize(input2.size)

        # 创建混合结果：在原图基础上，用遮罩混合推理结果
    result = input2.copy()
    result.paste(input1, (0, 0), mask)

    # 转换回numpy数组和BGR格式
    result_array = np.array(result)
    result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)

    return result_bgr, mask


def get_smoothened_boxes(boxes, T):
    """
    平滑人脸检测框以减少抖动
    
    参数:
        boxes (list): 边界框列表
        T (int): 时间窗口大小
        
    返回:
        list: 平滑后的边界框列表
    """
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T :]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images):
    """
    检测所有图像中的人脸并返回裁剪后的人脸区域
    返回:
        list: 包含人脸图像和坐标的列表
    """

    
    results = []
    pady1, pady2, padx1, padx2 = args.pads
    
    tqdm_partial = partial(tqdm, position=0, leave=True)
    for image, (rect) in tqdm_partial(
        zip(images, face_rect(images)),
        total=len(images),
        desc="检测每一帧中的人脸",
        ncols=100,
    ):
        if rect is None:
            cv2.imwrite("temp/faulty_frame.jpg", image)
            raise ValueError(
                "未检测到人脸！确保视频在所有帧中都包含人脸。"
            )
        
        # 应用填充
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])
    
    boxes = np.array(results)
    
    # 平滑处理
    if str(args.nosmooth) == "False":
        boxes = get_smoothened_boxes(boxes, T=5)
    
    # 裁剪人脸区域
    results = [
        [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
        for image, (x1, y1, x2, y2) in zip(images, boxes)
    ]

    return results


def datagen(frames, mels):
    """
    数据生成器，为模型准备批量数据
    
    参数:
        frames (list): 视频帧列表
        mels (list): Mel频谱图列表
        
    生成器:
        tuple: (图像批次, Mel批次, 帧批次, 坐标批次)
    """
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    # print("\r" + " " * 100, end="\r")
    
    # print(f"datagen args.box[0] = {args.box[0]}")
    # 人脸检测
    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames)
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print("使用指定的边界框而不是人脸检测...")
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
    
    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()
         
        # 调整人脸大小到模型输入尺寸
        face = cv2.resize(face, (args.img_size, args.img_size))
        
        img_batch.append(face) # 人脸图片
        mel_batch.append(m) # 音频mel
        frame_batch.append(frame_to_save) # 原图
        coords_batch.append(coords) # 人脸坐标
        
        # 当批次大小达到要求时，生成数据
        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
            
            # 创建遮罩图像（下半部分置零）
            img_masked = img_batch.copy()
            img_masked[:, args.img_size // 2 :] = 0
            
            # 拼接遮罩图像和原图像
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
            )
            
            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    
    # 处理剩余的数据
    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)
        
        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2 :] = 0
        
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
        mel_batch = np.reshape(
            mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
        )
        
        yield img_batch, mel_batch, frame_batch, coords_batch


# ============================================================================
# 7. 常量定义
# ============================================================================

mel_step_size = 16  # Mel频谱图步长


def _load(checkpoint_path):
    """
    加载模型检查点
    
    参数:
        checkpoint_path (str): 检查点文件路径
        
    返回:
        dict: 模型检查点
    """
    if device != "cpu":
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage
        )
    return checkpoint


# ============================================================================
# 8. 主函数 - 核心推理流程
# ============================================================================

def main():
    """
    主推理函数，执行完整的唇形同步流程
    """
    args.img_size = 96  # 模型输入图像尺寸
    frame_number = 11   # 帧编号（未使用）
    
    # 检查输入是否为静态图像
    if os.path.isfile(args.face) and args.face.split(".")[1] in ["jpg", "png", "jpeg"]:
        args.static = True
    
    # 验证输入文件
    if not os.path.isfile(args.face):
        raise ValueError("--face参数必须是有效的视频/图像文件路径")
    
    # 处理静态图像输入
    elif args.face.split(".")[1] in ["jpg", "png", "jpeg"]:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    else:
        # 处理视频输入
        if args.fullres != 1:
            print("调整视频大小...")
        
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        
        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            
            # 调整分辨率
            if args.fullres != 1:
                aspect_ratio = frame.shape[1] / frame.shape[0]
                frame = cv2.resize(
                    frame, (int(args.out_height * aspect_ratio), args.out_height)
                )
            
            # 旋转处理
            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
            
            # 裁剪处理
            y1, y2, x1, x2 = args.crop
            if x2 == -1:
                x2 = frame.shape[1]
            if y2 == -1:
                y2 = frame.shape[0]
            
            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)
    
    # 音频预处理
    if not args.audio.endswith(".wav"):
        print("将音频转换为.wav格式")
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error", 
            "-i", args.audio, "temp/temp.wav",
        ])
        args.audio = "temp/temp.wav"
    
    print("分析音频...")
    wav = audio.load_wav(args.audio, 16000)  # 加载音频，采样率16kHz
    mel = audio.melspectrogram(wav)          # 提取Mel频谱图
    
    # 检查Mel频谱图是否包含NaN值
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError(
            "Mel频谱图包含NaN值！使用TTS语音？请在wav文件中添加小的epsilon噪声后重试"
        )
    
    # 将Mel频谱图分割成块
    mel_chunks = []
    mel_idx_multiplier = 80.0 / fps  # Mel频谱图与视频帧的对应关系
    print(f"fps={fps}   mel_idx_multiplier={ mel_idx_multiplier}")
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    
    # 确保帧数与Mel块数匹配
    full_frames = full_frames[: len(mel_chunks)]

    print(f"len(mel_chunks)={len(mel_chunks)}   full_frames={ len(mel_chunks)}")

    # 预览模式处理
    if str(args.preview_settings) == "True":
        full_frames = [full_frames[0]]
        mel_chunks = [mel_chunks[0]]
    
    print(str(len(full_frames)) + " 帧需要处理")
    batch_size = args.wav2lip_batch_size
    print(f" batch_size={batch_size}")

    # 创建数据生成器
    if str(args.preview_settings) == "True":
        gen = datagen(full_frames, mel_chunks)
    else:
        gen = datagen(full_frames.copy(), mel_chunks)
    
    # 主推理循环
    for i, (img_batch, mel_batch, frames, coords) in enumerate(
        tqdm(
            gen,
            total=int(np.ceil(float(len(mel_chunks)) / batch_size)),
            desc="处理Wav2Lip",
            ncols=100,
        )
    ):
        # 初始化设置（仅在第一次迭代时执行）
        if i == 0:
            # if not args.quality == "Fast":
            #     print(f"遮罩大小: {args.mask_dilation}, 羽化: {args.mask_feathering}")
            #     if not args.quality == "Improved":
            #         print("加载", args.sr_model)
            #         run_params = load_sr()  # 加载超分辨率模型

            print(f"遮罩大小: {args.mask_dilation}, 羽化: {args.mask_feathering}")
            print("加载", args.sr_model)
            run_params = load_sr()  # 加载超分辨率模型
            
            print("开始处理...")
            frame_h, frame_w = full_frames[0].shape[:-1]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter("temp/result.mp4", fourcc, fps, (frame_w, frame_h))
        
        # 准备模型输入
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        
        # 模型推理
        with torch.no_grad():
            pred = model(mel_batch, img_batch)
        
        # 后处理预测结果
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        
        # 处理每个预测结果
        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            
            # 调试模式：将背景转为灰度
            if str(args.debug_mask) == "True":
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
            
            # 调整预测结果尺寸
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            cf = f[y1:y2, x1:x2]

            # 超分辨率增强
            if args.quality == "Enhanced":
                p = upscale(p, run_params)

            # 应用遮罩混合
            if args.quality in ["Enhanced", "Improved"]:
                if str(args.mouth_tracking) == "True":
                    p, last_mask = create_tracked_mask(p, cf)
                else:
                    p, last_mask = create_mask(p, cf)

            # args.mask_dilation = 30000
            # p = upscale(p, run_params)
            # p, last_mask = create_mask(p, cf)


            # cv2.imshow("preview p", p)
            # cv2.imshow("preview last_mask", last_mask)

            # 将处理后的嘴部区域放回原图
            f[y1:y2, x1:x2] = p

            if args.preview_window == "Face":
                cv2.imshow("preview Face", p)
            elif args.preview_window == "Full":
                cv2.imshow("preview Full", f)
            elif args.preview_window == "Both":
                cv2.imshow("preview Face", p)
                cv2.imshow("preview Full", f)
                
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                exit()
            
            # 预览模式处理
            if str(args.preview_settings) == "True":
                cv2.imwrite("temp/preview.jpg", f)
                
                cv2.imshow("预览 - 按Q键关闭", f)
                if cv2.waitKey(-1) & 0xFF == ord('q'):
                    exit()
            else:
                out.write(f)  # 写入视频文件
    
    # 清理资源
    cv2.destroyAllWindows()
    out.release()
    
    # 最终视频合成（非预览模式）
    if str(args.preview_settings) == "False":
        print("转换为最终视频")
        subprocess.check_call([
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", "temp/result.mp4",
            "-i", args.audio,
            "-c:v", "libx264",
            args.outfile
        ])


# ============================================================================
# 9. 程序入口点
# ============================================================================

if __name__ == "__main__":
    args = parser.parse_args()  # 解析命令行参数
    do_load(args.checkpoint_path)  # 加载模型
    main()  # 执行主函数


# ============================================================================
# 10. 相关模型功能说明
# ============================================================================

"""
相关模型功能作用和使用方式：
python inference_cn.py --checkpoint_path "checkpoints/Wav2Lip.pth"  --quality "Enhanced" --face "res/my.jpg" --audio "res/A.wav"  --outfile "output/result.mp4"  --static True

1. Wav2Lip 主模型 (wav2lip.py)
   - 功能：核心的唇形同步模型，基于音频Mel频谱图生成对应的嘴部动作
   - 输入：Mel频谱图 + 人脸图像（下半部分遮罩）
   - 输出：同步后的嘴部区域图像
   - 使用：通过load_model()函数加载，在推理时调用model(mel_batch, img_batch)

2. RetinaFace 人脸检测器
   - 功能：检测图像中的人脸位置和关键点
   - 模型文件：checkpoints/mobilenet.pth
   - 输出：人脸边界框、68个关键点、置信度分数
   - 使用：detector = RetinaFace(...), faces = detector(images)

3. 面部关键点预测器 (predictor.pkl)
   - 功能：基于dlib的68点面部关键点检测
   - 输入：人脸区域图像
   - 输出：68个面部关键点坐标
   - 使用：shape = predictor(image, face_rect)

4. 嘴部检测器 (mouth_detector.pkl)
   - 功能：专门用于检测嘴部区域的检测器
   - 输入：图像
   - 输出：嘴部边界框
   - 使用：faces = mouth_detector(image)

5. GFPGAN 超分辨率模型
   - 功能：提升生成图像的质量和分辨率
   - 模型文件：checkpoints/GFPGANv1.4.pth
   - 使用：通过enhance.py模块的upscale()函数调用

6. 音频处理模块 (audio.py)
   - 功能：音频预处理，提取Mel频谱图特征
   - 主要函数：
     - load_wav(): 加载音频文件
     - melspectrogram(): 提取Mel频谱图
   - 参数：采样率16kHz，Mel频谱图维度80

使用流程：
1. 加载所有预训练模型
2. 预处理输入视频和音频
3. 提取音频的Mel频谱图特征
4. 检测视频中每帧的人脸位置
5. 将人脸图像和Mel特征输入Wav2Lip模型
6. 对生成的嘴部区域进行后处理（遮罩混合、超分辨率）
7. 将处理后的嘴部区域合成回原视频
8. 输出最终的唇形同步视频

质量模式说明：
- Fast: 快速模式，不使用遮罩混合和超分辨率
- Improved: 改进模式，使用遮罩混合但不使用超分辨率
- Enhanced: 增强模式，同时使用遮罩混合和超分辨率
"""
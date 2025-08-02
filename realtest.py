import sys
import torch
import os
import cv2
import numpy as np
import pyaudio
from PIL import Image
import wave
import audio
import logging
# RetinaFace人脸检测器
from batch_face import RetinaFace
# 序列化库，用于加载预训练模型
import pickle
from models import Wav2Lip
from gfpgan import GFPGANer

import mediapipe as mp

# 参数详细说明
# 1. static_image_mode=False
# False（视频模式）：
#
# 针对视频流优化，利用时间连续性
# 使用跟踪算法，减少重复检测
# 性能更好，适合实时应用
# 会在跟踪失败时重新检测
# True（静态模式）：
#
# 每帧都进行完整的人脸检测
# 不使用跟踪，每张图像独立处理
# 精度更高但速度较慢
# 适合处理单张图片
# 2. max_num_faces=1
# 设置最多检测的人脸数量
# 1：只检测一张人脸（最大的或最清晰的）
# 2-10：可检测多张人脸
# 数量越多，计算开销越大
# 3. refine_landmarks=True
# True：启用精细化关键点检测
#
# 返回468个高精度关键点
# 包含眼部虹膜、嘴唇内外轮廓等细节
# 计算量较大但精度更高
# False：基础关键点检测
#
# 返回468个基础关键点
# 精度稍低但速度更快
# 4. min_detection_confidence=0.5
# 人脸检测的最小置信度阈值（0.0-1.0）
# 0.5：中等严格程度
# 较低值（0.1-0.4）：更容易检测到人脸，但可能有误检
# 较高值（0.6-0.9）：检测更严格，减少误检但可能漏检

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)
# 嘴部关键点索引（MediaPipe 468点模型）
MOUTH_LANDMARKS = [
    61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
    402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267,
    269, 270, 267, 271, 272, 271, 272
]
def detect_mouth_mediapipe(image):
    results = face_mesh.process(image)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 提取嘴部关键点
            mouth_points = []
            for idx in MOUTH_LANDMARKS:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                mouth_points.append((x, y))

            # 计算嘴部边界框
            mouth_points = np.array(mouth_points)
            return mouth_points
    return None



# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 加载模型
device = 'cuda'
gpu_id = 0


# 嘴唇关键点预测器（68个关键点）
# 点 0-16: 下巴轮廓
# 点 17-21: 右眉毛
# 点 22-26: 左眉毛
# 点 27-35: 鼻子
# 点 36-41: 右眼
# 点 42-47: 左眼
# 点 48-67: 嘴巴
# 嘴部外轮廓: 48-59 (12个点)
# 嘴部内轮廓: 60-67 (8个点)
with open(os.path.join(project_root,"checkpoints", "predictor.pkl"), "rb") as f:
    predictor = pickle.load(f)

# 嘴唇检测器
with open(os.path.join(project_root,"checkpoints", "mouth_detector.pkl"), "rb") as f:
    mouth_detector = pickle.load(f)

# 人脸检测器
detector = RetinaFace(
    gpu_id=gpu_id,
    model_path=os.path.join(project_root, "checkpoints", "mobilenet.pth"),
    network="mobilenet"
)


run_params = GFPGANer(
        model_path=os.path.join(project_root, "checkpoints", "GFPGANv1.4.pth"),
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )

# Wav2Lip主模型
with open(os.path.join(project_root,"checkpoints", "wav2lip_GAN.pk1"), "rb") as f:
# with open(os.path.join(project_root,"checkpoints", "nwav2lip_gan.pth"), "rb") as f:
    model = pickle.load(f)

# model = Wav2Lip()
# checkpoint = torch.load(os.path.join(project_root, "checkpoints", "Wav2Lip_GAN.pth"))
# s = checkpoint["state_dict"]
# new_s = {}
# for k, v in s.items():
#     new_s[k.replace("module.", "")] = v
# model.load_state_dict(new_s)
# model = model.to(device)
# model.eval() # 评估模式


mel_step_size = 16  # Mel频谱图步长
img_size = 96  #  Wav2Lip 模型输入图像尺寸 -训练时使用的是 96x96

mask_dilation = 200  # 嘴部周围遮罩的大小
mask_feathering = 400 #嘴部周围遮罩的羽化量

audio_player = pyaudio.PyAudio()
audio_stream = audio_player.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=16000,
                output=True,
                frames_per_buffer=1024
            )


#########################
def show_img_rect(title,img,x,y,w,h, color=(0, 255, 255)):
    display_img = img.copy()
    cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 2)
    cv2.imshow(title, cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def detector_mouth_point_v2(original_img):
    original_rgb = cv2.cvtColor(original_img.copy(), cv2.COLOR_BGR2RGB)
    return detect_mouth_mediapipe(original_rgb)


def detector_mouth_point(original_img):
    '''检测嘴部'''
    original_rgb = cv2.cvtColor(original_img.copy(), cv2.COLOR_BGR2RGB)
    faces = mouth_detector(original_rgb)
    if len(faces) == 0:
        # 没有检测到嘴部
        logging.info(f"没有检测到嘴部")
        return None
    face = faces[0]
    shape = predictor(original_rgb, face)  # 获取68个面部关键点
    # 提取嘴部关键点（第48-67个点）
    mouth_points = np.array(
        [[shape.part(i).x, shape.part(i).y] for i in range(48, 68)]
    )
    return mouth_points

def create_tracked_mask(img, original_img):
    """
    创建跟踪式遮罩（每帧都重新检测嘴部）

    参数:
        img (numpy.ndarray): 当前处理的图像
        original_img (numpy.ndarray): 原始图像

    返回:
        tuple: 处理后的图像
    """
    logging.info(f"create_tracked_mask img={img.shape}, original_img={original_img.shape}")

    # 颜色空间转换：BGR -> RGB
    img_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    original_rgb = cv2.cvtColor(original_img.copy(), cv2.COLOR_BGR2RGB)

    # 检测嘴部坐标
    mouth_points = detector_mouth_point_v2(original_rgb)
    # 计算嘴部边界框
    x, y, w, h = cv2.boundingRect(mouth_points)
    logging.info(f"计算嘴部边界框 x={x}, y={y}),w={w}, h={h}")

    # 根据边界框大小 设置核大小
    kernel_size = int(max(w, h) * mask_dilation / 1000)
    # kernel_size = max(3, min(kernel_size, 51))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    blur = int(max(w, h) * mask_feathering / 1000)
    # blur = max(3, min(blur, 101))
    if blur % 2 == 0:
        blur += 1

    logging.info(
        f' mask_dilation={mask_dilation} -> kernel_size={kernel_size}'
        f' mask_feathering={mask_feathering} -> blur={blur}')

    # 创建嘴部二值遮罩
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
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

    # 应用高斯模糊进行羽化
    masked_diff = cv2.GaussianBlur(masked_diff, (blur, blur), 0)
    mask = Image.fromarray(masked_diff)

    # 转换为PIL图像进行混合
    input1 = Image.fromarray(img_rgb) # 推理结果
    input2 = Image.fromarray(original_rgb) # 原图

    # 确保尺寸一致
    if input1.size != input2.size:
        input1 = input1.resize(input2.size)
    if mask.size != input2.size:
        mask = mask.resize(input2.size)


    # 创建混合结果：在原图基础上，用遮罩混合推理结果
    result = input2.copy() # 原图
    result.paste(input1, (0, 0), mask) # 推理结果+遮罩

    # 转换回numpy数组和BGR格式
    result_array = np.array(result)
    result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)

    return result_bgr

def upscale(image):
    _, _, output = run_params.enhance(
        image, has_aligned=False, only_center_face=False, paste_back=True
    )
    return output

def face_detect(image):
    '''获取人脸矩形大小'''
    frame_h, frame_w = image.shape[:-1]
    # logging.info(f'原图尺寸 {frame_w}x{frame_h}')

    faces = detector([image])
    # 取第一个检测到的人脸矩形区域
    box, landmarks, score = faces[0][0]
    # logging.info(f'人脸矩形区域 box={box} landmarks={landmarks} score={score}')
    face_box = tuple(map(int, box))

    x1, y1, x2, y2 = face_box

    # 裁剪人脸区域
    face_image = image[y1:y2, x1:x2]
    face_h, face_w = face_image.shape[:-1]
    # logging.info(f'人脸尺寸  {face_w}x{face_h}')

    return [face_image, (y1, y2, x1, x2)]

def model_pred(mel, frame, face, coord):
    # 调整人脸图片大小到模型输入尺寸96x96
    face = cv2.resize(face, (img_size, img_size))
    face_h, face_w = face.shape[:-1]
    # logging.info(f'调整后人脸尺寸 {face_w}x{face_h}')

    img_batch, mel_batch = np.asarray([face]), np.asarray([mel])
    # 创建遮罩图像（下半部分置零）
    img_masked = img_batch.copy()
    img_masked[:, img_size // 2:] = 0
    # 拼接遮罩图像和原图像
    img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
    mel_batch = np.reshape(
        mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
    )
    frames = [frame]
    coords = [coord]

    # 准备模型输入
    img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
    mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
    # 模型推理
    with torch.no_grad():
        pred = model(mel_batch, img_batch)

    # 后处理预测结果
    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
    p, f, c = pred[0], frames[0], coords[0]  # 预测结果 原图 嘴巴区域在原图的坐标
    y1, y2, x1, x2 = c

    # 调整预测结果尺寸
    logger.info(f"预测结果尺寸 {p.shape}")
    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
    logger.info(f"预测结果尺寸 调整后"f" {p.shape}")
    p = upscale(p)

    cf = f[y1:y2, x1:x2]

    # p = upscale(p)
    p = create_tracked_mask(p, cf)

    # 将处理后的嘴部区域放回原图
    f[y1:y2, x1:x2] = p

    logging.info("预测唇形同步完成")
    return cf,f

########################
def load_wav_generate(audio_file_path):
    with wave.open(audio_file_path, 'rb') as wav_file:
        logging.info(f"开始加载音频文件: {audio_file_path}")
        # 获取音频参数
        channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        frames = wav_file.getnframes()
        logging.info(f"音频参数: 声道={channels}, 位深={sampwidth * 8}bit, 采样率={framerate}Hz, 帧数={frames}")

        # 读取所有音频数据
        audio_bytes = wav_file.readframes(frames)
        audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # 分块处理音频数据
        chunk_duration = 0.2  # 每块x秒
        chunk_samples = int(chunk_duration * 16000)
        total_chunks = len(audio_data) // chunk_samples + (1 if len(audio_data) % chunk_samples > 0 else 0)
        logging.info(f"音频总时长: {len(audio_data) / 16000:.2f}秒, 分为{total_chunks}块")

        for i in range(0, len(audio_data), chunk_samples):
            chunk = audio_data[i:i + chunk_samples]
            if len(chunk) > 0:
                yield chunk

########################

image = cv2.imread('res/my.jpg')
audio_chunks = load_wav_generate('res/A.wav')

for index,chunk in enumerate(audio_chunks):
    mel = audio.melspectrogram(chunk)
    # logging.info(f"正在处理第{index+1}块音频 ")
    # logging.info(f"Mel 形状: {mel.shape}  范围: {mel.min():.6f} , {mel.max():.6f}")  # 期望: (80, time_frames)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        logging.info(f'Mel频谱图包含NaN值 , index={index+1} Mel 形状 {mel.shape} 范围 [{mel.min():.6f}')
        continue

    # 不足16 补充到16
    if mel.shape[1] < mel_step_size:
        # 填充到16帧
        padding_needed = mel_step_size - mel.shape[1]
        padded_mel = np.pad(mel, ((0, 0), (0, padding_needed)), mode='edge')
        mel = padded_mel[:, :mel_step_size]

    # 对原图进行复制
    frame = image.copy()
    # 人脸图片和坐标元组
    face, coord = face_detect(frame).copy()

    cf,f = model_pred(mel,frame,face,coord)

    # cv2.imshow("人脸预览 - 按Q键中止", p)
    cv2.imshow("f", f)

    audio_stream.write(chunk.tobytes())

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        exit()



################################################
audio_stream.stop_stream()
audio_stream.close()
audio_player.terminate()

cv2.destroyAllWindows()
del model
del detector.model










import logging
import queue
import time
import cv2
import os
from PIL import Image
import numpy as np
import torch
from batch_face import RetinaFace
import pickle
from gfpgan import GFPGANer
from joblib import Logger

import audio
from models import Wav2Lip
import mediapipe as mp

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

device = 'cuda'
gpu_id = 0

mel_step_size = 16  # Mel频谱图步长
img_size = 96  #  Wav2Lip 模型输入图像尺寸 -训练时使用的是 96x96
sample_rate = 16000


face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3
)
# 嘴部关键点索引（MediaPipe 468点模型）
MOUTH_LANDMARKS = [
    61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
    402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267,
    269, 270, 267, 271, 272, 271, 272
]

class LipSyncProcessor:
    """
    实时唇形同步处理器
    project_root 项目根目录
    mask_dilation 嘴部周围遮罩的大小
    mask_feathering 嘴部周围遮罩的羽化量
    """
    def __init__(self,project_root, mask_dilation=80, mask_feathering=300):

        self.mask_dilation = mask_dilation
        self.mask_feathering = mask_feathering

        # 嘴部检测器
        with open(os.path.join(project_root, "checkpoints", "mouth_detector.pkl"), "rb") as f:
            self.mouth_detector = pickle.load(f)
            logger.info(f'mouth_detector 嘴部检测器 加载成功')

        # 人脸检测器
        self.detector = RetinaFace(
            gpu_id=gpu_id,
            model_path=os.path.join(project_root, "checkpoints", "mobilenet.pth"),
            network="mobilenet"
        )
        logger.info(f'mobilenet 人脸检测 加载成功')

        # 面部关键点预测器（68个关键点）
        with open(os.path.join(project_root, "checkpoints", "predictor.pkl"), "rb") as f:
            self.predictor = pickle.load(f)
            logger.info(f'predictor 面部关键点预测器 加载成功')

        self.run_params = GFPGANer(
            model_path=os.path.join(project_root, "checkpoints", "GFPGANv1.4.pth"),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        logger.info(f'GFPGANer 超清晰度 加载成功')

        with open(os.path.join(project_root,"checkpoints", "wav2lip_GAN.pk1"), "rb") as f:
            self.model = pickle.load(f)
            logger.info(f'Wav2Lip 唇音同步 加载成功')

        # self.model = Wav2Lip()
        # checkpoint = torch.load(os.path.join(project_root, "checkpoints", "Wav2Lip_GAN.pth"))
        # s = checkpoint["state_dict"]
        # new_s = {}
        # for k, v in s.items():
        #     new_s[k.replace("module.", "")] = v
        # self.model.load_state_dict(new_s)
        # self.model = self.model.to(device)
        # self.model.eval()  # 评估模式
        # logger.info(f'Wav2Lip 加载成功')

        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_queue = queue.Queue()
        self.warmup_system(project_root)


    # def receive_audio(self, audio_chunk):
    #     '''音频添加到缓冲区'''
    #     self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
    # def get_audio_chunk(self):
    #     '''
    #     从缓冲区获取0.2秒 缓冲区不足0.2秒取所有
    #     '''
    #     min_samples = int(0.2 * sample_rate)
    #     if len(self.audio_buffer) >= min_samples:
    #         chunk = self.audio_buffer[:min_samples]
    #         # 从缓冲区移除已取的数据
    #         self.audio_buffer = self.audio_buffer[min_samples:]
    #     else:
    #         # logger.info(f"缓冲区不足0.2秒，取所有数据")
    #         # 缓冲区不足，取所有数据
    #         chunk = self.audio_buffer.copy()
    #         # 清空缓冲区
    #         self.audio_buffer = np.array([], dtype=np.float32)
    #     return chunk

    def receive_audio(self, audio_chunk):
        # logger.info(f"接收音频数据 {len(audio_chunk)}")
        self.audio_queue.put(audio_chunk)

    def get_audio_chunk(self):
        try:
            # chunk = self.audio_queue.get(timeout=1/30)
            chunk = self.audio_queue.get_nowait()
            if chunk is not None and len(chunk) > 0:
                return chunk
        except:
            return None


    def create_audio_mel(self):
        chunk = self.get_audio_chunk()

        if chunk is None or len(chunk) == 0:
            return None,None
        # 生成mel频谱图
        mel = audio.melspectrogram(chunk)
        # logging.info(f"Mel 形状: {mel.shape}  范围: {mel.min():.6f} , {mel.max():.6f}")  # 期望: (80, time_frames)
        
        # 检查mel频谱图是否包含NaN值
        if np.isnan(mel.reshape(-1)).sum() > 0:
            logger.warning(f'Mel频谱图包含NaN值 Mel 形状 {mel.shape} 范围: {mel.min():.6f} , {mel.max():.6f}')
            return None,None

        # 确保mel频谱图正好是16帧
        if mel.shape[1] < mel_step_size:
            logger.debug(f'Mel频谱图不足16帧，Mel 形状 {mel.shape}，进行填充')
            # 填充到16帧
            padding_needed = mel_step_size - mel.shape[1]
            padded_mel = np.pad(mel, ((0, 0), (0, padding_needed)), mode='edge')
            mel = padded_mel[:, :mel_step_size]
        # elif mel.shape[1] > mel_step_size:
        #     logger.debug(f'Mel频谱图超过16帧，Mel 形状 {mel.shape}，进行截取')
        #     # 截取前16帧
        #     mel = mel[:, :mel_step_size]

        return mel, chunk


    def synced_frame(self,image):
        '''一帧画面 尝试同步'''
        mel, chunk = self.create_audio_mel()
        if mel is None:

            return image, None
        try:

            f = self.model_pred(mel, image)
            if f is not None:
                return f,chunk
        except Exception as e:
            # import traceback
            # traceback.print_exc()
            logging.error(f"处理帧时发生错误: {e}")

        return image,chunk


    def detect_mouth_mediapipe(self,image):
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

    def create_tracked_mask(self,img, original_img):
        """
        创建跟踪式遮罩（每帧都重新检测嘴部）
        参数:
            img (numpy.ndarray): 当前处理的图像
            original_img (numpy.ndarray): 原始图像
        返回:
            tuple: 处理后的图像
        """
        # 颜色空间转换：BGR -> RGB
        img_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        original_rgb = cv2.cvtColor(original_img.copy(), cv2.COLOR_BGR2RGB)

        # 检测嘴唇
        # faces = self.mouth_detector(original_rgb)
        # if len(faces) == 0:
        #     # 没有检测到嘴部
        #     return img
        # face = faces[0]
        # logger.info(f"嘴部检测 face = {face} ") # face = [(18, 66) (233, 281)]
        # shape = self.predictor(original_rgb, face)  # 获取68个面部关键点
        # # 提取嘴部关键点（第48-67个点）
        # mouth_points = np.array(
        #     [ [shape.part(i).x, shape.part(i).y] for i in range(48, 68) ]
        # )
        mouth_points = self.detect_mouth_mediapipe(original_rgb)
        if mouth_points is None:
            # 没有检测到嘴部
            logger.info(f"没有检测到嘴部")
            return img
        # 计算嘴部边界框
        x, y, w, h = cv2.boundingRect(mouth_points)

        # 根据边界框大小 设置核大小
        kernel_size = int(max(w, h) * self.mask_dilation / 100)
        kernel_size = max(3, min(kernel_size, 51))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        blur = int(max(w, h) * self.mask_feathering / 100)
        blur = max(3, min(blur, 101))
        if blur % 2 == 0:
            blur += 1

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
        input1 = Image.fromarray(img_rgb)  # 推理结果
        input2 = Image.fromarray(original_rgb)  # 原图

        # 确保尺寸一致
        if input1.size != input2.size:
            input1 = input1.resize(input2.size)
        if mask.size != input2.size:
            mask = mask.resize(input2.size)

        # 创建混合结果：在原图基础上，用遮罩混合推理结果
        result = input2.copy()  # 原图
        result.paste(input1, (0, 0), mask)  # 推理结果+遮罩

        # 转换回numpy数组和BGR格式
        result_array = np.array(result)
        result_bgr = cv2.cvtColor(result_array, cv2.COLOR_RGB2BGR)

        return result_bgr


    def upscale(self,image):
        _, _, output =  self.run_params.enhance(
            image, has_aligned=False, only_center_face=False, paste_back=True
        )
        return output


    def face_detect(self,image):
        '''获取人脸矩形大小'''
        faces = self.detector([image])
        if not faces:
            logger.error(f"没有检测到人脸 original_rgb= {image.shape}")
            return None,None
        # 取第一个检测到的人脸矩形区域
        box, landmarks, score = faces[0][0]
        # logging.info(f'人脸矩形区域 box={box} landmarks={landmarks} score={score}')
        face_box = tuple(map(int, box))
        x1, y1, x2, y2 = face_box
        # x1, y1, x2, y2 = x1+96, y1+100, x2-96, y2+10
        # 裁剪人脸区域
        face_image = image[y1:y2, x1:x2]
        # logging.info(f'人脸矩形区域 face_image={face_image.shape}')
        return [face_image, (y1, y2, x1, x2)]

    def model_pred(self, mel, image):
        frame = image.copy()

        s1 = time.perf_counter()
        # 人脸图片,坐标元组
        face, coord = self.face_detect(frame) # .copy()
        s2 = time.perf_counter()
        # logging.info(f"人脸检测耗时: {s2 - s1:.6f}s")

        # cv2.imshow('face',face)
        # cv2.waitKey(0)


        if face is None:
            return frame

        # 调整人脸图片大小到模型输入尺寸96x96
        face = cv2.resize(face, (img_size, img_size))
        img_batch = np.asarray([face])
        mel_batch = np.asarray([mel])

        # 创建遮罩图像（下半部分置零）
        img_masked = img_batch.copy()
        img_masked[:, img_size // 2:] = 0

        # 拼接遮罩图像和原图像
        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0

        mel_batch = np.reshape(
            mel_batch,
            [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
        )

        # 准备模型输入
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
        s1 = time.perf_counter()
        # 模型推理
        with torch.no_grad():
            pred = self.model(mel_batch, img_batch)
        s2 = time.perf_counter()
        # logging.info(f"模型推理耗时: {s2 - s1:.6f}s")

        # 处理预测结果
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
        p, f, c = pred[0], frame, coord  # 预测结果 原图 嘴巴区域在原图的坐标

        y1, y2, x1, x2 = c

        # 调整预测结果尺寸
        p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

        # cv2.imshow('pred', p)
        # cv2.waitKey(0)

        # s1 = time.perf_counter()
        # p = self.upscale(p)
        # s2 = time.perf_counter()
        # logging.info(f"超清晰度耗时: {s2 - s1:.6f}s")

        cf = f[y1:y2, x1:x2]

        s1 = time.perf_counter()
        p = self.create_tracked_mask(p, cf)
        s2 = time.perf_counter()
        # logging.info(f"羽化遮罩耗时: {s2 - s1:.6f}s")

        # 将处理后的嘴部区域放回原图
        try:
            f[y1:y2, x1:x2] = p
        except:

            import traceback
            traceback.print_exc()
            logging.error(f"处理失败 face={face.shape} p={p.shape}")
            cv2.imshow('pred', p)
            cv2.waitKey(0)
            return image


        return f



    def cleanup(self):
           del self.model
           del self.detector.model

    def warmup_system(self,project_root):
        """系统预热：预热所有模型组件"""
        logger.info("=== 系统预热开始 ===")
        #使用随机噪声mel
        mel = np.random.normal(0, 0.1, (80, 16)).astype(np.float32)
        mel = np.clip(mel, -5.0, 5.0)
        image = cv2.imread(os.path.join(project_root, "res", "my.jpg"))  # 使用您的测试图片
        f = self.model_pred(mel,image)

        # cv2.imshow('prev load', f)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()














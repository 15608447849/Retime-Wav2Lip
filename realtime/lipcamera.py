import queue
import threading
import time
import logging
import os
import sys
import numpy as np
import pyaudio
import requests
import resampy
import cv2
from collections import deque

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

fps = 30
delay_seconds = 5

class FramePlayer:
    """画面捕获器 - 支持延迟播放的视频帧处理器"""

    def __init__(self, lip):
        self.lip = lip
        self.source = None
        self.frame_callback = None
        self.capture = None
        self.capture_thread = None

        # 初始化帧缓存系统
        self.frame_chunk_buffer = None

        # 重置延迟缓存
        self.init_delay_frame_chunk_buffer()

        self.play_thread = threading.Thread(target=self.play_frame, daemon=True)
        self.play_thread.start()

    def start(self, source, frame_callback):
        if self.capture_thread:
            return False
        self.source = source
        self.frame_callback = frame_callback
        try:
            self.capture = cv2.VideoCapture(self.source)
            if not self.capture.isOpened():
                raise Exception(f"无法打开视频源: {self.source}")

            if isinstance(self.source, int):
                # 摄像头参数
                width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = self.capture.get(cv2.CAP_PROP_FPS)
                logger.info(f"Camera initialized: {width}x{height} @ {fps:.1f}fps")

            self.capture_thread = threading.Thread(target=self.cap_frame, daemon=True)
            self.capture_thread.start()
            return True
        except Exception as e:
            logger.error(f"FramePlayer启动失败: {e}")
        return False

    def stop(self):
        if self.capture_thread:
            self.capture_thread = None
        if self.capture:
            self.capture.release()
            self.capture = None
        self.frame_callback = None


    def cap_frame(self):
        """视频捕获"""
        try:
            while self.capture:

                # 读取一帧图像数据
                ret, frame = self.capture.read()
                if not ret:
                    # 如果是视频文件，重新开始播放
                    if isinstance(self.source, str):
                        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        # 摄像头无数据，退出循环
                        break
                # frame = cv2.imread('res/my.jpg')
                if self.lip:
                    try:
                        frame, chunk = self.lip.synced_frame(frame)
                        self.add_frame_chunk_buffer(frame, chunk)
                        continue
                    except Exception as e:
                        logger.error(f"帧捕获处理 回调错误: {e}")
        except Exception as e:
            logger.error(f"FramePlayer帧捕获错误: {e}")

    def play_frame(self):
        '''视频播放'''
        try:
            while self.play_thread:
                try:
                    if self.frame_callback:
                        frame, chunk = self.get_frame_chunk_buffer()

                        if frame is not None:
                            self.frame_callback(frame, chunk)
                except Exception as e:
                    logger.error(f"播放延迟音视频错误: {e}")

                time.sleep(1 / fps)
            logger.info("播放延迟音视频线程结束")
        except Exception as e:
            logger.error(f"播放延迟音视频线程致命错误: {e}")

    def cleanup(self):
        self.stop()
        if self.play_thread:
            self.play_thread = None

    def init_delay_frame_chunk_buffer(self):
        '''重置缓存列表 添加空帧 用于延迟播放'''
        # 计算缓存大小：fps * 延迟秒数
        buffer_size = fps * delay_seconds  # 30 * 5 = 150帧

        # 使用deque创建缓冲区
        self.frame_chunk_buffer = deque()

        # 预填充空帧和空音频块，确保延迟效果
        for i in range(buffer_size):
            self.frame_chunk_buffer.append((None, None))

    def add_frame_chunk_buffer(self, frame, chunk):
        ''' 添加帧及音频缓存'''
        try:
            # 添加到缓存队列（deque会自动移除最老的帧）
            self.frame_chunk_buffer.append((frame if frame is not None else None,
                                            chunk if chunk is not None else None))
        except Exception as e:
            logger.error(f"添加帧到缓存时发生错误: {e}")

    def get_frame_chunk_buffer(self):
        ''' 返回帧及音频缓存'''
        try:
            # 检查缓存是否为空
            if len(self.frame_chunk_buffer) == 0:
                return None, None
            # 从队列左侧取出最老的帧（FIFO - 先进先出）
            frame, chunk = self.frame_chunk_buffer.popleft()
            return frame, chunk
        except Exception as e:
            logger.error(f"从缓存获取帧时发生错误: {e}")
            return None, None

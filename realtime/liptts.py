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

sample_rate=16000
chunk_size=1024


class TTSPlayer:
    """TTS音频播放器"""
    def __init__(self,lip):
        self.lip = lip
        self.audio = pyaudio.PyAudio()
        self.audio_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=sample_rate,
                output=True,
                frames_per_buffer=chunk_size
            )
        logger.info(f"音频流初始化成功: 采样率{sample_rate}Hz")

        self.text_queue = queue.Queue()
        self.text_thread = threading.Thread(target=self.loop_tts_text, daemon=True)
        self.text_thread.start()
    def loop_tts_text(self):
        while True:
            try:
                server_url,audio_path,text = self.text_queue.get()
                if text:
                    self.conv_tts_stream(server_url,audio_path,text)
            except Exception as e:
                logger.error(f"TTS文本处理错误: {e}")
                continue
    def conv_tts_stream(self, server_url: str, audio_path: str, text: str):
        try:
            request_data = {
                "text": text,
                "audio_paths": [audio_path] if audio_path else []
            }

            logger.info(f"连接TTS服务器: {server_url} text: {text}")

            response = requests.post(
                server_url,
                json=request_data,
                stream=True,
                headers={"content-type": "application/json"},
                timeout=30
            )

            response.raise_for_status()

            # 如果希望每500ms处理一次数据：
            target_latency = 0.3  # 秒
            bytes_per_sample = 4  # float32占4字节

            # 0.5 * 16000 * 4 = 32000 字节
            optimal_chunk_size = target_latency * sample_rate * bytes_per_sample
            logger.info(f"optimal_chunk_size: {optimal_chunk_size}")

            for chunk in response.iter_content(chunk_size=int(optimal_chunk_size)):
                # logger.info(f"tts chunk: {len(chunk)}")
                if chunk:
                    try:
                        audio_data = np.frombuffer(chunk, dtype=np.float32) / 32768.0
                        resampled_data = resampy.resample(audio_data, sr_orig=24000, sr_new=sample_rate)
                        if self.lip:
                            self.lip.receive_audio(resampled_data)
                    except Exception as e:
                        logger.error(f"处理音频数据错误: {e}")
                        continue
        except Exception as e:
            logger.error(f"获取音频流错误: {e}")

    def send_tts_text(self, server_url: str, audio_path: str, text: str):
        self.text_queue.put((server_url,audio_path,text))

    def cleanup(self):
        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except Exception as e:
                logger.error(f"关闭音频流错误: {e}")
            finally:
                self.audio_stream = None
        if self.audio:
            self.audio.terminate()








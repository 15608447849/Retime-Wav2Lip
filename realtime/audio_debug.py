#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time Audio Stream Test Tool for Easy-Wav2Lip Project
实时音频流测试工具 - 用于测试TTS实时音频流的接收和播放

=== 音频基础知识 Audio Fundamentals ===
1. 采样率 (Sample Rate): 每秒采样次数
   - 常见值：16kHz(语音)、44.1kHz(CD音质)、48kHz(专业音频)
   - 原理：将连续的模拟音频信号转换为离散的数字信号
   - 奈奎斯特定理：采样率至少是最高频率的2倍才能完整重建信号

2. 位深度 (Bit Depth): 每个采样点的数据位数
   - 常见值：16-bit(CD质量)、24-bit(专业录音)、32-bit(浮点)
   - 影响：位深度越高，动态范围越大，音质越好
   - 计算：16-bit可表示65536个不同的音量级别

3. 声道 (Channels): 音频通道数
   - 单声道(Mono): 1个通道，所有扬声器播放相同内容
   - 立体声(Stereo): 2个通道，左右声道可以不同
   - 多声道：5.1、7.1等环绕声系统

4. 音频格式:
   - PCM: 原始音频数据，未压缩，质量最高
   - WAV: 包含PCM数据的容器格式
   - MP3: 有损压缩格式，文件小但质量略降
   - 流式音频：边接收边播放，适合实时应用

5. 音频缓冲 (Audio Buffer):
   - 作用：平衡音频生产和消费速度差异
   - 缓冲区太小：可能出现音频断续(underrun)
   - 缓冲区太大：增加延迟，影响实时性
   - 最佳实践：根据网络条件和处理能力动态调整

6. 重采样 (Resampling):
   - 目的：转换不同采样率的音频数据
   - 上采样：增加采样率，通过插值生成新样本点
   - 下采样：降低采样率，通过滤波和抽取减少样本点
   - 质量vs性能：高质量重采样需要更多计算资源
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import requests
import threading
import time
import numpy as np
import pyaudio
import resampy
import queue
import json
import logging
from typing import Optional, Generator
from enum import Enum

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class State(Enum):
    """
    播放状态枚举
    
    状态说明：
    - IDLE: 空闲状态，未进行任何操作
    - CONNECTING: 正在连接TTS服务器
    - STREAMING: 正在接收和播放音频流
    - STOPPED: 用户主动停止播放
    - ERROR: 发生错误，播放中断
    """
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    STOPPED = "stopped"
    ERROR = "error"

class AudioStreamPlayer:
    """
    实时音频流播放器
    
    功能：
    1. 接收TTS服务返回的实时音频流
    2. 处理音频数据（重采样、格式转换）
    3. 实时播放音频
    4. 管理音频缓冲区
    5. 监控播放状态和性能
    """
    
    def __init__(self, sample_rate=16000, chunk_size=1024):
        """
        初始化音频播放器
        
        参数：
        - sample_rate: 目标采样率，默认16kHz（适合语音处理）
        - chunk_size: 音频块大小，影响延迟和稳定性
        """
        self.sample_rate = sample_rate  # 目标采样率
        self.chunk_size = chunk_size    # 音频块大小
        self.state = State.IDLE         # 当前播放状态
        
        # PyAudio相关对象
        self.audio = pyaudio.PyAudio()  # PyAudio实例
        self.stream = None              # 音频输出流
        
        # 线程和队列管理
        self.audio_queue = queue.Queue(maxsize=5000)  # 音频数据队列，限制大小防止内存溢出
        self.play_thread = None         # 播放线程
        self.fetch_thread = None        # 数据获取线程
        
        # 性能监控
        self.total_chunks = 0           # 总接收块数
        self.start_time = None          # 开始时间
        self.first_chunk_time = None    # 首个音频块时间
        
    def init_audio_stream(self):
        """
        初始化PyAudio输出流
        
        配置说明：
        - format: 音频格式，使用32位浮点数
        - channels: 单声道输出
        - rate: 采样率
        - output: 输出流（播放）
        - frames_per_buffer: 每次回调的帧数，影响延迟
        """
        try:
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,           # 32位浮点格式
                channels=1,                         # 单声道
                rate=self.sample_rate,              # 采样率
                output=True,                        # 输出流
                frames_per_buffer=self.chunk_size   # 缓冲区大小
            )
            logger.info(f"Audio stream initialized: {self.sample_rate}Hz, chunk_size={self.chunk_size}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize audio stream: {e}")
            return False
    
    def fetch_audio_stream(self, server_url: str, request_data: dict):
        """
        从TTS服务器获取音频流数据
        
        参数：
        - server_url: TTS服务器URL
        - request_data: 请求数据（包含文本和音频路径）
        
        工作流程：
        1. 发送POST请求到TTS服务器
        2. 以流式方式接收响应数据
        3. 处理音频数据（重采样、格式转换）
        4. 将处理后的数据放入播放队列
        """
        try:
            self.state = State.CONNECTING
            self.start_time = time.perf_counter()
            first_chunk = True
            
            logger.info(f"Connecting to TTS server: {server_url}")
            logger.info(f"Request data: {request_data}")
            
            # 发送流式POST请求
            response = requests.post(
                server_url,
                json=request_data,
                stream=True,                        # 启用流式接收
                headers={"content-type": "application/json"},
                timeout=30                          # 30秒超时
            )
            
            # 检查HTTP响应状态
            response.raise_for_status()
            self.state = State.STREAMING
            
            # 处理流式音频数据
            for chunk in response.iter_content(chunk_size=9600):  # 24000*20ms*2 = 960 bytes for 20ms at 24kHz
                # 检查播放状态
                if self.state != State.STREAMING:
                    break
                
                # 记录首个音频块的延迟
                if first_chunk:
                    self.first_chunk_time = time.perf_counter()
                    delay = self.first_chunk_time - self.start_time
                    logger.info(f"Time to first chunk: {delay:.3f}s")
                    first_chunk = False
                
                # 处理音频数据
                if chunk:
                    try:
                        # 将字节数据转换为numpy数组
                        # 假设接收的是24kHz 16-bit PCM数据
                        audio_data = np.frombuffer(chunk, dtype=np.float32) / 32768.0
                        
                        # 重采样从24kHz到目标采样率
                        if len(audio_data) > 0:
                            resampled_data = resampy.resample(
                                audio_data, 
                                sr_orig=24000, 
                                sr_new=self.sample_rate
                            )
                            
                            # 将处理后的数据放入队列
                            try:
                                self.audio_queue.put(resampled_data, timeout=0.1)
                                self.total_chunks += 1
                            except queue.Full:
                                logger.warning("Audio queue full, dropping chunk")
                    
                    except Exception as e:
                        logger.error(f"Error processing audio chunk: {e}")
                        continue
            
            logger.info(f"Audio stream fetch completed. Total chunks: {self.total_chunks}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            self.state = State.ERROR
        except Exception as e:
            logger.error(f"Error fetching audio stream: {e}")
            self.state = State.ERROR
        finally:
            # 发送结束信号
            try:
                self.audio_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
    
    def play_audio_stream(self):
        """
        音频播放线程函数
        
        工作流程：
        1. 从音频队列获取数据
        2. 写入PyAudio输出流
        3. 监控播放状态
        4. 处理播放完成或错误情况
        """
        try:
            logger.info("Audio playback thread started")
            
            while self.state in [State.CONNECTING, State.STREAMING]:
                try:
                    # 从队列获取音频数据，超时1秒
                    audio_data = self.audio_queue.get(timeout=1.0)
                    
                    # None表示流结束
                    if audio_data is None:
                        logger.info("End of audio stream")
                        break
                    
                    # 播放音频数据
                    if self.stream and self.state == State.STREAMING:
                        # 确保数据格式正确
                        if audio_data.dtype != np.float32:
                            audio_data = audio_data.astype(np.float32)
                        
                        # 写入音频流进行播放
                        self.stream.write(audio_data.tobytes())
                    
                except queue.Empty:
                    # 队列为空，继续等待
                    continue
                except Exception as e:
                    logger.error(f"Error playing audio: {e}")
                    break
            
            logger.info("Audio playback thread finished")
            
        except Exception as e:
            logger.error(f"Fatal error in audio playback: {e}")
            self.state = State.ERROR
        finally:
            # 确保状态更新
            if self.state == State.STREAMING:
                self.state = State.IDLE
    
    def start_streaming(self, server_url: str, request_data: dict):
        """
        开始音频流播放
        
        参数：
        - server_url: TTS服务器URL
        - request_data: 请求数据
        
        返回：
        - True: 启动成功
        - False: 启动失败
        """
        if self.state != State.IDLE:
            logger.warning("Player is not in idle state")
            return False
        
        # 初始化音频流
        if not self.init_audio_stream():
            return False
        
        # 清空队列和重置计数器
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        self.total_chunks = 0
        self.start_time = None
        self.first_chunk_time = None
        
        # 启动数据获取线程
        self.fetch_thread = threading.Thread(
            target=self.fetch_audio_stream,
            args=(server_url, request_data),
            daemon=True
        )
        
        # 启动音频播放线程
        self.play_thread = threading.Thread(
            target=self.play_audio_stream,
            daemon=True
        )
        
        try:
            self.fetch_thread.start()
            self.play_thread.start()
            logger.info("Audio streaming started")
            return True
        except Exception as e:
            logger.error(f"Failed to start streaming threads: {e}")
            self.stop_streaming()
            return False
    
    def stop_streaming(self):
        """
        停止音频流播放
        
        清理步骤：
        1. 设置停止状态
        2. 等待线程结束
        3. 关闭音频流
        4. 清理资源
        """
        logger.info("Stopping audio streaming...")
        self.state = State.STOPPED
        
        # 等待线程结束
        if self.fetch_thread and self.fetch_thread.is_alive():
            self.fetch_thread.join(timeout=2.0)
        
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=2.0)
        
        # 关闭音频流
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error closing audio stream: {e}")
            finally:
                self.stream = None
        
        self.state = State.IDLE
        logger.info("Audio streaming stopped")
    
    def get_status(self) -> dict:
        """
        获取播放器状态信息
        
        返回：
        包含状态、性能指标等信息的字典
        """
        status = {
            'state': self.state.value,
            'total_chunks': self.total_chunks,
            'queue_size': self.audio_queue.qsize(),
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size
        }
        
        if self.start_time:
            status['elapsed_time'] = time.perf_counter() - self.start_time
        
        if self.first_chunk_time and self.start_time:
            status['first_chunk_delay'] = self.first_chunk_time - self.start_time
        
        return status
    
    def cleanup(self):
        """
        清理资源
        """
        self.stop_streaming()
        if self.audio:
            self.audio.terminate()

class AudioStreamTestGUI:
    """
    音频流测试图形界面
    
    功能：
    1. 提供文本输入界面
    2. 配置TTS服务器参数
    3. 控制音频播放
    4. 显示状态和日志信息
    5. 监控播放性能
    """
    
    def __init__(self):
        """
        初始化GUI界面
        """
        self.root = tk.Tk()
        self.root.title("Real-time Audio Stream Test - Easy-Wav2Lip")
        self.root.geometry("800x600")
        
        # 音频播放器实例
        self.player = AudioStreamPlayer()
        
        # 状态更新定时器
        self.status_update_job = None
        
        # 创建界面
        self.create_widgets()
        
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        """
        创建GUI组件
        """
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # 服务器配置区域
        config_frame = ttk.LabelFrame(main_frame, text="Server Configuration", padding="5")
        config_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)
        
        # 服务器URL输入
        ttk.Label(config_frame, text="Server URL:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.server_url_var = tk.StringVar(value="http://127.0.0.1:11996/tts_live_stream")
        self.server_url_entry = ttk.Entry(config_frame, textvariable=self.server_url_var, width=50)
        self.server_url_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        
        # 音频路径输入
        ttk.Label(config_frame, text="Audio Path:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.audio_path_var = tk.StringVar(value="/mnt/d/liunxfs/index-tts-vllm/tests/sample_prompt.wav")
        self.audio_path_entry = ttk.Entry(config_frame, textvariable=self.audio_path_var, width=50)
        self.audio_path_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=(5, 0))
        
        # 文本输入区域
        text_frame = ttk.LabelFrame(main_frame, text="Text Input", padding="5")
        text_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
        
        # 文本输入框
        self.text_input = scrolledtext.ScrolledText(text_frame, height=6, wrap=tk.WORD)
        self.text_input.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.text_input.insert(tk.END, "Hello, this is a test message for TTS streaming.")
        
        # 控制按钮区域
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 发送按钮
        self.send_button = ttk.Button(control_frame, text="Send & Play", command=self.start_streaming)
        self.send_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 停止按钮
        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_streaming, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 清除日志按钮
        self.clear_button = ttk.Button(control_frame, text="Clear Log", command=self.clear_log)
        self.clear_button.pack(side=tk.LEFT)
        
        # 状态显示区域
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="5")
        status_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        status_frame.columnconfigure(1, weight=1)
        
        # 状态标签
        ttk.Label(status_frame, text="State:").grid(row=0, column=0, sticky=tk.W)
        self.status_label = ttk.Label(status_frame, text="IDLE", foreground="blue")
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(status_frame, text="Chunks:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.chunks_label = ttk.Label(status_frame, text="0")
        self.chunks_label.grid(row=0, column=3, sticky=tk.W, padx=(5, 0))
        
        ttk.Label(status_frame, text="Queue:").grid(row=0, column=4, sticky=tk.W, padx=(20, 0))
        self.queue_label = ttk.Label(status_frame, text="0")
        self.queue_label.grid(row=0, column=5, sticky=tk.W, padx=(5, 0))
        
        # 日志显示区域
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="5")
        log_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # 日志文本框
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=2)
        
        # 开始状态更新
        self.update_status()
    
    def log_message(self, message: str, level: str = "INFO"):
        """
        在日志区域显示消息
        
        参数：
        - message: 日志消息
        - level: 日志级别
        """
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def clear_log(self):
        """
        清除日志内容
        """
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def start_streaming(self):
        """
        开始音频流播放
        """
        # 获取输入参数
        server_url = self.server_url_var.get().strip()
        audio_path = self.audio_path_var.get().strip()
        text = self.text_input.get(1.0, tk.END).strip()
        
        # 验证输入
        if not server_url:
            messagebox.showerror("Error", "Please enter server URL")
            return
        
        if not text:
            messagebox.showerror("Error", "Please enter text to synthesize")
            return
        
        # 构建请求数据
        request_data = {
            "text": text,
            "audio_paths": [audio_path] if audio_path else []
        }
        
        self.log_message(f"Starting TTS streaming...")
        self.log_message(f"Server: {server_url}")
        self.log_message(f"Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # 启动播放
        if self.player.start_streaming(server_url, request_data):
            self.send_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.log_message("Streaming started successfully")
        else:
            self.log_message("Failed to start streaming", "ERROR")
            messagebox.showerror("Error", "Failed to start audio streaming")
    
    def stop_streaming(self):
        """
        停止音频流播放
        """
        self.log_message("Stopping streaming...")
        self.player.stop_streaming()
        
        self.send_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.log_message("Streaming stopped")
    
    def update_status(self):
        """
        更新状态显示
        """
        try:
            status = self.player.get_status()
            
            # 更新状态标签
            state = status['state'].upper()
            self.status_label.config(text=state)
            
            # 根据状态设置颜色
            color_map = {
                'IDLE': 'blue',
                'CONNECTING': 'orange',
                'STREAMING': 'green',
                'STOPPED': 'red',
                'ERROR': 'red'
            }
            self.status_label.config(foreground=color_map.get(state, 'black'))
            
            # 更新其他信息
            self.chunks_label.config(text=str(status['total_chunks']))
            self.queue_label.config(text=str(status['queue_size']))
            
            # 自动停止检测
            if state in ['IDLE', 'ERROR'] and self.stop_button['state'] == tk.NORMAL:
                self.send_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
                
                if state == 'ERROR':
                    self.log_message("Streaming ended with error", "ERROR")
                elif status['total_chunks'] > 0:
                    self.log_message(f"Streaming completed. Total chunks: {status['total_chunks']}")
        
        except Exception as e:
            logger.error(f"Error updating status: {e}")
        
        # 安排下次更新
        self.status_update_job = self.root.after(500, self.update_status)
    
    def on_closing(self):
        """
        窗口关闭事件处理
        """
        # 停止播放
        self.player.stop_streaming()
        
        # 取消定时器
        if self.status_update_job:
            self.root.after_cancel(self.status_update_job)
        
        # 清理资源
        self.player.cleanup()
        
        # 关闭窗口
        self.root.destroy()
    
    def run(self):
        """
        运行GUI主循环
        """
        self.root.mainloop()

def main():
    """
    主函数 - 程序入口点
    """
    print("=== Real-time Audio Stream Test Tool ===")
    print("This tool tests TTS real-time audio streaming and playback")
    print("Make sure your TTS server is running before testing")
    
    try:
        # 创建并运行GUI
        app = AudioStreamTestGUI()
        app.run()
    except KeyboardInterrupt:
        print("\nUser interrupted the program")
    except Exception as e:
        print(f"Error running application: {e}")
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
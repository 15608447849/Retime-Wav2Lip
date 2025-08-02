#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Debug Tool for Easy-Wav2Lip Project
摄像头调试工具 - 用于测试摄像头设备，显示实时视频帧

=== 视频基础知识 Video Fundamentals ===
1. FPS (Frames Per Second): 每秒帧数
   - 表示视频每秒钟包含多少张静态图片
   - 常见值：24fps(电影)、30fps(电视)、60fps(游戏)
   - 人眼感知：>24fps看起来连续，>60fps非常流畅
   - 技术原理：通过快速播放连续的静态图片产生运动错觉

2. 分辨率 (Resolution): 图像的像素尺寸
   - 格式：宽度 x 高度 (如 1920x1080)
   - 常见分辨率：720p(1280x720)、1080p(1920x1080)、4K(3840x2160)
   - 分辨率越高，图像越清晰，但处理需要更多计算资源
   - 影响因素：存储空间、传输带宽、处理性能

3. 像素 (Pixel): 图像的最小单位
   - 每个像素包含颜色信息(RGB: 红绿蓝)
   - 像素越多，图像越细腻
   - 数字图像本质：像素矩阵，每个位置存储颜色值

4. 颜色空间 (Color Space):
   - RGB: 红绿蓝三原色，计算机常用，加色模式
   - BGR: 蓝绿红，OpenCV默认格式（注意顺序相反）
   - HSV: 色调、饱和度、明度，更符合人眼感知
   - 转换原因：不同应用场景需要不同的颜色表示方式

5. 摄像头参数 Camera Parameters:
   - 亮度(Brightness): 图像整体明暗程度，影响曝光效果
   - 对比度(Contrast): 明暗差异程度，影响图像层次感
   - 饱和度(Saturation): 颜色鲜艳程度，影响色彩丰富度
   - 曝光(Exposure): 传感器接收光线的时间，影响图像明暗
   - 自动曝光: 摄像头自动调节曝光参数以获得最佳效果

6. 视频流处理 Video Stream Processing:
   - 实时处理：边接收边处理，延迟低但计算要求高
   - 缓冲机制：临时存储帧数据，平衡输入输出速度差异
   - 帧丢弃：处理不及时时丢弃旧帧，保持实时性
"""

import cv2  # OpenCV: 计算机视觉库，用于图像和视频处理
import sys  # 系统相关功能
import time  # 时间相关功能
import numpy as np  # 数值计算库，处理图像数组

class CameraDebugger:
    """
    摄像头调试器类
    
    主要功能：
    1. 检测系统中可用的摄像头设备
    2. 初始化摄像头并设置参数
    3. 实时显示摄像头画面
    4. 提供交互功能（截图、参数调节等）
    5. 监控摄像头性能指标
    """
    
    def __init__(self):
        """
        初始化摄像头调试器
        
        成员变量说明：
        - self.cap: OpenCV VideoCapture对象，用于控制摄像头
        - self.is_running: 布尔值，控制预览循环的运行状态
        """
        self.cap = None  # 摄像头捕获对象，初始为空
        self.is_running = False  # 预览运行状态标志
        
    def list_cameras(self):
        """
        检测并列出系统中所有可用的摄像头设备
        
        工作原理：
        1. 遍历可能的摄像头索引（0-9）
        2. 尝试打开每个索引对应的摄像头
        3. 测试是否能成功读取帧数据
        4. 获取摄像头的基本参数信息
        
        返回值：
        - 成功：包含摄像头信息的列表
        - 失败：None
        """
        print("Detecting available cameras...")
        available_cameras = []
        
        # 检测前10个可能的摄像头索引
        # 原理：操作系统通常按顺序分配摄像头设备号
        for i in range(10):
            # 创建VideoCapture对象尝试连接摄像头
            cap = cv2.VideoCapture(i)
            
            # 检查摄像头是否成功打开
            if cap.isOpened():
                # 尝试读取一帧数据来验证摄像头是否真正可用
                ret, frame = cap.read()
                if ret:  # ret为True表示成功读取帧
                    # 获取摄像头的基本参数
                    # CAP_PROP_*是OpenCV定义的摄像头属性常量
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 帧宽度
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 帧高度
                    fps = cap.get(cv2.CAP_PROP_FPS)                  # 帧率
                    
                    # 将摄像头信息添加到可用列表
                    available_cameras.append({
                        'index': i,      # 摄像头索引号
                        'width': width,  # 分辨率宽度
                        'height': height,# 分辨率高度
                        'fps': fps       # 帧率
                    })
                    print(f"Camera {i}: {width}x{height} @ {fps:.1f}fps")
                
                # 释放摄像头资源，避免占用
                cap.release()
        
        # 检查是否找到可用摄像头
        if not available_cameras:
            print("No available cameras detected!")
            return None
        
        return available_cameras
    
    def init_camera(self, camera_index=0):
        """
        初始化指定索引的摄像头
        
        参数：
        - camera_index: 摄像头索引号，默认为0（通常是主摄像头）
        
        工作流程：
        1. 创建VideoCapture对象连接摄像头
        2. 设置摄像头参数（分辨率、帧率等）
        3. 验证参数设置是否成功
        4. 返回初始化结果
        
        返回值：
        - True: 初始化成功
        - False: 初始化失败
        """
        try:
            # 创建VideoCapture对象
            # 参数可以是：设备索引、视频文件路径、网络流URL等
            self.cap = cv2.VideoCapture(camera_index)
            
            # 检查摄像头是否成功打开
            if not self.cap.isOpened():
                print(f"Cannot open camera {camera_index}")
                return False
            
            # 设置摄像头参数
            # 注意：不是所有摄像头都支持所有参数设置
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 设置帧宽度为640像素
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 设置帧高度为480像素
            self.cap.set(cv2.CAP_PROP_FPS, 30)            # 设置帧率为30fps
            
            # 获取实际设置的参数（可能与设置值不同）
            # 原因：硬件限制或驱动不支持某些参数值
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Camera initialized: {width}x{height} @ {fps:.1f}fps")
            return True
            
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            return False
    
    def start_preview(self):
        """
        开始摄像头实时预览
        
        功能特性：
        1. 实时显示摄像头画面
        2. 在画面上叠加状态信息
        3. 计算并显示实际FPS
        4. 提供交互式按键控制
        5. 绘制辅助线帮助对焦
        
        按键功能：
        - 空格键：截图保存
        - 's'键：切换信息显示
        - 'r'键：重置摄像头
        - 'q'键或ESC：退出预览
        """
        # 检查摄像头是否已初始化
        if not self.cap or not self.cap.isOpened():
            print("Camera not initialized or cannot be opened")
            return
        
        # 显示操作说明
        print("Starting camera preview...")
        print("Key controls:")
        print("  SPACE: Take screenshot")
        print("  's': Toggle info display")
        print("  'r': Reset camera")
        print("  'q' or ESC: Exit")
        
        # 初始化预览状态变量
        self.is_running = True      # 预览运行标志
        show_info = True            # 是否显示状态信息
        frame_count = 0             # 帧计数器
        start_time = time.time()    # FPS计算起始时间
        
        # 主预览循环
        while self.is_running:
            # 从摄像头读取一帧数据
            # ret: 布尔值，表示是否成功读取
            # frame: numpy数组，包含图像数据（BGR格式）
            ret, frame = self.cap.read()
            
            # 检查是否成功读取帧
            if not ret:
                print("Cannot read frame from camera")
                break
            
            # 更新帧计数和时间
            frame_count += 1
            current_time = time.time()
            
            # 计算实际FPS（每30帧计算一次）
            # 原理：FPS = 帧数 / 时间间隔
            if frame_count % 30 == 0:  # 每30帧计算一次，减少计算频率
                elapsed_time = current_time - start_time
                actual_fps = 30 / elapsed_time if elapsed_time > 0 else 0
                start_time = current_time  # 重置计时起点
            else:
                actual_fps = 0  # 非计算帧显示0
            
            # 在图像上绘制状态信息
            if show_info:
                # 获取图像尺寸
                # frame.shape返回(高度, 宽度, 通道数)
                height, width = frame.shape[:2]
                
                # 准备要显示的文本信息
                info_text = [
                    f"Resolution: {width}x{height}",  # 分辨率信息
                    f"Frame: {frame_count}",          # 当前帧数
                    f"FPS: {actual_fps:.1f}" if actual_fps > 0 else "Calculating...",  # 实际FPS
                    f"Time: {time.strftime('%H:%M:%S')}"  # 当前时间
                ]
                
                # 在图像上绘制文本
                y_offset = 30  # 文本起始Y坐标
                for i, text in enumerate(info_text):
                    # cv2.putText参数说明：
                    # - frame: 目标图像
                    # - text: 要绘制的文本
                    # - (x, y): 文本位置坐标
                    # - font: 字体类型
                    # - scale: 字体大小比例
                    # - color: 颜色(BGR格式)
                    # - thickness: 线条粗细
                    cv2.putText(frame, text, (10, y_offset + i * 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 绘制中心十字线（用于对焦参考）
                center_x, center_y = width // 2, height // 2
                # 绘制水平线
                cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 0, 255), 1)
                # 绘制垂直线
                cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 0, 255), 1)
            
            # 显示处理后的帧
            # 窗口标题使用英文避免乱码
            cv2.imshow('Camera Debug - Easy-Wav2Lip', frame)
            
            # 处理键盘输入
            # cv2.waitKey(1)等待1毫秒的按键输入
            # & 0xFF确保只取低8位，兼容不同系统
            key = cv2.waitKey(1) & 0xFF
            
            # 根据按键执行相应操作
            if key == ord('q') or key == 27:  # 'q'键或ESC键(ASCII码27)退出
                break
            elif key == ord(' '):  # 空格键截图
                # 生成带时间戳的文件名
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                filename = f'screenshot_{timestamp}.jpg'
                # 保存当前帧为JPEG图片
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('s'):  # 's'键切换信息显示
                show_info = not show_info  # 布尔值取反
                print(f"Info display: {'ON' if show_info else 'OFF'}")
            elif key == ord('r'):  # 'r'键重置摄像头
                print("Resetting camera...")
                # 释放当前摄像头
                self.cap.release()
                time.sleep(0.5)  # 等待500毫秒确保资源释放
                # 重新初始化摄像头
                if self.init_camera():
                    print("Camera reset successful")
                else:
                    print("Camera reset failed")
                    break
        
        # 退出预览循环后清理资源
        self.stop_preview()
    
    def stop_preview(self):
        """
        停止预览并清理资源
        
        清理步骤：
        1. 设置运行标志为False
        2. 释放摄像头资源
        3. 关闭所有OpenCV窗口
        """
        self.is_running = False
        if self.cap:
            self.cap.release()  # 释放摄像头资源
        cv2.destroyAllWindows()  # 关闭所有OpenCV创建的窗口
        print("Camera preview stopped")
    
    def test_camera_properties(self):
        """
        测试并显示摄像头的各种属性参数
        
        功能：
        1. 读取摄像头支持的各种参数
        2. 显示参数的当前值
        3. 帮助了解摄像头性能和特性
        
        测试的属性包括：
        - 基本参数：分辨率、帧率
        - 图像参数：亮度、对比度、饱和度等
        - 控制参数：曝光、自动曝光等
        """
        if not self.cap or not self.cap.isOpened():
            print("Camera not initialized")
            return
        
        print("\n=== Camera Properties Test ===")
        
        # 定义要测试的摄像头属性
        # 格式：(OpenCV属性常量, 属性描述)
        properties = [
            (cv2.CAP_PROP_FRAME_WIDTH, "Frame Width"),      # 帧宽度
            (cv2.CAP_PROP_FRAME_HEIGHT, "Frame Height"),    # 帧高度
            (cv2.CAP_PROP_FPS, "Frame Rate"),               # 帧率
            (cv2.CAP_PROP_BRIGHTNESS, "Brightness"),        # 亮度
            (cv2.CAP_PROP_CONTRAST, "Contrast"),            # 对比度
            (cv2.CAP_PROP_SATURATION, "Saturation"),        # 饱和度
            (cv2.CAP_PROP_HUE, "Hue"),                      # 色调
            (cv2.CAP_PROP_EXPOSURE, "Exposure"),            # 曝光
            (cv2.CAP_PROP_AUTO_EXPOSURE, "Auto Exposure"),  # 自动曝光
        ]
        
        # 遍历所有属性并尝试获取值
        for prop_id, prop_name in properties:
            try:
                # 使用cap.get()方法获取属性值
                value = self.cap.get(prop_id)
                print(f"{prop_name}: {value}")
            except Exception as e:
                # 某些属性可能不被支持，捕获异常并显示
                print(f"{prop_name}: Cannot get value ({e})")

def main():
    """
    主函数 - 程序入口点
    
    执行流程：
    1. 创建摄像头调试器实例
    2. 检测可用摄像头设备
    3. 让用户选择要使用的摄像头
    4. 初始化选定的摄像头
    5. 测试摄像头属性
    6. 开始实时预览
    7. 处理异常和清理资源
    """
    print("=== Easy-Wav2Lip Camera Debug Tool ===\n")
    
    # 创建摄像头调试器实例
    debugger = CameraDebugger()
    
    # 检测系统中可用的摄像头
    cameras = debugger.list_cameras()
    if not cameras:
        print("Program exit - No cameras found")
        return
    
    # 摄像头选择逻辑
    if len(cameras) == 1:
        # 只有一个摄像头，自动选择
        camera_index = cameras[0]['index']
        print(f"Auto-selected camera {camera_index}")
    else:
        # 多个摄像头，让用户选择
        print(f"\nDetected {len(cameras)} camera devices")
        while True:
            try:
                # 获取用户输入
                camera_index = int(input("Please select camera index (enter number): "))
                # 验证输入的索引是否有效
                if any(cam['index'] == camera_index for cam in cameras):
                    break
                else:
                    print("Invalid camera index, please try again")
            except ValueError:
                print("Please enter a valid number")
    
    # 初始化选定的摄像头并开始预览
    if debugger.init_camera(camera_index):
        # 测试摄像头属性（显示技术参数）
        debugger.test_camera_properties()
        
        # 开始实时预览
        try:
            debugger.start_preview()
        except KeyboardInterrupt:
            # 用户按Ctrl+C中断程序
            print("\nUser interrupted the program")
        except Exception as e:
            # 捕获其他异常
            print(f"Error during preview: {e}")
        finally:
            # 确保资源被正确清理
            debugger.stop_preview()
    else:
        print("Camera initialization failed, program exit")

# Python程序入口点
# 当直接运行此脚本时执行main函数
# 当作为模块导入时不执行
if __name__ == "__main__":
    main()
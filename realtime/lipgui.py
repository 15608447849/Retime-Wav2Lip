import time
import tkinter as tk
from time import sleep
from tkinter import ttk, scrolledtext, messagebox, filedialog
from PIL import Image,ImageTk
import logging
import cv2

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LipSyncApp:
    """主应用程序"""

    def __init__(self, lip=None, tts=None, capture=None):
        self.root = tk.Tk()
        self.root.title("实时TTS唇形同步调试工具")
        self.root.geometry("1920x1080")

        self.lip = lip
        self.tts = tts
        self.capture = capture

        # 创建界面
        self.create_widgets()
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # 右侧视频显示
        video_frame = ttk.LabelFrame(main_frame, text="视频预览", padding="5")
        video_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        self.video_label = ttk.Label(video_frame, text="选择视频文件开始", anchor=tk.CENTER)
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 左侧面板
        left_frame = ttk.Frame(main_frame, width=150)
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_frame.columnconfigure(0, weight=1)
        # 创建左侧面板组件
        self.create_left_panel(left_frame)

        # 配置主框架权重
        main_frame.rowconfigure(0, weight=1)

    def create_left_panel(self, parent):
        # TTS服务器配置
        tts_frame = ttk.LabelFrame(parent, text="TTS服务器配置", padding="5")
        tts_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        tts_frame.columnconfigure(1, weight=1)

        ttk.Label(tts_frame, text="服务器URL:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.server_url_var = tk.StringVar(value="http://127.0.0.1:11996/tts_live_stream")
        ttk.Entry(tts_frame, textvariable=self.server_url_var).grid(row=0, column=1, sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Label(tts_frame, text="音频路径:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.audio_path_var = tk.StringVar(value="/mnt/d/liunxfs/index-tts-vllm/tests/sample_prompt.wav")
        ttk.Entry(tts_frame, textvariable=self.audio_path_var).grid(row=1, column=1, sticky=(tk.W, tk.E))

        # 文本输入
        text_frame = ttk.LabelFrame(parent, text="文本输入", padding="5")
        text_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self.text_input = scrolledtext.ScrolledText(text_frame, height=6, wrap=tk.WORD)
        self.text_input.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.text_input.insert(tk.END, "我的流年似水，谁的一路相随。在这个充满遗忘的季节，谁又在回忆最初的那片天空，和那颗回不到天空的星星。流云般的生活，淡淡的。无关色彩。这种催人失落的岁月，慢慢吞噬着梦想。流云般的过客，以并不华丽的姿态瞬间即逝。途上的风景让人留恋，却得不到留念。谁的伤人泪水，让我今夜沉醉。黑色的曲调本来就是应运而生，并不独指时代的变迁。这个让人不得不寂寞的世纪。流云般的年华，麻痹着多数人的心灵。一切无关冷暖，无关痛痒，无关喜怒，无关对错。这独自删改的故事，本来就该如流云般消逝。这只是个让人醉生梦死的归宿。流云留下了风和雨，冲刷了时间罪过。却有多少雨能淋湿心房，有多少风能吹出芬芳。却是时间能让双鬓发黄。流云带走了自己曾留下的一切，不必记得。流云去，流云来。这是需要遗忘的理由。黑夜带走寂寞，留下失落。我的年华落幕，谁在各走各路。流云，流云。流云！")

        # 质量设置
        quality_frame = ttk.LabelFrame(parent, text="处理设置", padding="5")
        quality_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        quality_frame.columnconfigure(1, weight=1)

        # 视频源
        video_frame = ttk.LabelFrame(parent, text="视频源", padding="5")
        video_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        video_frame.columnconfigure(0, weight=1)

        # 视频源类型选择 "file" 或 "camera"
        self.video_source_type = tk.StringVar(value="file")
        source_frame = ttk.Frame(video_frame)
        source_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        source_frame.columnconfigure(1, weight=1)

        tk.Radiobutton(source_frame, text="视频文件", variable=self.video_source_type, value="file",
                       command=self.on_source_type_change).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(source_frame, text="摄像头", variable=self.video_source_type, value="camera",
                        command=self.on_source_type_change).grid(row=0, column=1, sticky=tk.W)
        # 视频文件选择
        self.video_path_var = tk.StringVar()
        self.video_path_frame = ttk.Frame(video_frame)
        self.video_path_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.video_path_frame.columnconfigure(0, weight=1)

        self.video_path_entry = ttk.Entry(self.video_path_frame, textvariable=self.video_path_var, state="readonly")
        self.video_path_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.browse_button = ttk.Button(self.video_path_frame, text="浏览", command=self.browse_video)
        self.browse_button.grid(row=0, column=1)
        self.browse_cancel_button = ttk.Button(self.video_path_frame, text="取消", command=self.browse_cancel_video)
        self.browse_cancel_button.grid(row=0, column=2)

        # 摄像头选择
        self.camera_frame = ttk.Frame(video_frame)
        self.camera_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.camera_frame.columnconfigure(1, weight=1)

        ttk.Label(self.camera_frame, text="摄像头ID:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.camera_id_var = tk.StringVar(value="0")
        self.camera_id_entry = ttk.Entry(self.camera_frame, textvariable=self.camera_id_var, width=10)
        self.camera_id_entry.grid(row=0, column=1, sticky=tk.W)

        # 控制按钮
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))


        self.open_capture = ttk.Button(control_frame, text="打开视频源", command=self.open_cap)
        self.open_capture.pack(side=tk.LEFT, padx=(0, 10))

        self.send_text = ttk.Button(control_frame, text="发送文本", command=self.send_tts)
        self.send_text.pack(side=tk.LEFT, padx=(0, 10))

        # 配置权重
        parent.rowconfigure(2, weight=1)
        parent.rowconfigure(6, weight=1)


    def browse_video(self):
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mov *.mkv"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            self.video_path_var.set(file_path)
    def browse_cancel_video(self):
        self.video_path_var.set("")
    def get_video_source(self):
        """获取视频源"""
        if self.video_source_type.get() == "file":
            return self.video_path_var.get().strip()
        else:
            try:
                return int(self.camera_id_var.get())
            except ValueError:
                return 0  # 默认摄像头ID

    def on_source_type_change(self):
        """视频源类型改变时的处理函数"""
        if self.video_source_type.get() == "file":
            self.camera_frame.grid_remove()
            self.video_path_frame.grid()
        else:
            self.video_path_frame.grid_remove()
            self.camera_frame.grid()


    def open_cap(self):
        '''打开视频源'''

        if not self.capture:
            messagebox.showerror("错误", "组件未加载")
            return

        self.capture.stop()
        video = self.video_path_var.get()
        camera = self.camera_id_var.get()
        self.capture.start( int(camera) if video.strip() == '' else video,
                           lambda frame,chunk: self.display_frame_player_audio(frame,chunk))

    def send_tts(self):
        server_url = self.server_url_var.get().strip()
        audio_path = self.audio_path_var.get().strip()
        text = self.text_input.get(1.0, tk.END).strip()
        # 验证输入
        if not server_url:
            messagebox.showerror("错误", "请输入TTS服务器URL")
            return
        if not text:
            messagebox.showerror("错误", "请输入要合成的文本")
            return
        if not self.tts:
            messagebox.showerror("错误", "组件未加载")
            return
        self.tts.send_tts_text(server_url, audio_path, text)


    def display_frame_player_audio(self, frame, chunk):
        """唇形同步处理视频帧播放"""
        # logger.info(f"frame = {'有' if frame is not None else '无'} "
        #             f"chunk = {'有' if chunk is not None else '无'}")
        if chunk is not None:
            # self.tts.audio_queue.put(chunk)
            self.tts.audio_stream.write(chunk.tobytes())
        if frame is not None:
            self.display_frame(frame)


    def display_frame(self, frame):
        """在GUI中显示视频帧"""
        if frame is not None:
            try:
                # 调整帧大小
                frame = cv2.resize(frame, (int( 640*1.25 ), int(480 *1.25)))

                # 转换颜色空间
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 转换为PIL图像
                pil_image = Image.fromarray(frame_rgb)

                # 转换为Tkinter可显示的格式
                tk_image = ImageTk.PhotoImage(pil_image)

                # 更新显示
                self.video_label.configure(image=tk_image, text="")
                self.video_label.image = tk_image  # 保持引用

            except Exception as e:
                logger.error(f"显示帧错误: {e}")


    def on_closing(self):
        """窗口关闭事件处理"""
        # 清理资源
        if self.lip:
            self.lip.cleanup()

        if self.capture:
            self.capture.stop()

        if self.tts:
            self.tts.cleanup()

        # 关闭窗口
        self.root.destroy()

    def run(self):
        """运行主循环"""
        self.root.mainloop()



import os
import sys

from realtime.lipgui import LipSyncApp
from realtime.lipproc import LipSyncProcessor
from realtime.liptts import TTSPlayer
from realtime.lipcamera import FramePlayer

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def main():
    try:
        print(f"project_root = {project_root}")
        lip = LipSyncProcessor(project_root)
        tts = TTSPlayer(lip)
        cap = FramePlayer(lip)

        # 创建并运行应用
        LipSyncApp(lip,tts,cap).run()
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"程序运行错误: {e}")


if __name__ == "__main__":
    main()
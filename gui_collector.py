import sys
import os
import time
import datetime
import cv2
import numpy as np
from PIL import Image
import imagehash
import mss

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QRadioButton, QButtonGroup,
    QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QFileDialog,
    QTextEdit, QFrame, QSplitter
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap, QImage, QFont

import win32gui
import win32api

# 确保数据保存目录存在
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "raw_images")
os.makedirs(SAVE_DIR, exist_ok=True)


def get_open_windows():
    """枚举当前系统所有可见窗口"""
    windows = []
    def enum_callback(hwnd, extra):
        try:
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if isinstance(title, bytes):
                    title = title.decode('utf-8', errors='ignore')
                title = str(title).strip()
                if title and title not in ["Program Manager", "Settings", "NVIDIA GeForce Overlay"]:
                    windows.append((hwnd, title))
        except Exception:
            pass
    try:
        win32gui.EnumWindows(enum_callback, None)
    except Exception as e:
        print(f"EnumWindows Error: {e}")
    return windows


def capture_window_hwnd(hwnd):
    """使用 mss 抓取指定 HWND 窗口的画面"""
    if not win32gui.IsWindow(hwnd):
        return None
    # 检查窗口是否处于最小化状态
    if win32gui.IsIconic(hwnd):
        return None
    try:
        rect = win32gui.GetWindowRect(hwnd)
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top
        if width <= 10 or height <= 10:
            return None
        
        # 处理最小化或异常坐标情况
        if left < -10000 or top < -10000:
            return None

        monitor = {"top": top, "left": left, "width": width, "height": height}
        with mss.mss() as sct:
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    except Exception as e:
        print(f"截图异常: {e}")
        return None


class CaptureApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("冒险岛 YOLO 图像数据集采集助手 v1.0")
        self.resize(880, 620)
        self.setMinimumSize(800, 550)

        self.selected_hwnd = None
        self.captured_count = 0
        self.last_hash = None
        self.is_capturing = False

        # 初始化 F9 快捷键按压状态检测
        self.f9_was_pressed = False

        # 设置现代化暗黑主题 QSS
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2e;
            }
            QWidget {
                color: #cdd6f4;
                font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
                background-color: #181825;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: #89b4fa;
            }
            QPushButton {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                color: #cdd6f4;
            }
            QPushButton:hover {
                background-color: #45475a;
                border-color: #89b4fa;
            }
            QPushButton:pressed {
                background-color: #585b70;
            }
            QPushButton#btnStart {
                background-color: #a6e3a1;
                color: #11111b;
                border: none;
            }
            QPushButton#btnStart:hover {
                background-color: #94e2d5;
            }
            QPushButton#btnStop {
                background-color: #f38ba8;
                color: #11111b;
                border: none;
            }
            QPushButton#btnStop:hover {
                background-color: #eba0ac;
            }
            QComboBox {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 6px;
                color: #cdd6f4;
            }
            QComboBox::drop-down {
                border: none;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 5px;
                color: #cdd6f4;
            }
            QTextEdit {
                background-color: #11111b;
                border: 1px solid #313244;
                border-radius: 6px;
                color: #a6adc8;
                font-family: "Consolas", monospace;
                font-size: 12px;
            }
            QLabel#lblPreview {
                background-color: #11111b;
                border: 2px dashed #45475a;
                border-radius: 8px;
            }
        """)

        self.init_ui()

        # 全局定时器：用于检测快捷键 F9 和定时截屏
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timer_tick)
        self.timer.start(50)  # 50ms 轮询检测

        self.auto_capture_timer = QTimer()
        self.auto_capture_timer.timeout.connect(self.do_capture)

        self.refresh_window_list()
        self.update_count_display()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # 左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # 1. 窗口绑定组
        win_group = QGroupBox("目标游戏窗口绑定")
        win_layout = QVBoxLayout(win_group)
        
        h_win = QHBoxLayout()
        self.cb_windows = QComboBox()
        self.btn_refresh = QPushButton("刷新窗口")
        self.btn_refresh.clicked.connect(self.refresh_window_list)
        h_win.addWidget(self.cb_windows, 4)
        h_win.addWidget(self.btn_refresh, 1)
        win_layout.addLayout(h_win)
        left_layout.addWidget(win_group)

        # 2. 采集模式组
        mode_group = QGroupBox("采集模式与参数")
        mode_layout = QVBoxLayout(mode_group)

        self.rb_manual = QRadioButton("手动触发模式（全局快捷键 [F9] 截屏）")
        self.rb_auto = QRadioButton("自动定时模式（按设定的时间间隔截屏）")
        self.rb_manual.setChecked(True)

        self.bg_mode = QButtonGroup()
        self.bg_mode.addButton(self.rb_manual, 1)
        self.bg_mode.addButton(self.rb_auto, 2)
        self.bg_mode.buttonToggled.connect(self.on_mode_changed)

        mode_layout.addWidget(self.rb_manual)
        mode_layout.addWidget(self.rb_auto)

        h_interval = QHBoxLayout()
        h_interval.setContentsMargins(20, 0, 0, 0)
        h_interval.addWidget(QLabel("时间间隔 (秒):"))
        self.sp_interval = QDoubleSpinBox()
        self.sp_interval.setRange(0.2, 10.0)
        self.sp_interval.setValue(1.0)
        self.sp_interval.setSingleStep(0.2)
        self.sp_interval.setEnabled(False)
        h_interval.addWidget(self.sp_interval)
        mode_layout.addLayout(h_interval)

        # 过滤去重选项
        self.chk_dedup = QCheckBox("启用智能相似度去重（忽略静止不动画面）")
        self.chk_dedup.setChecked(True)
        mode_layout.addWidget(self.chk_dedup)

        h_hash = QHBoxLayout()
        h_hash.setContentsMargins(20, 0, 0, 0)
        h_hash.addWidget(QLabel("去重阈值 (Diff Threshold, 越小越严格):"))
        self.sp_hash = QSpinBox()
        self.sp_hash.setRange(0, 20)
        self.sp_hash.setValue(5)
        h_hash.addWidget(self.sp_hash)
        mode_layout.addLayout(h_hash)

        left_layout.addWidget(mode_group)

        # 3. 控制按钮与保存位置
        ctrl_group = QGroupBox("控制与存储")
        ctrl_layout = QVBoxLayout(ctrl_group)

        h_btn = QHBoxLayout()
        self.btn_start = QPushButton("启动采集")
        self.btn_start.setObjectName("btnStart")
        self.btn_start.clicked.connect(self.start_capture)

        self.btn_stop = QPushButton("暂停采集")
        self.btn_stop.setObjectName("btnStop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_capture)

        h_btn.addWidget(self.btn_start)
        h_btn.addWidget(self.btn_stop)
        ctrl_layout.addLayout(h_btn)

        h_folder = QHBoxLayout()
        self.btn_open_dir = QPushButton("打开数据文件夹")
        self.btn_open_dir.clicked.connect(self.open_save_dir)
        h_folder.addWidget(self.btn_open_dir)
        ctrl_layout.addLayout(h_folder)

        left_layout.addWidget(ctrl_group)

        # 日志输出框
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        left_layout.addWidget(self.txt_log, 1)

        # 右侧预览面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        lbl_title = QLabel("最新截屏预览 (LIVE THUMBNAIL)")
        lbl_title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        lbl_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(lbl_title)

        self.lbl_preview = QLabel()
        self.lbl_preview.setObjectName("lblPreview")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setText("暂无捕获的画面\n\n选择窗口后点击 [启动采集]\n或按下 [F9] 快捷键")
        right_layout.addWidget(self.lbl_preview, 1)

        self.lbl_counter = QLabel("已保存样本总数: 0 张")
        self.lbl_counter.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_counter.setStyleSheet("color: #a6e3a1;")
        self.lbl_counter.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.lbl_counter)

        # 添加左右分割线布局
        main_layout.addWidget(left_panel, 5)
        main_layout.addWidget(right_panel, 4)

        self.log(f"程序就绪！图片保存位置: {SAVE_DIR}")

    def log(self, text):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.txt_log.append(f"[{timestamp}] {text}")

    def refresh_window_list(self):
        self.cb_windows.clear()
        windows = get_open_windows()
        for hwnd, title in windows:
            self.cb_windows.addItem(f"{title} (HWND: {hwnd})", hwnd)
        if windows:
            self.log(f"已扫描到 {len(windows)} 个有效活动窗口")
        else:
            self.log("未查找到可见窗口！请确认游戏已打开")

    def on_mode_changed(self):
        is_auto = self.rb_auto.isChecked()
        self.sp_interval.setEnabled(is_auto)
        if self.is_capturing and is_auto:
            self.auto_capture_timer.start(int(self.sp_interval.value() * 1000))
        elif not is_auto:
            self.auto_capture_timer.stop()
        else:
            self.log("提示：已切换到【自动定时模式】，请选择目标窗口后点击【启动采集】按钮开始自动截图。")

    def start_capture(self):
        if self.cb_windows.count() == 0:
            self.log("错误: 没有可选择的游戏窗口！")
            return
        
        self.selected_hwnd = self.cb_windows.currentData()
        if not self.selected_hwnd or not win32gui.IsWindow(self.selected_hwnd):
            self.log("错误: 所选窗口句柄无效！请刷新重试")
            return

        win_title = self.cb_windows.currentText()
        self.is_capturing = True
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.cb_windows.setEnabled(False)

        if self.rb_auto.isChecked():
            ms = int(self.sp_interval.value() * 1000)
            self.auto_capture_timer.start(ms)
            self.log(f"启动自动定时采集模式！绑定目标: {win_title}，间隔 {self.sp_interval.value()}s")
            # 立即执行一次截屏，获取首帧反馈
            self.do_capture(trigger_reason="启动自动采集首帧")
        else:
            self.log(f"启动手动快捷键采集模式！绑定目标: {win_title}，请按 [F9] 截屏")

    def stop_capture(self):
        self.is_capturing = False
        self.auto_capture_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.cb_windows.setEnabled(True)
        self.log("已暂停采集功能。")

    def on_timer_tick(self):
        """轮询检测全局快捷键 F9 (VK_F9 = 0x78)"""
        if not self.is_capturing:
            return

        # 仅在手动模式下检测 F9
        if self.rb_manual.isChecked():
            f9_state = win32api.GetAsyncKeyState(0x78) & 0x8000
            if f9_state and not self.f9_was_pressed:
                self.f9_was_pressed = True
                self.do_capture(trigger_reason="F9 快捷键")
            elif not f9_state:
                self.f9_was_pressed = False

    def do_capture(self, trigger_reason="定时触发"):
        if not self.selected_hwnd or not win32gui.IsWindow(self.selected_hwnd):
            self.log("目标窗口已失效或关闭，停止采集！")
            self.stop_capture()
            return

        img = capture_window_hwnd(self.selected_hwnd)
        if img is None:
            self.log("截图失败！请确认游戏窗口未最小化或隐藏")
            return

        # 校验相似度去重
        if self.chk_dedup.isChecked():
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            curr_hash = imagehash.phash(pil_img)
            if self.last_hash is not None:
                diff = curr_hash - self.last_hash
                if diff <= self.sp_hash.value():
                    self.log(f"跳过无变化画面 (Diff: {diff} <= {self.sp_hash.value()}，画面静止时默认不去重保存)")
                    return
            self.last_hash = curr_hash

        # 保存图片（兼容包含中文的路径）
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
        filename = f"maple_{timestamp_str}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)

        try:
            success, enc_img = cv2.imencode(".jpg", img)
            if success:
                enc_img.tofile(filepath)
                self.captured_count += 1
                self.update_count_display()
                self.log(f"成功采集样本: {filename} [{trigger_reason}]")
                self.update_preview(img)
            else:
                self.log(f"错误: 编码图片失败，未保存: {filename}")
        except Exception as e:
            self.log(f"保存文件失败: {e}")

    def update_preview(self, img_bgr):
        h, w, ch = img_bgr.shape
        bytes_per_line = ch * w
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # 保持比例缩放适应预览框大小
        scaled_pixmap = pixmap.scaled(
            self.lbl_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.lbl_preview.setPixmap(scaled_pixmap)

    def update_count_display(self):
        # 统计文件夹中存在的实际图片数量
        files = [f for f in os.listdir(SAVE_DIR) if f.endswith(('.jpg', '.png'))]
        self.captured_count = len(files)
        self.lbl_counter.setText(f"已保存样本总数: {self.captured_count} 张")

    def open_save_dir(self):
        os.startfile(SAVE_DIR)


if __name__ == "__main__":
    import traceback
    def sys_exception_hook(exctype, value, tb):
        print("Uncaught Exception:")
        traceback.print_exception(exctype, value, tb)
        sys.__excepthook__(exctype, value, tb)
    sys.excepthook = sys_exception_hook

    app = QApplication(sys.argv)
    window = CaptureApp()
    window.show()
    sys.exit(app.exec_())

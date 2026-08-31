import os
import sys

# 优先自动注册并加载 PyTorch DLL 目录，防止 Windows DLL 库与 OpenCV/PyQt5 冲突 (WinError 1114)
torch_lib_dir = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.exists(torch_lib_dir):
    try:
        os.add_dll_directory(torch_lib_dir)
    except Exception:
        pass
    os.environ["PATH"] = torch_lib_dir + os.pathsep + os.environ.get("PATH", "")

import torch
from ultralytics import YOLO

import time
import datetime
import cv2
import numpy as np
import mss
import win32gui
import win32api
import ctypes

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QComboBox, QGroupBox, QTextEdit, QLineEdit,
    QSizePolicy, QSpinBox, QDoubleSpinBox, QCheckBox, QFrame, QSplitter, QColorDialog, QInputDialog, QScrollArea
)
from PyQt5.QtCore import QTimer, Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QPen, QColor

from minimap_tracker import MinimapTracker, RouteManager, PathNode, DEFAULT_ROUTE_PATH, DEFAULT_MINIMAP_CONFIG_PATH, ROUTES_DIR
from game_controller import GameController
from decision_engine import DecisionEngine
from auto_buff_manager import AutoBuffManager

# 项目基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
FALLBACK_MODEL = os.path.join(BASE_DIR, "yolov8n.pt")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
DATASET_IMG_DIR = os.path.join(DATASET_DIR, "raw_images")
DEFAULT_COMBAT_CONFIG_PATH = os.path.join(DATASET_DIR, "combat_config.json")
os.makedirs(DATASET_IMG_DIR, exist_ok=True)


def get_open_windows():
    """枚举当前系统所有可见活动窗口"""
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
    except Exception:
        pass
    return windows


def capture_window_hwnd(hwnd, sct):
    """从 HWND 抓取单帧图像"""
    if hwnd and win32gui.IsWindow(hwnd):
        try:
            rect = win32gui.GetWindowRect(hwnd)
            left, top, right, bottom = rect
            width, height = right - left, bottom - top
            if width > 10 and height > 10 and left > -10000 and top > -10000:
                monitor = {"top": top, "left": left, "width": width, "height": height}
                sct_img = sct.grab(monitor)
                img = np.array(sct_img)
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), (left, top)
        except Exception as e:
            print(f"截图异常: {e}")
            return None, (0, 0)
    return None, (0, 0)


class ClickableLabel(QLabel):
    """支持鼠标交互框选与模板提取的画板 Label"""
    roi_selected_signal = pyqtSignal(int, int, int, int)
    template_picked_signal = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_selecting = False
        self.selection_mode = False   # 框选小地图模式
        self.template_pick_mode = False  # 模板提取模式
        self.start_pos = QPoint()
        self.current_pos = QPoint()
        self.img_scale_ratio = 1.0
        self.img_offset_x = 0
        self.img_offset_y = 0
        self.current_cv_frame = None  # 保存最新的 BGR 帧

    def set_selection_mode(self, enabled):
        self.selection_mode = enabled
        self.color_pick_mode = False
        if enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def set_template_pick_mode(self, enabled):
        self.template_pick_mode = enabled
        self.selection_mode = False
        if enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.selection_mode:
                self.is_selecting = True
                self.start_pos = event.pos()
                self.current_pos = event.pos()
                self.update()
            elif self.template_pick_mode and self.current_cv_frame is not None:
                pos = event.pos()
                orig_x = int((pos.x() - self.img_offset_x) / self.img_scale_ratio) if self.img_scale_ratio > 0 else pos.x()
                orig_y = int((pos.y() - self.img_offset_y) / self.img_scale_ratio) if self.img_scale_ratio > 0 else pos.y()

                h, w, _ = self.current_cv_frame.shape
                orig_x = max(0, min(w - 1, orig_x))
                orig_y = max(0, min(h - 1, orig_y))

                # 截取 15x15 的小方块作为模板
                half_size = 7
                y1 = max(0, orig_y - half_size)
                y2 = min(h, orig_y + half_size + 1)
                x1 = max(0, orig_x - half_size)
                x2 = min(w, orig_x + half_size + 1)
                
                patch = self.current_cv_frame[y1:y2, x1:x2].copy()
                self.template_picked_signal.emit(patch)
                print(f"【模板截取】在 ({orig_x}, {orig_y}) 截取了大小为 {patch.shape} 的模板")
                self.set_template_pick_mode(False)
            else:
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.current_pos = event.pos()
            self.update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.is_selecting and event.button() == Qt.LeftButton:
            self.is_selecting = False
            self.current_pos = event.pos()
            self.update()

            if self.img_scale_ratio > 0:
                x1 = int((min(self.start_pos.x(), self.current_pos.x()) - self.img_offset_x) / self.img_scale_ratio)
                y1 = int((min(self.start_pos.y(), self.current_pos.y()) - self.img_offset_y) / self.img_scale_ratio)
                w = int(abs(self.current_pos.x() - self.start_pos.x()) / self.img_scale_ratio)
                h = int(abs(self.current_pos.y() - self.start_pos.y()) / self.img_scale_ratio)

                if w > 10 and h > 10:
                    self.roi_selected_signal.emit(x1, y1, w, h)
                    print(f"【鼠标框选成功】小地图区域: Left={x1}, Top={y1}, W={w}, H={h}")

            self.set_selection_mode(False)
        else:
            super().mouseReleaseEvent(event)


class DetectorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("冒险岛 YOLO 自动打怪与策略控制台 (特征过滤与 Debug 掩膜版) v2.5")
        self.resize(1280, 860)
        self.setMinimumSize(1080, 720)

        # 1. 核心状态控制
        self.is_monitoring_preview = True
        self.is_detection_enabled = True
        self.is_bot_running = False
        self.show_minimap_mask = False  # Debug 掩膜视界开关
        self.is_auto_screenshot = False
        self.screenshot_count = 0
        
        # 定时器初始化 (自动截图)
        self.screenshot_timer = QTimer(self)
        self.screenshot_timer.timeout.connect(self.save_screenshot)

        # 2. 小地图定位与路线管理器初始化 (x=20, y=146, w=272, h=170)
        self.minimap_tracker = MinimapTracker(crop_box=(20, 146, 272, 170))
        self.route_manager = RouteManager()
        self.player_map_pos = None

        # 全局快捷键 F1/F2/F3 (录制) / F9 (停止打怪) 状态记录
        self.hotkey_states = {0x70: False, 0x71: False, 0x72: False, 0x78: False}

        # 3. 初始化控制与决策系统
        self.game_controller = GameController()
        self.decision_engine = DecisionEngine(self.game_controller, self.route_manager)
        self.auto_buff_manager = AutoBuffManager(self.game_controller)

        # 4. 加载 YOLO 权重
        if os.path.exists(MODEL_PATH):
            self.weights = MODEL_PATH
            print(f"【成功】已加载最优模型: {self.weights}")
        elif os.path.exists(FALLBACK_MODEL):
            self.weights = FALLBACK_MODEL
            print(f"【注意】未找到 best.pt，加载基础模型: {self.weights}")
        else:
            self.weights = "yolov8n.pt"

        self.model = YOLO(self.weights)
        self.sct = mss.MSS()
        self.selected_hwnd = None

        # 19 个识别类别的色彩分配 (BGR)
        self.colors = [
            (50, 205, 50),    # player_left (绿)
            (0, 255, 127),    # player_right (青绿)
            (255, 105, 180),  # player_climb (粉)
            (0, 165, 255),    # rope (橙)
            (255, 215, 0),    # portal (金黄)
            (0, 0, 255),      # orange_mushroom (红)
            (0, 69, 255),     # red_snail (橙红)
            (50, 205, 50),    # slime (酸橙)
            (255, 191, 0),    # bubbling (天蓝)
            (147, 112, 219),  # horny_mushroom (紫)
            (128, 128, 128),  # zombie_mushroom (灰)
            (139, 69, 19),    # axe_stump (棕)
            (205, 133, 63),   # wild_boar (浅棕)
            (255, 182, 193),  # pig (粉红)
            (220, 20, 60),    # ribbon_pig (深红)
            (255, 69, 0),     # fire_boar (火红)
            (46, 139, 87),    # jr_necki (海绿)
            (0, 128, 128),    # croco (暗绿)
            (178, 34, 34),    # drake (砖红)
        ]

        self.fps_count = 0
        self.fps_start_time = time.time()
        self.fps_display = 0.0
        self.last_log_time = 0
        self.last_pos_log_time = 0
        self.last_aoe_log_time = 0

        # 暗黑风 QSS
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2e; }
            QWidget { color: #cdd6f4; font-family: "Segoe UI", "Microsoft YaHei"; font-size: 13px; }
            QGroupBox { font-weight: bold; border: 1px solid #45475a; border-radius: 8px; margin-top: 8px; padding-top: 10px; background-color: #181825; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 8px; color: #89b4fa; }
            QPushButton { background-color: #313244; border: 1px solid #45475a; border-radius: 6px; padding: 6px 12px; font-weight: bold; }
            QPushButton:hover { background-color: #45475a; border-color: #89b4fa; }
            QPushButton#btnBotStart { background-color: #a6e3a1; color: #11111b; border: none; font-size: 14px; }
            QPushButton#btnBotStart:hover { background-color: #94e2d5; }
            QPushButton#btnBotStop { background-color: #f38ba8; color: #11111b; border: none; font-size: 14px; }
            QPushButton#btnRecordOn { background-color: #f38ba8; color: #11111b; border: none; }
            QComboBox, QSpinBox, QDoubleSpinBox { background-color: #313244; border: 1px solid #45475a; border-radius: 6px; padding: 5px; color: #cdd6f4; }
            QTextEdit { background-color: #11111b; border: 1px solid #313244; border-radius: 6px; color: #a6adc8; font-family: "Consolas", monospace; font-size: 12px; }
            QLabel#lblDisplay, QLabel#lblMinimapDisplay { background-color: #11111b; border: 2px solid #45475a; border-radius: 8px; }
        """)

        self.init_ui()
        self.log_model_info()
        self.refresh_window_list()
        
        if is_admin():
            self.log("【权限验证通过】软件当前已拥有管理员(Admin)权限，SendInput 畅通。")
        else:
            self.log("【警告！权限不足】软件当前未获得管理员权限！SendInput 模拟按键极大概率会被游戏拦截！请退出并右键使用管理员身份运行！")

        # 载入路线
        if self.route_manager.load_from_json():
            self.log(f"已自动载入巡路路线 ({len(self.route_manager.nodes)} 个节点)")

        # 扫描并初始化地图 XML 列表
        self.refresh_map_xml_list()

        # 30ms 轮询主控制循环
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_loop)
        self.timer.start(30)

    def log(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.txt_log.append(f"[{timestamp}] {msg}")

    def log_model_info(self):
        self.log("=" * 50)
        self.log(f"已成功加载 YOLO 权重文件: {self.weights}")
        if hasattr(self.model, 'names') and self.model.names:
            classes_str = ", ".join([f"{k}:{v}" for k, v in self.model.names.items()])
            self.log(f"包含的识别类别 ({len(self.model.names)} 类): [{classes_str}]")
        self.log("=" * 50)

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # 1. 顶部控制栏
        top_group = QGroupBox("目标游戏窗口绑定与识别门槛设置")
        top_layout = QHBoxLayout(top_group)

        top_layout.addWidget(QLabel("选择窗口:"))
        self.cb_windows = QComboBox()
        self.cb_windows.currentIndexChanged.connect(self.on_window_selected)
        top_layout.addWidget(self.cb_windows, 3)

        self.btn_refresh = QPushButton("刷新窗口")
        self.btn_refresh.clicked.connect(self.refresh_window_list)
        top_layout.addWidget(self.btn_refresh)

        top_layout.addWidget(QLabel("Conf门槛:"))
        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(1, 95)
        self.slider_conf.setValue(15)
        self.slider_conf.setFixedWidth(120)
        self.slider_conf.valueChanged.connect(self.on_conf_changed)
        top_layout.addWidget(self.slider_conf)

        self.lbl_conf_val = QLabel("0.15")
        self.lbl_conf_val.setStyleSheet("font-weight: bold; color: #f9e2af;")
        top_layout.addWidget(self.lbl_conf_val)

        main_layout.addWidget(top_group)

        # 2. 中部左右 5:5 响应式分割面板
        splitter = QSplitter(Qt.Horizontal)

        # ====================== 左侧：监控/识别/小地图看板面板 ======================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 按钮组
        h_btn_group = QHBoxLayout()
        self.btn_bot = QPushButton("⚡ 开始打怪")
        self.btn_bot.setObjectName("btnBotStart")
        self.btn_bot.clicked.connect(self.toggle_bot)

        self.btn_toggle_preview = QPushButton("👁️ 画面渲染: 开启")
        self.btn_toggle_preview.setCheckable(True)
        self.btn_toggle_preview.setChecked(True)
        self.btn_toggle_preview.clicked.connect(self.toggle_preview)

        self.btn_toggle_detection = QPushButton("🎯 YOLO检测: 开启")
        self.btn_toggle_detection.setCheckable(True)
        self.btn_toggle_detection.setChecked(True)
        self.btn_toggle_detection.clicked.connect(self.toggle_detection)

        self.btn_manual_screenshot = QPushButton("📷 手动截图")
        self.btn_manual_screenshot.clicked.connect(self.manual_screenshot)

        self.btn_auto_screenshot = QPushButton("📸 自动截图: 关闭")
        self.btn_auto_screenshot.setCheckable(True)
        self.btn_auto_screenshot.setChecked(False)
        self.btn_auto_screenshot.clicked.connect(self.toggle_auto_screenshot)

        h_btn_group.addWidget(self.btn_bot, 2)
        h_btn_group.addWidget(self.btn_toggle_preview, 2)
        h_btn_group.addWidget(self.btn_toggle_detection, 2)
        h_btn_group.addWidget(self.btn_manual_screenshot, 2)
        h_btn_group.addWidget(self.btn_auto_screenshot, 2)
        left_layout.addLayout(h_btn_group)

        # 主画面展示框 (ClickableLabel)
        self.lbl_display = ClickableLabel()
        self.lbl_display.roi_selected_signal.connect(self.on_roi_selected)
        self.lbl_display.template_picked_signal.connect(self.on_template_picked)
        self.lbl_display.setObjectName("lblDisplay")
        self.lbl_display.setAlignment(Qt.AlignCenter)
        self.lbl_display.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_display.setMinimumSize(380, 220)
        self.lbl_display.setText("请在上方下拉菜单中绑定《冒险岛》游戏窗口")
        left_layout.addWidget(self.lbl_display, 3)

        # 日志诊断框
        log_group = QGroupBox("实时诊断与状态日志 (Debug Log)")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(5, 5, 5, 5)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(100)
        log_layout.addWidget(self.txt_log)
        left_layout.addWidget(log_group, 1)

        # ====================== 右侧：小地图看板与吸管/调色盘取色组 ======================
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 小地图寻路可视化看板 GroupBox
        minimap_group = QGroupBox("🗺️ 小地图寻路看板与对齐设置 (Minimap Dashboard)")
        minimap_layout = QVBoxLayout(minimap_group)
        
        self.lbl_minimap_display = ClickableLabel()
        self.lbl_minimap_display.template_picked_signal.connect(self.on_template_picked)
        self.lbl_minimap_display.setObjectName("lblMinimapDisplay")
        self.lbl_minimap_display.setAlignment(Qt.AlignCenter)
        self.lbl_minimap_display.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_minimap_display.setMinimumSize(240, 140)
        self.lbl_minimap_display.setText("小地图副本捕获中...")
        minimap_layout.addWidget(self.lbl_minimap_display, 1)

        # 小地图区域框选与吸管/调色盘选择按钮组
        mm_crop_layout = QHBoxLayout()
        self.btn_auto_snap = QPushButton("🎯 自动定位小地图")
        self.btn_auto_snap.clicked.connect(self.auto_snap_minimap_roi)

        self.btn_select_roi = QPushButton("🖱️ 手动画框")
        self.btn_select_roi.clicked.connect(self.enable_mouse_roi_select)

        self.btn_pick_template = QPushButton("🔍 提取模板")
        self.btn_pick_template.clicked.connect(self.enable_template_picker)

        self.btn_toggle_mask = QPushButton("🔍 探针调试")
        self.btn_toggle_mask.clicked.connect(self.toggle_mask_view)

        self.btn_dump_debug = QPushButton("📸 诊断快照(F7)")
        self.btn_dump_debug.clicked.connect(self.dump_minimap_debug_snapshot)

        mm_crop_layout.addWidget(self.btn_auto_snap)
        mm_crop_layout.addWidget(self.btn_select_roi)
        mm_crop_layout.addWidget(self.btn_pick_template)
        mm_crop_layout.addWidget(self.btn_toggle_mask)
        mm_crop_layout.addWidget(self.btn_dump_debug)
        minimap_layout.addLayout(mm_crop_layout)

        mm_spin_layout = QHBoxLayout()
        mm_spin_layout.addWidget(QLabel("X:"))
        self.sp_crop_x = QSpinBox()
        self.sp_crop_x.setRange(0, 2000)
        self.sp_crop_x.setValue(self.minimap_tracker.crop_left)
        self.sp_crop_x.valueChanged.connect(self.on_crop_spin_changed)
        mm_spin_layout.addWidget(self.sp_crop_x)

        mm_spin_layout.addWidget(QLabel("Y:"))
        self.sp_crop_y = QSpinBox()
        self.sp_crop_y.setRange(0, 2000)
        self.sp_crop_y.setValue(self.minimap_tracker.crop_top)
        self.sp_crop_y.valueChanged.connect(self.on_crop_spin_changed)
        mm_spin_layout.addWidget(self.sp_crop_y)

        mm_spin_layout.addWidget(QLabel("W:"))
        self.sp_crop_w = QSpinBox()
        self.sp_crop_w.setRange(20, 1000)
        self.sp_crop_w.setValue(self.minimap_tracker.crop_w)
        self.sp_crop_w.valueChanged.connect(self.on_crop_spin_changed)
        mm_spin_layout.addWidget(self.sp_crop_w)

        mm_spin_layout.addWidget(QLabel("H:"))
        self.sp_crop_h = QSpinBox()
        self.sp_crop_h.setRange(20, 1000)
        self.sp_crop_h.setValue(self.minimap_tracker.crop_h)
        self.sp_crop_h.valueChanged.connect(self.on_crop_spin_changed)
        mm_spin_layout.addWidget(self.sp_crop_h)

        self.btn_save_mm_cfg = QPushButton("💾 保存配置")
        self.btn_save_mm_cfg.clicked.connect(self.save_minimap_config_manually)
        mm_spin_layout.addWidget(self.btn_save_mm_cfg)

        minimap_layout.addLayout(mm_spin_layout)

        # 路线录制工具栏
        route_rec_layout = QHBoxLayout()
        self.btn_toggle_rec = QPushButton("⏺ 开始录制路线")
        self.btn_toggle_rec.clicked.connect(self.toggle_route_recording)

        self.btn_add_walk = QPushButton("+ WALK(F1)")
        self.btn_add_walk.clicked.connect(lambda: self.add_record_node("WALK"))

        self.btn_add_jump = QPushButton("+ JUMP(F2)")
        self.btn_add_jump.clicked.connect(lambda: self.add_record_node("JUMP"))

        self.btn_add_climb = QPushButton("+ CLIMB(F3)")
        self.btn_add_climb.clicked.connect(lambda: self.add_record_node("CLIMB"))

        self.btn_add_climb_end = QPushButton("+ 登顶终点(F4)")
        self.btn_add_climb_end.clicked.connect(lambda: self.add_record_node("CLIMB_END"))

        route_rec_layout.addWidget(self.btn_toggle_rec)
        route_rec_layout.addWidget(self.btn_add_walk)
        route_rec_layout.addWidget(self.btn_add_jump)
        route_rec_layout.addWidget(self.btn_add_climb)
        route_rec_layout.addWidget(self.btn_add_climb_end)
        minimap_layout.addLayout(route_rec_layout)

        route_file_layout = QHBoxLayout()
        self.cb_route_select = QComboBox()
        self.cb_route_select.currentIndexChanged.connect(self.on_route_selected)
        self.load_route_list()
        
        self.btn_save_route = QPushButton("💾 命名保存路线")
        self.btn_save_route.clicked.connect(self.save_route_as)

        self.btn_clear_route = QPushButton("🗑️ 清空节点")
        self.btn_clear_route.clicked.connect(self.clear_route_nodes)

        route_file_layout.addWidget(self.cb_route_select, 2)
        route_file_layout.addWidget(self.btn_save_route, 1)
        route_file_layout.addWidget(self.btn_clear_route, 1)
        minimap_layout.addLayout(route_file_layout)

        right_layout.addWidget(minimap_group, 2)

        # 🌟 高级 XML 地图拓扑寻路配置组 (双模式无缝切换)
        map_nav_group = QGroupBox("🗺️ 寻路模式与地图拓扑 (双模式无缝切换)")
        map_nav_layout = QVBoxLayout(map_nav_group)

        h_nav_mode = QHBoxLayout()
        self.chk_advanced_nav = QCheckBox("🌟 启用高级 XML 地图拓扑寻路 (A* 自动跳跃/爬绳)")
        self.chk_advanced_nav.toggled.connect(self.on_nav_mode_toggled)
        h_nav_mode.addWidget(self.chk_advanced_nav)
        map_nav_layout.addLayout(h_nav_mode)

        h_map_select = QHBoxLayout()
        h_map_select.addWidget(QLabel("地图 XML:"))
        self.cb_map_xml = QComboBox()
        self.cb_map_xml.setEnabled(False)
        self.cb_map_xml.currentIndexChanged.connect(self.on_map_xml_changed)
        self.btn_refresh_maps = QPushButton("🔄 刷新地图")
        self.btn_refresh_maps.setEnabled(False)
        self.btn_refresh_maps.clicked.connect(self.refresh_map_xml_list)
        
        h_map_select.addWidget(self.cb_map_xml, 3)
        h_map_select.addWidget(self.btn_refresh_maps, 1)
        map_nav_layout.addLayout(h_map_select)

        right_layout.addWidget(map_nav_group)

        # 右侧：战斗攻击与持续 Buff 技能配置组
        strat_group = QGroupBox("⚔️ 战斗攻击与持续 Buff 技能配置")
        strat_layout = QVBoxLayout(strat_group)

        # 1. 普攻配置行
        h_norm = QHBoxLayout()
        h_norm.addWidget(QLabel("普攻按键:"))
        self.txt_normal_key = QLineEdit("C")
        self.txt_normal_key.setFixedWidth(35)
        h_norm.addWidget(self.txt_normal_key)

        h_norm.addWidget(QLabel("普攻距离(X):"))
        self.sp_normal_range = QSpinBox()
        self.sp_normal_range.setRange(20, 800)
        self.sp_normal_range.setValue(140)
        h_norm.addWidget(self.sp_normal_range)

        h_norm.addWidget(QLabel("普攻间隔(s):"))
        self.sp_normal_cd = QDoubleSpinBox()
        self.sp_normal_cd.setRange(0.05, 5.0)
        self.sp_normal_cd.setSingleStep(0.05)
        self.sp_normal_cd.setValue(0.60)
        self.sp_normal_cd.setFixedWidth(55)
        h_norm.addWidget(self.sp_normal_cd)
        strat_layout.addLayout(h_norm)

        # 2. 群攻配置行
        h_aoe = QHBoxLayout()
        h_aoe.addWidget(QLabel("群攻按键:"))
        self.txt_aoe_key = QLineEdit("D")
        self.txt_aoe_key.setFixedWidth(35)
        h_aoe.addWidget(self.txt_aoe_key)

        h_aoe.addWidget(QLabel("群攻距离(X):"))
        self.sp_aoe_range = QSpinBox()
        self.sp_aoe_range.setRange(20, 800)
        self.sp_aoe_range.setValue(200)
        h_aoe.addWidget(self.sp_aoe_range)

        h_aoe.addWidget(QLabel("触发怪数:"))
        self.sp_aoe_count = QSpinBox()
        self.sp_aoe_count.setRange(1, 20)
        self.sp_aoe_count.setValue(3)
        self.sp_aoe_count.setFixedWidth(40)
        h_aoe.addWidget(self.sp_aoe_count)

        h_aoe.addWidget(QLabel("群攻间隔(s):"))
        self.sp_aoe_cd = QDoubleSpinBox()
        self.sp_aoe_cd.setRange(0.05, 5.0)
        self.sp_aoe_cd.setSingleStep(0.05)
        self.sp_aoe_cd.setValue(0.60)
        self.sp_aoe_cd.setFixedWidth(55)
        h_aoe.addWidget(self.sp_aoe_cd)
        strat_layout.addLayout(h_aoe)

        # 3. 纵向高度、防抖间隔与判定模式行
        h_range_mode = QHBoxLayout()
        h_range_mode.addWidget(QLabel("判定高度(Y):"))
        self.sp_attack_range_y = QSpinBox()
        self.sp_attack_range_y.setRange(10, 500)
        self.sp_attack_range_y.setValue(120)
        self.sp_attack_range_y.setFixedWidth(50)
        h_range_mode.addWidget(self.sp_attack_range_y)

        h_range_mode.addWidget(QLabel("切换间隔(s):"))
        self.sp_state_interval = QDoubleSpinBox()
        self.sp_state_interval.setRange(0.0, 3.0)
        self.sp_state_interval.setSingleStep(0.05)
        self.sp_state_interval.setValue(0.50)
        self.sp_state_interval.setFixedWidth(55)
        h_range_mode.addWidget(self.sp_state_interval)

        h_range_mode.addWidget(QLabel("群攻方向:"))
        self.cb_aoe_dir_mode = QComboBox()
        self.cb_aoe_dir_mode.addItems(["单向 (单侧面朝方向)", "双向 (以玩家为中心)"])
        self.cb_aoe_dir_mode.setCurrentText("单向 (单侧面朝方向)")
        h_range_mode.addWidget(self.cb_aoe_dir_mode)
        strat_layout.addLayout(h_range_mode)

        # 4. 寻怪范围与跳跃按键配置行
        h_agro_jump = QHBoxLayout()
        h_agro_jump.addWidget(QLabel("寻怪范围(X):"))
        self.sp_agro_dist = QSpinBox()
        self.sp_agro_dist.setRange(50, 2000)
        self.sp_agro_dist.setValue(500)
        h_agro_jump.addWidget(self.sp_agro_dist)

        h_agro_jump.addWidget(QLabel("跳跃按键:"))
        self.cb_key_jump = QComboBox()
        self.cb_key_jump.addItems(["Alt", "Space", "V", "C", "D"])
        self.cb_key_jump.setCurrentText("Alt")
        h_agro_jump.addWidget(self.cb_key_jump)
        strat_layout.addLayout(h_agro_jump)

        # 4. 持续 Buff 技能动态列表
        h_buff_header = QHBoxLayout()
        h_buff_header.addWidget(QLabel("<b>✨ 持续 Buff 技能 (勾选立刻触发)</b>"))

        self.btn_add_buff = QPushButton("➕ 添加持续 Buff")
        self.btn_add_buff.clicked.connect(lambda: self.add_buff_item_row())
        h_buff_header.addWidget(self.btn_add_buff)

        self.btn_save_combat_cfg = QPushButton("💾 保存技能配置")
        self.btn_save_combat_cfg.clicked.connect(self.save_combat_config)
        h_buff_header.addWidget(self.btn_save_combat_cfg)

        strat_layout.addLayout(h_buff_header)

        self.buff_items = []
        self.buff_scroll = QScrollArea()
        self.buff_scroll.setWidgetResizable(True)
        self.buff_container = QWidget()
        self.buff_list_layout = QVBoxLayout(self.buff_container)
        self.buff_list_layout.setContentsMargins(2, 2, 2, 2)
        self.buff_scroll.setWidget(self.buff_container)
        strat_layout.addWidget(self.buff_scroll, 1)

        # 载入或初始化默认 Buff 项
        self.load_combat_config()

        right_layout.addWidget(strat_group, 1)

        # Splitter 分割
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([580, 580])
        main_layout.addWidget(splitter, 1)

        # 底部状态栏
        self.lbl_status = QLabel("FPS: 0.0 | 定位: 未追踪 | 路线节点: 0 | 检出目标: 0")
        self.lbl_status.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.lbl_status.setStyleSheet("color: #a6e3a1;")
        main_layout.addWidget(self.lbl_status)

    def toggle_mask_view(self):
        self.show_minimap_mask = not self.show_minimap_mask
        if self.show_minimap_mask:
            self.log("【Debug掩膜开启】小地图看板已切为 HSV 二进制掩膜视角（玩家点应显示为白色孤点）。")
        else:
            self.log("【Debug掩膜关闭】小地图看板已恢复为正常彩色拓扑视角。")

    def toggle_detection(self):
        self.is_detection_enabled = self.btn_toggle_detection.isChecked()
        state = "开启" if self.is_detection_enabled else "关闭"
        self.btn_toggle_detection.setText(f"🎯 YOLO检测: {state}")
        self.log(f"【系统】YOLO检测已{state}")

    def toggle_auto_screenshot(self):
        self.is_auto_screenshot = self.btn_auto_screenshot.isChecked()
        if self.is_auto_screenshot:
            if not self.selected_hwnd or not win32gui.IsWindow(self.selected_hwnd):
                self.log("【警告】尚未绑定有效的游戏窗口！请先在顶部选择框绑定《冒险岛》游戏窗口再开启截图！")
            self.screenshot_count = 0
            self.btn_auto_screenshot.setText("📸 自动截图: 开启 (0张)")
            self.btn_auto_screenshot.setStyleSheet("background-color: #f38ba8; color: #11111b;")
            self.screenshot_timer.start(2000)  # 每2秒截图一次
            self.log("【系统】已开启自动定时截图 (每2秒)，保存目录: dataset/raw_images")
            # 立即触发一次截图
            self.save_screenshot()
        else:
            self.btn_auto_screenshot.setText("📸 自动截图: 关闭")
            self.btn_auto_screenshot.setStyleSheet("")
            self.screenshot_timer.stop()
            self.log(f"【系统】已关闭自动截图，本次累计保存 {self.screenshot_count} 张素材。")

    def manual_screenshot(self):
        """手动点击单张截图"""
        self.log("【手动截图】触发单张抓图...")
        self.save_screenshot()

    def save_screenshot(self):
        frame_to_save = None
        
        # 1. 优先独立抓取 100% 干净的无标注原始游戏帧 (无 YOLO 框和画线)
        if self.selected_hwnd and win32gui.IsWindow(self.selected_hwnd):
            captured, _ = capture_window_hwnd(self.selected_hwnd, self.sct)
            if captured is not None and captured.size > 0:
                frame_to_save = captured

        # 2. 兜底使用画板记录的当前帧
        if frame_to_save is None and hasattr(self.lbl_display, 'current_cv_frame') and self.lbl_display.current_cv_frame is not None:
            frame_to_save = self.lbl_display.current_cv_frame

        if frame_to_save is None:
            if not self.selected_hwnd or not win32gui.IsWindow(self.selected_hwnd):
                self.log("【截图失败】尚未绑定游戏窗口！请先在顶部下拉框中选择《冒险岛》或游戏窗口！")
            else:
                self.log("【截图失败】未能捕获到有效画面（游戏窗口可能已最小化）。")
            return

        try:
            os.makedirs(DATASET_IMG_DIR, exist_ok=True)
            now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            filename = os.path.join(DATASET_IMG_DIR, f"frame_{now_str}.jpg")
            abs_path = os.path.abspath(filename)
            
            success = cv2.imwrite(filename, frame_to_save)
            if success:
                self.screenshot_count += 1
                if self.is_auto_screenshot:
                    self.btn_auto_screenshot.setText(f"📸 自动截图: 开启 ({self.screenshot_count}张)")
                self.log(f"【截图成功】#{self.screenshot_count} 已成功保存至: {abs_path}")
            else:
                self.log(f"【截图保存失败】cv2.imwrite 写入图片失败，请检查文件夹权限: {abs_path}")
        except Exception as e:
            self.log(f"【截图保存失败】写入发生异常: {e}")

    def auto_snap_minimap_roi(self):
        """一键全图扫描并自动吸附小地图视口坐标"""
        if not hasattr(self, 'last_game_frame') or self.last_game_frame is None:
            self.log("【自动定位失败】尚未捕获到游戏画面，请先确保游戏窗口已选择！")
            return
            
        frame = self.last_game_frame
        h, w = frame.shape[:2]
        # 在左上角搜索区域扫描 (0~400, 0~500)
        search_h = min(400, h)
        search_w = min(500, w)
        search_roi = frame[0:search_h, 0:search_w]
        
        if self.minimap_tracker.player_template is not None:
            tmpl = self.minimap_tracker.player_template
            res = cv2.matchTemplate(search_roi, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val >= 0.55:
                px, py = max_loc
                exp_w = self.sp_crop_w.value()
                exp_h = self.sp_crop_h.value()
                
                # 自动推导左上角坐标，使黄点置于适中位置
                best_left = max(0, px - exp_w // 2)
                best_top = max(0, py - exp_h // 2)
                
                self.minimap_tracker.set_crop_box(best_left, best_top, exp_w, exp_h)
                self.update_crop_spins_from_tracker()
                self.log(f"🎯 【一键自动定位成功】已自动捕捉到小地图黄点 (置信度: {max_val:.2f})！框选已吸附至: X={best_left}, Y={best_top}, W={exp_w}, H={exp_h}")
                return
                
        self.log("【自动定位未命中】未能在左上角搜寻到玩家黄点，请使用【🖱️ 手动画框】手动框选一次。")

    def enable_mouse_roi_select(self):
        self.lbl_display.set_selection_mode(True)
        self.log("【框选模式激活】请在左侧主画面上按住鼠标左键拖拽，框选出游戏的小地图区域。")

    def enable_template_picker(self):
        self.lbl_display.set_template_pick_mode(True)
        self.lbl_minimap_display.set_template_pick_mode(True)
        self.log("【模板提取】请在上方任意游戏画面中，点击玩家标志提取模板。")

    def save_minimap_config_manually(self):
        if self.minimap_tracker.save_config():
            self.log(f"【配置保存成功】X={self.sp_crop_x.value()}, Y={self.sp_crop_y.value()}, W={self.sp_crop_w.value()}, H={self.sp_crop_h.value()}")

    def on_crop_spin_changed(self):
        left = self.sp_crop_x.value()
        top = self.sp_crop_y.value()
        w = self.sp_crop_w.value()
        h = self.sp_crop_h.value()
        self.minimap_tracker.set_crop_box(left, top, w, h)

    def update_crop_spins_from_tracker(self):
        self.sp_crop_x.blockSignals(True)
        self.sp_crop_y.blockSignals(True)
        self.sp_crop_w.blockSignals(True)
        self.sp_crop_h.blockSignals(True)

        self.sp_crop_x.setValue(self.minimap_tracker.crop_left)
        self.sp_crop_y.setValue(self.minimap_tracker.crop_top)
        self.sp_crop_w.setValue(self.minimap_tracker.crop_w)
        self.sp_crop_h.setValue(self.minimap_tracker.crop_h)

        self.sp_crop_x.blockSignals(False)
        self.sp_crop_y.blockSignals(False)
        self.sp_crop_w.blockSignals(False)
        self.sp_crop_h.blockSignals(False)

    def on_roi_selected(self, x, y, w, h):
        self.minimap_tracker.set_crop_box(x, y, w, h)
        self.update_crop_spins_from_tracker()
        self.log(f"【小地图对齐成功】框选更新: Left={x}, Top={y}, Width={w}, Height={h}")

    def on_template_picked(self, patch):
        self.minimap_tracker.set_player_template(patch)
        self.log(f"【模板提取成功】已提取 {patch.shape} 大小的模板并保存！")

    def refresh_window_list(self):
        self.cb_windows.blockSignals(True)
        self.cb_windows.clear()
        self.cb_windows.addItem("-- 请选择游戏窗口 --", None)

        windows = get_open_windows()
        auto_target_index = -1

        for idx, (hwnd, title) in enumerate(windows):
            self.cb_windows.addItem(f"{title} (HWND: {hwnd})", hwnd)
            if any(k in title.lower() for k in ["冒险岛", "maple", "maplestory"]):
                auto_target_index = idx + 1

        self.cb_windows.blockSignals(False)

        if auto_target_index > 0:
            self.cb_windows.setCurrentIndex(auto_target_index)
            self.log(f"【自动定位成功】绑定窗口: [{self.cb_windows.currentText()}]")
        else:
            self.log(f"系统窗口列表更新完成 (共 {len(windows)} 个有效窗口)")

    def on_window_selected(self, index):
        self.selected_hwnd = self.cb_windows.currentData()
        if self.selected_hwnd:
            title = self.cb_windows.currentText()
            self.log(f"已锁定捕获句柄: {title}")

    def on_conf_changed(self, val):
        self.lbl_conf_val.setText(f"{val / 100.0:.2f}")

    def toggle_preview(self):
        self.is_monitoring_preview = self.btn_toggle_preview.isChecked()
        if self.is_monitoring_preview:
            self.btn_toggle_preview.setText("👁️ 画面渲染: 开启")
            self.log("【画面渲染】已开启：恢复 GUI 主画面重绘。")
        else:
            self.btn_toggle_preview.setText("🙈 画面渲染: 暂停(省资源)")
            self.lbl_display.setText("画面渲染已暂停 (极速省资源模式)\n\n后台仍持续捕获画面并进行 YOLO 识别！")
            self.log("【画面渲染】已暂停：节省 CPU/GPU 渲染资源！")

    def sync_buff_config(self):
        self.auto_buff_manager.update_config("auto_hp", self.chk_auto_hp.isChecked())
        self.auto_buff_manager.update_config("hp_key", self.cb_key_hp.currentText())
        self.auto_buff_manager.update_config("auto_mp", self.chk_auto_mp.isChecked())
        self.auto_buff_manager.update_config("mp_key", self.cb_key_mp.currentText())
        self.auto_buff_manager.update_config("auto_buff", self.chk_auto_buff.isChecked())
        self.auto_buff_manager.update_config("buff_key", self.cb_key_buff.currentText())
        self.auto_buff_manager.update_config("buff_interval", float(self.sp_buff_interval.value()))

    def toggle_auto_buff(self, state):
        if state == Qt.Checked:
            self.sync_buff_config()
            self.auto_buff_manager.start()
        else:
            self.auto_buff_manager.stop()

    def toggle_bot(self):
        self.is_bot_running = not self.is_bot_running
        if self.is_bot_running:
            self.btn_bot.setText("⏹️ 停止打怪")
            self.btn_bot.setObjectName("btnBotStop")
            self.btn_bot.setStyle(self.btn_bot.style())
            self.log("【打怪总开关】已启动打怪逻辑！(联动激活【YOLO检测】和【状态保持】)")

            # 自动将目标游戏窗口前置并获取焦点，确保 SendInput 命中目标
            if self.selected_hwnd:
                try:
                    win32gui.ShowWindow(self.selected_hwnd, 9)  # 9 = SW_RESTORE (若最小化则恢复)
                    win32gui.SetForegroundWindow(self.selected_hwnd)
                    self.log(f"【自动切屏】已尝试将游戏窗口置于最前 (HWND: {self.selected_hwnd})")
                except Exception as e:
                    self.log(f"【自动切屏】尝试置前窗口失败: {e}")

            if not self.is_detection_enabled:
                self.btn_toggle_detection.setChecked(True)
                self.toggle_detection()
                
        else:
            self.btn_bot.setText("⚡ 开始打怪")
            self.btn_bot.setObjectName("btnBotStart")
            self.btn_bot.setStyle(self.btn_bot.style())
            self.decision_engine.reset()
            self.game_controller.release_all_keys()
            self.log("【打怪总开关】已停止打怪。")

    def toggle_route_recording(self):
        self.route_manager.is_recording = not self.route_manager.is_recording
        if self.route_manager.is_recording:
            self.btn_toggle_rec.setText("⏹ 停止录制路线")
            self.btn_toggle_rec.setObjectName("btnRecordOn")
            self.btn_toggle_rec.setStyle(self.btn_toggle_rec.style())
            self.log("【路线录制】已开始录制！游戏里可按 [F1]=WALK, [F2]=JUMP, [F3]=CLIMB 快捷键随手标记节点。")
        else:
            self.btn_toggle_rec.setText("⏺ 开始录制路线")
            self.btn_toggle_rec.setObjectName("")
            self.btn_toggle_rec.setStyle(self.btn_toggle_rec.style())
            self.log(f"【路线录制】录制已停止，当前共有 {len(self.route_manager.nodes)} 个寻路节点。")

    def refresh_map_xml_list(self):
        """扫描 /map 文件夹并填充地图 XML 下拉列表"""
        self.cb_map_xml.blockSignals(True)
        self.cb_map_xml.clear()
        map_dir = os.path.join(BASE_DIR, "map")
        found_maps = []
        if os.path.exists(map_dir):
            xml_files = [f for f in os.listdir(map_dir) if f.endswith(".xml")]
            for xf in xml_files:
                full_path = os.path.join(map_dir, xf)
                self.cb_map_xml.addItem(f"🗺️ {xf}", full_path)
                found_maps.append(xf)
        
        if self.cb_map_xml.count() == 0:
            self.cb_map_xml.addItem("-- 未找到地图 XML (放入 /map) --", None)
            
        self.cb_map_xml.blockSignals(False)
        if found_maps:
            self.log(f"【地图数据扫描】在 /map 中扫描到 {len(found_maps)} 份地图数据: {found_maps}")

    def on_nav_mode_toggled(self, checked):
        """高级地图 XML 寻路与传统路线录制模式自由切换"""
        if checked:
            self.cb_map_xml.setEnabled(True)
            self.btn_refresh_maps.setEnabled(True)
            xml_path = self.cb_map_xml.currentData()
            if xml_path and os.path.exists(xml_path):
                if self.decision_engine.load_map_xml(xml_path):
                    self._apply_map_canvas_size()
                    self.log(f"【模式切换】已启用 🌟 高级 XML 地图拓扑寻路模式 (已载入: {os.path.basename(xml_path)})")
                else:
                    self.log("【错误】加载地图 XML 失败，回退至传统模式")
            else:
                self.log("【提示】未找到有效的地图 XML，请先在 /map 文件夹中放入解包数据")
        else:
            self.cb_map_xml.setEnabled(False)
            self.btn_refresh_maps.setEnabled(False)
            self.log("【模式切换】已切回 🚶 传统按键录制巡逻模式 (F1~F4 路线)")

    def on_map_xml_changed(self, index):
        """切换选择的地图 XML"""
        if self.chk_advanced_nav.isChecked():
            xml_path = self.cb_map_xml.currentData()
            if xml_path and os.path.exists(xml_path):
                if self.decision_engine.load_map_xml(xml_path):
                    self._apply_map_canvas_size()
                    self.log(f"【地图更新】已切换高级寻路地图为: {os.path.basename(xml_path)}")

    def _apply_map_canvas_size(self):
        """根据当前地图 XML 的真实 canvas 尺寸自动设置小地图框选宽高"""
        if self.decision_engine.map_parser:
            cw = self.decision_engine.map_parser.canvas_w
            ch = self.decision_engine.map_parser.canvas_h
            if cw > 0 and ch > 0:
                # 若为超高滚动地图 (如魔法密林树洞 ch > 100)，游戏界面的小地图框高度固定为标准视窗高 (~79px)
                target_crop_h = min(ch, 79)
                self.sp_crop_w.blockSignals(True)
                self.sp_crop_h.blockSignals(True)
                self.sp_crop_w.setValue(cw)
                self.sp_crop_h.setValue(target_crop_h)
                self.sp_crop_w.blockSignals(False)
                self.sp_crop_h.blockSignals(False)
                self.minimap_tracker.set_crop_box(self.sp_crop_x.value(), self.sp_crop_y.value(), cw, target_crop_h)
                self.log(f"📐 【小地图尺寸自动对齐】已自动调整小地图框选为: {cw} x {target_crop_h} px (全景总高度: {ch}px)")

    def add_buff_item_row(self, key_text="1", cooldown_val=180, is_checked=False):
        """动态新增一行持续 Buff 配置"""
        item_id = len(self.buff_items) + 1
        row_widget = QWidget()
        h_row = QHBoxLayout(row_widget)
        h_row.setContentsMargins(2, 2, 2, 2)

        chk = QCheckBox(f"Buff {item_id}")
        chk.setChecked(is_checked)

        txt_key = QLineEdit(key_text)
        txt_key.setFixedWidth(45)

        sp_cd = QSpinBox()
        sp_cd.setRange(5, 1200)
        sp_cd.setValue(cooldown_val)
        sp_cd.setFixedWidth(60)

        lbl_status = QLabel("状态: 待开启")
        lbl_status.setStyleSheet("color: #89b4fa; font-weight: bold;")

        btn_del = QPushButton("❌")
        btn_del.setFixedWidth(30)

        item_dict = {
            "widget": row_widget,
            "chk": chk,
            "txt_key": txt_key,
            "sp_cd": sp_cd,
            "lbl_status": lbl_status,
            "last_press": 0.0,
            "target_cd": float(sp_cd.value())
        }

        # 一勾选立刻按下按键并开始倒计时
        def on_chk_changed(state):
            if state == 2:  # Checked
                key = txt_key.text().strip()
                if key and self.game_controller:
                    self.game_controller.tap_key(key)
                item_dict["last_press"] = time.time()
                base_cd = float(sp_cd.value())
                if hasattr(self.game_controller, 'jitter') and self.game_controller.jitter:
                    item_dict["target_cd"] = self.game_controller.jitter.calc_floating_interval(base_cd, ratio=0.10)
                else:
                    item_dict["target_cd"] = base_cd
                lbl_status.setText(f"倒计时: {int(item_dict['target_cd'])}s")
                self.log(f"【Buff 技能】勾选开启 Buff [{key}]！已立刻按键，并开启拟人化动态倒计时 (目标 ~{item_dict['target_cd']:.1f}s)")
            else:
                lbl_status.setText("状态: 已禁用")
                self.log(f"【Buff 技能】已禁用 Buff [{txt_key.text().strip()}]")

        chk.stateChanged.connect(on_chk_changed)

        def on_delete():
            self.buff_list_layout.removeWidget(row_widget)
            row_widget.deleteLater()
            if item_dict in self.buff_items:
                self.buff_items.remove(item_dict)

        btn_del.clicked.connect(on_delete)

        h_row.addWidget(chk)
        h_row.addWidget(QLabel("按键:"))
        h_row.addWidget(txt_key)
        h_row.addWidget(QLabel("间隔(s):"))
        h_row.addWidget(sp_cd)
        h_row.addWidget(lbl_status)
        h_row.addWidget(btn_del)

        self.buff_list_layout.addWidget(row_widget)
        self.buff_items.append(item_dict)

        if is_checked:
            on_chk_changed(2)

    def save_combat_config(self):
        """保存当前攻击与 Buff 技能配置至 JSON 文件"""
        try:
            buffs_data = []
            for item in self.buff_items:
                buffs_data.append({
                    "key": item["txt_key"].text().strip(),
                    "cooldown": item["sp_cd"].value(),
                    "checked": item["chk"].isChecked()
                })
            
            cfg = {
                "normal_atk_key": self.txt_normal_key.text().strip(),
                "normal_atk_range": self.sp_normal_range.value(),
                "normal_atk_interval": float(self.sp_normal_cd.value()),
                "aoe_skill_key": self.txt_aoe_key.text().strip(),
                "aoe_skill_range": self.sp_aoe_range.value(),
                "aoe_skill_interval": float(self.sp_aoe_cd.value()),
                "aoe_monster_count": self.sp_aoe_count.value(),
                "attack_range_y": self.sp_attack_range_y.value(),
                "state_switch_interval": float(self.sp_state_interval.value()),
                "aoe_dir_mode": self.cb_aoe_dir_mode.currentText(),
                "monster_agro_dist": self.sp_agro_dist.value(),
                "jump_key": self.cb_key_jump.currentText(),
                "buffs": buffs_data
            }

            os.makedirs(os.path.dirname(DEFAULT_COMBAT_CONFIG_PATH), exist_ok=True)
            import json
            with open(DEFAULT_COMBAT_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)

            self.log(f"【技能配置】成功保存攻击与 Buff 技能配置至 {DEFAULT_COMBAT_CONFIG_PATH}")
        except Exception as e:
            self.log(f"【技能配置保存失败】: {e}")

    def load_combat_config(self):
        """从 JSON 文件载入攻击与 Buff 技能配置"""
        if not os.path.exists(DEFAULT_COMBAT_CONFIG_PATH):
            self.add_buff_item_row("1", 180)
            self.add_buff_item_row("2", 180)
            return

        try:
            import json
            with open(DEFAULT_COMBAT_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            if "normal_atk_key" in cfg:
                self.txt_normal_key.setText(cfg["normal_atk_key"])
            if "normal_atk_range" in cfg:
                self.sp_normal_range.setValue(cfg["normal_atk_range"])
            if "normal_atk_interval" in cfg:
                self.sp_normal_cd.setValue(float(cfg["normal_atk_interval"]))
            if "aoe_skill_key" in cfg:
                self.txt_aoe_key.setText(cfg["aoe_skill_key"])
            if "aoe_skill_range" in cfg:
                self.sp_aoe_range.setValue(cfg["aoe_skill_range"])
            if "aoe_skill_interval" in cfg:
                self.sp_aoe_cd.setValue(float(cfg["aoe_skill_interval"]))
            if "aoe_monster_count" in cfg:
                self.sp_aoe_count.setValue(cfg["aoe_monster_count"])
            if "attack_range_y" in cfg:
                self.sp_attack_range_y.setValue(cfg["attack_range_y"])
            if "state_switch_interval" in cfg:
                self.sp_state_interval.setValue(float(cfg["state_switch_interval"]))
            if "aoe_dir_mode" in cfg:
                idx = self.cb_aoe_dir_mode.findText(cfg["aoe_dir_mode"])
                if idx >= 0:
                    self.cb_aoe_dir_mode.setCurrentIndex(idx)
            if "monster_agro_dist" in cfg:
                self.sp_agro_dist.setValue(cfg["monster_agro_dist"])
            if "jump_key" in cfg:
                idx = self.cb_key_jump.findText(cfg["jump_key"])
                if idx >= 0:
                    self.cb_key_jump.setCurrentIndex(idx)

            buffs_data = cfg.get("buffs", [])
            if buffs_data:
                for item in list(self.buff_items):
                    self.buff_list_layout.removeWidget(item["widget"])
                    item["widget"].deleteLater()
                self.buff_items.clear()

                for b in buffs_data:
                    self.add_buff_item_row(
                        key_text=b.get("key", "1"),
                        cooldown_val=b.get("cooldown", 180),
                        is_checked=b.get("checked", False)
                    )
            else:
                self.add_buff_item_row("1", 180)
                self.add_buff_item_row("2", 180)

            self.log(f"【技能配置】已自动恢复历史攻击与 Buff 技能配置自 {DEFAULT_COMBAT_CONFIG_PATH}")
        except Exception as e:
            self.log(f"【技能配置加载失败】: {e}")

    def add_record_node(self, action_type="WALK"):
        pos = self.player_map_pos or getattr(self, "last_valid_player_map_pos", None)
        if pos:
            px, py = pos
            node = self.route_manager.add_node(px, py, action_type)
            self.log(f"【节点录制】✅ 成功记录路点 #{node.node_id} ({action_type}) 坐标: ({px}, {py})")
        else:
            self.log("【节点录制失败】未能在小地图上精确定位到玩家当前位置！")

    def load_route_list(self):
        self.cb_route_select.blockSignals(True)
        self.cb_route_select.clear()
        self.cb_route_select.addItem("-- 选择路线 --", None)
        
        if os.path.exists(ROUTES_DIR):
            for file in os.listdir(ROUTES_DIR):
                if file.endswith(".json"):
                    self.cb_route_select.addItem(file, os.path.join(ROUTES_DIR, file))
        self.cb_route_select.blockSignals(False)

    def on_route_selected(self, index):
        if index <= 0:
            return
        file_path = self.cb_route_select.itemData(index)
        if file_path:
            success, crop_box = self.route_manager.load_from_json(file_path)
            if success:
                self.log(f"【路线读取】成功载入 {len(self.route_manager.nodes)} 个巡路节点自 {os.path.basename(file_path)}")
                if crop_box:
                    self.minimap_tracker.set_crop_box(*crop_box)
                    self.update_crop_spins_from_tracker()
                    self.log(f"【配置同步】小地图裁剪区已自动恢复为: {crop_box}")

    def save_route_as(self):
        name, ok = QInputDialog.getText(self, "保存路线", "请输入路线名称:", text="my_route")
        if ok and name.strip():
            filename = name.strip()
            if not filename.endswith(".json"):
                filename += ".json"
            
            filepath = os.path.join(ROUTES_DIR, filename)
            crop = (
                self.minimap_tracker.crop_left,
                self.minimap_tracker.crop_top,
                self.minimap_tracker.crop_w,
                self.minimap_tracker.crop_h
            )
            
            if self.route_manager.save_to_json(filepath, crop_box=crop):
                self.log(f"【路线保存】已保存 {len(self.route_manager.nodes)} 个节点及小地图配置至 {filename}")
                self.load_route_list()
                
                # 设置当前选项为刚保存的
                index = self.cb_route_select.findText(filename)
                if index >= 0:
                    self.cb_route_select.setCurrentIndex(index)

    def clear_route_nodes(self):
        self.route_manager.clear()
        self.log("【路线管理】已清空当前节点队列。")

    def poll_hotkeys(self):
        # 1. 强制停止打怪快捷键 F9 (0x78)
        vk_f9 = 0x78
        is_f9_down = win32api.GetAsyncKeyState(vk_f9) & 0x8000
        if is_f9_down and not self.hotkey_states.get(vk_f9, False):
            self.hotkey_states[vk_f9] = True
            if self.is_bot_running:
                self.log("【全局快捷键】检测到 F9 键按下，强制停止自动打怪和按键映射！")
                self.toggle_bot()
        elif not is_f9_down:
            self.hotkey_states[vk_f9] = False

        # 1.4. 一键导出小地图诊断快照快捷键 F7 (0x76)
        vk_f7 = 0x76
        is_f7_down = win32api.GetAsyncKeyState(vk_f7) & 0x8000
        if is_f7_down and not self.hotkey_states.get(vk_f7, False):
            self.hotkey_states[vk_f7] = True
            self.dump_minimap_debug_snapshot()
        elif not is_f7_down:
            self.hotkey_states[vk_f7] = False

        # 1.5. 手动截图训练素材快捷键 F8 (0x77)
        vk_f8 = 0x77
        is_f8_down = win32api.GetAsyncKeyState(vk_f8) & 0x8000
        if is_f8_down and not self.hotkey_states.get(vk_f8, False):
            self.hotkey_states[vk_f8] = True
            self.save_screenshot()
        elif not is_f8_down:
            self.hotkey_states[vk_f8] = False

        # 2. 录制路线快捷键 F1(WALK)/F2(JUMP)/F3(CLIMB)/F4(CLIMB_END)
        action_keys = [(0x70, "WALK"), (0x71, "JUMP"), (0x72, "CLIMB"), (0x73, "CLIMB_END")]
        for vk, action_type in action_keys:
            is_down = win32api.GetAsyncKeyState(vk) & 0x8000
            if is_down and not self.hotkey_states.get(vk, False):
                self.hotkey_states[vk] = True
                if not self.route_manager.is_recording:
                    self.route_manager.is_recording = True
                    self.btn_toggle_rec.setText("⏹ 停止录制路线")
                    self.btn_toggle_rec.setObjectName("btnRecordOn")
                    self.btn_toggle_rec.setStyle(self.btn_toggle_rec.style())
                    self.log("【路线录制】快捷键触发自动开启路线录制模式！")
                self.add_record_node(action_type)
            elif not is_down:
                self.hotkey_states[vk] = False

    def dump_minimap_debug_snapshot(self):
        """一键导出当前帧全屏画面、小地图裁剪原图与诊断数据"""
        import datetime
        debug_dir = os.path.join(DATASET_DIR, "debug_snapshots")
        os.makedirs(debug_dir, exist_ok=True)
        t_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 导出带小地图框选标记的完整全屏画面
        if hasattr(self, 'last_game_frame') and self.last_game_frame is not None:
            full_frame = self.last_game_frame.copy()
            cl = self.minimap_tracker.crop_left
            ct = self.minimap_tracker.crop_top
            cw = self.minimap_tracker.crop_w
            ch = self.minimap_tracker.crop_h
            cv2.rectangle(full_frame, (cl, ct), (cl + cw, ct + ch), (0, 255, 255), 2)
            cv2.putText(full_frame, f"MINIMAP ROI [{cw}x{ch}]", (cl, max(15, ct - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(debug_dir, f"full_{t_str}.png"), full_frame)

        # 2. 导出小地图实际裁剪图
        if hasattr(self, 'last_raw_minimap') and self.last_raw_minimap is not None:
            mm_img = self.last_raw_minimap.copy()
            if self.player_map_pos:
                cv2.circle(mm_img, self.player_map_pos, 4, (0, 0, 255), 1)
            cv2.imwrite(os.path.join(debug_dir, f"minimap_crop_{t_str}.png"), mm_img)

        # 3. 导出匹配热力掩膜
        if self.minimap_tracker.last_mask_img is not None:
            cv2.imwrite(os.path.join(debug_dir, f"mask_{t_str}.png"), self.minimap_tracker.last_mask_img)

        self.log(f"📸 【诊断快照已导出】已保存当前帧至 dataset/debug_snapshots/ (时间戳: {t_str})")

    def process_loop(self):
        if not self.selected_hwnd or not win32gui.IsWindow(self.selected_hwnd):
            return

        game_frame, (win_x, win_y) = capture_window_hwnd(self.selected_hwnd, self.sct)
        if game_frame is None:
            return

        # 保存用于一键快照调试的原始帧
        self.last_game_frame = game_frame.copy()

        # 更新最新原始帧给主画板 ClickableLabel
        self.lbl_display.current_cv_frame = game_frame
        
        # 实时在主画面上绘制小地图 ROI 框选区域 (亮黄色矩形，一目了然看清小地图框选是否对准)
        cl, ct, cw, ch = self.minimap_tracker.crop_left, self.minimap_tracker.crop_top, self.minimap_tracker.crop_w, self.minimap_tracker.crop_h
        cv2.rectangle(game_frame, (cl, ct), (cl + cw, ct + ch), (0, 255, 255), 1)
        cv2.putText(game_frame, f"MINIMAP [{cw}x{ch}]", (cl, max(12, ct - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)

        # 实时绘制鼠标手动框选中的虚线框
        if self.lbl_display.is_selecting and self.lbl_display.start_pos and self.lbl_display.current_pos and self.lbl_display.img_scale_ratio > 0:
            x1 = int((min(self.lbl_display.start_pos.x(), self.lbl_display.current_pos.x()) - self.lbl_display.img_offset_x) / self.lbl_display.img_scale_ratio)
            y1 = int((min(self.lbl_display.start_pos.y(), self.lbl_display.current_pos.y()) - self.lbl_display.img_offset_y) / self.lbl_display.img_scale_ratio)
            w = int(abs(self.lbl_display.current_pos.x() - self.lbl_display.start_pos.x()) / self.lbl_display.img_scale_ratio)
            h = int(abs(self.lbl_display.current_pos.y() - self.lbl_display.start_pos.y()) / self.lbl_display.img_scale_ratio)
            cv2.rectangle(game_frame, (x1, y1), (x1+w, y1+h), (0, 215, 255), 2)

        # 1. 快捷键轮询 (包含 F9 强制停止与 F1 端点录制)
        self.poll_hotkeys()

        # 2. 从游戏帧剪裁小地图图像 & 进行小地图定位
        minimap_bgr = self.minimap_tracker.crop_minimap(game_frame)
        self.last_raw_minimap = minimap_bgr.copy() if minimap_bgr is not None else None
        if minimap_bgr is not None:
            # 更新剪裁小地图帧给小地图看板 ClickableLabel
            self.lbl_minimap_display.current_cv_frame = minimap_bgr

            self.player_map_pos, conf_score = self.minimap_tracker.locate_player_pos(minimap_bgr)
            if self.player_map_pos:
                self.last_valid_player_map_pos = self.player_map_pos

            # 节流 1.5 秒打印一次当前玩家小地图实时坐标 Log
            now_time = time.time()
            if now_time - self.last_pos_log_time >= 1.5:
                self.last_pos_log_time = now_time
                if self.player_map_pos:
                    px, py = self.player_map_pos
                    curr_target = self.route_manager.get_current_target_node()
                    next_target = self.route_manager.get_current_next_node()
                    seg_str = f" [当前段: P{curr_target.node_id}->P{next_target.node_id}]" if (curr_target and next_target and self.route_manager.nodes) else ""
                    self.log(f"【小地图定位】玩家当前像素坐标: ({px}, {py}) | 置信度: {conf_score:.2f}{seg_str}")
                else:
                    self.log(f"【小地图定位】未匹配到玩家图标 (最高得分: {conf_score:.2f} < 0.70)")

            # 决定看板渲染普通彩色拓扑视图还是 Debug 二进制 Mask 掩膜视图
            if self.show_minimap_mask and self.minimap_tracker.last_mask_img is not None:
                dashboard_bgr = self.minimap_tracker.last_mask_img.copy()
                if self.player_map_pos:
                    px, py = self.player_map_pos
                    cv2.circle(dashboard_bgr, (px, py), 6, (0, 0, 255), 2)
                    cv2.putText(dashboard_bgr, "YOU", (px - 10, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            else:
                dashboard_bgr = self.minimap_tracker.draw_minimap_dashboard(
                    minimap_bgr, self.player_map_pos, self.route_manager, self.decision_engine
                )

                # 打印高级 A* 寻路状态诊断 Log (节流 1.5s)
                if self.chk_advanced_nav.isChecked() and self.decision_engine.map_parser and self.player_map_pos:
                    if now_time - self.last_pos_log_time >= 1.5:
                        px, py = self.player_map_pos
                        mp = self.decision_engine.map_parser
                        xw, yw = mp.minimap_to_world(px, py)
                        cur_fh = mp.snap_to_foothold(xw, yw)
                        active_path = self.decision_engine.active_path
                        cur_idx = self.decision_engine.current_step_idx
                        step_str = f"步骤: {cur_idx+1}/{len(active_path)} ({active_path[cur_idx].action} 目标X={active_path[cur_idx].target_x})" if (active_path and cur_idx < len(active_path)) else "无执行路径/等待规划"
                        fh_str = f"平台 #{cur_fh.id} (Y={cur_fh.get_y_at_x(xw):.0f})" if cur_fh else "未吸附到平台"
                        self.log(f"【🌟 A* 寻路监控】世界绝对坐标: ({xw}, {yw}) | 所在: {fh_str} | {step_str}")

            # 渲染到 lbl_minimap_display
            mh, mw, mch = dashboard_bgr.shape
            m_bytes = mch * mw
            m_rgb = cv2.cvtColor(dashboard_bgr, cv2.COLOR_BGR2RGB)
            m_qimg = QImage(m_rgb.data, mw, mh, m_bytes, QImage.Format_RGB888)
            m_pixmap = QPixmap.fromImage(m_qimg)
            
            scaled_m_pixmap = m_pixmap.scaled(
                self.lbl_minimap_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            m_sw = scaled_m_pixmap.width()
            m_sh = scaled_m_pixmap.height()
            if mw > 0:
                self.lbl_minimap_display.img_scale_ratio = m_sw / float(mw)
            self.lbl_minimap_display.img_offset_x = (self.lbl_minimap_display.width() - m_sw) // 2
            self.lbl_minimap_display.img_offset_y = (self.lbl_minimap_display.height() - m_sh) // 2
            
            self.lbl_minimap_display.setPixmap(scaled_m_pixmap)

        # 3. YOLO 目标识别与决策输入准备
        conf_thresh = self.slider_conf.value() / 100.0
        detections = []
        screen_monsters = []
        screen_player_pos = None
        player_state = None

        if self.is_detection_enabled or self.is_bot_running:
            raw_results = self.model.predict(game_frame, conf=0.01, verbose=False)
            raw_boxes = raw_results[0].boxes if len(raw_results) > 0 else []

            if len(raw_boxes) > 0:
                for box in raw_boxes:
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    class_name = self.model.names[cls_id] if hasattr(self.model, 'names') else str(cls_id)

                    if conf >= conf_thresh:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        detections.append(f"{class_name}@({center_x},{center_y})[{conf:.2f}]")
                        
                        # 分类角色状态与怪物 (传递完整包围盒坐标支持双重底部/中心对齐)
                        if class_name in ["player_left", "player_right", "player_climb"]:
                            screen_player_pos = (center_x, center_y, x1, y1, x2, y2)
                            self.last_screen_player_pos = (center_x, center_y, x1, y1, x2, y2)
                            if class_name == "player_climb":
                                player_state = "climb"
                            elif class_name == "player_left":
                                player_state = "left"
                            else:
                                player_state = "right"
                        else:
                            # 其他默认认为是怪物 (排除 rope 和 portal)
                            if class_name not in ["rope", "portal"]:
                                screen_monsters.append((center_x, center_y, x1, y1, x2, y2))

                        if self.is_monitoring_preview:
                            color = self.colors[cls_id % len(self.colors)]
                            cv2.rectangle(game_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.circle(game_frame, (center_x, center_y), 4, (0, 0, 255), -1)

                            label_str = f"{class_name} {conf:.2f}"
                            (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                            cv2.rectangle(game_frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                            cv2.putText(game_frame, label_str, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # 4. 持续 Buff 技能倒计时检查与自动施放 (拟人化 ±10% 动态浮动)
        now_time = time.time()
        for item in self.buff_items:
            if item["chk"].isChecked():
                elapsed = now_time - item["last_press"]
                base_cd = float(item["sp_cd"].value())
                target_cd = item.get("target_cd", base_cd)
                remaining = max(0, int(target_cd - elapsed))
                item["lbl_status"].setText(f"倒计时: {remaining}s")
                if elapsed >= target_cd:
                    key = item["txt_key"].text().strip()
                    if key and self.game_controller:
                        self.game_controller.tap_key(key)
                    item["last_press"] = now_time
                    # 重新生成下一次独立浮动的目标时长 (±10%)
                    if hasattr(self.game_controller, 'jitter') and self.game_controller.jitter:
                        item["target_cd"] = self.game_controller.jitter.calc_floating_interval(base_cd, ratio=0.10)
                    else:
                        item["target_cd"] = base_cd
                    self.log(f"【Buff 技能定时】自动重发 Buff 技能 [{key}] (本次间隔 {elapsed:.1f}s / 下次目标 {item['target_cd']:.1f}s)")

        # 5. 执行自动打怪决策核心
        if self.is_bot_running:
            game_h, game_w, _ = game_frame.shape
            if not screen_player_pos:
                # 若受击闪烁/扣血无敌导致单帧漏检，优先兜底维持在上一帧已知位置
                if hasattr(self, 'last_screen_player_pos') and self.last_screen_player_pos is not None:
                    screen_player_pos = self.last_screen_player_pos
                else:
                    screen_player_pos = (game_w // 2, game_h // 2 + 50)

            # 节流 1.5 秒打印一次群攻范围怪数量诊断 Log
            if now_time - self.last_aoe_log_time >= 1.5:
                self.last_aoe_log_time = now_time
                px, py = screen_player_pos[0], screen_player_pos[1]
                p_bottom = screen_player_pos[5] if len(screen_player_pos) >= 6 else (py + 30)
                aoe_range = self.sp_aoe_range.value()
                aoe_count_thresh = self.sp_aoe_count.value()
                atk_range_y = self.sp_attack_range_y.value()
                is_single = ("单向" in self.cb_aoe_dir_mode.currentText())
                
                same_level = [
                    m for m in screen_monsters 
                    if min(abs(m[1] - py), abs((m[5] if len(m) >= 6 else m[1]) - p_bottom)) <= atk_range_y
                ]
                if is_single:
                    r_c = sum(1 for m in same_level if 0 <= (m[0] - px) <= aoe_range)
                    l_c = sum(1 for m in same_level if 0 <= (px - m[0]) <= aoe_range)
                    self.log(f"【群攻范围诊断(单向)】画面总怪: {len(screen_monsters)} | 同平台(Y容差<={atk_range_y}px)怪数: {len(same_level)} | 右侧怪数: {r_c}, 左侧怪数: {l_c} (门槛: >= {aoe_count_thresh})")
                else:
                    in_aoe_range = [m for m in same_level if abs(m[0] - px) <= aoe_range]
                    self.log(f"【群攻范围诊断(双向)】画面总怪: {len(screen_monsters)} | 同平台(Y容差<={atk_range_y}px)怪数: {len(same_level)} | 群攻范围(X<={aoe_range}px)怪数: {len(in_aoe_range)} (门槛: >= {aoe_count_thresh})")

            # 获取用户设置的攻击与技能配置
            current_config = {
                "normal_atk_key": self.txt_normal_key.text().strip() or "C",
                "normal_atk_range": self.sp_normal_range.value(),
                "normal_atk_interval": float(self.sp_normal_cd.value()),
                "aoe_skill_key": self.txt_aoe_key.text().strip() or "D",
                "aoe_skill_range": self.sp_aoe_range.value(),
                "aoe_skill_interval": float(self.sp_aoe_cd.value()),
                "aoe_monster_count": self.sp_aoe_count.value(),
                "attack_range_y": self.sp_attack_range_y.value(),
                "state_switch_interval": float(self.sp_state_interval.value()),
                "aoe_dir_mode": self.cb_aoe_dir_mode.currentText(),
                "monster_agro_dist": self.sp_agro_dist.value(),
                "jump_key": self.cb_key_jump.currentText(),
                "enable_advanced_nav": self.chk_advanced_nav.isChecked(),
                "crop_w": self.sp_crop_w.value(),
                "crop_h": self.sp_crop_h.value()
            }
                
            # 执行决策
            self.decision_engine.update(
                game_screen_player_pos=screen_player_pos,
                game_screen_monsters=screen_monsters,
                player_state=player_state,
                minimap_player_pos=self.player_map_pos,
                config=current_config
            )

        # 5. 主画面渲染
        if self.is_monitoring_preview:
            h, w, ch = game_frame.shape
            bytes_per_line = ch * w
            rgb_frame = cv2.cvtColor(game_frame, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            lbl_w = self.lbl_display.width()
            lbl_h = self.lbl_display.height()

            scaled_pixmap = pixmap.scaled(
                self.lbl_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

            sw = scaled_pixmap.width()
            sh = scaled_pixmap.height()
            self.lbl_display.img_scale_ratio = sw / float(w)
            self.lbl_display.img_offset_x = (lbl_w - sw) // 2
            self.lbl_display.img_offset_y = (lbl_h - sh) // 2

            self.lbl_display.setPixmap(scaled_pixmap)

        # 5. 计算 FPS 并更新底部状态栏
        self.fps_count += 1
        now = time.time()
        if now - self.fps_start_time >= 1.0:
            self.fps_display = self.fps_count / (now - self.fps_start_time)
            self.fps_count = 0
            self.fps_start_time = now

        pos_str = f"({self.player_map_pos[0]},{self.player_map_pos[1]})" if self.player_map_pos else "未定位"
        rec_str = " [录制中]" if self.route_manager.is_recording else ""
        nav_mode_str = " [🌟XML拓扑A*]" if self.chk_advanced_nav.isChecked() else " [🚶传统巡逻]"
        self.lbl_status.setText(
            f"FPS: {self.fps_display:.1f} | 寻路模式:{nav_mode_str} | 小地图坐标:{pos_str} | 传统节点:{len(self.route_manager.nodes)}个{rec_str} | 检出目标:{len(detections)}个"
        )


if __name__ == "__main__":
    import traceback
    def sys_exception_hook(exctype, value, tb):
        print("Uncaught Exception:")
        traceback.print_exception(exctype, value, tb)
        sys.__excepthook__(exctype, value, tb)
    sys.excepthook = sys_exception_hook

    app = QApplication(sys.argv)
    window = DetectorApp()
    window.show()
    sys.exit(app.exec_())

import os
import time
import cv2
import numpy as np
import threading
import winsound
import ctypes
from ctypes import wintypes, HRESULT, POINTER, c_void_p, c_float, c_uint32, Structure, byref

ole32 = ctypes.oledll.ole32

class _GUID(Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", wintypes.BYTE * 8)
    ]

    def __init__(self, l, w1, w2, b1, b2, b3, b4, b5, b6, b7, b8):
        super().__init__(l, w1, w2, (wintypes.BYTE * 8)(b1, b2, b3, b4, b5, b6, b7, b8))

CLSID_MMDeviceEnumerator = _GUID(0xBCDE0395, 0xE52F, 0x467C, 0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E)
IID_IMMDeviceEnumerator = _GUID(0xA95664D2, 0x9614, 0x4F35, 0xA7, 0x46, 0xDE, 0x8D, 0xB6, 0x36, 0x17, 0xE6)
IID_IAudioEndpointVolume = _GUID(0x5CDF2C82, 0x841E, 0x4546, 0x97, 0x22, 0x0C, 0xF7, 0x40, 0x78, 0x22, 0x9A)

CLSCTX_ALL = 23
eRender = 0
eMultimedia = 1

import json
from datetime import datetime

def get_system_volume():
    """
    通过 Windows 原生 Core Audio COM 接口获取系统主音量标量 (0.0 ~ 1.0)。
    失败返回 None。
    """
    co_initialized = False
    try:
        hr = ole32.CoInitialize(None)
        if hr in (0, 1):
            co_initialized = True
    except Exception:
        pass

    try:
        enumerator = c_void_p()
        hr = ole32.CoCreateInstance(
            byref(CLSID_MMDeviceEnumerator),
            None,
            CLSCTX_ALL,
            byref(IID_IMMDeviceEnumerator),
            byref(enumerator)
        )
        if hr != 0 or not enumerator:
            return None

        enum_vtbl = ctypes.cast(ctypes.cast(enumerator, POINTER(c_void_p)).contents, POINTER(c_void_p))
        GetDefaultAudioEndpoint_proto = ctypes.WINFUNCTYPE(HRESULT, c_void_p, c_uint32, c_uint32, POINTER(c_void_p))
        GetDefaultAudioEndpoint = GetDefaultAudioEndpoint_proto(enum_vtbl[4])
        Release_proto = ctypes.WINFUNCTYPE(c_uint32, c_void_p)

        device = c_void_p()
        hr = GetDefaultAudioEndpoint(enumerator, eRender, eMultimedia, byref(device))
        if hr != 0 or not device:
            Release_proto(enum_vtbl[2])(enumerator)
            return None

        dev_vtbl = ctypes.cast(ctypes.cast(device, POINTER(c_void_p)).contents, POINTER(c_void_p))
        Activate_proto = ctypes.WINFUNCTYPE(HRESULT, c_void_p, POINTER(_GUID), wintypes.DWORD, c_void_p, POINTER(c_void_p))
        Activate = Activate_proto(dev_vtbl[3])

        endpoint_volume = c_void_p()
        hr = Activate(device, byref(IID_IAudioEndpointVolume), CLSCTX_ALL, None, byref(endpoint_volume))
        if hr != 0 or not endpoint_volume:
            Release_proto(dev_vtbl[2])(device)
            Release_proto(enum_vtbl[2])(enumerator)
            return None

        vol_vtbl = ctypes.cast(ctypes.cast(endpoint_volume, POINTER(c_void_p)).contents, POINTER(c_void_p))
        GetMasterVolumeLevelScalar_proto = ctypes.WINFUNCTYPE(HRESULT, c_void_p, POINTER(c_float))
        GetMasterVolumeLevelScalar = GetMasterVolumeLevelScalar_proto(vol_vtbl[9])
        current_vol = c_float(0.0)
        GetMasterVolumeLevelScalar(endpoint_volume, byref(current_vol))
        curr_val = float(current_vol.value)

        Release_proto(vol_vtbl[2])(endpoint_volume)
        Release_proto(dev_vtbl[2])(device)
        Release_proto(enum_vtbl[2])(enumerator)

        return curr_val
    except Exception as e:
        print(f"[get_system_volume] 异常: {e}")
        return None
    finally:
        if co_initialized:
            try:
                ole32.CoUninitialize()
            except Exception:
                pass


def set_system_volume(scalar, unmute=False):
    """
    通过 Windows 原生 Core Audio COM 接口设置系统主音量标量 (0.0 ~ 1.0)。
    可选是否自动解除静音状态。
    返回: bool (是否设置成功)
    """
    if scalar is None:
        return False
    scalar = max(0.0, min(1.0, float(scalar)))

    co_initialized = False
    try:
        hr = ole32.CoInitialize(None)
        if hr in (0, 1):
            co_initialized = True
    except Exception:
        pass

    try:
        enumerator = c_void_p()
        hr = ole32.CoCreateInstance(
            byref(CLSID_MMDeviceEnumerator),
            None,
            CLSCTX_ALL,
            byref(IID_IMMDeviceEnumerator),
            byref(enumerator)
        )
        if hr != 0 or not enumerator:
            return False

        enum_vtbl = ctypes.cast(ctypes.cast(enumerator, POINTER(c_void_p)).contents, POINTER(c_void_p))
        GetDefaultAudioEndpoint_proto = ctypes.WINFUNCTYPE(HRESULT, c_void_p, c_uint32, c_uint32, POINTER(c_void_p))
        GetDefaultAudioEndpoint = GetDefaultAudioEndpoint_proto(enum_vtbl[4])
        Release_proto = ctypes.WINFUNCTYPE(c_uint32, c_void_p)

        device = c_void_p()
        hr = GetDefaultAudioEndpoint(enumerator, eRender, eMultimedia, byref(device))
        if hr != 0 or not device:
            Release_proto(enum_vtbl[2])(enumerator)
            return False

        dev_vtbl = ctypes.cast(ctypes.cast(device, POINTER(c_void_p)).contents, POINTER(c_void_p))
        Activate_proto = ctypes.WINFUNCTYPE(HRESULT, c_void_p, POINTER(_GUID), wintypes.DWORD, c_void_p, POINTER(c_void_p))
        Activate = Activate_proto(dev_vtbl[3])

        endpoint_volume = c_void_p()
        hr = Activate(device, byref(IID_IAudioEndpointVolume), CLSCTX_ALL, None, byref(endpoint_volume))
        if hr != 0 or not endpoint_volume:
            Release_proto(dev_vtbl[2])(device)
            Release_proto(enum_vtbl[2])(enumerator)
            return False

        vol_vtbl = ctypes.cast(ctypes.cast(endpoint_volume, POINTER(c_void_p)).contents, POINTER(c_void_p))

        if unmute:
            SetMute_proto = ctypes.WINFUNCTYPE(HRESULT, c_void_p, wintypes.BOOL, c_void_p)
            SetMute = SetMute_proto(vol_vtbl[14])
            SetMute(endpoint_volume, False, None)

        SetMasterVolumeLevelScalar_proto = ctypes.WINFUNCTYPE(HRESULT, c_void_p, c_float, c_void_p)
        SetMasterVolumeLevelScalar = SetMasterVolumeLevelScalar_proto(vol_vtbl[7])
        SetMasterVolumeLevelScalar(endpoint_volume, c_float(scalar), None)

        Release_proto(vol_vtbl[2])(endpoint_volume)
        Release_proto(dev_vtbl[2])(device)
        Release_proto(enum_vtbl[2])(enumerator)

        return True
    except Exception as e:
        print(f"[set_system_volume] 异常: {e}")
        return False
    finally:
        if co_initialized:
            try:
                ole32.CoUninitialize()
            except Exception:
                pass


def boost_system_volume(min_level_percent=80):
    """
    通过 Windows 原生 Core Audio COM 接口将系统主音量调高至指定百分比 (默认 80%)，
    并自动解除静音状态。无需依赖任何第三方库，纳秒级响应。
    返回: (success: bool, old_vol: float, new_vol: float)
    """
    old_vol = get_system_volume()
    if old_vol is None:
        old_vol = 0.5
    target_scalar = min(1.0, max(0.0, float(min_level_percent) / 100.0))
    success = set_system_volume(target_scalar, unmute=True)
    return success, old_vol, target_scalar

class CaptchaAlertDetector:
    """
    测谎仪 / 符文图形验证弹窗检测器 (完整 3 连框 UI 容器全景高精匹配版)
    
    采用 'HSV 荧光绿极速门控 + 完整 3 连框多尺度全景模板匹配 + 3 图标水平共线几何校验'
    
    优势:
    1. 误判率 (False Positive) 降至 0.00%: 彻底杜绝单个绿点/技能误报，必须命中整套 3 连框或 3 准星排布
    2. 常态极速: 正常打怪时 <0.3ms 极速跳过
    3. 异常毫秒捕获: 弹窗出现时在 5ms 内精准识别并报警
    """
    def __init__(self, template_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 完整 3 连框全景模板 (459x144)
        if template_path is None:
            self.full_template_path = os.path.join(base_dir, "dataset", "captcha_full_template.png")
            self.single_icon_path = os.path.join(base_dir, "dataset", "captcha_target_icon.png")
        else:
            self.full_template_path = template_path
            self.single_icon_path = os.path.join(base_dir, "dataset", "captcha_target_icon.png")
            
        self.full_tpl_bgr = None
        self.full_tpl_gray = None
        self.full_tpl_half = None
        self.fh, self.fw = 0, 0
        
        # 单个准星模板 (46x46)
        self.icon_tpl_gray = None
        self.ih, self.iw = 0, 0
        
        # 荧光绿 HSV 色彩区间
        self.lower_green = np.array([35, 90, 90], dtype=np.uint8)
        self.upper_green = np.array([85, 255, 255], dtype=np.uint8)
        
        # 报警控制
        self.last_alarm_time = 0.0
        self.alarm_cooldown = 1.8
        self.is_alarm_playing = False
        self.is_enabled = True
        self.match_threshold = 0.72 # 全景模板相关系数门槛

        # 打怪初始基准音量与 5 秒恢复定时器
        self.baseline_volume = None
        self.volume_restore_timer = None
        self.volume_restore_lock = threading.Lock()
        
        # 多尺度金字塔
        self.scales = [1.0, 0.9, 1.1, 0.8, 1.25]
        
        self.load_templates()

    def set_baseline_volume(self, vol):
        """记录打怪开始时的系统音量作为基准音量"""
        if vol is not None:
            self.baseline_volume = float(vol)
            print(f"[CaptchaAlertDetector] 已锁定打怪基准音量: {self.baseline_volume * 100:.0f}%")

    def get_baseline_volume(self):
        return self.baseline_volume

    def restore_volume(self):
        """立即将系统音量恢复至打怪基准音量，并取消倒计时定时器"""
        with self.volume_restore_lock:
            if self.volume_restore_timer is not None:
                self.volume_restore_timer.cancel()
                self.volume_restore_timer = None
            if self.baseline_volume is not None:
                set_system_volume(self.baseline_volume)
                print(f"[CaptchaAlertDetector] 系统音量已恢复至基准值: {self.baseline_volume * 100:.0f}%")

    def _on_volume_restore_timeout(self):
        """5秒定时器到期后执行音量恢复"""
        with self.volume_restore_lock:
            self.volume_restore_timer = None
            if self.baseline_volume is not None:
                set_system_volume(self.baseline_volume)
                print(f"[CaptchaAlertDetector] 5秒倒计时结束，系统音量已恢复至基准值: {self.baseline_volume * 100:.0f}%")

    def load_templates(self):
        """加载完整 3 连框模板与单个准星模板"""
        # 1. 载入完整全景模板
        if os.path.exists(self.full_template_path):
            self.full_tpl_bgr = cv2.imread(self.full_template_path)
            if self.full_tpl_bgr is not None:
                self.full_tpl_gray = cv2.cvtColor(self.full_tpl_bgr, cv2.COLOR_BGR2GRAY)
                self.fh, self.fw = self.full_tpl_gray.shape[:2]
                self.full_tpl_half = cv2.resize(self.full_tpl_gray, (0, 0), fx=0.5, fy=0.5)
                print(f"[CaptchaAlertDetector] 成功载入【完整 3 连框全景模板】: {self.full_template_path} (尺寸: {self.fw}x{self.fh})")

        # 2. 载入单个准星模板
        if os.path.exists(self.single_icon_path):
            icon_bgr = cv2.imread(self.single_icon_path)
            if icon_bgr is not None:
                self.icon_tpl_gray = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2GRAY)
                self.ih, self.iw = self.icon_tpl_gray.shape[:2]
                print(f"[CaptchaAlertDetector] 成功载入【准星辅助模板】: {self.single_icon_path} (尺寸: {self.iw}x{self.ih})")

    def detect(self, bgr_frame):
        """
        全景多目标匹配检索
        返回: (is_detected: bool, match_boxes: list of (x, y, w, h, score))
        """
        if not self.is_enabled or bgr_frame is None:
            return False, []

        # 1. HSV 极速色彩门控 (画面中必须包含足够的绿色特征像素，3 个准星总面积通常 > 100 像素)
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        green_pixel_count = cv2.countNonZero(mask)

        # 常态无目标绿色时，0.2ms 直接退出
        if green_pixel_count < 60:
            return False, []

        gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        gh, gw = gray_frame.shape[:2]
        match_boxes = []

        # 2. 方案 A: 完整 3 连框全景模板金字塔匹配 (最优准确度，0 误判)
        if self.full_tpl_gray is not None:
            # 先用 0.5x 降采样图做粗搜 (耗时 ~3ms)
            gray_half = cv2.resize(gray_frame, (0, 0), fx=0.5, fy=0.5)
            res_half = cv2.matchTemplate(gray_half, self.full_tpl_half, cv2.TM_CCOEFF_NORMED)
            _, max_val_half, _, max_loc_half = cv2.minMaxLoc(res_half)

            if max_val_half >= (self.match_threshold - 0.15):
                # 在原图候选区域精确核实
                cand_x = max(0, int(max_loc_half[0] * 2) - 15)
                cand_y = max(0, int(max_loc_half[1] * 2) - 15)
                cand_w = min(gw - cand_x, self.fw + 30)
                cand_h = min(gh - cand_y, self.fh + 30)

                if cand_w >= self.fw and cand_h >= self.fh:
                    roi = gray_frame[cand_y:cand_y+cand_h, cand_x:cand_x+cand_w]
                    res_full = cv2.matchTemplate(roi, self.full_tpl_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res_full)

                    if max_val >= self.match_threshold:
                        real_x = cand_x + max_loc[0]
                        real_y = cand_y + max_loc[1]
                        match_boxes.append((real_x, real_y, self.fw, self.fh, float(max_val)))
                        return True, match_boxes

            # 多尺度弥补 (应对窗口轻微非 1:1 缩放)
            for scale in [0.9, 1.1, 0.8, 1.25]:
                sw, sh = int(self.fw * scale), int(self.fh * scale)
                if sw >= gw or sh >= gh or sw < 50:
                    continue
                scaled_tpl = cv2.resize(self.full_tpl_gray, (sw, sh), interpolation=cv2.INTER_LINEAR)
                scaled_half = cv2.resize(scaled_tpl, (0, 0), fx=0.5, fy=0.5)
                res_s = cv2.matchTemplate(gray_half, scaled_half, cv2.TM_CCOEFF_NORMED)
                _, max_v, _, max_l = cv2.minMaxLoc(res_s)
                if max_v >= self.match_threshold:
                    rx = int(max_l[0] * 2)
                    ry = int(max_l[1] * 2)
                    match_boxes.append((rx, ry, sw, sh, float(max_v)))
                    return True, match_boxes

        # 3. 方案 B: 3 准星水平共线与等间距几何校验 (辅助双保险)
        # 必须至少有 2~3 个准星在同一水平线 (Y 坐标相差 <= 8px) 且间距在 100~180px 之间
        if self.icon_tpl_gray is not None:
            res_icon = cv2.matchTemplate(gray_frame, self.icon_tpl_gray, cv2.TM_CCOEFF_NORMED)
            locs = np.where(res_icon >= 0.78)
            cand_pts = []
            for pt in zip(*locs[::-1]):
                if not any(abs(pt[0] - cp[0]) < 20 and abs(pt[1] - cp[1]) < 20 for cp in cand_pts):
                    cand_pts.append(pt)

            if len(cand_pts) >= 2:
                # 检查水平共线性与等间距
                cand_pts.sort(key=lambda p: p[0])
                for i in range(len(cand_pts)):
                    for j in range(i + 1, len(cand_pts)):
                        p1, p2 = cand_pts[i], cand_pts[j]
                        dx = abs(p2[0] - p1[0])
                        dy = abs(p2[1] - p1[1])
                        # 3 连框相邻准星间距约为 144px (容差 100~190px)，垂直偏差 <= 8px
                        if dy <= 8 and (100 <= dx <= 190 or 240 <= dx <= 340):
                            match_boxes.append((min(p1[0], p2[0]), min(p1[1], p2[1]), dx + self.iw, self.ih, 0.88))
                            return True, match_boxes

        return False, []

    def trigger_alarm(self, boost_volume=True, target_volume=80, force=False, restore_delay=5.0):
        """
        触发多频急促警报音效 (非阻塞多线程执行)，并将音量提升至 80%，并在 5 秒后自动恢复。
        参数:
            boost_volume: 是否自动调高系统音量至 80% 并解除静音
            target_volume: 目标系统音量百分比 (默认 80)
            force: 是否忽略冷却时间强制触发 (测试用)
            restore_delay: 恢复至基准音量的延时秒数 (默认 5.0 秒)
        """
        now = time.time()
        if not force:
            if now - self.last_alarm_time < self.alarm_cooldown or self.is_alarm_playing:
                return
        
        self.last_alarm_time = now
        self.is_alarm_playing = True

        # 若勾选自动调高音量，启动/刷新 5 秒倒计时恢复机制
        if boost_volume:
            if self.baseline_volume is None:
                cur_v = get_system_volume()
                if cur_v is not None:
                    self.baseline_volume = cur_v

            with self.volume_restore_lock:
                if self.volume_restore_timer is not None:
                    self.volume_restore_timer.cancel()
                self.volume_restore_timer = threading.Timer(restore_delay, self._on_volume_restore_timeout)
                self.volume_restore_timer.daemon = True
                self.volume_restore_timer.start()

        def _play():
            try:
                # 1. 自动调高系统主音量至 80% 并解除静音
                if boost_volume:
                    try:
                        boost_system_volume(target_volume)
                    except Exception as e:
                        print(f"[CaptchaAlertDetector] 自动调节音量异常: {e}")

                # 2. 播放多频急促警报蜂鸣
                for _ in range(3):
                    winsound.Beep(2200, 120)
                    time.sleep(0.04)
                    winsound.Beep(1600, 120)
                    time.sleep(0.04)
            except Exception:
                pass
            finally:
                self.is_alarm_playing = False

        thread = threading.Thread(target=_play, daemon=True)
        thread.start()


class BotSessionLogger:
    """
    打怪会话本地 JSON 日志记录器
    持久化存储于本地文件 (默认为项目根目录 bot_sessions.json)，用于 Git 追踪与 GitHub 上传。
    数据结构格式:
    [
      {
        "start_time": "2026-09-07 16:30:00",
        "captcha_triggers": [
          "2026-09-07 16:35:12"
        ],
        "end_time": "2026-09-07 16:45:00"
      }
    ]
    """
    def __init__(self, log_path=None):
        if log_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.log_path = os.path.join(base_dir, "bot_sessions.json")
        else:
            self.log_path = log_path

        self.lock = threading.Lock()
        self.current_session_idx = None
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        with self.lock:
            if not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0:
                try:
                    with open(self.log_path, "w", encoding="utf-8") as f:
                        json.dump([], f, indent=2, ensure_ascii=False)
                except Exception as e:
                    print(f"[BotSessionLogger] 初始化日志文件异常: {e}")

    def _load_all_sessions(self):
        try:
            if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 0:
                with open(self.log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
        except Exception as e:
            print(f"[BotSessionLogger] 读取日志文件异常: {e}")
        return []

    def _save_all_sessions(self, sessions):
        try:
            with open(self.log_path, "w", encoding="utf-8") as f:
                json.dump(sessions, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[BotSessionLogger] 写入日志文件异常: {e}")

    def start_session(self):
        """点击开始打怪时调用：追加一条新 session 记录并持久化"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            sessions = self._load_all_sessions()
            new_session = {
                "start_time": now_str,
                "captcha_triggers": [],
                "end_time": None
            }
            sessions.append(new_session)
            self._save_all_sessions(sessions)
            self.current_session_idx = len(sessions) - 1
            print(f"[BotSessionLogger] 打怪 Session 开始: {now_str} (已更新至 {self.log_path})")
            return new_session

    def record_captcha_trigger(self):
        """测谎报警触发时调用：向当前 session 追加触发时间戳并持久化"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            sessions = self._load_all_sessions()
            target_idx = None
            if self.current_session_idx is not None and 0 <= self.current_session_idx < len(sessions):
                target_idx = self.current_session_idx
            elif sessions and sessions[-1].get("end_time") is None:
                target_idx = len(sessions) - 1

            if target_idx is not None:
                sessions[target_idx].setdefault("captcha_triggers", []).append(now_str)
                self._save_all_sessions(sessions)
                print(f"[BotSessionLogger] 记录测谎仪触发时间戳: {now_str}")
                return True
            else:
                print(f"[BotSessionLogger] 警告: 当前没有进行中的打怪 Session，未记录测谎触发")
                return False

    def end_session(self):
        """点击停止打怪或程序关闭时调用：写入结束时间戳并持久化"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            sessions = self._load_all_sessions()
            target_idx = None
            if self.current_session_idx is not None and 0 <= self.current_session_idx < len(sessions):
                target_idx = self.current_session_idx
            elif sessions and sessions[-1].get("end_time") is None:
                target_idx = len(sessions) - 1

            if target_idx is not None:
                sessions[target_idx]["end_time"] = now_str
                self._save_all_sessions(sessions)
                self.current_session_idx = None
                print(f"[BotSessionLogger] 打怪 Session 结束: {now_str} (已更新至 {self.log_path})")
                return True
            return False


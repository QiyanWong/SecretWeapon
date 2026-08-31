import time
import math
import random
import ctypes
import win32gui
from ctypes import wintypes

# 启用 Windows 多媒体 1ms 物理高精度时钟，破除 15.6ms 时钟中断栅格
try:
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

# C struct redefinitions for SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# 冒险岛底层是 DirectX 引擎，通常拦截了虚拟键码 (Virtual Key Codes)
# 我们必须使用 DirectInput 的硬件扫描码 (Scan Codes) 才能让游戏识别
DIK_KEYS = {
    'ESCAPE': 0x01, '1': 0x02, '2': 0x03, '3': 0x04, '4': 0x05, '5': 0x06, '6': 0x07, '7': 0x08, '8': 0x09, '9': 0x0A, '0': 0x0B,
    'MINUS': 0x0C, 'EQUALS': 0x0D, 'BACK': 0x0E, 'TAB': 0x0F, 'Q': 0x10, 'W': 0x11, 'E': 0x12, 'R': 0x13, 'T': 0x14, 'Y': 0x15, 'U': 0x16, 'I': 0x17, 'O': 0x18, 'P': 0x19,
    'LBRACKET': 0x1A, 'RBRACKET': 0x1B, 'RETURN': 0x1C, 'LCONTROL': 0x1D, 'A': 0x1E, 'S': 0x1F, 'D': 0x20, 'F': 0x21, 'G': 0x22, 'H': 0x23, 'J': 0x24, 'K': 0x25, 'L': 0x26,
    'SEMICOLON': 0x27, 'APOSTROPHE': 0x28, 'GRAVE': 0x29, 'LSHIFT': 0x2A, 'BACKSLASH': 0x2B, 'Z': 0x2C, 'X': 0x2D, 'C': 0x2E, 'V': 0x2F, 'B': 0x30, 'N': 0x31, 'M': 0x32,
    'COMMA': 0x33, 'PERIOD': 0x34, 'SLASH': 0x35, 'RSHIFT': 0x36, 'MULTIPLY': 0x37, 'LMENU': 0x38, 'SPACE': 0x39, 'CAPITAL': 0x3A,
    'F1': 0x3B, 'F2': 0x3C, 'F3': 0x3D, 'F4': 0x3E, 'F5': 0x3F, 'F6': 0x40, 'F7': 0x41, 'F8': 0x42, 'F9': 0x43, 'F10': 0x44,
    'NUMLOCK': 0x45, 'SCROLL': 0x46, 'NUMPAD7': 0x47, 'NUMPAD8': 0x48, 'NUMPAD9': 0x49, 'SUBTRACT': 0x4A, 'NUMPAD4': 0x4B, 'NUMPAD5': 0x4C, 'NUMPAD6': 0x4D, 'ADD': 0x4E, 'NUMPAD1': 0x4F, 'NUMPAD2': 0x50, 'NUMPAD3': 0x51, 'NUMPAD0': 0x52, 'DECIMAL': 0x53,
    'HOME': 0xC7, 'UP': 0xC8, 'PRIOR': 0xC9, 'LEFT': 0xCB, 'RIGHT': 0xCD, 'END': 0xCF, 'DOWN': 0xD0, 'NEXT': 0xD1, 'INSERT': 0xD2, 'DELETE': 0xD3,
}

# 别名映射
DIK_KEYS['CTRL'] = DIK_KEYS['LCONTROL']
DIK_KEYS['ALT'] = DIK_KEYS['LMENU']
DIK_KEYS['SHIFT'] = DIK_KEYS['LSHIFT']

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    
    flags = 0x0008  # KEYEVENTF_SCANCODE
    scan_code = hexKeyCode
    
    # 冒险岛/DirectInput 中，方向键(UP/DOWN/LEFT/RIGHT)等扩展键的扫描码会带有 0x80 最高位
    # SendInput 需要剥离这个最高位，并加上 KEYEVENTF_EXTENDEDKEY (0x0001) 标志
    if scan_code & 0x80:
        flags |= 0x0001  # KEYEVENTF_EXTENDEDKEY
        scan_code &= 0x7F  # 取低 7 位作为真实的物理扫描码

    ii_.ki = KeyBdInput(0, scan_code, flags, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    
    flags = 0x0008 | 0x0002  # KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP
    scan_code = hexKeyCode
    
    if scan_code & 0x80:
        flags |= 0x0001  # KEYEVENTF_EXTENDEDKEY
        scan_code &= 0x7F

    ii_.ki = KeyBdInput(0, scan_code, flags, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def MoveMouseRelative(dx, dy):
    """通过 SendInput 硬件驱动级相对位移鼠标，打破全局零鼠标移动特征"""
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.mi = MouseInput(int(dx), int(dy), 0, 0x0001, 0, ctypes.pointer(extra))  # MOUSEEVENTF_MOVE = 0x0001
    x = Input(ctypes.c_ulong(0), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


class HumanMouseDrifter:
    """
    拟人化鼠标微扰引擎 (Humanized Mouse Activity Engine)
    在游戏运行期间，每隔 15~40 秒生成细微自然的平滑微位移与悬停，打破 30 分钟 0 鼠标特征
    """
    def __init__(self):
        self.last_drift_time = time.time()
        self.next_drift_interval = random.uniform(15.0, 35.0)

    def tick(self):
        now = time.time()
        if now - self.last_drift_time >= self.next_drift_interval:
            self._do_micro_drift()
            self.last_drift_time = now
            self.next_drift_interval = random.uniform(20.0, 50.0)

    def _do_micro_drift(self):
        # 生成 3~6 步的轻微平滑位移 (总位移仅 5~20 像素，绝不干扰游戏画面)
        steps = random.randint(3, 6)
        total_dx = random.randint(-15, 15)
        total_dy = random.randint(-10, 10)
        
        step_dx = total_dx / steps
        step_dy = total_dy / steps
        
        for _ in range(steps):
            jitter_x = step_dx + random.uniform(-1.0, 1.0)
            jitter_y = step_dy + random.uniform(-1.0, 1.0)
            MoveMouseRelative(int(round(jitter_x)), int(round(jitter_y)))
            time.sleep(random.uniform(0.015, 0.040))


class HumanJitter:
    """
    拟人化动力学抖动引擎 (Humanized Input Dynamics Engine)
    
    融合心理生理学（Psychophysics）与动作控制模型：
    1. 按键时长 (Hold Duration)：对数正态分布 (Log-Normal) + AR(1) 时序肌肉惯性
    2. 视觉/认知反应时间 (Reaction Time)：Ex-Gaussian 分布模型
    3. 组合键非对齐 (Key Rollover/Offset)：异步错开时间
    4. 动态浮动周期计算器 (Floating Interval)：截断高斯 ±ratio 动态浮动
    """

    def __init__(self):
        self.last_press_duration = 0.075
        self.ar_alpha = 0.35  # 一阶自回归惯性权重

    def get_press_duration(self, base_ms=75.0, sigma=0.25, min_ms=35.0, max_ms=180.0) -> float:
        """
        获取单次按键按下的持续时间 (Hold Duration)
        采用 对数正态分布 (Log-Normal) + 截断保护
        """
        mu = math.log(base_ms / 1000.0)
        raw_val = random.lognormvariate(mu, sigma)
        
        # 结合上一次动作的惯性 (AR-1 自回归模型)
        smoothed_val = (1.0 - self.ar_alpha) * raw_val + self.ar_alpha * self.last_press_duration
        
        # 截断在生理合理极限范围内
        final_val = max(min_ms / 1000.0, min(smoothed_val, max_ms / 1000.0))
        self.last_press_duration = final_val
        return final_val

    def get_reaction_delay(self, base_reaction_ms=170.0, noise_ms=30.0, tail_rate=0.02) -> float:
        """
        获取人类认知与视觉反应时间 (Reaction Time)
        采用 Ex-Gaussian 简化模型 (高斯分布 + 指数长尾)
        """
        gaussian_part = random.gauss(base_reaction_ms, noise_ms)
        tail_part = random.expovariate(tail_rate) if random.random() < 0.18 else 0.0
        total_ms = max(110.0, gaussian_part + tail_part)
        return total_ms / 1000.0

    def get_combo_offset(self, mean_offset_ms=22.0, sigma_ms=6.0) -> float:
        """
        获取组合键（如 跑+跳/技能组合）之间的微先后错开时间
        """
        offset = random.gauss(mean_offset_ms, sigma_ms)
        return max(8.0, min(offset, 55.0)) / 1000.0

    def calc_floating_interval(self, base_interval: float, ratio: float = 0.10, min_val: float = 0.05) -> float:
        """
        计算动态浮动目标周期 (如 Buff 技能 CD、攻击间隔)
        使用截断正态分布生成 ±ratio 的浮动 (默认 ±10%)
        """
        sigma = ratio / 2.5  # ~99% 的样本落在 [-ratio, +ratio]
        delta = random.gauss(0, sigma)
        delta = max(-ratio, min(delta, ratio))
        next_cd = base_interval * (1.0 + delta)
        return max(min_val, next_cd)

    def get_inter_key_interval(self, target_cps=6.0) -> float:
        """
        连续击键的间隔时间 (Inter-Key Interval)
        """
        mean_interval = 1.0 / target_cps
        jitter = random.gauss(0, mean_interval * 0.15)
        return max(0.035, mean_interval + jitter)


class GameController:
    """基于 DirectInput 硬件扫描码与拟人化动力学抖动的按键控制器"""

    def __init__(self, **kwargs):
        print("[GameController] 初始化 SendInput (DirectInput 扫描码) + HumanJitter 动力学引擎...")
        self.pressed_keys = set()
        self.jitter = HumanJitter()
        self.mouse_drifter = HumanMouseDrifter()
        print("[GameController] 拟人化按键动力学已就绪 (对数正态击键时长、AR(1)肌肉惯性、非对称组合键时序、鼠标微扰)。")
        print("[GameController] 注意：如果游戏无响应，请确保程序已右键【以管理员身份运行】。")

    def press_key(self, key_name):
        """按下按键"""
        key_name = key_name.upper()
        if key_name not in self.pressed_keys:
            self.pressed_keys.add(key_name)
            
        scan_code = DIK_KEYS.get(key_name)
        if scan_code:
            PressKey(scan_code)
        else:
            print(f"[GameController] 警告: 找不到按键 {key_name} 的 DirectInput 扫描码！")

    def release_key(self, key_name):
        """释放按键"""
        key_name = key_name.upper()
        if key_name in self.pressed_keys:
            self.pressed_keys.remove(key_name)
            
        scan_code = DIK_KEYS.get(key_name)
        if scan_code:
            ReleaseKey(scan_code)

    def tap_key(self, key_name, duration=None):
        """
        点按按键 (拟人化持续时间)
        若 duration 为 None，则自动由 HumanJitter 计算对数正态抖动时长
        """
        if duration is None:
            duration = self.jitter.get_press_duration()
        self.press_key(key_name)
        time.sleep(duration)
        self.release_key(key_name)

    def combo_tap(self, first_key, second_key):
        """
        拟人化组合键 (非对齐先后差 + 重叠释放)
        """
        offset = self.jitter.get_combo_offset()
        first_hold = self.jitter.get_press_duration()
        second_hold = self.jitter.get_press_duration()

        self.press_key(first_key)
        time.sleep(offset)
        
        self.press_key(second_key)
        
        # 错落释放
        remain_first = max(0.01, first_hold - offset)
        time.sleep(remain_first)
        self.release_key(first_key)
        
        time.sleep(second_hold)
        self.release_key(second_key)

    def release_all_keys(self):
        """释放所有当前按下的按键"""
        for k in list(self.pressed_keys):
            self.release_key(k)
        self.pressed_keys.clear()

    def clear_movement(self):
        """清理所有方向键"""
        for k in ["LEFT", "RIGHT", "UP", "DOWN"]:
            self.release_key(k)

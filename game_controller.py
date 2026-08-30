import time
import ctypes
import win32gui
from ctypes import wintypes

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


class GameController:
    """直接使用 SendInput (DirectInput 硬件扫描码) 的按键控制器"""

    def __init__(self, **kwargs):
        print("[GameController] 初始化 SendInput (DirectInput 扫描码) 模块...")
        self.pressed_keys = set()
        print("[GameController] 已完全放弃 G-Hub 宏！现在通过系统 API 纯净模拟按键。")
        print("[GameController] 注意：如果游戏无响应，请确保程序已右键【以管理员身份运行】。")

    def press_key(self, key_name):
        """按下按键"""
        key_name = key_name.upper()
        if key_name not in self.pressed_keys:
            fg_hwnd = win32gui.GetForegroundWindow()
            fg_title = win32gui.GetWindowText(fg_hwnd)
            print(f"[GameController] ⬇ 按下: {key_name} (当前窗口: '{fg_title}')")
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
            print(f"[GameController] ⬆ 释放: {key_name}")
            self.pressed_keys.remove(key_name)
            
        scan_code = DIK_KEYS.get(key_name)
        if scan_code:
            ReleaseKey(scan_code)

    def tap_key(self, key_name, duration=0.08):
        """点按按键"""
        self.press_key(key_name)
        time.sleep(duration)
        self.release_key(key_name)

    def release_all_keys(self):
        """释放所有当前按下的按键"""
        for k in list(self.pressed_keys):
            self.release_key(k)
        self.pressed_keys.clear()

    def clear_movement(self):
        """清理所有方向键"""
        for k in ["LEFT", "RIGHT", "UP", "DOWN"]:
            self.release_key(k)

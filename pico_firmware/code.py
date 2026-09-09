import board
import digitalio
import supervisor
import sys
import time
import usb_hid

# 1. 板载 LED 指示灯
try:
    led = digitalio.DigitalInOut(board.LED)
    led.direction = digitalio.Direction.OUTPUT
    led.value = True
    time.sleep(0.1)
    led.value = False
except Exception:
    led = None

# 2. 获取内置 USB HID 键盘设备 (Usage Page 1, Usage 6)
kbd = None
for d in usb_hid.devices:
    if d.usage_page == 0x01 and d.usage == 0x06:
        kbd = d
        break

# 标准 USB HID 键码表
HID_KEYCODES = {
    'A': 0x04, 'B': 0x05, 'C': 0x06, 'D': 0x07, 'E': 0x08, 'F': 0x09, 'G': 0x0A,
    'H': 0x0B, 'I': 0x0C, 'J': 0x0D, 'K': 0x0E, 'L': 0x0F, 'M': 0x10, 'N': 0x11,
    'O': 0x12, 'P': 0x13, 'Q': 0x14, 'R': 0x15, 'S': 0x16, 'T': 0x17, 'U': 0x18,
    'V': 0x19, 'W': 0x1A, 'X': 0x1B, 'Y': 0x1C, 'Z': 0x1D,
    '1': 0x1E, '2': 0x1F, '3': 0x20, '4': 0x21, '5': 0x22, '6': 0x23, '7': 0x24,
    '8': 0x25, '9': 0x26, '0': 0x27,
    'RETURN': 0x28, 'ENTER': 0x28, 'ESCAPE': 0x29, 'BACK': 0x2A, 'BACKSPACE': 0x2A,
    'TAB': 0x2B, 'SPACE': 0x2C,
    'MINUS': 0x2D, 'EQUALS': 0x2E, 'LBRACKET': 0x2F, 'RBRACKET': 0x30,
    'BACKSLASH': 0x31, 'SEMICOLON': 0x33, 'APOSTROPHE': 0x34, 'GRAVE': 0x35,
    'COMMA': 0x36, 'PERIOD': 0x37, 'SLASH': 0x38, 'CAPITAL': 0x39,
    'RIGHT': 0x4F, 'LEFT': 0x50, 'DOWN': 0x51, 'UP': 0x52,
    'INSERT': 0x49, 'HOME': 0x4A, 'PAGEUP': 0x4B, 'PRIOR': 0x4B,
    'DELETE': 0x4C, 'END': 0x4D, 'PAGEDOWN': 0x4E, 'NEXT': 0x4E,
    'F1': 0x3A, 'F2': 0x3B, 'F3': 0x3C, 'F4': 0x3D, 'F5': 0x3E, 'F6': 0x3F,
    'F7': 0x40, 'F8': 0x41, 'F9': 0x42, 'F10': 0x43, 'F11': 0x44, 'F12': 0x45,
}

MODIFIER_MASKS = {
    'LCTRL': 0x01, 'CTRL': 0x01, 'LCONTROL': 0x01,
    'LSHIFT': 0x02, 'SHIFT': 0x02,
    'LALT': 0x04, 'ALT': 0x04, 'LMENU': 0x04,
    'LGUI': 0x08, 'WIN': 0x08,
    'RCTRL': 0x10, 'RSHIFT': 0x20, 'RALT': 0x40, 'RGUI': 0x80
}

pressed_keys = set()
active_modifiers = 0
report = bytearray(8)

def send_report():
    if not kbd:
        print("ERR:NO_KBD")
        return
    report[0] = active_modifiers
    report[1] = 0x00
    keys = list(pressed_keys)[:6]
    for i in range(6):
        report[2 + i] = keys[i] if i < len(keys) else 0x00
    try:
        kbd.send_report(report)
    except Exception as e:
        print(f"ERR_SEND:{e}")

# 初始清空按键
send_report()

buf = ""
while True:
    if supervisor.runtime.serial_bytes_available:
        ch = sys.stdin.read(1)
        if ch == '\n' or ch == '\r':
            cmd = buf.strip()
            buf = ""
            if not cmd:
                continue

            # 心跳检测 / 设备识别
            if cmd == "PING":
                print("PONG:PICO_KEYBOARD")
                if led:
                    led.value = True
                    time.sleep(0.05)
                    led.value = False
                continue

            if cmd == "STATUS":
                devs = [f"{d.usage_page:#x}:{d.usage:#x}" for d in usb_hid.devices]
                print(f"DEVICES:{devs},KBD:{bool(kbd)}")
                continue

            # 按下指令 P:KEY
            if cmd.startswith("P:"):
                k = cmd[2:].upper()
                if k in MODIFIER_MASKS:
                    active_modifiers |= MODIFIER_MASKS[k]
                elif k in HID_KEYCODES:
                    pressed_keys.add(HID_KEYCODES[k])
                send_report()
                print(f"OK:P:{k}")
                if led:
                    led.value = True

            # 释放指令 R:KEY
            elif cmd.startswith("R:"):
                k = cmd[2:].upper()
                if k in MODIFIER_MASKS:
                    active_modifiers &= ~MODIFIER_MASKS[k]
                elif k in HID_KEYCODES:
                    pressed_keys.discard(HID_KEYCODES[k])
                send_report()
                print(f"OK:R:{k}")
                if led and not pressed_keys and not active_modifiers:
                    led.value = False

            # 全部释放指令 C:CLEAR
            elif cmd == "C:CLEAR":
                pressed_keys.clear()
                active_modifiers = 0
                send_report()
                print("OK:CLEAR")
                if led:
                    led.value = False

        else:
            buf += ch
    else:
        time.sleep(0.001)

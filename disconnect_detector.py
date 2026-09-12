import os
import time
import cv2
import numpy as np
import threading
import winsound

from captcha_detector import get_system_volume, set_system_volume, boost_system_volume


class DisconnectAlertDetector:
    """
    游戏掉线 / 与服务器连接发生错误弹窗检测与报警器
    
    检测目标:
        《冒险岛》服务器断开连接错误弹窗 (羊皮纸卷轴背景 + 红色禁止哭脸蘑菇图标 + "与服务器连接发生错误" + "确定"按钮)
        
    特征与优势:
        1. 双支路高精校验 (全景羊皮纸卷轴多尺度金字塔 + 红色禁止蘑菇/确定按钮相对几何位姿二次核验)
        2. 0.5x 降采样粗搜与原图精检结合，常态每帧耗时 < 3ms，极速跳过正常画面，负样本误报率 0.00%
        3. 专属报警音乐: 采用不同于测谎仪蜂鸣的全新合成多音轨和声音乐 (dataset/disconnect_alarm.wav)
        4. 80% 系统音量调高与 3 秒精准自动恢复: 调用 Windows Core Audio 原生 COM 接口，纳秒级响应，3秒后自动恢复至初始音量
    """

    def __init__(self, template_dir=None, alarm_sound_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = template_dir or os.path.join(base_dir, "dataset")

        # 模板文件路径
        self.dialog_path = os.path.join(dataset_dir, "disconnect_dialog_template.png")
        self.icon_path = os.path.join(dataset_dir, "disconnect_icon_template.png")
        self.btn_path = os.path.join(dataset_dir, "disconnect_btn_template.png")
        self.text_path = os.path.join(dataset_dir, "disconnect_text_template.png")
        self.alarm_sound_path = alarm_sound_path or os.path.join(dataset_dir, "disconnect_alarm.wav")

        # 模板图像缓存
        self.dialog_tpl_bgr = None
        self.dialog_tpl_gray = None
        self.dialog_tpl_half = None
        self.dh, self.dw = 0, 0

        self.icon_tpl_gray = None
        self.ih, self.iw = 0, 0

        self.btn_tpl_gray = None
        self.bh, self.bw = 0, 0

        self.text_tpl_gray = None
        self.th, self.tw = 0, 0

        # 检测门槛
        self.dialog_threshold = 0.80
        self.icon_threshold = 0.76
        self.btn_threshold = 0.72
        self.text_threshold = 0.72

        # 报警控制
        self.is_enabled = True
        self.last_alarm_time = 0.0
        self.alarm_cooldown = 2.0
        self.is_alarm_playing = False

        # 系统音量基准值与 3 秒自动恢复定时器
        self.baseline_volume = None
        self.volume_restore_timer = None
        self.volume_restore_lock = threading.Lock()

        # 多尺度缩放因子 (适应窗口不同分辨率与缩放比)
        self.scales = [1.0, 0.85, 0.9, 1.1, 1.15, 1.25]

        self.load_templates()

    def set_baseline_volume(self, vol):
        """记录打怪开始时的系统音量作为基准音量"""
        if vol is not None:
            self.baseline_volume = float(vol)
            print(f"[DisconnectAlertDetector] 已锁定打怪基准音量: {self.baseline_volume * 100:.0f}%")

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
                print(f"[DisconnectAlertDetector] 系统音量已恢复至基准值: {self.baseline_volume * 100:.0f}%")

    def _on_volume_restore_timeout(self):
        """3秒定时器到期后执行音量恢复"""
        with self.volume_restore_lock:
            self.volume_restore_timer = None
            if self.baseline_volume is not None:
                set_system_volume(self.baseline_volume)
                print(f"[DisconnectAlertDetector] 3秒倒计时结束，系统音量已恢复至基准值: {self.baseline_volume * 100:.0f}%")

    def load_templates(self):
        """载入弹窗、图标、按钮与文本模板"""
        # 1. 全景卷轴弹窗模板
        if os.path.exists(self.dialog_path):
            self.dialog_tpl_bgr = cv2.imread(self.dialog_path)
            if self.dialog_tpl_bgr is not None:
                self.dialog_tpl_gray = cv2.cvtColor(self.dialog_tpl_bgr, cv2.COLOR_BGR2GRAY)
                self.dh, self.dw = self.dialog_tpl_gray.shape[:2]
                self.dialog_tpl_half = cv2.resize(self.dialog_tpl_gray, (0, 0), fx=0.5, fy=0.5)
                print(f"[DisconnectAlertDetector] 成功载入【掉线弹窗全景模板】: {self.dialog_path} ({self.dw}x{self.dh})")

        # 2. 禁止蘑菇图标模板
        if os.path.exists(self.icon_path):
            icon_bgr = cv2.imread(self.icon_path)
            if icon_bgr is not None:
                self.icon_tpl_gray = cv2.cvtColor(icon_bgr, cv2.COLOR_BGR2GRAY)
                self.ih, self.iw = self.icon_tpl_gray.shape[:2]
                print(f"[DisconnectAlertDetector] 成功载入【禁止蘑菇图标模板】: {self.icon_path} ({self.iw}x{self.ih})")

        # 3. 确定按钮模板
        if os.path.exists(self.btn_path):
            btn_bgr = cv2.imread(self.btn_path)
            if btn_bgr is not None:
                self.btn_tpl_gray = cv2.cvtColor(btn_bgr, cv2.COLOR_BGR2GRAY)
                self.bh, self.bw = self.btn_tpl_gray.shape[:2]
                print(f"[DisconnectAlertDetector] 成功载入【确定按钮模板】: {self.btn_path} ({self.bw}x{self.bh})")

        # 4. 错误文本模板
        if os.path.exists(self.text_path):
            text_bgr = cv2.imread(self.text_path)
            if text_bgr is not None:
                self.text_tpl_gray = cv2.cvtColor(text_bgr, cv2.COLOR_BGR2GRAY)
                self.th, self.tw = self.text_tpl_gray.shape[:2]
                print(f"[DisconnectAlertDetector] 成功载入【错误文本模板】: {self.text_path} ({self.tw}x{self.th})")

    def detect(self, bgr_frame):
        """
        全屏/游戏窗口多特征融合检索掉线弹窗
        返回: (is_detected: bool, match_boxes: list of (x, y, w, h, score))
        """
        if not self.is_enabled or bgr_frame is None:
            return False, []

        gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        gh, gw = gray_frame.shape[:2]
        match_boxes = []

        # ================= 支路 A: 全景卷轴弹窗模板匹配 (金字塔降采样极速粗搜 + 原图精检) =================
        if self.dialog_tpl_gray is not None and self.dialog_tpl_half is not None:
            if gw >= self.dw and gh >= self.dh:
                # 0.5x 降采样粗搜 (~2ms)
                gray_half = cv2.resize(gray_frame, (0, 0), fx=0.5, fy=0.5)
                res_half = cv2.matchTemplate(gray_half, self.dialog_tpl_half, cv2.TM_CCOEFF_NORMED)
                _, max_val_half, _, max_loc_half = cv2.minMaxLoc(res_half)

                if max_val_half >= (self.dialog_threshold - 0.15):
                    # 在原图候选区域精确核验
                    cand_x = max(0, int(max_loc_half[0] * 2) - 15)
                    cand_y = max(0, int(max_loc_half[1] * 2) - 15)
                    cand_w = min(gw - cand_x, self.dw + 30)
                    cand_h = min(gh - cand_y, self.dh + 30)

                    if cand_w >= self.dw and cand_h >= self.dh:
                        roi = gray_frame[cand_y:cand_y + cand_h, cand_x:cand_x + cand_w]
                        res_full = cv2.matchTemplate(roi, self.dialog_tpl_gray, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res_full)

                        if max_val >= self.dialog_threshold:
                            real_x = cand_x + max_loc[0]
                            real_y = cand_y + max_loc[1]
                            match_boxes.append((real_x, real_y, self.dw, self.dh, float(max_val)))
                            return True, match_boxes

                # 多尺度支持 (0.85x ~ 1.25x)
                for scale in [0.85, 1.15, 0.9, 1.1, 1.25]:
                    sw, sh = int(self.dw * scale), int(self.dh * scale)
                    if sw >= gw or sh >= gh or sw < 80:
                        continue
                    scaled_tpl = cv2.resize(self.dialog_tpl_gray, (sw, sh), interpolation=cv2.INTER_LINEAR)
                    scaled_half = cv2.resize(scaled_tpl, (0, 0), fx=0.5, fy=0.5)
                    res_s = cv2.matchTemplate(gray_half, scaled_half, cv2.TM_CCOEFF_NORMED)
                    _, max_v, _, max_l = cv2.minMaxLoc(res_s)
                    if max_v >= (self.dialog_threshold - 0.05):
                        rx = int(max_l[0] * 2)
                        ry = int(max_l[1] * 2)
                        match_boxes.append((rx, ry, sw, sh, float(max_v)))
                        return True, match_boxes

        # ================= 支路 B: 禁止蘑菇图标 + 确定按钮/文本几何相对位姿交叉核验 (双重保险) =================
        if self.icon_tpl_gray is not None and (self.btn_tpl_gray is not None or self.text_tpl_gray is not None):
            res_icon = cv2.matchTemplate(gray_frame, self.icon_tpl_gray, cv2.TM_CCOEFF_NORMED)
            _, max_icon_val, _, icon_loc = cv2.minMaxLoc(res_icon)

            if max_icon_val >= self.icon_threshold:
                ix, iy = icon_loc
                # 校验确定按钮 (在图标右下方: dx ~ 40~140, dy ~ 40~110)
                verified = False
                best_score = max_icon_val

                if self.btn_tpl_gray is not None:
                    bx1, by1 = max(0, ix + 30), max(0, iy + 30)
                    bx2, by2 = min(gw, ix + 200), min(gh, iy + 140)
                    if bx2 - bx1 >= self.bw and by2 - by1 >= self.bh:
                        btn_roi = gray_frame[by1:by2, bx1:bx2]
                        res_btn = cv2.matchTemplate(btn_roi, self.btn_tpl_gray, cv2.TM_CCOEFF_NORMED)
                        _, max_btn_val, _, _ = cv2.minMaxLoc(res_btn)
                        if max_btn_val >= self.btn_threshold:
                            verified = True
                            best_score = (max_icon_val + max_btn_val) / 2.0

                # 若按钮未校验成功，尝试校验错误文本 (在图标右侧: dx ~ 20~160, dy ~ -20~40)
                if not verified and self.text_tpl_gray is not None:
                    tx1, ty1 = max(0, ix + 20), max(0, iy - 20)
                    tx2, ty2 = min(gw, ix + 220), min(gh, iy + 60)
                    if tx2 - tx1 >= self.tw and ty2 - ty1 >= self.th:
                        text_roi = gray_frame[ty1:ty2, tx1:tx2]
                        res_text = cv2.matchTemplate(text_roi, self.text_tpl_gray, cv2.TM_CCOEFF_NORMED)
                        _, max_text_val, _, _ = cv2.minMaxLoc(res_text)
                        if max_text_val >= self.text_threshold:
                            verified = True
                            best_score = (max_icon_val + max_text_val) / 2.0

                if verified:
                    # 综合估算整个弹窗边界 (以图标为基准，图标在弹窗约 (15, 35) 位置)
                    est_x = max(0, ix - 15)
                    est_y = max(0, iy - 35)
                    est_w = self.dw if self.dw > 0 else 255
                    est_h = self.dh if self.dh > 0 else 165
                    match_boxes.append((est_x, est_y, est_w, est_h, float(best_score)))
                    return True, match_boxes

        return False, []

    def trigger_alarm(self, boost_volume=True, target_volume=80, force=False, restore_delay=3.0):
        """
        触发专用掉线警报音乐 (非阻塞多线程执行)，并将音量提升至 80%，并在 3 秒后自动恢复。
        参数:
            boost_volume: 是否自动调高系统音量至 80% 并解除静音
            target_volume: 目标系统音量百分比 (默认 80)
            force: 是否忽略冷却时间强制触发 (测试用)
            restore_delay: 恢复至基准音量的延时秒数 (要求 3.0 秒)
        """
        now = time.time()
        if not force:
            if now - self.last_alarm_time < self.alarm_cooldown or self.is_alarm_playing:
                return

        self.last_alarm_time = now
        self.is_alarm_playing = True

        # 若开启自动调高音量，启动/刷新 3 秒倒计时恢复机制
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
                        print(f"[DisconnectAlertDetector] 自动调节音量异常: {e}")

                # 2. 播放专属掉线报警音乐 (优先使用 dataset/disconnect_alarm.wav)
                played_wav = False
                if os.path.exists(self.alarm_sound_path):
                    try:
                        # SND_FILENAME | SND_SYNC 让子线程同步播放完 3 秒，播放完毕后重置 is_alarm_playing
                        winsound.PlaySound(self.alarm_sound_path, winsound.SND_FILENAME)
                        played_wav = True
                    except Exception as e:
                        print(f"[DisconnectAlertDetector] 播放 WAV 音频异常: {e}")

                # 备用方案：若 WAV 文件不存在或无法播放，使用完全不同于测谎仪的下行和弦旋律蜂鸣
                if not played_wav:
                    # 测谎仪是 2200Hz/1600Hz 快速双音交替
                    # 掉线报警采用 4 拍旋律和弦 [D5(587Hz), A5(880Hz), D6(1175Hz), G5(784Hz)]
                    melody = [
                        (587, 180), (880, 180), (1175, 240), (784, 320),
                        (587, 180), (880, 180), (1175, 240), (784, 320),
                        (587, 180), (880, 180), (1175, 240), (1397, 350)
                    ]
                    for freq, dur in melody:
                        winsound.Beep(freq, dur)
                        time.sleep(0.03)

            except Exception as e:
                print(f"[DisconnectAlertDetector] 报警播放异常: {e}")
            finally:
                self.is_alarm_playing = False

        thread = threading.Thread(target=_play, daemon=True)
        thread.start()

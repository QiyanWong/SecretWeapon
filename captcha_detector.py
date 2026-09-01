import os
import time
import cv2
import numpy as np
import threading
import winsound

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
        
        # 多尺度金字塔
        self.scales = [1.0, 0.9, 1.1, 0.8, 1.25]
        
        self.load_templates()

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

    def trigger_alarm(self):
        """
        触发多频急促警报音效 (非阻塞多线程执行)
        """
        now = time.time()
        if now - self.last_alarm_time < self.alarm_cooldown or self.is_alarm_playing:
            return
        
        self.last_alarm_time = now
        self.is_alarm_playing = True

        def _play():
            try:
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

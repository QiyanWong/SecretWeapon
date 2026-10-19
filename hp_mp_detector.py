import os
import cv2
import numpy as np


class HpMpDetector:
    """
    2D 游戏角色生命值 (HP) 与魔法值 (MP) 毫秒级高精度视觉监测器
    基于状态栏模板定位 + HSV 动态连续填充扫描算法
    """

    def __init__(self, template_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if template_path is None:
            template_path = os.path.join(base_dir, "dataset", "hp_tmpl.png")
        self.template_path = template_path

        self.template = None
        if os.path.exists(self.template_path):
            self.template = cv2.imread(self.template_path)

        # 槽位相对于 HP 模板左上角匹配点的偏移与尺寸 (基于基准分辨率标准测量)
        # HP 槽内部有效区域
        self.hp_offset_x = 6
        self.hp_offset_y = 24
        self.bar_width = 247
        self.bar_height = 23

        # MP 槽内部有效区域
        self.mp_offset_x = 258
        self.mp_offset_y = 24

        # 缓存与防抖
        self.cached_loc = None  # (match_x, match_y)
        self.miss_count = 0
        self.last_hp = None
        self.last_mp = None
        self.is_detected = False

    def _calc_bar_fill_ratio(self, mask: np.ndarray) -> float:
        """
        计算条形进度条从左往右的连续有效颜色填充比例
        通过列像素统计 + 连续性断点扫描，有效剔除右侧噪点与字符重叠影响
        """
        h, w = mask.shape[:2]
        if w == 0 or h == 0:
            return 0.0

        # 每列只要有 >= 3 个有效颜色像素，即视为该列存在填充
        col_has_color = np.sum(mask, axis=0) >= 3

        max_x = 0
        consecutive_gap = 0
        for x, has in enumerate(col_has_color):
            if has:
                max_x = x + 1
                consecutive_gap = 0
            else:
                consecutive_gap += 1
                # 连续超过 3 列空白，说明能量条填充在此处终止
                if consecutive_gap > 3:
                    break

        ratio = float(max_x) / float(w)
        return max(0.0, min(1.0, ratio))

    def detect(self, game_frame: np.ndarray):
        """
        从游戏帧中识别当前 HP 与 MP 百分比
        返回: (hp_pct, mp_pct, is_detected, debug_info)
        """
        if game_frame is None or game_frame.size == 0 or self.template is None:
            return self.last_hp, self.last_mp, False, None

        frame_h, frame_w = game_frame.shape[:2]
        if frame_h < 100 or frame_w < 300:
            return self.last_hp, self.last_mp, False, None

        # 1. 限制搜索区域为底部 180 像素 (2D Game 状态栏通常固定在底部)
        search_top = max(0, frame_h - 180)
        search_roi = game_frame[search_top:frame_h, :]

        match_x, match_y = None, None
        best_score = 0.0

        # 2. 优先在缓存坐标的局部小窗口内快速校验 (耗时 < 0.2ms)
        if self.cached_loc is not None:
            cx, cy = self.cached_loc
            rx1 = max(0, cx - 25)
            rx2 = min(frame_w, cx + 50)
            ry1 = max(0, (cy - search_top) - 15)
            ry2 = min(search_roi.shape[0], (cy - search_top) + 25)

            if rx2 - rx1 >= self.template.shape[1] and ry2 - ry1 >= self.template.shape[0]:
                sub_search = search_roi[ry1:ry2, rx1:rx2]
                res_sub = cv2.matchTemplate(sub_search, self.template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res_sub)
                if max_val >= 0.85:
                    match_x = rx1 + max_loc[0]
                    match_y = search_top + ry1 + max_loc[1]
                    best_score = max_val
                    self.cached_loc = (match_x, match_y)
                    self.miss_count = 0

        # 3. 若缓存未命中，进行底部全区域模板匹配
        if match_x is None:
            res = cv2.matchTemplate(search_roi, self.template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= 0.80:
                match_x = max_loc[0]
                match_y = search_top + max_loc[1]
                best_score = max_val
                self.cached_loc = (match_x, match_y)
                self.miss_count = 0
            else:
                self.miss_count += 1
                if self.miss_count > 10:
                    self.cached_loc = None
                self.is_detected = False
                return self.last_hp, self.last_mp, False, None

        self.is_detected = True

        # 4. 提取 HP / MP 槽区域
        hp_x = match_x + self.hp_offset_x
        hp_y = match_y + self.hp_offset_y
        mp_x = match_x + self.mp_offset_x
        mp_y = match_y + self.mp_offset_y

        bw = self.bar_width
        bh = self.bar_height

        # 越界保护
        if (hp_x + bw > frame_w or hp_y + bh > frame_h or
            mp_x + bw > frame_w or mp_y + bh > frame_h):
            return self.last_hp, self.last_mp, False, None

        hp_crop = game_frame[hp_y:hp_y + bh, hp_x:hp_x + bw]
        mp_crop = game_frame[mp_y:mp_y + bh, mp_x:mp_x + bw]

        # 5. 复合颜色空间红色掩膜 (HP: 强红通道优势 + HSV 饱和度过滤)
        hp_hsv = cv2.cvtColor(hp_crop, cv2.COLOR_BGR2HSV)
        hp_b = hp_crop[:, :, 0].astype(int)
        hp_g = hp_crop[:, :, 1].astype(int)
        hp_r = hp_crop[:, :, 2].astype(int)
        mask_red = (
            ((hp_hsv[:, :, 0] <= 12) | (hp_hsv[:, :, 0] >= 168)) &
            (hp_hsv[:, :, 1] >= 50) &
            (hp_r > hp_g + 20) &
            (hp_r > hp_b + 20)
        )
        raw_hp = self._calc_bar_fill_ratio(mask_red)

        # 6. 复合颜色空间蓝色掩膜 (MP: 强蓝通道优势 + HSV 蓝色色调区间)
        mp_hsv = cv2.cvtColor(mp_crop, cv2.COLOR_BGR2HSV)
        mp_b = mp_crop[:, :, 0].astype(int)
        mp_r = mp_crop[:, :, 2].astype(int)
        mask_blue = (
            (mp_hsv[:, :, 0] >= 85) & (mp_hsv[:, :, 0] <= 135) &
            (mp_hsv[:, :, 1] >= 40) &
            (mp_b > mp_r + 15)
        )
        raw_mp = self._calc_bar_fill_ratio(mask_blue)

        # 7. 智能防抖与低血极速响应策略 (安全第一，掉血即时响应，回血平滑过渡)
        if self.last_hp is None:
            self.last_hp = raw_hp
        else:
            # 若瞬间突变为 0% 且上一帧在 15% 以上，很可能是受击全屏白光/受击无敌单帧闪烁，做单次防抖
            if raw_hp == 0.0 and self.last_hp > 0.15 and not getattr(self, '_hp_zero_flag', False):
                self._hp_zero_flag = True
            else:
                self._hp_zero_flag = False
                self.last_hp = raw_hp

        if self.last_mp is None:
            self.last_mp = raw_mp
        else:
            self.last_mp = raw_mp

        exp_x = match_x + 510
        exp_y = match_y - 2
        exp_w = 245
        exp_h = 24

        debug_info = {
            "match_loc": (match_x, match_y),
            "match_score": best_score,
            "hp_rect": (hp_x, hp_y, bw, bh),
            "mp_rect": (mp_x, mp_y, bw, bh),
            "exp_rect": (exp_x, exp_y, exp_w, exp_h),
            "raw_hp": raw_hp,
            "raw_mp": raw_mp,
        }

        return self.last_hp, self.last_mp, True, debug_info

    def draw_overlay(self, frame: np.ndarray, hp_pct: float, mp_pct: float, debug_info: dict):
        """在画面上直观绘制 HP / MP 监控指示边框与实时百分比浮字"""
        if frame is None or debug_info is None:
            return

        hp_x, hp_y, bw, bh = debug_info["hp_rect"]
        mp_x, mp_y, _, _ = debug_info["mp_rect"]

        # HP 区域高亮边框与标签 (鲜艳红色)
        cv2.rectangle(frame, (hp_x, hp_y), (hp_x + bw, hp_y + bh), (0, 0, 255), 1)
        hp_text = f"HP {hp_pct * 100:.1f}%"
        cv2.putText(frame, hp_text, (hp_x, max(12, hp_y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)

        # MP 区域高亮边框与标签 (明亮蓝色)
        cv2.rectangle(frame, (mp_x, mp_y), (mp_x + bw, mp_y + bh), (255, 200, 0), 1)
        mp_text = f"MP {mp_pct * 100:.1f}%"
        cv2.putText(frame, mp_text, (mp_x, max(12, mp_y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 200, 0), 1, cv2.LINE_AA)

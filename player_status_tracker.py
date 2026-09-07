"""
player_status_tracker.py - 角色状态、生命守护与收益统计引擎

负责：
1. 底部状态栏视觉锚定（基于 HP 图标模板匹配，毫秒级定位）。
2. HP / MP 槽颜色占比扫描（0.1ms 级运算，无需 OCR，精准测算 0.0% ~ 100.0%）。
3. 自动补血/补蓝执行器（阈值判定 + 防抖保护 + 独立调用 GameController）。
4. EXP 百分比提取与效率（EXP/小时）测速。
5. 药水消耗与背包金币统计计算器（支持独立重置、药水单价核算、纯利润评估）。
"""

import os
import time
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class PlayerStatusTracker:
    def __init__(self, game_controller=None):
        self.game_controller = game_controller

        # 状态总开关与打怪解耦
        self.enabled = False

        # HP 喝药配置
        self.hp_threshold = 70.0      # 默认低于 70% 喝血
        self.hp_key = "Delete"        # 喝血按键
        self.hp_potion_price = 320    # 单价 (金币)
        self.hp_cd = 0.5              # 喝药冷却 (秒)
        self.last_hp_pot_time = 0.0

        # MP 喝药配置
        self.mp_threshold = 50.0      # 默认低于 50% 喝蓝
        self.mp_key = "PageDown"      # 喝蓝按键
        self.mp_potion_price = 200    # 单价 (金币)
        self.mp_cd = 0.5              # 喝药冷却 (秒)
        self.last_mp_pot_time = 0.0

        # 当前实时检测到的百分比
        self.current_hp_pct = 100.0
        self.current_mp_pct = 100.0
        self.current_exp_pct = 0.0
        self.current_exp_abs = 0

        # 锚点与相对布局缓存 (hp_icon_x, hp_icon_y)
        self.hp_anchor = None
        self.is_anchor_locked = False
        self.last_status_conf = 0.0
        self.last_anchor_search_time = 0.0

        # 快捷栏与背包自定义 ROI (可在界面上手动框选或自动对齐)
        self.quickslot_anchor = None
        self.last_quickslot_conf = 0.0
        self.last_quickslot_search_time = 0.0
        self.quickslot_hp_roi = None  # (x, y, w, h)
        self.quickslot_mp_roi = None  # (x, y, w, h)
        self.inventory_meso_roi = None  # (x, y, w, h)
        self.show_debug_overlay = True

        # 统计数据：药水
        self.initial_hp_count = None
        self.current_hp_count = 0
        self.used_hp_potions = 0

        self.initial_mp_count = None
        self.current_mp_count = 0
        self.used_mp_potions = 0

        # 统计数据：EXP
        self.exp_start_time = time.time()
        self.initial_exp_pct = None
        self.initial_exp_abs = None
        self.gained_exp_abs = 0
        self.gained_exp_pct = 0.0
        self.exp_per_hour = 0.0

        # 统计数据：金币
        self.meso_start_time = time.time()
        self.initial_meso = None
        self.current_meso = 0
        self.gained_meso = 0
        self.meso_per_hour = 0.0

        # 加载匹配模板 (支持优先从 dataset/assets 读取)
        self.tmpl_hp = self._load_template("hp_icon.png")
        self.tmpl_quickslot = self._load_template("quickslot_tmpl.png")

    def _load_template(self, filename):
        candidates = [
            os.path.join(BASE_DIR, "dataset", "assets", filename),
            os.path.join(BASE_DIR, "scratch", filename),
            os.path.join(BASE_DIR, "synthetic_assets", filename),
        ]
        for p in candidates:
            if os.path.exists(p):
                im = cv2.imread(p)
                if im is not None:
                    return im
        return None

    # ========================== 独立重置接口 ==========================
    def reset_exp_stats(self):
        """独立重置经验统计"""
        self.exp_start_time = time.time()
        self.initial_exp_pct = self.current_exp_pct
        self.initial_exp_abs = self.current_exp_abs
        self.gained_exp_abs = 0
        self.gained_exp_pct = 0.0
        self.exp_per_hour = 0.0

    def reset_meso_stats(self):
        """独立重置金币统计"""
        self.meso_start_time = time.time()
        self.initial_meso = self.current_meso
        self.gained_meso = 0
        self.meso_per_hour = 0.0

    def reset_potion_stats(self):
        """独立重置药水消耗统计"""
        self.initial_hp_count = self.current_hp_count
        self.used_hp_potions = 0
        self.initial_mp_count = self.current_mp_count
        self.used_mp_potions = 0

    def reset_all_stats(self):
        """一键全部重置"""
        self.reset_exp_stats()
        self.reset_meso_stats()
        self.reset_potion_stats()

    # ========================== 状态条视觉定位与解析 ==========================
    def locate_status_bar(self, frame_bgr):
        """
        在游戏画面底部 250px 区域内匹配 'HP' 图标，实现全分辨率自适应
        """
        if frame_bgr is None:
            return None

        h, w = frame_bgr.shape[:2]
        scan_h = min(250, h)
        roi = frame_bgr[h - scan_h : h, :]

        if self.tmpl_hp is None:
            self.tmpl_hp = self._load_template("hp_icon.png")

        if self.tmpl_hp is not None:
            res = cv2.matchTemplate(roi, self.tmpl_hp, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            self.last_status_conf = float(max_val)
            if max_val >= 0.65:
                abs_x = max_loc[0]
                abs_y = h - scan_h + max_loc[1]
                self.hp_anchor = (abs_x, abs_y)
                self.is_anchor_locked = True
                return self.hp_anchor

        self.is_anchor_locked = False
        return self.hp_anchor

    def locate_quickslot(self, frame_bgr):
        """
        在游戏画面右下角匹配快捷栏位置
        """
        if frame_bgr is None:
            return None

        h, w = frame_bgr.shape[:2]
        scan_h = min(250, h)
        # 快捷栏永远位于右半侧底部
        roi_right = frame_bgr[h - scan_h : h, w // 2 : w]

        if self.tmpl_quickslot is None:
            self.tmpl_quickslot = self._load_template("quickslot_tmpl.png")

        if self.tmpl_quickslot is not None:
            res = cv2.matchTemplate(roi_right, self.tmpl_quickslot, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            self.last_quickslot_conf = float(max_val)
            if max_val >= 0.70:
                abs_x = (w // 2) + max_loc[0]
                abs_y = h - scan_h + max_loc[1]
                self.quickslot_anchor = (abs_x, abs_y)
                return self.quickslot_anchor

        return self.quickslot_anchor

    def update_hp_mp_exp(self, frame_bgr):
        """
        核心 0.2ms 级视觉扫描：根据 HP 图标相对位置计算 HP/MP/EXP 百分比
        """
        if frame_bgr is None:
            return self.current_hp_pct, self.current_mp_pct, self.current_exp_pct

        now = time.time()
        if self.hp_anchor is None or not self.is_anchor_locked or (now - self.last_anchor_search_time > 2.0):
            self.locate_status_bar(frame_bgr)
            self.last_anchor_search_time = now

        if self.quickslot_anchor is None or (now - self.last_quickslot_search_time > 3.0):
            self.locate_quickslot(frame_bgr)
            self.last_quickslot_search_time = now

        if self.hp_anchor is None or not self.is_anchor_locked:
            return self.current_hp_pct, self.current_mp_pct, self.current_exp_pct

        hx, hy = self.hp_anchor
        h, w = frame_bgr.shape[:2]

        try:
            # 1. 截取 HP 槽 (内部长条: y: +27 ~ +34, x: +2 ~ +161)
            # 槽有效宽度为 159 像素
            hp_y1, hp_y2 = hy + 27, hy + 34
            hp_x1, hp_x2 = hx + 2, hx + 161
            if 0 <= hp_y1 < hp_y2 <= h and 0 <= hp_x1 < hp_x2 <= w:
                hp_bar = frame_bgr[hp_y1:hp_y2, hp_x1:hp_x2]
                is_red = (hp_bar[:, :, 2] > 160) & (hp_bar[:, :, 1] < 60) & (hp_bar[:, :, 0] < 60)
                red_cols = np.sum(np.any(is_red, axis=0))
                self.current_hp_pct = max(0.0, min(100.0, (red_cols / 159.0) * 100.0))

            # 2. 截取 MP 槽 (内部长条: x: +164 ~ +323)
            # 槽有效宽度为 159 像素
            mp_y1, mp_y2 = hy + 27, hy + 34
            mp_x1, mp_x2 = hx + 164, hx + 323
            if 0 <= mp_y1 < mp_y2 <= h and 0 <= mp_x1 < mp_x2 <= w:
                mp_bar = frame_bgr[mp_y1:mp_y2, mp_x1:mp_x2]
                is_blue = (mp_bar[:, :, 0] > 165) & (mp_bar[:, :, 1] > 80) & (mp_bar[:, :, 2] < 60)
                blue_cols = np.sum(np.any(is_blue, axis=0))
                self.current_mp_pct = max(0.0, min(100.0, (blue_cols / 159.0) * 100.0))

            # 3. 截取 EXP 槽 (内部长条: x: +334 ~ +507)
            # 槽有效宽度为 173 像素
            exp_y1, exp_y2 = hy + 27, hy + 34
            exp_x1, exp_x2 = hx + 334, hx + 507
            if 0 <= exp_y1 < exp_y2 <= h and 0 <= exp_x1 < exp_x2 <= w:
                exp_bar = frame_bgr[exp_y1:exp_y2, exp_x1:exp_x2]
                is_yellow = (exp_bar[:, :, 1] > 160) & (exp_bar[:, :, 2] > 160) & (exp_bar[:, :, 0] < 90)
                yellow_cols = np.sum(np.any(is_yellow, axis=0))
                self.current_exp_pct = max(0.0, min(100.0, (yellow_cols / 173.0) * 100.0))

                # 更新经验收益效率
                if self.initial_exp_pct is None:
                    self.initial_exp_pct = self.current_exp_pct
                self.gained_exp_pct = max(0.0, self.current_exp_pct - self.initial_exp_pct)
                elapsed_h = max(0.001, (now - self.exp_start_time) / 3600.0)
                self.exp_per_hour = self.gained_exp_pct / elapsed_h

        except Exception as e:
            pass

        return self.current_hp_pct, self.current_mp_pct, self.current_exp_pct

    # ========================== 实时可视化 Debug Overlay ==========================
    def draw_debug_overlay(self, frame_bgr):
        """
        在主画面上以高质量半透明 HUD 与矩形标记绘制当前状态守护与收益识别详情
        """
        if frame_bgr is None or not self.show_debug_overlay:
            return

        h, w = frame_bgr.shape[:2]

        # 1. 绘制顶部 HUD 综合诊断仪表盘 (放置在左上方或避开小地图)
        hud_w, hud_h = 440, 125
        hud_x, hud_y = 20, 15
        if hud_x + hud_w <= w and hud_y + hud_h <= h:
            hud_roi = frame_bgr[hud_y : hud_y + hud_h, hud_x : hud_x + hud_w]
            overlay = np.zeros_like(hud_roi)
            overlay[:] = (18, 18, 28)  # 深色半透明底色
            cv2.addWeighted(overlay, 0.78, hud_roi, 0.22, 0, hud_roi)
            cv2.rectangle(frame_bgr, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (80, 80, 120), 1)

            mode_str = "ACTIVE (Drink ON)" if self.enabled else "MONITORING (Drink OFF)"
            mode_color = (120, 255, 120) if self.enabled else (100, 220, 255)

            lock_str = f"LOCKED ({self.last_status_conf:.2f})" if self.is_anchor_locked else f"LOST ({self.last_status_conf:.2f})"
            lock_color = (100, 255, 100) if self.is_anchor_locked else (80, 80, 255)

            cv2.putText(frame_bgr, "STATUS MONITOR HUD (DEBUG)", (hud_x + 10, hud_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 220), 2)
            cv2.putText(frame_bgr, f"Mode: {mode_str}", (hud_x + 10, hud_y + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, mode_color, 1)
            cv2.putText(frame_bgr, f"Status Bar: {lock_str}", (hud_x + 235, hud_y + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, lock_color, 1)

            cv2.putText(frame_bgr, f"HP: {self.current_hp_pct:.1f}% (Thresh<={int(self.hp_threshold)}%, Key:{self.hp_key})",
                        (hud_x + 10, hud_y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 255), 1)
            cv2.putText(frame_bgr, f"MP: {self.current_mp_pct:.1f}% (Thresh<={int(self.mp_threshold)}%, Key:{self.mp_key})",
                        (hud_x + 10, hud_y + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 50), 1)

            meso_roi_str = f"X={self.inventory_meso_roi[0]},Y={self.inventory_meso_roi[1]}" if self.inventory_meso_roi else "Not Selected"
            cv2.putText(frame_bgr, f"EXP: {self.current_exp_pct:.2f}% (+{self.gained_exp_pct:.2f}%) | Meso ROI: {meso_roi_str}",
                        (hud_x + 10, hud_y + 106), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 255), 1)

        # 2. 状态栏标注 (若锁定则绘制各条框选，若未锁定则绘制告警横幅)
        if self.is_anchor_locked and self.hp_anchor is not None:
            hx, hy = self.hp_anchor
            # 状态栏整体框
            cv2.rectangle(frame_bgr, (hx - 4, hy - 4), (hx + 514, hy + 38), (255, 200, 0), 2)
            cv2.putText(frame_bgr, f"STATUS BAR [Conf: {self.last_status_conf:.2f}]",
                        (hx, max(20, hy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 0), 2)

            # HP 槽高亮
            hp_y1, hp_y2 = hy + 27, hy + 34
            hp_x1, hp_x2 = hx + 2, hx + 161
            cv2.rectangle(frame_bgr, (hp_x1, hp_y1), (hp_x2, hp_y2), (0, 0, 255), 2)
            cv2.putText(frame_bgr, f"HP: {self.current_hp_pct:.1f}%",
                        (hp_x1, max(20, hp_y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            # MP 槽高亮
            mp_y1, mp_y2 = hy + 27, hy + 34
            mp_x1, mp_x2 = hx + 164, hx + 323
            cv2.rectangle(frame_bgr, (mp_x1, mp_y1), (mp_x2, mp_y2), (255, 140, 0), 2)
            cv2.putText(frame_bgr, f"MP: {self.current_mp_pct:.1f}%",
                        (mp_x1, max(20, mp_y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 140, 0), 2)

            # EXP 槽高亮
            exp_y1, exp_y2 = hy + 27, hy + 34
            exp_x1, exp_x2 = hx + 334, hx + 507
            cv2.rectangle(frame_bgr, (exp_x1, exp_y1), (exp_x2, exp_y2), (0, 255, 255), 2)
            cv2.putText(frame_bgr, f"EXP: {self.current_exp_pct:.2f}%",
                        (exp_x1, max(20, exp_y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
        else:
            # 告警提示：未锁定状态栏
            warn_msg = f"[STATUS BAR LOST] Conf: {self.last_status_conf:.2f} < 0.65"
            cv2.putText(frame_bgr, warn_msg, (w // 2 - 220, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        # 3. 快捷栏药水槽位标注 (若已识别)
        if self.quickslot_anchor is not None:
            qx, qy = self.quickslot_anchor
            cv2.rectangle(frame_bgr, (qx - 15, qy - 10), (qx + 190, qy + 115), (0, 220, 100), 2)
            cv2.putText(frame_bgr, f"QUICKSLOT [Conf: {self.last_quickslot_conf:.2f}]",
                        (qx - 15, max(20, qy - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 100), 2)

            # Del (血药槽) 框选
            cv2.rectangle(frame_bgr, (qx + 35, qy + 45), (qx + 85, qy + 95), (0, 165, 255), 1)
            cv2.putText(frame_bgr, "Del[HP]", (qx + 35, qy + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 165, 255), 1)

            # End (蓝药槽) 框选
            cv2.rectangle(frame_bgr, (qx + 88, qy + 45), (qx + 138, qy + 95), (255, 200, 0), 1)
            cv2.putText(frame_bgr, "End[MP]", (qx + 88, qy + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 200, 0), 1)

        # 4. 背包金币区域 ROI 标注 (若用户已框选)
        if self.inventory_meso_roi is not None:
            mx, my, mw, mh = self.inventory_meso_roi
            cv2.rectangle(frame_bgr, (mx, my), (mx + mw, my + mh), (0, 215, 255), 2)
            cv2.putText(frame_bgr, f"MESO ROI [{self.current_meso:,}]",
                        (mx, max(20, my - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 2)

    # ========================== 自动补血/补蓝执行器 ==========================
    def check_and_drink_potions(self):
        """
        在 process_loop 中每帧高频调用。
        只有在 self.enabled == True 时才触发按键，独立于自动打怪。
        """
        if not self.enabled or self.game_controller is None:
            return None

        now = time.time()
        triggered = []

        # 1. 检查 HP 保护
        if self.current_hp_pct <= self.hp_threshold:
            if (now - self.last_hp_pot_time) >= self.hp_cd:
                self.last_hp_pot_time = now
                self._send_key(self.hp_key)
                self.used_hp_potions += 1
                triggered.append(f"HP {self.current_hp_pct:.1f}% <= {self.hp_threshold}% 喝血[{self.hp_key}]")

        # 2. 检查 MP 保护
        if self.current_mp_pct <= self.mp_threshold:
            if (now - self.last_mp_pot_time) >= self.mp_cd:
                self.last_mp_pot_time = now
                self._send_key(self.mp_key)
                self.used_mp_potions += 1
                triggered.append(f"MP {self.current_mp_pct:.1f}% <= {self.mp_threshold}% 喝蓝[{self.mp_key}]")

        return triggered if triggered else None

    def _send_key(self, key_name):
        """调用 GameController 注入按键"""
        if not self.game_controller:
            return
        key = key_name.strip()
        # 兼容常见按键名映射
        key_map = {
            "delete": "Delete",
            "del": "Delete",
            "pagedown": "PageDown",
            "pgdn": "PageDown",
            "pageup": "PageUp",
            "pgup": "PageUp",
            "insert": "Insert",
            "ins": "Insert",
            "home": "Home",
            "end": "End",
        }
        mapped_key = key_map.get(key.lower(), key)
        self.game_controller.tap_key(mapped_key)

    # ========================== 金币与净利润核算 ==========================
    def update_meso_amount(self, current_meso):
        """供金币识别后更新数值"""
        self.current_meso = current_meso
        now = time.time()
        if self.initial_meso is None:
            self.initial_meso = current_meso
        self.gained_meso = max(0, self.current_meso - self.initial_meso)
        elapsed_h = max(0.001, (now - self.meso_start_time) / 3600.0)
        self.meso_per_hour = self.gained_meso / elapsed_h

    def get_potion_cost(self):
        """计算药水消耗总金额"""
        return (self.used_hp_potions * self.hp_potion_price) + (self.used_mp_potions * self.mp_potion_price)

    def get_net_profit(self):
        """计算纯净利润 (金币增量 - 药水成本)"""
        return self.gained_meso - self.get_potion_cost()

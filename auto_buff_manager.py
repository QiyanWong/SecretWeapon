import time
import random
import threading


class AutoBuffManager:
    """
    智能生命 (HP) / 魔法 (MP) 阈值监测与自动药水恢复系统
    支持低于设定百分比阈值自动喝药、自定义按键、CD 冷却防狂喝与 ±10% 拟人化击键抖动
    """

    def __init__(self, game_controller):
        self.gc = game_controller
        self.is_running = False
        self.thread = None

        self.config = {
            "auto_hp": True,
            "hp_threshold": 60,       # 低于此百分比 (0~100) 自动加血
            "hp_key": "Delete",       # 喝红药自定义按键
            "hp_cooldown": 0.6,       # 喝红药冷却时长 (秒)

            "auto_mp": True,
            "mp_threshold": 40,       # 低于此百分比 (0~100) 自动加蓝
            "mp_key": "PageDown",     # 喝蓝药自定义按键
            "mp_cooldown": 0.6,       # 喝蓝药冷却时长 (秒)

            "jitter_ratio": 0.10      # 默认 ±10% 动态拟人浮动
        }

        self.last_hp_time = 0.0
        self.last_mp_time = 0.0

        # 当前实时监测到的生命/蓝量缓存
        self.current_hp = 1.0
        self.current_mp = 1.0
        self.is_gauge_detected = False

        # 下一次拟人化动态冷却时长
        self.next_hp_interval = self._calc_next_interval(self.config["hp_cooldown"])
        self.next_mp_interval = self._calc_next_interval(self.config["mp_cooldown"])

        self._lock = threading.Lock()

    def _calc_next_interval(self, base_interval: float, ratio: float = 0.10) -> float:
        """根据基准时长计算下一次施放的动态浮动阈值 (截断高斯 ±ratio)"""
        if hasattr(self.gc, 'jitter') and self.gc.jitter:
            return self.gc.jitter.calc_floating_interval(base_interval, ratio=ratio)
        sigma = ratio / 2.5
        delta = random.gauss(0, sigma)
        delta = max(-ratio, min(delta, ratio))
        return max(0.15, base_interval * (1.0 + delta))

    def update_config(self, key, value):
        with self._lock:
            self.config[key] = value
            if key == "hp_cooldown":
                self.next_hp_interval = self._calc_next_interval(float(value), self.config.get("jitter_ratio", 0.10))
            elif key == "mp_cooldown":
                self.next_mp_interval = self._calc_next_interval(float(value), self.config.get("jitter_ratio", 0.10))

    def check_gauges(self, hp_pct: float, mp_pct: float, is_detected: bool, log_callback=None):
        """
        每帧由主循环传入最新的生命与蓝量百分比 (0.0~1.0)
        一旦低于设定百分比阈值且冷却完毕，立即以毫秒级非阻塞方式触发喝药按键
        """
        self.current_hp = hp_pct
        self.current_mp = mp_pct
        self.is_gauge_detected = is_detected

        if not is_detected:
            return

        now = time.time()
        jitter = self.config.get("jitter_ratio", 0.10)

        with self._lock:
            # 1. 自动喝红 (HP 低于百分比阈值触发)
            if self.config.get("auto_hp", False):
                hp_thresh = float(self.config.get("hp_threshold", 60))
                current_hp_pct = hp_pct * 100.0

                if current_hp_pct < hp_thresh:
                    if now - self.last_hp_time >= self.next_hp_interval:
                        hp_key = str(self.config.get("hp_key", "Delete")).strip()
                        if hp_key and self.gc:
                            self.gc.tap_key(hp_key)
                            self.last_hp_time = now
                            base_cd = float(self.config.get("hp_cooldown", 0.6))
                            self.next_hp_interval = self._calc_next_interval(base_cd, jitter)
                            msg = f"💖【自动补给】生命值过低 ({current_hp_pct:.1f}% < {hp_thresh:.0f}%) -> 喝红药 [{hp_key}] (下次冷却: {self.next_hp_interval:.2f}s)"
                            if log_callback:
                                try:
                                    log_callback(msg)
                                except Exception:
                                    pass
                            else:
                                try:
                                    print(msg)
                                except Exception:
                                    pass

            # 2. 自动喝蓝 (MP 低于百分比阈值触发)
            if self.config.get("auto_mp", False):
                mp_thresh = float(self.config.get("mp_threshold", 40))
                current_mp_pct = mp_pct * 100.0

                if current_mp_pct < mp_thresh:
                    if now - self.last_mp_time >= self.next_mp_interval:
                        mp_key = str(self.config.get("mp_key", "PageDown")).strip()
                        if mp_key and self.gc:
                            self.gc.tap_key(mp_key)
                            self.last_mp_time = now
                            base_cd = float(self.config.get("mp_cooldown", 0.6))
                            self.next_mp_interval = self._calc_next_interval(base_cd, jitter)
                            msg = f"💧【自动补给】魔法值过低 ({current_mp_pct:.1f}% < {mp_thresh:.0f}%) -> 喝蓝药 [{mp_key}] (下次冷却: {self.next_mp_interval:.2f}s)"
                            if log_callback:
                                try:
                                    log_callback(msg)
                                except Exception:
                                    pass
                            else:
                                try:
                                    print(msg)
                                except Exception:
                                    pass

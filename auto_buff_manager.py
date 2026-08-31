import time
import random
import threading

class AutoBuffManager:
    """独立于主决策引擎的状态保持系统 (具备 ±10% 拟人化动态浮动与击键抖动)"""
    def __init__(self, game_controller):
        self.gc = game_controller
        self.is_running = False
        self.thread = None
        
        self.config = {
            "auto_hp": False,
            "hp_key": "1",
            "hp_interval": 10.0,
            
            "auto_mp": False,
            "mp_key": "2",
            "mp_interval": 15.0,
            
            "auto_buff": False,
            "buff_key": "3",
            "buff_interval": 120.0,
            
            "jitter_ratio": 0.10  # 默认 ±10% 动态浮动
        }
        
        self.last_hp_time = 0
        self.last_mp_time = 0
        self.last_buff_time = 0
        
        # 初始计算动态目标触发周期
        self.next_buff_interval = self._calc_next_interval(120.0)
        self.next_hp_interval = self._calc_next_interval(10.0)
        self.next_mp_interval = self._calc_next_interval(15.0)

    def _calc_next_interval(self, base_interval: float, ratio: float = 0.10) -> float:
        """根据基准时长计算下一次施放的动态浮动阈值 (截断高斯 ±ratio)"""
        if hasattr(self.gc, 'jitter') and self.gc.jitter:
            return self.gc.jitter.calc_floating_interval(base_interval, ratio=ratio)
        sigma = ratio / 2.5
        delta = random.gauss(0, sigma)
        delta = max(-ratio, min(delta, ratio))
        return max(1.0, base_interval * (1.0 + delta))

    def update_config(self, key, value):
        if key in self.config:
            self.config[key] = value
            if key == "buff_interval":
                self.next_buff_interval = self._calc_next_interval(float(value), self.config.get("jitter_ratio", 0.10))
            elif key == "hp_interval":
                self.next_hp_interval = self._calc_next_interval(float(value), self.config.get("jitter_ratio", 0.10))
            elif key == "mp_interval":
                self.next_mp_interval = self._calc_next_interval(float(value), self.config.get("jitter_ratio", 0.10))

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            print("【状态保持系统】已启动 (拟人化 ±10% 动态浮动已生效)")

    def stop(self):
        if self.is_running:
            self.is_running = False
            if self.thread:
                self.thread.join(timeout=1.0)
            print("【状态保持系统】已停止")

    def _run_loop(self):
        while self.is_running:
            now = time.time()
            jitter_ratio = self.config.get("jitter_ratio", 0.10)
            
            # 1. 自动 Buff (时间触发 + 动态浮动)
            if self.config["auto_buff"] and (now - self.last_buff_time >= self.next_buff_interval):
                buff_key = self.config["buff_key"]
                print(f"【拟人 Buff】施放技能 -> {buff_key} (本次实测间隔: {now - self.last_buff_time:.1f}s / 目标: {self.next_buff_interval:.1f}s)")
                
                # 动态击键时长 + 触发
                self.gc.tap_key(buff_key)
                self.last_buff_time = now
                
                # 重新计算下一次的浮动目标
                self.next_buff_interval = self._calc_next_interval(self.config["buff_interval"], jitter_ratio)
                
                # 随机施法后摇 (350ms ~ 550ms)
                time.sleep(random.uniform(0.35, 0.55))
                
            # 2. 自动喝红 (当前为定时 + 动态浮动)
            if self.config["auto_hp"] and (now - self.last_hp_time >= self.next_hp_interval):
                hp_key = self.config["hp_key"]
                print(f"【自动恢复】喝红药 -> {hp_key}")
                self.gc.tap_key(hp_key)
                self.last_hp_time = now
                self.next_hp_interval = self._calc_next_interval(self.config["hp_interval"], jitter_ratio)
                time.sleep(random.uniform(0.15, 0.28))
                
            # 3. 自动喝蓝 (定时 + 动态浮动)
            if self.config["auto_mp"] and (now - self.last_mp_time >= self.next_mp_interval):
                mp_key = self.config["mp_key"]
                print(f"【自动恢复】喝蓝药 -> {mp_key}")
                self.gc.tap_key(mp_key)
                self.last_mp_time = now
                self.next_mp_interval = self._calc_next_interval(self.config["mp_interval"], jitter_ratio)
                time.sleep(random.uniform(0.15, 0.28))
                
            time.sleep(0.2)  # 200ms 轮询粒度更平滑细腻

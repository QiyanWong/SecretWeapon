import time
import threading

class AutoBuffManager:
    """独立于主决策引擎的状态保持系统 (自动喝药与自动 Buff)"""
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
            "buff_interval": 120.0
        }
        
        self.last_hp_time = 0
        self.last_mp_time = 0
        self.last_buff_time = 0

    def update_config(self, key, value):
        if key in self.config:
            self.config[key] = value

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()
            print("【状态保持系统】已启动")

    def stop(self):
        if self.is_running:
            self.is_running = False
            if self.thread:
                self.thread.join(timeout=1.0)
            print("【状态保持系统】已停止")

    def _run_loop(self):
        while self.is_running:
            now = time.time()
            
            # 自动 Buff (时间触发)
            if self.config["auto_buff"] and (now - self.last_buff_time > self.config["buff_interval"]):
                print(f"【自动 Buff】施放技能 -> {self.config['buff_key']}")
                self.gc.tap_key(self.config["buff_key"], 0.2)
                self.last_buff_time = now
                time.sleep(0.5)  # 等待施法后摇
                
            # 自动喝红 (当前占位为定时，实际应接入像素判定)
            # if self.config["auto_hp"] and hp_percent < 50:
            if self.config["auto_hp"] and (now - self.last_hp_time > self.config["hp_interval"]):
                print(f"【自动恢复】喝红药 -> {self.config['hp_key']}")
                self.gc.tap_key(self.config["hp_key"], 0.1)
                self.last_hp_time = now
                time.sleep(0.2)
                
            # 自动喝蓝
            if self.config["auto_mp"] and (now - self.last_mp_time > self.config["mp_interval"]):
                print(f"【自动恢复】喝蓝药 -> {self.config['mp_key']}")
                self.gc.tap_key(self.config["mp_key"], 0.1)
                self.last_mp_time = now
                time.sleep(0.2)
                
            time.sleep(0.5) # 每半秒检测一次触发条件

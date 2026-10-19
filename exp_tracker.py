import re
import time
import queue
import threading
import cv2
import numpy as np


def parse_exp_string(text: str):
    """
    从 OCR 识别文本中提取经验值绝对数字与百分比
    支持如 'EXP 277837[21.77%]', '647859[43.25%]', '123456 (50.00%)' 等各种格式
    返回: (exp_val: int or None, pct_val: float or None)
    """
    if not text:
        return None, None

    pct_val = None
    exp_val = None

    # 1. 优先提取括号内的百分比数值
    pct_match = re.search(r'\[?(\d+(?:\.\d+)?)%\]?', text)
    if pct_match:
        try:
            pct_val = float(pct_match.group(1))
        except ValueError:
            pass

    # 2. 移除百分比与括号部分后提取绝对经验整数
    clean = re.sub(r'\[.*?\]', '', text)
    clean = re.sub(r'\(.*?\)', '', clean)
    clean = re.sub(r'EXP|exp|Exp|[:,\s]', '', clean)
    digit_match = re.search(r'\d+', clean)
    if digit_match:
        try:
            exp_val = int(digit_match.group(0))
        except ValueError:
            pass

    return exp_val, pct_val


class ExpTracker:
    """
    基于 RapidOCR 的角色经验获取率与升级预估统计器
    仅在自动打怪期间记录有效战斗时长，精准剔除经验条跳变误差与挂机闲逛干扰
    支持绝对经验值统计、百分比时薪、升级预估 (ETA) 及一键重置
    """

    def __init__(self, log_callback=None):
        self.log_callback = log_callback
        self._lock = threading.Lock()

        # 战斗计时状态 (仅在自动打怪期间累加)
        self.is_combat_active = False
        self.total_combat_seconds = 0.0
        self.last_tick_time = 0.0

        # 经验统计数据
        self.start_exp = None
        self.start_pct = None
        self.current_exp = None
        self.current_pct = None
        self.raw_ocr_text = ""

        self.last_seen_exp = None
        self.last_seen_pct = None
        self.level_up_count = 0

        # 异步 OCR 线程管理
        self._ocr_engine = None
        self._task_queue = queue.Queue(maxsize=1)
        self._is_worker_running = True
        self._last_sample_time = 0.0
        self._sample_interval = 1.5  # OCR 采样间隔 (秒)

        self._worker_thread = threading.Thread(target=self._ocr_worker_loop, daemon=True)
        self._worker_thread.start()

    def _get_ocr_engine(self):
        """延迟初始化 RapidOCR，避免在主程序启动时产生阻塞"""
        if self._ocr_engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
                self._ocr_engine = RapidOCR()
            except Exception as e:
                self._log(f"【OCR初始化异常】未能加载 RapidOCR: {e}")
        return self._ocr_engine

    def _log(self, msg: str):
        if self.log_callback:
            try:
                self.log_callback(msg)
            except Exception:
                pass
        else:
            try:
                print(msg)
            except Exception:
                pass

    def start_combat(self):
        """当用户点击【开始打怪】时触发：开启纯打怪有效计时"""
        with self._lock:
            if not self.is_combat_active:
                self.is_combat_active = True
                self.last_tick_time = time.time()
                # 若之前未获取过基准，若已有当前值则直接锁为起点
                if self.start_exp is None and self.current_exp is not None:
                    self.start_exp = self.current_exp
                    self.start_pct = self.current_pct
                self._log("📈【经验时薪监控】已联动【开始打怪】激活！仅统计纯打怪有效时长。")

    def stop_combat(self):
        """当用户点击【停止打怪】时触发：暂停有效战斗时长计时"""
        with self._lock:
            if self.is_combat_active:
                now = time.time()
                if self.last_tick_time > 0:
                    self.total_combat_seconds += max(0.0, now - self.last_tick_time)
                self.is_combat_active = False
                self.last_tick_time = 0.0
                elapsed_min = self.total_combat_seconds / 60.0
                self._log(f"⏸️【经验时薪监控】打怪已暂停，当前累计有效时长: {elapsed_min:.1f} 分钟。")

    def reset(self):
        """重置经验统计所有计数器"""
        with self._lock:
            self.total_combat_seconds = 0.0
            if self.is_combat_active:
                self.last_tick_time = time.time()
            else:
                self.last_tick_time = 0.0

            self.start_exp = self.current_exp
            self.start_pct = self.current_pct
            self.last_seen_exp = self.current_exp
            self.last_seen_pct = self.current_pct
            self.level_up_count = 0
            self._log("🔄【经验统计重置】已清空战斗时长与累计经验，重新锁定当前经验为起点！")

    def update_tick(self):
        """由主定时器每帧或周期性调用，累加打怪有效时间"""
        with self._lock:
            if self.is_combat_active:
                now = time.time()
                if self.last_tick_time > 0:
                    delta = now - self.last_tick_time
                    if 0 < delta < 2.0:  # 过滤休眠或挂起引起的异常时间跳变
                        self.total_combat_seconds += delta
                self.last_tick_time = now

    def feed_frame(self, game_frame: np.ndarray, exp_rect: tuple):
        """
        在主循环中送入当前帧与经验文字区域坐标 (exp_x, exp_y, exp_w, exp_h)
        仅在打怪期间且超过采样冷却时，异步派发给后台线程进行 OCR 解析
        """
        if not self.is_combat_active or game_frame is None or exp_rect is None:
            return

        now = time.time()
        if now - self._last_sample_time < self._sample_interval:
            return

        ex, ey, ew, eh = exp_rect
        fh, fw = game_frame.shape[:2]
        if ex < 0 or ey < 0 or ex + ew > fw or ey + eh > fh:
            return

        exp_crop = game_frame[ey:ey + eh, ex:ex + ew].copy()
        try:
            # 非阻塞投递：若前一次任务尚在进行，直接跳过本帧
            self._task_queue.put_nowait(exp_crop)
            self._last_sample_time = now
        except queue.Full:
            pass

    def _ocr_worker_loop(self):
        """后台异步 OCR 解析工作线程，杜绝主 GUI 线程任何卡顿"""
        while self._is_worker_running:
            try:
                crop = self._task_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                engine = self._get_ocr_engine()
                if engine is None or crop is None or crop.size == 0:
                    continue

                # 适度放大增强 OCR 对微点阵文字的识别率
                ch, cw = crop.shape[:2]
                scaled = cv2.resize(crop, (cw * 2, ch * 2), interpolation=cv2.INTER_LINEAR)
                result, _ = engine(scaled)

                if result:
                    full_text = " ".join(item[1] for item in result).strip()
                    exp_val, pct_val = parse_exp_string(full_text)
                    if exp_val is not None or pct_val is not None:
                        self._handle_ocr_result(exp_val, pct_val, full_text)
            except Exception as e:
                pass

    def _handle_ocr_result(self, exp_val, pct_val, raw_text):
        """安全整合最新识别结果"""
        with self._lock:
            self.raw_ocr_text = raw_text

            # 首次识别到数据，且处于打怪中，自动锚定为起点
            if self.current_exp is None and exp_val is not None:
                self.current_exp = exp_val
                if self.is_combat_active and self.start_exp is None:
                    self.start_exp = exp_val

            if self.current_pct is None and pct_val is not None:
                self.current_pct = pct_val
                if self.is_combat_active and self.start_pct is None:
                    self.start_pct = pct_val

            # 检测角色升级 (经验大幅跳水)
            if exp_val is not None and self.current_exp is not None:
                if exp_val < self.current_exp and (self.current_exp - exp_val > 1000):
                    self.level_up_count += 1
                    self._log(f"🎉【角色升级】检测到经验归零/升级！(已升 {self.level_up_count} 级)")
                self.current_exp = exp_val

            if pct_val is not None:
                if self.current_pct is not None and pct_val < self.current_pct and (self.current_pct - pct_val > 5.0):
                    if self.level_up_count == 0:
                        self.level_up_count += 1
                        self._log("🎉【角色升级】检测到经验百分比重置！")
                self.current_pct = pct_val

    def get_stats(self) -> dict:
        """
        获取当前格式化后的全部经验与时薪数据
        返回字典包含时长、获得经验、时薪、ETA等
        """
        with self._lock:
            secs = int(self.total_combat_seconds)
            h = secs // 3600
            m = (secs % 3600) // 60
            s = secs % 60
            elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"

            gained_exp = 0
            if self.current_exp is not None and self.start_exp is not None:
                gained_exp = max(0, self.current_exp - self.start_exp)

            gained_pct = 0.0
            if self.current_pct is not None and self.start_pct is not None:
                gained_pct = max(0.0, self.current_pct - self.start_pct)

            exp_per_hour = 0.0
            pct_per_hour = 0.0
            eta_str = "计算中..."

            hours = self.total_combat_seconds / 3600.0
            # 打怪时长达到 5 秒以上才计算时薪，避免瞬时无穷大
            if self.total_combat_seconds >= 5.0:
                if gained_exp > 0:
                    exp_per_hour = gained_exp / hours
                if gained_pct > 0:
                    pct_per_hour = gained_pct / hours

                # 计算预计升级时间 (ETA)
                if self.current_pct is not None and pct_per_hour > 0.01:
                    remain_pct = max(0.0, 100.0 - self.current_pct)
                    eta_hours = remain_pct / pct_per_hour
                    eta_secs = int(eta_hours * 3600)
                    eh = eta_secs // 3600
                    em = (eta_secs % 3600) // 60
                    if eh > 0:
                        eta_str = f"{eh}小时{em}分"
                    else:
                        eta_str = f"{em}分钟"
                elif self.current_pct is not None and self.current_pct >= 99.99:
                    eta_str = "即将升级"

            return {
                "is_active": self.is_combat_active,
                "elapsed_seconds": self.total_combat_seconds,
                "elapsed_str": elapsed_str,
                "current_exp": self.current_exp,
                "current_pct": self.current_pct,
                "start_exp": self.start_exp,
                "start_pct": self.start_pct,
                "gained_exp": gained_exp,
                "gained_pct": gained_pct,
                "exp_per_hour": exp_per_hour,
                "pct_per_hour": pct_per_hour,
                "eta_str": eta_str,
                "level_ups": self.level_up_count,
                "raw_ocr": self.raw_ocr_text
            }

    def draw_overlay(self, frame: np.ndarray, exp_rect: tuple):
        """在监控画面上绘制亮黄色经验文字标注框与时薪速报"""
        if frame is None or exp_rect is None:
            return

        ex, ey, ew, eh = exp_rect
        stats = self.get_stats()

        # 亮黄色经验条指示边框
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 235, 255), 1)

        # 浮动提示标签
        cur_str = f"{stats['current_exp']:,}" if stats['current_exp'] else "---"
        pct_str = f"{stats['current_pct']:.2f}%" if stats['current_pct'] else "--"
        tag = f"EXP {cur_str} [{pct_str}]"
        if stats['exp_per_hour'] > 0:
            tag += f" | {stats['exp_per_hour']/1000.0:.1f}k/h"

        cv2.putText(frame, tag, (ex, max(12, ey - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 235, 255), 1, cv2.LINE_AA)

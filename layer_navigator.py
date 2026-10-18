"""
多层平台打怪导航器 (替代旧的 A* 全图寻路)

设计原则 (刻意保持简单):
  1. 地图被抽象为若干 "平台" (连续相接的 Foothold 组成一个平台/层)。
  2. 平台之间只有两种跃迁:
       - 向下: 永远使用 下跳 (DOWN + JUMP)
       - 向上: 永远使用 爬梯子/绳子
  3. 打怪层由用户标记 (F5)。没有任何标记时，只在开始时所在的平台打怪
     (等价于旧的 "平台打怪" 模式)；被击落/掉落后会自动导航回该平台。
  4. 所有平台永远启用安全区 (由 DecisionEngine 负责拦截)。
  5. 攀爬判定完全不依赖 YOLO 姿态，只依赖小地图坐标的运动学特征:
       - 普通跳跃在 JUMP_AIRTIME 内必然落回地面；
         超过该时间仍高于起跳高度 => 已挂在梯子上。
       - 登顶: 按住 UP 时纵坐标停止变化 => 到顶；
         再用 "横向探针" (按方向键看 X 是否移动) 确认真正站上平台，
         否则执行 方向+跳跃 强制脱绳。
  6. 梯子坐标自校准: 抓住梯子后角色 X 会被游戏强制吸附到梯子中轴，
     此刻的小地图 X 即为该梯子的真值，持久化保存，下次直接使用；
     未校准的梯子使用已校准梯子的平均偏差进行修正。
"""

import heapq
import json
import math
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILE_DIR = os.path.join(BASE_DIR, "dataset", "map_profiles")

RUNNING = "RUNNING"
SUCCESS = "SUCCESS"
FAIL = "FAIL"


# ============================================================================
# 帧上下文
# ============================================================================
class NavContext:
    """每帧传入导航器的感知数据"""
    def __init__(self, now, mx, my, xw, yw, jump_key="Alt",
                 margin_left=150, margin_right=150, crop_w=None, crop_h=None,
                 turns_per_layer=1, max_dwell=25.0):
        self.now = now
        self.mx = int(mx)            # 小地图像素 X (运动学判定用，原始传感器数据)
        self.my = int(my)            # 小地图像素 Y
        self.xw = xw                 # 世界坐标 X (几何判定用)
        self.yw = yw                 # 世界坐标 Y
        self.jump_key = jump_key
        self.ml = margin_left
        self.mr = margin_right
        self.crop_w = crop_w
        self.crop_h = crop_h
        self.turns_per_layer = turns_per_layer
        self.max_dwell = max_dwell


# ============================================================================
# 平台 / 边 / 平台图
# ============================================================================
class Platform:
    """一个连续可行走的平台 (由多段首尾相接的 Foothold 组成)"""
    def __init__(self, pid, fhs):
        self.id = pid
        self.fhs = sorted(fhs, key=lambda f: f.x1)
        self.fh_ids = {f.id for f in fhs}
        self.x_min = min(f.x1 for f in fhs)
        self.x_max = max(f.x2 for f in fhs)
        self.avg_y = sum(f.avg_y for f in fhs) / float(len(fhs))

    @property
    def width(self):
        return self.x_max - self.x_min

    @property
    def min_x(self):
        return self.x_min

    @property
    def max_x(self):
        return self.x_max

    @property
    def center_x(self):
        return (self.x_min + self.x_max) / 2.0

    def y_at(self, x, margin=0):
        """平台在 x 处的地面高度；x 不在平台上返回 None"""
        best = None
        for f in self.fhs:
            if f.x1 - margin <= x <= f.x2 + margin:
                y = f.get_y_at_x(x)
                if best is None or y < best:
                    best = y
        return best

    def safe_bounds(self, ml, mr):
        """计算安全区 [safe_left, safe_right] (极窄平台按比例收缩，保证安全区不消失)"""
        try:
            ml = max(0, int(ml))
            mr = max(0, int(mr))
        except (ValueError, TypeError):
            ml, mr = 150, 150
        total_w = self.width
        aml, amr = ml, mr
        if total_w < ml + mr + 40:
            if ml + mr > 0:
                scale = max(0.0, (total_w - 40) / float(ml + mr))
                aml, amr = int(ml * scale), int(mr * scale)
            else:
                aml, amr = 0, 0
            if aml + amr >= total_w - 10:
                aml = max(5, int(total_w * 0.25))
                amr = max(5, int(total_w * 0.25))
        sl = self.x_min + aml
        sr = self.x_max - amr
        if sl >= sr:
            c = self.center_x
            sl, sr = c - 10, c + 10
        return sl, sr

    def __repr__(self):
        return f"Platform(#{self.id}, x=[{self.x_min},{self.x_max}], y≈{self.avg_y:.0f})"


class Edge:
    DROP = "DROP"
    CLIMB = "CLIMB"

    EXIT_TOP = "TOP"     # 爬到绳顶，按住 UP 自动翻上平台
    EXIT_SIDE = "SIDE"   # 爬到平台高度后，向平台方向跳跃离开绳子

    def __init__(self, kind, src, dst, ladder=None, drop_xs=None,
                 exit_mode=None, exit_y=None, exit_dir=None, entry_y=None):
        self.kind = kind
        self.src = src
        self.dst = dst
        self.ladder = ladder
        self.drop_xs = drop_xs or []
        self.exit_mode = exit_mode
        self.exit_y = exit_y          # 目标平台在绳子旁的地面高度 (世界坐标)
        self.exit_dir = exit_dir      # 离开绳子的方向 "LEFT"/"RIGHT"
        self.entry_y = entry_y        # 上绳平台在绳子处的地面高度
        self.key = (kind, src, dst, ladder.id if ladder is not None else None)

    def __repr__(self):
        if self.kind == Edge.CLIMB:
            return f"CLIMB-{self.exit_mode}(#{self.src}->#{self.dst} via lr{self.ladder.id}@x={self.ladder.x})"
        return f"DROP(#{self.src}->#{self.dst})"


class PlatformGraph:
    JOIN_DX = 20          # 两段 Foothold 端点横向贴合容差
    JOIN_DY = 35          # 两段 Foothold 端点纵向落差容差
    LADDER_REACH = 110    # 悬挂绳底端距地面的最大可跳抓高度 (世界像素)
    BOARD_DX = 15         # 上绳平台必须覆盖绳子 X (容差)，否则走过去会掉下去
    EXIT_DX = 60          # 侧向离绳: 平台边缘距绳子的最大水平距离
    DROP_SAMPLE = 10      # 下跳点采样步长 (世界像素)

    def __init__(self, mp):
        self.mp = mp
        self.platforms = {}       # pid -> Platform
        self.fh_to_pid = {}       # fh_id -> pid
        self.edges_from = {}      # pid -> [Edge]
        self._build_platforms()
        self._build_climb_edges()
        self._build_drop_edges()
        n_climb = sum(1 for es in self.edges_from.values() for e in es if e.kind == Edge.CLIMB)
        n_drop = sum(1 for es in self.edges_from.values() for e in es if e.kind == Edge.DROP)
        print(f"🧭 [平台图] 平台 {len(self.platforms)} 个 | 爬梯边 {n_climb} 条 | 下跳边 {n_drop} 条")

    # ---------- 构建 ----------
    def _build_platforms(self):
        fhs = list(self.mp.horizontal_fhs)
        visited = set()
        for fh in fhs:
            if fh.id in visited:
                continue
            comp = [fh]
            visited.add(fh.id)
            queue = [fh]
            while queue:
                f = queue.pop()
                for o in fhs:
                    if o.id in visited:
                        continue
                    if (abs(o.x2 - f.x1) <= self.JOIN_DX and abs(o.y2 - f.y1) <= self.JOIN_DY) or \
                       (abs(o.x1 - f.x2) <= self.JOIN_DX and abs(o.y1 - f.y2) <= self.JOIN_DY):
                        visited.add(o.id)
                        queue.append(o)
                        comp.append(o)
            pid = min(f.id for f in comp)
            self.platforms[pid] = Platform(pid, comp)
            for f in comp:
                self.fh_to_pid[f.id] = pid
            self.edges_from[pid] = []

    def _build_climb_edges(self):
        for lr in self.mp.ladder_ropes:
            # 上绳平台: 覆盖绳子 X，且地面位于绳底下方 (或略高) 的最近平台
            bottom, best = None, 1e9
            for p in self.platforms.values():
                y = p.y_at(lr.x, margin=self.BOARD_DX)
                if y is None:
                    continue
                d = y - lr.y_bottom
                if -15 <= d <= self.LADDER_REACH and d < best:
                    best, bottom = d, p
            if bottom is None:
                continue
            bottom_y = bottom.y_at(lr.x, margin=self.BOARD_DX)

            # 出口平台: 绳子沿途经过的所有平台 (覆盖绳子或边缘紧贴绳子)
            for p in self.platforms.values():
                if p.id == bottom.id:
                    continue
                dx = max(0, p.x_min - lr.x, lr.x - p.x_max)
                if dx > self.EXIT_DX:
                    continue
                xx = min(max(lr.x, p.x_min), p.x_max)
                y = p.y_at(xx, margin=1)
                if y is None or y >= bottom_y - 20 or y < lr.y_top - 40:
                    continue
                if dx <= self.BOARD_DX and abs(y - lr.y_top) <= 40:
                    mode = Edge.EXIT_TOP
                else:
                    mode = Edge.EXIT_SIDE
                if dx > 0:
                    exit_dir = "RIGHT" if p.x_min > lr.x else "LEFT"
                else:
                    exit_dir = "RIGHT" if p.center_x >= lr.x else "LEFT"
                self.edges_from[bottom.id].append(Edge(
                    Edge.CLIMB, bottom.id, p.id, ladder=lr,
                    exit_mode=mode, exit_y=y, exit_dir=exit_dir, entry_y=bottom_y))

    def _near_ladder_top(self, x, y):
        """下跳点不能离梯子顶端太近 (DOWN 会抓住梯子往下爬)"""
        for lr in self.mp.ladder_ropes:
            if abs(x - lr.x) < 30 and abs(y - lr.y_top) < 60:
                return True
        return False

    def _build_drop_edges(self):
        for a in self.platforms.values():
            runs = {}
            x = a.x_min + self.DROP_SAMPLE
            while x <= a.x_max - self.DROP_SAMPLE:
                ya = a.y_at(x)
                if ya is not None and not self._near_ladder_top(x, ya):
                    b = self.platform_below(x, ya)
                    if b is not None:
                        runs.setdefault(b.id, []).append(x)
                x += self.DROP_SAMPLE
            for bid, xs in runs.items():
                s = set(xs)
                step2 = 2 * self.DROP_SAMPLE
                # 取连续区间内部的点，避免落在下层平台边缘
                good = [v for v in xs if (v - step2) in s and (v + step2) in s]
                self.edges_from[a.id].append(Edge(Edge.DROP, a.id, bid, drop_xs=good or xs))

    # ---------- 查询 ----------
    def platform_below(self, x, y):
        best, best_y = None, None
        for p in self.platforms.values():
            py = p.y_at(x)
            if py is None or py <= y + 5:
                continue
            if best_y is None or py < best_y:
                best, best_y = p, py
        return best

    def locate(self, xw, yw):
        fh = self.mp.snap_to_foothold(xw, yw, margin=45, margin_y=60)
        if fh is None:
            return None
        pid = self.fh_to_pid.get(fh.id)
        return self.platforms.get(pid) if pid is not None else None

    def drop_x_for(self, edge, from_x, ml, mr, preferred_x=None):
        """在候选下跳点中选择: 优先位于源平台安全区内，其次最接近指定下跳点 (默认平台中点)"""
        if not edge.drop_xs:
            return None
        sl, sr = self.platforms[edge.src].safe_bounds(ml, mr)
        safe = [v for v in edge.drop_xs if sl <= v <= sr]
        pool = safe or edge.drop_xs
        target_x = preferred_x if preferred_x is not None else self.platforms[edge.src].center_x
        return min(pool, key=lambda v: abs(v - target_x))

    def find_route(self, src, from_x, dst, ml, mr, drop_points=None, banned=()):
        """Dijkstra: 节点=平台，边=下跳/爬梯。返回 [(Edge, action_x), ...] 或 None"""
        if src == dst:
            return []
        dist = {src: 0.0}
        cnt = 0
        heap = [(0.0, cnt, src, from_x, [])]
        while heap:
            cost, _, pid, x, path = heapq.heappop(heap)
            if pid == dst:
                return path
            if cost > dist.get(pid, 1e18):
                continue
            for e in self.edges_from.get(pid, []):
                if e.key in banned:
                    continue
                if e.kind == Edge.CLIMB:
                    ax = e.ladder.x
                    act = 2.0 + abs(e.entry_y - e.exit_y) / 150.0
                else:
                    pref = (drop_points or {}).get(str(e.src))
                    ax = self.drop_x_for(e, x, ml, mr, preferred_x=pref)
                    if ax is None:
                        continue
                    act = 1.0
                nc = cost + abs(ax - x) / 100.0 + act
                if nc < dist.get(e.dst, 1e18):
                    dist[e.dst] = nc
                    cnt += 1
                    heapq.heappush(heap, (nc, cnt, e.dst, ax, path + [(e, ax)]))
        return None


# ============================================================================
# 每张地图的持久化配置 (打怪层标记 + 梯子校准)
# ============================================================================
class MapProfile:
    def __init__(self, map_key):
        os.makedirs(PROFILE_DIR, exist_ok=True)
        self.path = os.path.join(PROFILE_DIR, f"{map_key}.json")
        self.marked = []
        self.drop_points = {} # str(pid) -> int(x)
        self.ladders = {}     # str(lr_id) -> {"mx": int, "sig": [crop_w, crop_h, off_x, off_y]}
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.marked = [int(v) for v in data.get("marked_platforms", [])]
            self.drop_points = {str(k): int(v) for k, v in data.get("drop_points", {}).items()}
            self.ladders = data.get("ladder_calibration", {}) or {}
        except Exception as e:
            print(f"⚠️ [地图配置] 读取失败 {self.path}: {e}")

    def save(self):
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({"marked_platforms": self.marked,
                           "drop_points": self.drop_points,
                           "ladder_calibration": self.ladders}, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ [地图配置] 保存失败 {self.path}: {e}")


# ============================================================================
# 动作原语: 爬梯
# ============================================================================
class LadderClimber:
    """
    纯运动学横向跳跃扫梯抓梯原语 (逐帧驱动，不阻塞主线程)
    APPROACH (跑至起跳点) -> SWEEP_JUMP (横向起跳+UP) -> SWEEP_WAIT (滞空扫梯判定) -> CLIMB -> TOP_HOLD -> PROBE -> (JUMP_OFF -> SETTLE) -> VERIFY
    """
    APPROACH_TIMEOUT = 8.0  # 跑向起跳点的最长时间
    SWEEP_AIRTIME = 0.70    # 跳跃最长滞空时间
    DIR_HOLD_TIME = 0.12    # 起跳后维持水平方向的时长 (赋予水平惯性)
    STALL_TIME = 0.6        # 按住 UP 纵坐标不变超过此时间 => 到顶
    TOP_HOLD = 0.35         # 到顶后继续按 UP 让角色翻上平台
    PROBE_TIME = 0.35       # 横向探针: 按方向键的时长
    CLIMB_TIMEOUT = 14.0
    MAX_ATTEMPTS = 5
    MAX_JUMP_OFF = 2

    # 默认横向起跳距梯子中轴的距离 (世界像素)。
    # 调整为约 30px，贴近梯子发起横跳扫梯，抓梯成功率更高
    BASE_LAUNCH_DIST = 30

    def __init__(self, nav, edge, target_mx=None):
        self.nav = nav
        self.gc = nav.gc
        self.edge = edge
        self.lr = edge.ladder
        self.target_mx = int(target_mx) if target_mx is not None else None
        top = nav.graph.platforms[edge.dst]
        self.dir_inward = edge.exit_dir or ("RIGHT" if top.center_x >= self.lr.x else "LEFT")
        self.state = "APPROACH"
        self.t_state = None
        self.attempt = 0
        self.jump_offs = 0
        self.holding = None
        self.fail_reason = ""
        self.launch_x = None
        self.sweep_dir = None
        self.launch_xw = None
        self.launch_yw = None
        self.launch_my = None
        self.min_air_my = None
        self.min_air_yw = None

    # ---- 工具 ----
    def _enter(self, st, now):
        self.state = st
        self.t_state = now

    def _hold(self, d):
        o = "LEFT" if d == "RIGHT" else "RIGHT"
        self.gc.release_key(o)
        self.gc.press_key(d)
        self.holding = d

    def _release_lr(self):
        if self.holding:
            self.gc.release_key("LEFT")
            self.gc.release_key("RIGHT")
            self.holding = None

    def _fail(self, reason):
        self.gc.clear_movement()
        self.fail_reason = reason
        return FAIL

    def describe(self):
        if self.state in ("APPROACH", "SWEEP_JUMP", "SWEEP_WAIT"):
            s_dir = f"[{self.sweep_dir}]" if self.sweep_dir else ""
            return f"扫梯[{self.state}{s_dir}] lr{self.lr.id} 尝试={self.attempt+1}"
        return f"爬梯[{self.state}] lr{self.lr.id}({self.edge.exit_mode}) 尝试={self.attempt+1}"

    # ---- 主循环 ----
    def update(self, c):
        if self.t_state is None:
            self.t_state = c.now
        return getattr(self, "_st_" + self.state.lower())(c)

    def _plan_launch(self, c):
        bottom = self.nav.graph.platforms[self.edge.src]
        sl, sr = bottom.safe_bounds(c.ml, c.mr)
        lx = self.lr.x

        # 尝试重试时围绕 30px 微幅轮换起跳距离 (30 -> 24 -> 38 -> 20 -> 42)
        dist_mods = [0, -6, +8, -10, +12]
        cur_dist = max(18, self.BASE_LAUNCH_DIST + dist_mods[self.attempt % len(dist_mods)])

        can_left = (lx - cur_dist) >= sl
        can_right = (lx + cur_dist) <= sr

        # 优先朝向梯子起跳
        if c.xw <= lx:
            if can_left:
                return (lx - cur_dist, "RIGHT", cur_dist)
            elif can_right:
                return (lx + cur_dist, "LEFT", cur_dist)
        else:
            if can_right:
                return (lx + cur_dist, "LEFT", cur_dist)
            elif can_left:
                return (lx - cur_dist, "RIGHT", cur_dist)

        if can_left:
            return (lx - cur_dist, "RIGHT", cur_dist)
        elif can_right:
            return (lx + cur_dist, "LEFT", cur_dist)
        else:
            # 极窄平台自适应 (上限 30px)
            left_room = abs(lx - sl)
            right_room = abs(sr - lx)
            if left_room >= right_room:
                d = max(18, min(30, int(left_room * 0.8)))
                return (lx - d, "RIGHT", d)
            else:
                d = max(18, min(30, int(right_room * 0.8)))
                return (lx + d, "LEFT", d)

    def _st_approach(self, c):
        if c.now - self.t_state > self.APPROACH_TIMEOUT:
            return self._fail("走向起跳点超时")

        launch_x, sweep_dir, dist = self._plan_launch(c)
        self.launch_x = launch_x
        self.sweep_dir = sweep_dir

        dx = launch_x - c.xw
        dist_to_ladder = abs(self.lr.x - c.xw)
        is_between = (sweep_dir == "RIGHT" and launch_x <= c.xw <= self.lr.x) or \
                     (sweep_dir == "LEFT" and self.lr.x <= c.xw <= launch_x)

        # 接近起跳点 (收紧容差至 12px，确保起跳距梯子在 30px 左右) 或在梯子与起跳点之间且距离梯子充足，立即发起横跳
        if abs(dx) <= 12 or (is_between and dist_to_ladder >= 20):
            return self._start_sweep_jump(c)

        d = "RIGHT" if dx > 0 else "LEFT"
        self._hold(d)
        return RUNNING

    def _start_sweep_jump(self, c):
        self.launch_xw = c.xw
        self.launch_yw = c.yw
        self.launch_my = c.my
        self.min_air_my = c.my
        self.min_air_yw = c.yw

        # 1. 明确朝向梯子
        self._hold(self.sweep_dir)
        # 2. 按住 UP 键 (滞空扫过梯子抓梯核心)
        self.gc.press_key("UP")
        # 3. 点按跳跃
        self.gc.tap_key(c.jump_key, 0.08)

        print(f"🪜 [横跳扫梯] 发起横跳! 起跳X={c.xw:.0f}, 梯子X={self.lr.x}, 距梯={abs(self.lr.x - c.xw):.0f}px | 朝向={self.sweep_dir} + UP + 跳跃")
        self._enter("SWEEP_WAIT", c.now)
        return RUNNING

    def _st_sweep_wait(self, c):
        el = c.now - self.t_state

        # 起跳 120ms 后松开水平方向，靠惯性前进，防止抓梯后误按方向脱离梯子
        if el >= self.DIR_HOLD_TIME:
            self._release_lr()

        # 跟踪最高滞空高度
        if c.my < self.min_air_my:
            self.min_air_my = c.my
        if c.yw < self.min_air_yw:
            self.min_air_yw = c.yw

        # 抓梯成功判定: 滞空超过 300ms 且高度明显上升 (或抓住梯子往上爬)
        risen_my = self.launch_my - c.my
        risen_yw = self.launch_yw - c.yw

        if (el >= 0.35 and risen_my >= 1) or (risen_yw >= 25 and el >= 0.25):
            return self._caught(c)

        # 未抓住判定: 滞空超时或落回地面
        if el >= self.SWEEP_AIRTIME or (el >= 0.45 and c.my >= self.launch_my):
            self.gc.release_key("UP")
            self._release_lr()
            self.attempt += 1

            landing_offset = c.xw - self.lr.x
            dist_traveled = abs(c.xw - self.launch_xw)
            print(f"❌ [横跳扫梯] 第 {self.attempt} 次未抓住 | 起跳X={self.launch_xw:.0f}, 梯子X={self.lr.x}, 落地X={c.xw:.0f} (距梯 {landing_offset:+.0f}px, 跳跃位移={dist_traveled:.0f}px), 滞空最高上升={self.launch_yw - self.min_air_yw:.0f}px, 滞空={el*1000:.0f}ms")

            if self.attempt >= self.MAX_ATTEMPTS:
                return self._fail(f"连续 {self.attempt} 次横跳扫梯未抓住")

            self._enter("APPROACH", c.now)
            return RUNNING

        return RUNNING

    def _caught(self, c):
        el = c.now - self.t_state
        print(f"🎉 [横跳扫梯] 抓梯成功! (尝试第 {self.attempt+1} 次) | 抓住时坐标:世界({c.xw:.0f}, {c.yw:.0f}), 小地图({c.mx}, {c.my}) | 耗时: {el*1000:.0f}ms")
        self.nav.calibrate_ladder(self.lr, c.mx, c)
        self.min_my = c.my
        self.last_change = c.now
        self._enter("CLIMB", c.now)
        return RUNNING

    def _st_climb(self, c):
        self.gc.press_key("UP")
        if c.my < self.min_my:
            self.min_my = c.my
            self.last_change = c.now
        elif c.my > self.min_my + 2:
            return self._fail("攀爬中被击落/滑落")

        # 侧向离绳 (EXIT_SIDE): 高度到达目标平台地面时直接跳出
        if self.edge.exit_mode == Edge.EXIT_SIDE and self.edge.exit_y is not None:
            if c.yw <= self.edge.exit_y + 15:
                self.gc.release_key("UP")
                self._enter("JUMP_OFF", c.now)
                return RUNNING

        # 顶端翻身 (EXIT_TOP): 纵坐标停止变化 或 接近顶端
        if self.edge.exit_mode == Edge.EXIT_TOP:
            near_top = (self.edge.exit_y is not None and c.yw <= self.edge.exit_y + 10)
            if near_top or (c.now - self.last_change >= self.STALL_TIME):
                self._enter("TOP_HOLD", c.now)
                return RUNNING
        elif c.now - self.last_change >= self.STALL_TIME:
            # 兜底: 纵向到顶
            self._enter("TOP_HOLD", c.now)
            return RUNNING

        if c.now - self.t_state > self.CLIMB_TIMEOUT:
            return self._fail("攀爬超时")
        return RUNNING

    def _st_top_hold(self, c):
        if c.now - self.t_state < self.TOP_HOLD:
            self.gc.press_key("UP")
            return RUNNING
        self.gc.release_key("UP")
        self.probe_mx = c.mx
        self._hold(self.dir_inward)
        self._enter("PROBE", c.now)
        return RUNNING

    def _st_probe(self, c):
        if abs(c.mx - self.probe_mx) >= 1:
            self._release_lr()
            self._enter("VERIFY", c.now)
            return RUNNING
        if c.now - self.t_state >= self.PROBE_TIME:
            self._release_lr()
            self._enter("JUMP_OFF", c.now)
        return RUNNING

    def _st_jump_off(self, c):
        self.jump_offs += 1
        if self.jump_offs > self.MAX_JUMP_OFF:
            return self._fail("无法从梯子顶端脱离")
        print(f"🪜 [爬梯] 横向探针未移动，仍挂在梯子上 -> 执行 {self.dir_inward}+跳跃 强制脱绳")
        self._hold(self.dir_inward)
        self.gc.tap_key(c.jump_key, 0.08)
        self.settle_my = c.my
        self.settle_change = c.now
        self._enter("SETTLE", c.now)
        return RUNNING

    def _st_settle(self, c):
        el = c.now - self.t_state
        if el >= 0.35:
            self._release_lr()
        if c.my != self.settle_my:
            self.settle_my = c.my
            self.settle_change = c.now
        if el >= 0.5 and c.now - self.settle_change >= 0.3:
            # 落地后再做一次横向探针
            self.probe_mx = c.mx
            self._hold(self.dir_inward)
            self._enter("PROBE", c.now)
        return RUNNING

    def _st_verify(self, c):
        if c.now - self.t_state < 0.12:
            return RUNNING
        plat = self.nav.graph.locate(c.xw, c.yw)
        if plat is not None and plat.id == self.edge.dst:
            self.gc.clear_movement()
            return SUCCESS
        where = f"#{plat.id}" if plat else "未知"
        return self._fail(f"登顶后不在目标平台 #{self.edge.dst} (当前 {where})")


# ============================================================================
# 动作原语: 下跳
# ============================================================================
class DropDown:
    """ALIGN -> DOWN_HELD -> FALLING -> VERIFY"""
    ALIGN_TOL = 24          # 世界像素
    ALIGN_TIMEOUT = 8.0
    NO_FALL_TIMEOUT = 0.9   # 下跳后纵坐标没有增加 => 该处无法下跳
    LAND_STABLE = 0.3
    MAX_ATTEMPTS = 2

    def __init__(self, nav, edge, drop_x):
        self.nav = nav
        self.gc = nav.gc
        self.edge = edge
        self.drop_x = drop_x
        self.state = "ALIGN"
        self.t_state = None
        self.attempt = 0
        self.settle_since = None
        self.fail_reason = ""

    def _enter(self, st, now):
        self.state = st
        self.t_state = now

    def _fail(self, reason):
        self.gc.clear_movement()
        self.fail_reason = reason
        return FAIL

    def describe(self):
        return f"下跳[{self.state}] -> #{self.edge.dst} @x={self.drop_x}"

    def update(self, c):
        if self.t_state is None:
            self.t_state = c.now
        return getattr(self, "_st_" + self.state.lower())(c)

    def _st_align(self, c):
        if c.now - self.t_state > self.ALIGN_TIMEOUT:
            return self._fail("走向下跳点超时")
        dx = self.drop_x - c.xw
        if abs(dx) > self.ALIGN_TOL:
            d = "RIGHT" if dx > 0 else "LEFT"
            self.gc.release_key("LEFT" if d == "RIGHT" else "RIGHT")
            self.gc.press_key(d)
            self.settle_since = None
            return RUNNING
        self.gc.release_key("LEFT")
        self.gc.release_key("RIGHT")
        if self.settle_since is None:
            self.settle_since = c.now
            return RUNNING
        if c.now - self.settle_since < 0.12:
            return RUNNING
        self.start_my = c.my
        self.gc.press_key("DOWN")
        self._enter("DOWN_HELD", c.now)
        return RUNNING

    def _st_down_held(self, c):
        if c.now - self.t_state >= 0.06:
            self.gc.tap_key(c.jump_key, 0.08)
            self.falling = False
            self.down_released = False
            self.last_my = c.my
            self.last_change = c.now
            self._enter("FALLING", c.now)
        return RUNNING

    def _st_falling(self, c):
        el = c.now - self.t_state
        if el >= 0.25 and not self.down_released:
            self.gc.release_key("DOWN")
            self.down_released = True
        if c.my >= self.start_my + 2:
            self.falling = True
        if c.my != self.last_my:
            self.last_my = c.my
            self.last_change = c.now
        if self.falling and c.now - self.last_change >= self.LAND_STABLE:
            self._enter("VERIFY", c.now)
            return RUNNING
        if not self.falling and el >= self.NO_FALL_TIMEOUT:
            self.gc.release_key("DOWN")
            self.attempt += 1
            if self.attempt >= self.MAX_ATTEMPTS:
                return self._fail("该位置无法下跳 (可能禁止下跳)")
            self.settle_since = None
            self._enter("ALIGN", c.now)
        return RUNNING

    def _st_verify(self, c):
        plat = self.nav.graph.locate(c.xw, c.yw)
        if plat is not None and plat.id == self.edge.dst:
            return SUCCESS
        where = f"#{plat.id}" if plat else "未知"
        return self._fail(f"下跳落点不是目标平台 #{self.edge.dst} (当前 {where})")


# ============================================================================
# 多层导航器
# ============================================================================
class LayerNavigator:
    MAX_DWELL = 25.0        # 单层最长停留时间 (多层模式下到时强制换层)
    MIN_DWELL = 3.0         # 单层最短停留时间 (防刚踩上边缘立即误换层)
    SESSION_GAP = 3.0       # 超过该时间未被调用 => 视为新一轮挂机，重置会话
    PLATFORM_STABLE = 0.25  # 平台切换需稳定的时间 (过滤跳跃/受击造成的瞬时误吸附)
    BAN_TIME = 25.0         # 失败过多的边临时禁用时长 (默认与 MAX_DWELL 一致，运行时跟随配置)
    EDGE_MAX_FAILS = 3      # 失败尝试达到 3 次后再临时禁用

    def __init__(self, gc, mp):
        self.gc = gc
        self.mp = mp
        self.graph = PlatformGraph(mp)
        map_key = os.path.splitext(os.path.basename(mp.xml_path or "unknown"))[0]
        self.profile = MapProfile(map_key)
        self.marked = [p for p in self.profile.marked if p in self.graph.platforms]
        self.turns_per_layer = 1
        self.turn_count = 0
        # 确保所有已标记平台都默认具备中点下跳点
        for pid in self.marked:
            if str(pid) not in self.profile.drop_points and pid in self.graph.platforms:
                self.profile.drop_points[str(pid)] = int(self.graph.platforms[pid].center_x)
        self.reset_session()

    # ------------------------------------------------------------------ 会话
    def reset_session(self):
        if getattr(self, "primitive", None) is not None:
            self.gc.clear_movement()
        self.primitive = None
        self.home_pid = None
        self.target_idx = None
        self.route = None
        self.route_idx = 0
        self.hunt_pid = None
        self.arrive_time = 0.0
        self.touched = set()
        self.turn_count = 0
        self.banned = {}
        self.edge_fails = {}
        self.last_step = 0.0
        self.no_route_until = 0.0
        self.cur_pid = None
        self._cand_pid = None
        self._cand_since = 0.0
        self.status = "IDLE"
        self._last_log = 0.0

    # ------------------------------------------------------------------ 标记
    def toggle_mark_at(self, xw, yw):
        plat = self.graph.locate(xw, yw)
        if plat is None:
            return None, False
        if plat.id in self.marked:
            self.marked.remove(plat.id)
            self.profile.drop_points.pop(str(plat.id), None)
            on = False
        else:
            self.marked.append(plat.id)
            # 默认下跳点为该平台中点
            self.profile.drop_points[str(plat.id)] = int(plat.center_x)
            on = True
        self._save_marks()
        return plat, on

    def clear_marks(self):
        self.marked = []
        self.profile.drop_points = {}
        self._save_marks()

    def _save_marks(self):
        self.profile.marked = list(self.marked)
        self.profile.save()
        self.target_idx = None
        self.route = None

    def targets(self):
        if self.marked:
            return sorted(self.marked, key=lambda pid: self.graph.platforms[pid].avg_y)  # 自上而下
        return [self.home_pid] if self.home_pid is not None else []

    # ------------------------------------------------------------------ 梯子校准
    def _sig(self, c):
        return [c.crop_w, c.crop_h, self.mp.offset_x, self.mp.offset_y]

    def _pred_mx(self, lr, c):
        return self.mp.world_to_minimap(lr.x, lr.y_bottom, crop_w=c.crop_w, crop_h=c.crop_h)[0]

    def ladder_target_mx(self, lr, c):
        sig = self._sig(c)
        entry = self.profile.ladders.get(str(lr.id))
        if entry and entry.get("sig") == sig:
            return int(entry["mx"])
        # 未校准: 用已校准梯子的平均偏差修正 XML 预测值
        deltas = []
        for lr2 in self.mp.ladder_ropes:
            e2 = self.profile.ladders.get(str(lr2.id))
            if e2 and e2.get("sig") == sig:
                deltas.append(int(e2["mx"]) - self._pred_mx(lr2, c))
        corr = sorted(deltas)[len(deltas) // 2] if deltas else 0
        return self._pred_mx(lr, c) + corr

    def calibrate_ladder(self, lr, mx, c):
        pred = self._pred_mx(lr, c)
        old = self.profile.ladders.get(str(lr.id))
        self.profile.ladders[str(lr.id)] = {"mx": int(mx), "sig": self._sig(c)}
        self.profile.save()
        if not old or old.get("mx") != int(mx):
            print(f"📐 [梯子自校准] lr{lr.id}: XML 预测 mx={pred} -> 实测 mx={mx} (偏差 {mx - pred:+d}px)，已保存")

    # ------------------------------------------------------------------ 外部钩子
    def is_locked(self):
        """正在执行爬梯/下跳原语：禁止战斗打断、禁止安全区拦截"""
        return self.primitive is not None

    def transit_x_on(self, pid):
        """当前平台上的跃迁目标点 (世界 X)，供安全区放行"""
        if self.route and self.route_idx < len(self.route):
            e, ax = self.route[self.route_idx]
            if e.src == pid:
                return ax
        return None

    def on_combat(self):
        # 战斗不重置折返与巡逻计数，边巡逻边推怪是正常打怪行为
        pass

    def on_patrol_turn(self, side):
        self.touched.add(side)
        self.turn_count += 1

    def current_route_edges(self):
        if not self.route:
            return []
        return self.route[self.route_idx:]

    # ------------------------------------------------------------------ 主循环
    def _log(self, msg, now, force=False):
        if force or now - self._last_log >= 1.0:
            self._last_log = now
            print(msg)

    def _stable_platform(self, c):
        plat = self.graph.locate(c.xw, c.yw)
        if plat is None:
            return None
        if self.cur_pid is None:
            self.cur_pid = plat.id
        if plat.id != self.cur_pid:
            if self._cand_pid != plat.id:
                self._cand_pid = plat.id
                self._cand_since = c.now
            elif c.now - self._cand_since >= self.PLATFORM_STABLE:
                self.cur_pid = plat.id
                self._cand_pid = None
        else:
            self._cand_pid = None
        return self.graph.platforms[self.cur_pid]

    def _walk(self, d):
        self.gc.release_key("LEFT" if d == "RIGHT" else "RIGHT")
        self.gc.press_key(d)

    def step(self, c):
        """
        返回:
          "HUNT" - 当前就在打怪层，调用方执行平台内巡逻
          "BUSY" - 导航器本帧已接管按键 (走向梯子/下跳点、爬梯、下跳)
          "WAIT" - 暂时无法定位平台 (空中等)，调用方停止移动
        """
        now = c.now
        if getattr(c, "turns_per_layer", None):
            self.turns_per_layer = c.turns_per_layer
        if getattr(c, "max_dwell", None):
            self.MAX_DWELL = c.max_dwell

        if self.last_step > 0 and now - self.last_step > self.SESSION_GAP:
            self.reset_session()
        self.last_step = now

        # 1. 执行中的动作原语
        if self.primitive is not None:
            res = self.primitive.update(c)
            if res == RUNNING:
                self.status = self.primitive.describe()
                return "BUSY"
            edge = self.route[self.route_idx][0]
            if res == SUCCESS:
                print(f"✅ [导航] {edge} 完成")
                self.edge_fails.pop(edge.key, None)
                self.route_idx += 1
            else:
                fails = self.edge_fails.get(edge.key, 0) + 1
                self.edge_fails[edge.key] = fails
                print(f"❌ [导航] {edge} 失败: {self.primitive.fail_reason} (第 {fails} 次)")
                if fails >= self.EDGE_MAX_FAILS:
                    ban_duration = float(self.MAX_DWELL)
                    self.banned[edge.key] = now + ban_duration
                    self.edge_fails.pop(edge.key, None)
                    print(f"🚫 [导航] {edge} 连续失败 {self.EDGE_MAX_FAILS} 次，暂时禁用 {ban_duration:.0f}s (匹配单层最长打怪时间)，改走其他路线/原地打怪")
                self.route = None
            self.primitive = None
            self.cur_pid = None          # 跃迁后立即重新定位
            self.gc.clear_movement()
            return "BUSY"

        # 2. 定位当前平台
        plat = self._stable_platform(c)
        if plat is None:
            self.status = "等待吸附平台"
            return "WAIT"
        if self.home_pid is None:
            self.home_pid = plat.id
            if not self.marked:
                print(f"🏠 [导航] 未标记打怪层，锁定当前平台 #{plat.id} 为唯一打怪层")

        tg = self.targets()
        if not tg:
            return "HUNT"
        if self.target_idx is None or self.target_idx >= len(tg):
            if plat.id in tg:
                self.target_idx = tg.index(plat.id)
            else:   # 从高度最接近的目标层开始
                self.target_idx = min(range(len(tg)),
                                      key=lambda i: abs(self.graph.platforms[tg[i]].avg_y - plat.avg_y))
        target = tg[self.target_idx]

        # 3. 已在目标层: 打怪，并判定本层是否清理完毕
        if plat.id == target:
            if self.hunt_pid != target:
                self.hunt_pid = target
                self.arrive_time = now
                self.touched.clear()
                self.turn_count = 0
                self.route = None
                drop_x_info = self.profile.drop_points.get(str(target), int(plat.center_x))
                print(f"🎯 [导航] 到达打怪层 #{target} ({self.target_idx + 1}/{len(tg)}) | 默认下跳点: X={drop_x_info}")
            
            dwell = now - self.arrive_time
            done_turns = (self.turn_count >= self.turns_per_layer) and (dwell >= self.MIN_DWELL)
            done_lap = ({"L", "R"} <= self.touched) and (dwell >= self.MIN_DWELL)
            done_time = dwell >= self.MAX_DWELL

            # 若通往下一层的路线仍在失败冷却中，必须打满单层最长打怪时间才允许切换，避免过早换层撞墙
            next_target = tg[(self.target_idx + 1) % len(tg)] if len(tg) > 1 else target
            route_blocked_by_ban = False
            if self.banned:
                r_with_ban = self.graph.find_route(plat.id, c.xw, next_target, c.ml, c.mr,
                                                   drop_points=self.profile.drop_points, banned=self.banned)
                if not r_with_ban:
                    r_without_ban = self.graph.find_route(plat.id, c.xw, next_target, c.ml, c.mr,
                                                          drop_points=self.profile.drop_points, banned=())
                    if r_without_ban:
                        route_blocked_by_ban = True

            if route_blocked_by_ban:
                ready_to_switch = done_time
            else:
                ready_to_switch = (done_turns or done_lap or done_time)

            if len(tg) > 1 and ready_to_switch:
                if done_time:
                    why = f"停留达到上限 {self.MAX_DWELL:.0f}s"
                elif done_turns:
                    why = f"巡逻打怪已完成 {self.turn_count}/{self.turns_per_layer} 趟"
                else:
                    why = "遍历平台两侧"
                self.target_idx = (self.target_idx + 1) % len(tg)
                target = tg[self.target_idx]
                self.hunt_pid = None
                self.turn_count = 0
                print(f"🔁 [导航] 层 #{plat.id} {why}，前往下一打怪层 #{target}")
            else:
                self.status = f"打怪 层#{target} ({self.target_idx + 1}/{len(tg)}) [{self.turn_count}/{self.turns_per_layer}趟]"
                return "HUNT"
        else:
            self.hunt_pid = None

        # 4. 跃迁: 规划/执行路线
        need_plan = (self.route is None or self.route_idx >= len(self.route)
                     or self.route[self.route_idx][0].src != plat.id)
        if need_plan:
            if now < self.no_route_until:
                self.status = f"无路径到 #{target}，原地打怪"
                return "HUNT"
            self.banned = {k: t for k, t in self.banned.items() if t > now}
            route = self.graph.find_route(plat.id, c.xw, target, c.ml, c.mr, drop_points=self.profile.drop_points, banned=self.banned)
            if not route:
                ban_wait = 0.0
                if self.banned:
                    ban_wait = max(0.0, min(self.banned.values()) - now)

                retry_wait = max(ban_wait, 5.0)
                self.no_route_until = now + retry_wait

                if ban_wait > 0:
                    print(f"⚠️ [导航] 暂无从 #{plat.id} 到 #{target} 的可用路线 (存在失败冷却边，约 {ban_wait:.0f}s 后解禁重试)，期间留在本层打怪")
                else:
                    print(f"⚠️ [导航] 找不到从 #{plat.id} 到 #{target} 的路线 (仅支持 下跳/爬梯)，5s 后重试，期间原地打怪")

                if len(tg) > 1:
                    # 如果当前所在平台就是打怪层，优先锁回当前平台安心打怪，避免连续切层刷屏
                    if plat.id in tg:
                        self.target_idx = tg.index(plat.id)
                    else:
                        self.target_idx = (self.target_idx + 1) % len(tg)
                return "HUNT"
            self.route = route
            self.route_idx = 0
            print(f"🧭 [导航] #{plat.id} -> #{target}: " + " → ".join(str(e) for e, _ in route))

        edge, ax = self.route[self.route_idx]
        if edge.kind == Edge.CLIMB:
            if self.primitive is None:
                self.primitive = LadderClimber(self, edge)
            self.primitive.update(c)
            self.status = self.primitive.describe()
        else:
            if abs(ax - c.xw) <= 40:
                self.primitive = DropDown(self, edge, ax)
                self.primitive.update(c)
                self.status = self.primitive.describe()
            else:
                self._walk("RIGHT" if ax > c.xw else "LEFT")
                self.status = f"走向下跳点 x={ax} -> #{edge.dst}"
        return "BUSY"

import math
import heapq

class NavAction:
    WALK = "WALK"
    CLIMB_UP = "CLIMB_UP"
    CLIMB_DOWN = "CLIMB_DOWN"
    DROP_DOWN = "DROP_DOWN"
    JUMP = "JUMP"


class NavStep:
    """A* 寻路规划输出的单步动作"""
    def __init__(self, action, target_x, target_y, foothold_id=None, extra_info=None):
        self.action = action              # WALK, CLIMB_UP, CLIMB_DOWN, DROP_DOWN, JUMP
        self.target_x = int(target_x)     # 动作目标世界 X 坐标
        self.target_y = int(target_y)     # 动作目标世界 Y 坐标
        self.foothold_id = foothold_id    # 目标到达的平台 ID
        self.extra_info = extra_info or {}

    def __repr__(self):
        return f"NavStep({self.action}, X={self.target_x}, Y={self.target_y}, fh={self.foothold_id})"


class NavGraph:
    """
    2D 地图拓扑路网图 (连接所有平台、梯子、跳跃点与下跳点)
    """
    def __init__(self, map_parser):
        self.mp = map_parser
        self.adj = {} # {foothold_id: [(neighbor_fh_id, action_type, cost, via_x, via_y_start, via_y_end), ...]}
        self.build_graph()

    def build_graph(self):
        self.adj.clear()
        
        # 初始化所有平台节点
        for fh_id in self.mp.footholds.keys():
            self.adj[fh_id] = []

        # 0. 添加同层连续/无缝连接平台平移行走 (WALK)
        for fh1 in self.mp.horizontal_fhs:
            for fh2 in self.mp.horizontal_fhs:
                if fh1.id == fh2.id:
                    continue
                # 右接左 (端点贴合容差 10px, 高度差 <= 15px)
                if abs(fh1.x2 - fh2.x1) <= 10 and abs(fh1.y2 - fh2.y1) <= 15:
                    self.adj[fh1.id].append((fh2.id, NavAction.WALK, 5, fh1.x2, fh1.y2, fh2.y1))
                    self.adj[fh2.id].append((fh1.id, NavAction.WALK, 5, fh2.x1, fh2.y1, fh1.y2))

        # 1. 添加梯子/绳索双向连接 (CLIMB_UP / CLIMB_DOWN)
        for lr in self.mp.ladder_ropes:
            if lr.bottom_foothold and lr.top_foothold:
                bot_id = lr.bottom_foothold.id
                top_id = lr.top_foothold.id
                climb_dist = abs(lr.y_bottom - lr.y_top)
                
                # 向上攀爬 (底 -> 顶)
                cost_up = climb_dist * 1.2 + 20
                self.adj[bot_id].append((top_id, NavAction.CLIMB_UP, cost_up, lr.x, lr.y_bottom, lr.y_top))
                
                # 向下攀爬 (顶 -> 底)
                cost_down = climb_dist * 0.9 + 15
                self.adj[top_id].append((bot_id, NavAction.CLIMB_DOWN, cost_down, lr.x, lr.y_top, lr.y_bottom))

        # 2. 添加平台自然下跳连接 (DROP_DOWN)
        for fh_top in self.mp.horizontal_fhs:
            top_id = fh_top.id
            for fh_bot in self.mp.horizontal_fhs:
                if fh_top.id == fh_bot.id:
                    continue
                bot_id = fh_bot.id
                
                # 检查高度差 (顶部 Y 较小，底部 Y 较大)
                dy = fh_bot.avg_y - fh_top.avg_y
                if 40 <= dy <= 220:
                    # 检查水平重叠区间 [overlap_x1, overlap_x2]
                    ox1 = max(fh_top.x1, fh_bot.x1)
                    ox2 = min(fh_top.x2, fh_bot.x2)
                    if ox2 - ox1 >= 20: # 至少有 20 像素的下落重叠区域
                        mid_x = (ox1 + ox2) / 2.0
                        cost_drop = dy * 0.4 + 10 # 下落非常快
                        self.adj[top_id].append((bot_id, NavAction.DROP_DOWN, cost_drop, mid_x, fh_top.get_y_at_x(mid_x), fh_bot.get_y_at_x(mid_x)))

        # 3. 添加平级/近距离/树枝间跳跃连接 (JUMP)
        for fh_a in self.mp.horizontal_fhs:
            for fh_b in self.mp.horizontal_fhs:
                if fh_a.id == fh_b.id:
                    continue
                
                min_xa, max_xa = min(fh_a.x1, fh_a.x2), max(fh_a.x1, fh_a.x2)
                min_xb, max_xb = min(fh_b.x1, fh_b.x2), max(fh_b.x1, fh_b.x2)
                
                # 水平间距
                dx = max(0, min_xb - max_xa, min_xa - max_xb)
                dy = fh_b.avg_y - fh_a.avg_y

                # 平台跳跃 (支持向上跳跃 -100px 或向下跨跳 130px，横向间隙 <= 180px)
                if dx <= 180 and -100 <= dy <= 130:
                    jump_x_from = max_xa if min_xb > max_xa else min_xa
                    jump_x_to = min_xb if min_xb > max_xa else max_xb
                    cost_jump = dx + abs(dy) * 0.8 + 25
                    self.adj[fh_a.id].append((fh_b.id, NavAction.JUMP, cost_jump, jump_x_from, fh_a.get_y_at_x(jump_x_from), fh_b.get_y_at_x(jump_x_to)))


class AStarPathfinder:
    """
    A* 算法全局寻路器
    """
    def __init__(self, map_parser):
        self.mp = map_parser
        self.nav_graph = NavGraph(map_parser)

    def _find_nearest_foothold(self, wx, wy):
        """寻找离指定世界坐标最近的平台"""
        best_fh = None
        min_dist = 999999
        for fh in self.mp.horizontal_fhs:
            min_x, max_x = min(fh.x1, fh.x2), max(fh.x1, fh.x2)
            dx = max(0, min_x - wx, wx - max_x)
            dy = abs(wy - fh.avg_y)
            d = math.hypot(dx, dy)
            if d < min_dist:
                min_dist = d
                best_fh = fh
        return best_fh

    def find_path(self, start_world_x, start_world_y, target_world_x, target_world_y):
        """
        在全局拓扑图上搜索从起点世界坐标到终点世界坐标的最优动作序列
        """
        # 1. 查找起点与终点所在的平台 (优先吸附，未命中时兜底最近平台)
        start_fh = self.mp.snap_to_foothold(start_world_x, start_world_y) or self._find_nearest_foothold(start_world_x, start_world_y)
        target_fh = self.mp.snap_to_foothold(target_world_x, target_world_y) or self._find_nearest_foothold(target_world_x, target_world_y)

        if not start_fh or not target_fh:
            # 兜底：直接向目标点水平移动
            return [NavStep(NavAction.WALK, target_world_x, target_world_y)]

        # 2. 如果同在一块平台上，直接走过去
        if start_fh.id == target_fh.id:
            return [NavStep(NavAction.WALK, target_world_x, target_fh.get_y_at_x(target_world_x), foothold_id=target_fh.id)]

        # 3. 跨平台 A* 搜索
        # 优先队列元素: (f_score, current_fh_id, current_x, path_steps)
        start_node_x = start_world_x
        open_set = []
        heapq.heappush(open_set, (0, start_fh.id, start_node_x, []))
        
        g_scores = {start_fh.id: 0}
        visited = set()

        while open_set:
            f_score, curr_fh_id, curr_x, path = heapq.heappop(open_set)

            if curr_fh_id in visited:
                continue
            visited.add(curr_fh_id)

            curr_fh = self.mp.footholds[curr_fh_id]

            # 抵达目标平台！补充在该平台走向终点的最终 WALK 动作
            if curr_fh_id == target_fh.id:
                final_path = list(path)
                final_path.append(NavStep(NavAction.WALK, target_world_x, target_fh.get_y_at_x(target_world_x), foothold_id=target_fh.id))
                return self._simplify_path(final_path)

            # 遍历邻接边
            neighbors = self.nav_graph.adj.get(curr_fh_id, [])
            for (next_fh_id, action_type, edge_cost, via_x, via_y_start, via_y_end) in neighbors:
                if next_fh_id in visited:
                    continue

                # 平台内走动到起跳/攀爬点的距离开销
                walk_dist = abs(via_x - curr_x)
                tentative_g = g_scores[curr_fh_id] + walk_dist + edge_cost

                if next_fh_id not in g_scores or tentative_g < g_scores[next_fh_id]:
                    g_scores[next_fh_id] = tentative_g
                    
                    # 启发式函数 (Euclidean Distance to Target)
                    next_fh = self.mp.footholds[next_fh_id]
                    h_score = math.hypot(next_fh.x1 - target_world_x, next_fh.avg_y - target_world_y)
                    f_total = tentative_g + h_score

                    # 构建动作序列
                    new_path = list(path)
                    if action_type == NavAction.WALK:
                        # 连续平台平移，只需记录目标点
                        new_path.append(NavStep(NavAction.WALK, via_x, via_y_end, foothold_id=next_fh_id))
                    else:
                        # 梯子/下跳/跳跃：先走至机关点，再执行动作
                        ground_y = curr_fh.get_y_at_x(via_x) if hasattr(curr_fh, 'get_y_at_x') else via_y_start
                        new_path.append(NavStep(NavAction.WALK, via_x, ground_y, foothold_id=curr_fh_id))
                        extra_info = {
                            "y_bottom": via_y_start if action_type == NavAction.CLIMB_UP else via_y_end,
                            "y_top": via_y_end if action_type == NavAction.CLIMB_UP else via_y_start,
                            "ground_y": ground_y
                        }
                        new_path.append(NavStep(action_type, via_x, via_y_end, foothold_id=next_fh_id, extra_info=extra_info))

                    heapq.heappush(open_set, (f_total, next_fh_id, via_x, new_path))

        # 搜索无通路时兜底：水平走向终点 X
        return [NavStep(NavAction.WALK, target_world_x, target_world_y)]

    def _simplify_path(self, steps):
        """合并连续的同向 WALK 动作，让寻路路径干练顺畅"""
        if not steps or len(steps) <= 1:
            return steps

        simplified = []
        for s in steps:
            if simplified and simplified[-1].action == NavAction.WALK and s.action == NavAction.WALK:
                # 连续两个行走步骤，如果高度相同或接近，直接更新终点
                if abs(simplified[-1].target_y - s.target_y) <= 15:
                    simplified[-1] = s
                    continue
            simplified.append(s)

        return simplified

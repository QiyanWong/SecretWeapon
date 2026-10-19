import os
import sys
import json
import cv2
import numpy as np
import math

# 数据集与路线配置存储路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
ROUTES_DIR = os.path.join(DATASET_DIR, "routes")
os.makedirs(ROUTES_DIR, exist_ok=True)
DEFAULT_ROUTE_PATH = os.path.join(ROUTES_DIR, "map_route_default.json")
DEFAULT_MINIMAP_CONFIG_PATH = os.path.join(BASE_DIR, "dataset", "minimap_config.json")


class PathNode:
    """带动作属性与区间包围盒的拓扑寻路节点/线段"""
    def __init__(self, node_id, x, y, action_type="WALK", label="", x2=None, y2=None):
        self.node_id = node_id          # 节点序号
        self.x = int(x)                 # 起点/节点 X
        self.y = int(y)                 # 起点/节点 Y
        self.x2 = int(x2) if x2 is not None else int(x) # 路段终点 X (仅 WALK_SEGMENT)
        self.y2 = int(y2) if y2 is not None else int(y) # 路段终点 Y (仅 WALK_SEGMENT)
        self.action_type = action_type  # 动作类型: 'WALK_SEGMENT', 'JUMP', 'CLIMB', 'DOWN_JUMP'
        self.label = label if label else f"N{node_id}_{action_type}"

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "x": self.x,
            "y": self.y,
            "x2": self.x2,
            "y2": self.y2,
            "action_type": self.action_type,
            "label": self.label
        }

    @staticmethod
    def from_dict(d):
        return PathNode(
            node_id=d.get("node_id", 1),
            x=d.get("x", 0),
            y=d.get("y", 0),
            x2=d.get("x2", d.get("x", 0)),
            y2=d.get("y2", d.get("y", 0)),
            action_type=d.get("action_type", "WALK"),
            label=d.get("label", "")
        )


class RouteManager:
    """路线录制与有向闭环路段管理器"""
    def __init__(self):
        self.nodes = []
        self.current_target_index = 0
        self.is_recording = False

    def add_node(self, x, y, action_type="WALK", label=""):
        """按下 F1/F2/F3 插入顺序节点"""
        node_id = len(self.nodes) + 1
        node = PathNode(node_id, x, y, action_type, label)
        self.nodes.append(node)
        return node

    def remove_last_node(self):
        if self.nodes:
            removed = self.nodes.pop()
            if self.current_target_index >= len(self.nodes):
                self.current_target_index = max(0, len(self.nodes) - 1)
            return removed
        return None

    def clear(self):
        self.nodes.clear()
        self.current_target_index = 0

    def get_current_target_node(self):
        if self.nodes and 0 <= self.current_target_index < len(self.nodes):
            return self.nodes[self.current_target_index]
        return None

    def get_current_next_node(self):
        """获取当前节点向后的下一个节点 (闭环)"""
        if self.nodes and len(self.nodes) > 1:
            next_idx = (self.current_target_index + 1) % len(self.nodes)
            return self.nodes[next_idx]
        return self.get_current_target_node()

    def is_in_directed_segment_rectangle(self, segment_index, px, py, margin_x=4, margin_y=4):
        """判定玩家坐标 (px, py) 是否落入第 segment_index 段 (P_i -> P_{i+1}) 的有向对角矩形包围盒内"""
        if not self.nodes or not (0 <= segment_index < len(self.nodes)):
            return False
        
        n1 = self.nodes[segment_index]
        if len(self.nodes) == 1:
            return math.hypot(n1.x - px, n1.y - py) <= 6.0

        n2 = self.nodes[(segment_index + 1) % len(self.nodes)]
        min_x = min(n1.x, n2.x) - margin_x
        max_x = max(n1.x, n2.x) + margin_x
        min_y = min(n1.y, n2.y) - margin_y
        max_y = max(n1.y, n2.y) + margin_y
        return (min_x <= px <= max_x) and (min_y <= py <= max_y)

    def find_nearest_node_index(self, player_x, player_y):
        """寻找整条路线中离玩家坐标最近的节点索引"""
        if not self.nodes:
            return 0
        best_dist = float('inf')
        best_index = 0
        for i, node in enumerate(self.nodes):
            dist = math.hypot(node.x - player_x, node.y - player_y)
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return best_index

    def save_to_json(self, file_path=DEFAULT_ROUTE_PATH, crop_box=None):
        try:
            data = {
                "version": "1.1",
                "total_nodes": len(self.nodes),
                "nodes": [n.to_dict() for n in self.nodes]
            }
            if crop_box:
                data["minimap_crop"] = {
                    "left": crop_box[0],
                    "top": crop_box[1],
                    "width": crop_box[2],
                    "height": crop_box[3]
                }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"【成功】路线与配置数据已保存至: {file_path}")
            return True
        except Exception as e:
            print(f"保存路线异常: {e}")
            return False

    def load_from_json(self, file_path=DEFAULT_ROUTE_PATH):
        """返回 (success, crop_box)"""
        if not os.path.exists(file_path):
            return False, None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.nodes = [PathNode.from_dict(d) for d in data.get("nodes", [])]
            self.current_target_index = 0
            
            crop_box = None
            if "minimap_crop" in data:
                c = data["minimap_crop"]
                crop_box = (c.get("left", 20), c.get("top", 146), c.get("width", 272), c.get("height", 170))
                
            print(f"【成功】已成功载入路线文件 ({len(self.nodes)} 个节点): {file_path}")
            return True, crop_box
        except Exception as e:
            print(f"读取路线异常: {e}")
            return False, None


class MinimapTracker:
    """小地图定位与感知跟踪器 (支持模板匹配)"""
    def __init__(self, crop_box=(20, 146, 272, 170)):
        self.crop_left, self.crop_top, self.crop_w, self.crop_h = crop_box
        self.minimap_scale = 1.5  # 默认缩放比例，适应 1366x768 常用放大
        self.player_template = None
        self.last_mask_img = None
        self.last_player_pos = None
        
        self.template_path = os.path.join(DATASET_DIR, "player_template.png")
        self.load_config()
        self.load_player_template()

    def load_player_template(self):
        self.template_mask = None
        if os.path.exists(self.template_path):
            img = cv2.imread(self.template_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if len(img.shape) == 3 and img.shape[2] == 4:
                    self.player_template = img[:, :, :3]
                    self.template_mask = img[:, :, 3]
                    print(f"✨ 【官方透明模板载入成功】已启用 4 通道 RGBA Alpha 掩膜匹配引擎 (尺寸: {img.shape})")
                else:
                    self.player_template = img
                    print(f"【模板载入成功】已载入 3 通道 BGR 模板 (尺寸: {img.shape})")
        else:
            print(f"【模板读取】未找到玩家模板，请放入: {self.template_path}")

    def set_player_template(self, template_bgr):
        """动态寻找黄点边界，极度紧凑地裁剪模板，彻底剥离背景颜色"""
        if template_bgr is not None and template_bgr.size > 0:
            hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
            # 宽泛的黄色阈值
            lower_yellow = np.array([10, 80, 80])
            upper_yellow = np.array([45, 255, 255])
            mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # 找到黄点在 15x15 图片里的真实边界
            coords = cv2.findNonZero(mask)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                tight_bgr = template_bgr[y:y+h, x:x+w]
            else:
                ch, cw = template_bgr.shape[:2]
                tight_bgr = template_bgr[ch//2-3:ch//2+4, cw//2-3:cw//2+4]
            
            self.player_template = tight_bgr
            self.template_mask = None
            cv2.imwrite(self.template_path, tight_bgr)
            print(f"【模板提取成功】已保存至: {self.template_path}")

    def set_crop_box(self, left, top, width, height):
        self.crop_left = max(0, int(left))
        self.crop_top = max(0, int(top))
        self.crop_w = max(20, int(width))
        self.crop_h = max(20, int(height))
        self.save_config()

    def save_config(self, file_path=DEFAULT_MINIMAP_CONFIG_PATH):
        try:
            data = {
                "crop_left": self.crop_left,
                "crop_top": self.crop_top,
                "crop_w": self.crop_w,
                "crop_h": self.crop_h,
                "minimap_scale": getattr(self, "minimap_scale", 1.5)
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"【小地图配置已保存】: {data}")
            return True
        except Exception as e:
            print(f"保存小地图配置异常: {e}")
            return False

    def load_config(self, file_path=DEFAULT_MINIMAP_CONFIG_PATH):
        if not os.path.exists(file_path):
            return False
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.crop_left = data.get("crop_left", self.crop_left)
            self.crop_top = data.get("crop_top", self.crop_top)
            self.crop_w = data.get("crop_w", self.crop_w)
            self.crop_h = data.get("crop_h", self.crop_h)
            self.minimap_scale = float(data.get("minimap_scale", 1.5))
            print(f"【小地图配置已恢复】: Left={self.crop_left}, Top={self.crop_top}, W={self.crop_w}, H={self.crop_h}, Scale={self.minimap_scale:.2f}x")
            return True
        except Exception as e:
            print(f"读取小地图配置异常: {e}")
            return False

    def crop_minimap(self, game_frame_bgr):
        """从小地图区域裁剪 BGR 图像"""
        if game_frame_bgr is None:
            return None
        h, w, _ = game_frame_bgr.shape
        cl = max(0, min(w - 10, self.crop_left))
        ct = max(0, min(h - 10, self.crop_top))
        cr = max(cl + 10, min(w, cl + self.crop_w))
        cb = max(ct + 10, min(h, ct + self.crop_h))
        return game_frame_bgr[ct:cb, cl:cr].copy()

    def locate_player_pos(self, minimap_bgr):
        """
        三引擎高精定位 2D 游戏小地图玩家标志：
        引擎 1: 官方 RGBA 模板 + Alpha 掩膜归一化相关性匹配 (TM_CCORR_NORMED, 100% 隔离背景)
        引擎 2: HSV 金黄点高精度连通域提取 (自适应多帧闪烁)
        引擎 3: 传统 BGR 模板匹配与上一帧记忆中继
        """
        if minimap_bgr is None or minimap_bgr.size == 0:
            return None, 0.0

        # === 引擎 1: 官方 Alpha Mask 模板匹配 (最高优先级，100% 准确) ===
        if self.player_template is not None and self.player_template.size > 0:
            th, tw = self.player_template.shape[:2]
            if minimap_bgr.shape[0] >= th and minimap_bgr.shape[1] >= tw:
                if getattr(self, "template_mask", None) is not None:
                    res = cv2.matchTemplate(minimap_bgr, self.player_template, cv2.TM_CCORR_NORMED, mask=self.template_mask)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val >= 0.70:
                        cx = max_loc[0] + tw // 2
                        cy = max_loc[1] + th // 2
                        self.last_player_pos = (cx, cy)
                        return (cx, cy), float(max_val)
                else:
                    res = cv2.matchTemplate(minimap_bgr, self.player_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val >= 0.65:
                        cx = max_loc[0] + tw // 2
                        cy = max_loc[1] + th // 2
                        self.last_player_pos = (cx, cy)
                        return (cx, cy), float(max_val)

        # === 引擎 2: HSV 金黄点像素特征提取 ===
        hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([12, 65, 110])
        upper_yellow = np.array([42, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        self.last_mask_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if 2 <= area <= 65 and w <= 14 and h <= 14:
                cx = int(round(centroids[i][0]))
                cy = int(round(centroids[i][1]))
                self.last_player_pos = (cx, cy)
                return (cx, cy), 0.95

        # === 引擎 3: 记忆中继 ===
        if self.last_player_pos is not None:
            return self.last_player_pos, 0.50

        return None, 0.0

    def draw_minimap_dashboard(self, minimap_bgr, player_pos, route_manager, decision_engine=None):
        """
        在小地图副本图像上绘制有向闭环路段、XML 地图拓扑平台、梯子与 A* 实时寻路决策 HUD
        """
        if minimap_bgr is None:
            return None

        canvas = minimap_bgr.copy()
        ch, cw, _ = canvas.shape

        # ====================== 1. 地图 XML 平台拓扑与多层导航决策渲染 ======================
        if decision_engine and decision_engine.map_parser:
            mp = decision_engine.map_parser
            nav = getattr(decision_engine, "navigator", None)

            # (1) 绘制全图平台 (Footholds)
            for fh in mp.horizontal_fhs:
                x1_m, y1_m = mp.world_to_minimap(fh.x1, fh.y1, crop_w=cw, crop_h=ch)
                x2_m, y2_m = mp.world_to_minimap(fh.x2, fh.y2, crop_w=cw, crop_h=ch)
                if 0 <= x1_m < cw or 0 <= x2_m < cw or min(x1_m, x2_m) <= cw // 2 <= max(x1_m, x2_m):
                    cv2.line(canvas, (x1_m, y1_m), (x2_m, y2_m), (60, 180, 60), 1, cv2.LINE_AA)

            # (1.5) 高亮用户标记的打怪平台 (金黄色粗线) 与仅在标记层显示的跳跃攀爬区 (青色 JUMP)
            if nav and nav.marked:
                from layer_navigator import LadderClimber
                for mid in nav.marked:
                    mp_plat = nav.graph.platforms.get(mid)
                    if mp_plat:
                        for pfh in mp_plat.fhs:
                            fx1, fy1 = mp.world_to_minimap(pfh.x1, pfh.y1, crop_w=cw, crop_h=ch)
                            fx2, fy2 = mp.world_to_minimap(pfh.x2, pfh.y2, crop_w=cw, crop_h=ch)
                            cv2.line(canvas, (fx1, fy1), (fx2, fy2), (0, 215, 255), 2, cv2.LINE_AA)
                        cx_m, cy_m = mp.world_to_minimap(mp_plat.center_x, mp_plat.avg_y, crop_w=cw, crop_h=ch)
                        cv2.putText(canvas, f"★L{mid}", (cx_m - 10, max(12, cy_m - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 215, 255), 1)
                        
                        # 绘制该层默认下跳点 (亮橙色中点标记)
                        drop_x = nav.profile.drop_points.get(str(mid), int(mp_plat.center_x))
                        dx_m, dy_m = mp.world_to_minimap(drop_x, mp_plat.avg_y, crop_w=cw, crop_h=ch)
                        cv2.circle(canvas, (dx_m, dy_m), 3, (0, 165, 255), -1)
                        cv2.putText(canvas, "v", (dx_m - 3, dy_m + 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 165, 255), 1)

                        # 🌟 需求 1 & 2: 仅在被标记的平台上显示跳跃区 (如 818 与 914) 以及空中按 UP 判定线 (如 835 与 887)
                        drawn_ladders = set()
                        for edge in nav.graph.edges_from.get(mid, []):
                            if edge.kind == "CLIMB" and edge.ladder:
                                lr_id = getattr(edge.ladder, "id", None)
                                if lr_id is not None and lr_id in drawn_ladders:
                                    continue
                                if lr_id is not None:
                                    drawn_ladders.add(lr_id)
                                
                                jz = LadderClimber.get_jump_zone(mp_plat, edge.ladder)
                                # 1. 绘制左起跳区 (例如 818 附近朝右起跳)
                                if jz["can_left"] and jz["left_zone"]:
                                    zx1, zx2 = jz["left_zone"]
                                    zy = mp_plat.y_at(jz["left_target"]) or mp_plat.avg_y
                                    m1, my1 = mp.world_to_minimap(zx1, zy, crop_w=cw, crop_h=ch)
                                    m2, my2 = mp.world_to_minimap(zx2, zy, crop_w=cw, crop_h=ch)
                                    if m1 > m2: m1, m2 = m2, m1
                                    cv2.line(canvas, (m1, my1), (m2, my2), (255, 255, 0), 2, cv2.LINE_AA)
                                    cv2.line(canvas, (m1, my1 - 3), (m1, my1 + 3), (255, 255, 0), 1)
                                    cv2.line(canvas, (m2, my2 - 3), (m2, my2 + 3), (255, 255, 0), 1)
                                    lbl_x = max(0, min(cw - 24, (m1 + m2) // 2 - 8))
                                    cv2.putText(canvas, f"{jz['left_target']:.0f}>", (lbl_x, max(10, my1 - 3)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.22, (255, 255, 0), 1)

                                # 2. 绘制右起跳区 (例如 914 附近朝左起跳)
                                if jz["can_right"] and jz["right_zone"]:
                                    zx1, zx2 = jz["right_zone"]
                                    zy = mp_plat.y_at(jz["right_target"]) or mp_plat.avg_y
                                    m1, my1 = mp.world_to_minimap(zx1, zy, crop_w=cw, crop_h=ch)
                                    m2, my2 = mp.world_to_minimap(zx2, zy, crop_w=cw, crop_h=ch)
                                    if m1 > m2: m1, m2 = m2, m1
                                    cv2.line(canvas, (m1, my1), (m2, my2), (255, 255, 0), 2, cv2.LINE_AA)
                                    cv2.line(canvas, (m1, my1 - 3), (m1, my1 + 3), (255, 255, 0), 1)
                                    cv2.line(canvas, (m2, my2 - 3), (m2, my2 + 3), (255, 255, 0), 1)
                                    lbl_x = max(0, min(cw - 24, (m1 + m2) // 2 - 8))
                                    cv2.putText(canvas, f"<{jz['right_target']:.0f}", (lbl_x, max(10, my1 - 3)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.22, (255, 255, 0), 1)

                                # 3. 绘制空中按住 UP 触发线 (在平台上方 6px 处以黄绿色小圆点标示，如 835 与 887)
                                up_dist = 26
                                up_l = edge.ladder.x - up_dist
                                up_r = edge.ladder.x + up_dist
                                zy_up = mp_plat.avg_y - 6
                                ul_m, uly_m = mp.world_to_minimap(up_l, zy_up, crop_w=cw, crop_h=ch)
                                ur_m, ury_m = mp.world_to_minimap(up_r, zy_up, crop_w=cw, crop_h=ch)
                                if 0 <= ul_m < cw and 0 <= ur_m < cw:
                                    cv2.circle(canvas, (ul_m, uly_m), 2, (100, 255, 100), -1)
                                    cv2.circle(canvas, (ur_m, ury_m), 2, (100, 255, 100), -1)
                                    cv2.putText(canvas, "UP", ((ul_m + ur_m) // 2 - 4, max(8, uly_m - 2)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.20, (100, 255, 100), 1)

            # (2) 绘制全图梯子/绳索 (LadderRopes)
            for lr in mp.ladder_ropes:
                lx1_m, ly1_m = mp.world_to_minimap(lr.x, lr.y_top, crop_w=cw, crop_h=ch)
                lx2_m, ly2_m = mp.world_to_minimap(lr.x, lr.y_bottom, crop_w=cw, crop_h=ch)
                if 0 <= lx1_m < cw:
                    cv2.line(canvas, (lx1_m, ly1_m), (lx2_m, ly2_m), (238, 130, 238), 2, cv2.LINE_AA)
                    cv2.line(canvas, (lx1_m - 3, ly1_m), (lx1_m + 3, ly1_m), (255, 0, 255), 2)
                    cv2.line(canvas, (lx1_m - 3, ly2_m), (lx1_m + 3, ly2_m), (255, 0, 255), 2)

            # (3) 绘制跨层跃迁路径 (梯子/下跳)
            if nav and nav.route and nav.route_idx < len(nav.route):
                prev_pt = player_pos if player_pos else None
                for idx, (e, ax) in enumerate(nav.route[nav.route_idx:]):
                    tgt_y = e.exit_y if (e.kind == "CLIMB" and e.exit_y is not None) else nav.graph.platforms[e.dst].avg_y
                    ex_m, ey_m = mp.world_to_minimap(ax, tgt_y, crop_w=cw, crop_h=ch)
                    color = (255, 0, 255) if e.kind == "CLIMB" else (0, 165, 255)
                    if prev_pt is not None:
                        cv2.line(canvas, prev_pt, (ex_m, ey_m), color, 2, cv2.LINE_AA)
                    cv2.circle(canvas, (ex_m, ey_m), 4, color, -1)
                    prev_pt = (ex_m, ey_m)

            # (4) 绘制玩家当前吸附的 Foothold 平台与安全区
            if player_pos:
                px_disp, py_disp = player_pos
                px_w, py_w = mp.minimap_to_world(player_pos[0], player_pos[1], crop_w=cw, crop_h=ch)
                cur_fh = mp.snap_to_foothold(px_w, py_w)

                if cur_fh:
                    fh_x1_m, fh_y1_m = mp.world_to_minimap(cur_fh.x1, cur_fh.y1, crop_w=cw, crop_h=ch)
                    fh_x2_m, fh_y2_m = mp.world_to_minimap(cur_fh.x2, cur_fh.y2, crop_w=cw, crop_h=ch)
                    cv2.line(canvas, (fh_x1_m, fh_y1_m), (fh_x2_m, fh_y2_m), (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(canvas, f"FH#{cur_fh.id}", (fh_x1_m, max(12, fh_y1_m - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)

                # 🌟 默认永久开启的平台安全区 (亮青) 与两端危险区 (亮红粗线)
                d_margin_l = getattr(decision_engine, "current_danger_margin_left", getattr(decision_engine, "current_danger_margin", 150)) if decision_engine else 150
                d_margin_r = getattr(decision_engine, "current_danger_margin_right", getattr(decision_engine, "current_danger_margin", 150)) if decision_engine else 150
                p_bounds = getattr(decision_engine, "current_platform_bounds", None)
                if not p_bounds and decision_engine and hasattr(decision_engine, "get_platform_info"):
                    p_bounds = decision_engine.get_platform_info(px_w, py_w, danger_margin_left=d_margin_l, danger_margin_right=d_margin_r)

                if p_bounds:
                    pf_fh, x_min, x_max, safe_left, safe_right, x_mid = p_bounds
                    render_fh = cur_fh or pf_fh
                    
                    connected_fhs = getattr(decision_engine, "current_platform_fhs", None) or [render_fh]
                    # 高亮整段连续相连平台 (荧光绿粗线)
                    for pfh in connected_fhs:
                        pfx1_m, pfy1_m = mp.world_to_minimap(pfh.x1, pfh.y1, crop_w=cw, crop_h=ch)
                        pfx2_m, pfy2_m = mp.world_to_minimap(pfh.x2, pfh.y2, crop_w=cw, crop_h=ch)
                        cv2.line(canvas, (pfx1_m, pfy1_m), (pfx2_m, pfy2_m), (0, 255, 0), 2, cv2.LINE_AA)

                    def get_plat_y(x_coord):
                        for pfh in connected_fhs:
                            if min(pfh.x1, pfh.x2) - 10 <= x_coord <= max(pfh.x1, pfh.x2) + 10:
                                return pfh.get_y_at_x(x_coord)
                        return render_fh.get_y_at_x(x_coord)

                    y_xmin = get_plat_y(x_min)
                    y_sleft = get_plat_y(safe_left)
                    y_sright = get_plat_y(safe_right)
                    y_xmax = get_plat_y(x_max)

                    dl1_m, dly1_m = mp.world_to_minimap(x_min, y_xmin, crop_w=cw, crop_h=ch)
                    dl2_m, dly2_m = mp.world_to_minimap(safe_left, y_sleft, crop_w=cw, crop_h=ch)
                    dr1_m, dry1_m = mp.world_to_minimap(safe_right, y_sright, crop_w=cw, crop_h=ch)
                    dr2_m, dry2_m = mp.world_to_minimap(x_max, y_xmax, crop_w=cw, crop_h=ch)
                    
                    sl_m, sly_m = mp.world_to_minimap(safe_left, y_sleft, crop_w=cw, crop_h=ch)
                    sr_m, sry_m = mp.world_to_minimap(safe_right, y_sright, crop_w=cw, crop_h=ch)
                    
                    # 绘制左侧危险区 (亮红 4px)
                    if safe_left > x_min:
                        cv2.line(canvas, (dl1_m, dly1_m), (dl2_m, dly2_m), (0, 0, 255), 4, cv2.LINE_AA)
                        cv2.line(canvas, (dl1_m, dly1_m - 4), (dl1_m, dly1_m + 4), (0, 0, 255), 2)
                    
                    # 绘制右侧危险区 (亮红 4px)
                    if x_max > safe_right:
                        cv2.line(canvas, (dr1_m, dry1_m), (dr2_m, dry2_m), (0, 0, 255), 4, cv2.LINE_AA)
                        cv2.line(canvas, (dr2_m, dry2_m - 4), (dr2_m, dry2_m + 4), (0, 0, 255), 2)
                    
                    # 绘制中间安全巡逻区 (亮青 4px)
                    cv2.line(canvas, (sl_m, sly_m), (sr_m, sry_m), (255, 255, 0), 4, cv2.LINE_AA)
                    cv2.circle(canvas, (sl_m, sly_m), 4, (0, 255, 255), -1)
                    cv2.circle(canvas, (sr_m, sry_m), 4, (0, 255, 255), -1)
                    
                    # 绘制文字指示
                    cv2.putText(canvas, "SAFE", (sl_m + 2, max(12, sly_m - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                    if d_margin_l > 0:
                        cv2.putText(canvas, f"L:{d_margin_l}", (dl1_m, max(12, dly1_m - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
                    if d_margin_r > 0:
                        cv2.putText(canvas, f"R:{d_margin_r}", (max(0, dr2_m - 35), max(12, dry2_m - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
                    
                    if getattr(decision_engine, "is_escaping_platform_danger", False):
                        cv2.putText(canvas, "⚠️ ESCAPING DANGER ZONE", (10, max(15, ch - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

                # 绘制玩家定位准心 (十字 + 亮青双圆)
                cv2.circle(canvas, (px_disp, py_disp), 7, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(canvas, (px_disp, py_disp), 3, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.line(canvas, (px_disp - 9, py_disp), (px_disp + 9, py_disp), (0, 255, 255), 1)
                cv2.line(canvas, (px_disp, py_disp - 9), (px_disp, py_disp + 9), (0, 255, 255), 1)
                cv2.putText(canvas, f"({px_w},{py_w})", (px_disp - 20, max(14, py_disp - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

            # (5) 绘制顶部 HUD 决策状态条
            if nav:
                tg_cnt = len(nav.targets())
                status_str = nav.status
                step_info = f"[多层导航] {status_str} (打怪层:{tg_cnt}个)"
            else:
                step_info = "[平台巡逻] 安全区保护中"
                
            cv2.putText(canvas, step_info, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
            return canvas

        # ====================== 2. 传统模式巡逻路网与矩形包围盒渲染 ======================
        nodes = route_manager.nodes
        n_count = len(nodes)
        curr_idx = route_manager.current_target_index

        ACTION_COLORS = {
            "WALK": (255, 191, 0),       # 青蓝
            "JUMP": (50, 205, 50),       # 绿色
            "CLIMB": (238, 130, 238),    # 紫红
            "CLIMB_END": (255, 105, 180),# 深粉红 (攀爬终点/登顶)
            "DOWN_JUMP": (0, 165, 255)   # 橙色
        }

        # 1. 绘制每个有向路段的矩形包围盒与箭头 P_i -> P_{i+1}
        if n_count > 0:
            for i in range(n_count):
                n1 = nodes[i]
                n2 = nodes[(i + 1) % n_count]
                
                is_active = (i == curr_idx)
                
                if n_count > 1:
                    min_x, max_x = min(n1.x, n2.x) - 3, max(n1.x, n2.x) + 3
                    min_y, max_y = min(n1.y, n2.y) - 3, max(n1.y, n2.y) + 3
                    
                    rect_color = (0, 255, 255) if is_active else (70, 100, 70)
                    line_color = (0, 255, 255) if is_active else (180, 180, 90)
                    
                    # 绘制矩形包围盒
                    cv2.rectangle(canvas, (min_x, min_y), (max_x, max_y), rect_color, 1)
                    # 绘制有向箭头 P_i -> P_{i+1}
                    cv2.arrowedLine(canvas, (n1.x, n1.y), (n2.x, n2.y), line_color, 1, tipLength=0.2)

                # 节点标记
                pos = (n1.x, n1.y)
                color = ACTION_COLORS.get(n1.action_type, (200, 200, 200))
                cv2.circle(canvas, pos, 3, color, -1, cv2.LINE_AA)
                
                # 节点编号
                lbl_text = f"{n1.node_id}"
                cv2.putText(canvas, lbl_text, (n1.x + 3, n1.y - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # 2. 绘制玩家实时位置
        if player_pos:
            px, py = player_pos
            cv2.circle(canvas, (px, py), 6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 2, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, "YOU", (px - 10, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        return canvas

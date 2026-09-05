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
                "crop_h": self.crop_h
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
            print(f"【小地图配置已恢复】: Left={self.crop_left}, Top={self.crop_top}, W={self.crop_w}, H={self.crop_h}")
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
        三引擎高精定位冒险岛小地图玩家标志：
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
                    if max_val >= 0.50:
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

        # ====================== 1. 高级 XML 拓扑与 A* 寻路决策渲染 ======================
        if decision_engine and decision_engine.map_parser and getattr(decision_engine, "pathfinder", None):
            mp = decision_engine.map_parser
            
            # 判断是否属于超长/超高纵向滚动地图 (如魔法密林树洞 canvas_h > 120 且高度远大于视窗)
            is_tall_map = (mp.canvas_h > 120 and mp.canvas_h > mp.canvas_w * 1.5)
            
            if is_tall_map:
                # 🌟 全景上帝视角模式 (Panoramic God-View)：构建完整全景画布
                pano_w = max(180, int(mp.canvas_w * 1.8))
                pano_h = int(mp.canvas_h * (pano_w / float(mp.canvas_w)))
                canvas = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)
                canvas[:] = (20, 24, 28) # 炫酷暗黑科技背景
                
                # 绘制全景微弱网格线
                for gy in range(0, pano_h, 30):
                    cv2.line(canvas, (0, gy), (pano_w, gy), (30, 36, 42), 1)
                
                cw, ch = pano_w, pano_h

            # (1) 绘制全图平台 (Footholds)
            for fh in mp.horizontal_fhs:
                x1_m, y1_m = mp.world_to_minimap(fh.x1, fh.y1, crop_w=cw, crop_h=ch)
                x2_m, y2_m = mp.world_to_minimap(fh.x2, fh.y2, crop_w=cw, crop_h=ch)
                if 0 <= x1_m < cw or 0 <= x2_m < cw:
                    cv2.line(canvas, (x1_m, y1_m), (x2_m, y2_m), (60, 180, 60), 2 if is_tall_map else 1, cv2.LINE_AA)

            # (2) 绘制全图梯子/绳索 (LadderRopes)
            for lr in mp.ladder_ropes:
                lx1_m, ly1_m = mp.world_to_minimap(lr.x, lr.y_top, crop_w=cw, crop_h=ch)
                lx2_m, ly2_m = mp.world_to_minimap(lr.x, lr.y_bottom, crop_w=cw, crop_h=ch)
                if 0 <= lx1_m < cw:
                    cv2.line(canvas, (lx1_m, ly1_m), (lx2_m, ly2_m), (238, 130, 238), 2, cv2.LINE_AA)
                    cv2.line(canvas, (lx1_m - 4, ly1_m), (lx1_m + 4, ly1_m), (255, 0, 255), 2)
                    cv2.line(canvas, (lx1_m - 4, ly2_m), (lx1_m + 4, ly2_m), (255, 0, 255), 2)

            # (3) 绘制当前执行的 A* 路径折线序列
            active_path = getattr(decision_engine, "active_path", [])
            cur_step_idx = getattr(decision_engine, "current_step_idx", 0)
            
            if active_path:
                prev_pt = None
                if player_pos:
                    if is_tall_map:
                        px_w, py_w = mp.minimap_to_world(player_pos[0], player_pos[1], crop_w=minimap_bgr.shape[1], crop_h=minimap_bgr.shape[0])
                        prev_pt = mp.world_to_minimap(px_w, py_w, crop_w=cw, crop_h=ch)
                    else:
                        prev_pt = player_pos

                for s_idx, step in enumerate(active_path):
                    sx_m, sy_m = mp.world_to_minimap(step.target_x, step.target_y, crop_w=cw, crop_h=ch)
                    is_current = (s_idx == cur_step_idx)
                    pt_color = (0, 255, 255) if is_current else (220, 220, 0)
                    
                    if prev_pt is not None:
                        cv2.line(canvas, prev_pt, (sx_m, sy_m), pt_color, 2 if is_current else 1, cv2.LINE_AA)
                    
                    cv2.circle(canvas, (sx_m, sy_m), 4 if is_current else 3, pt_color, -1)
                    prev_pt = (sx_m, sy_m)

            # (4) 绘制玩家当前吸附的 Foothold 平台 (荧光绿高亮)
            if player_pos:
                orig_w, orig_h = minimap_bgr.shape[1], minimap_bgr.shape[0]
                px_w, py_w = mp.minimap_to_world(player_pos[0], player_pos[1], crop_w=orig_w, crop_h=orig_h)
                cur_fh = mp.snap_to_foothold(px_w, py_w)
                
                # 全景画布上的玩家点
                if is_tall_map:
                    px_disp, py_disp = mp.world_to_minimap(px_w, py_w, crop_w=cw, crop_h=ch)
                else:
                    px_disp, py_disp = player_pos

                if cur_fh:
                    fh_x1_m, fh_y1_m = mp.world_to_minimap(cur_fh.x1, cur_fh.y1, crop_w=cw, crop_h=ch)
                    fh_x2_m, fh_y2_m = mp.world_to_minimap(cur_fh.x2, cur_fh.y2, crop_w=cw, crop_h=ch)
                    cv2.line(canvas, (fh_x1_m, fh_y1_m), (fh_x2_m, fh_y2_m), (0, 255, 0), 3 if is_tall_map else 2, cv2.LINE_AA)
                    cv2.putText(canvas, f"FH#{cur_fh.id}", (fh_x1_m, max(12, fh_y1_m - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)

                # 🌟 平台打怪模式：高亮安全区 (亮青/黄) 与两端危险区 (亮红粗线)
                d_margin = getattr(decision_engine, "current_danger_margin", 150) if decision_engine else 150
                p_bounds = getattr(decision_engine, "current_platform_bounds", None)
                if not p_bounds and decision_engine and hasattr(decision_engine, "get_platform_info"):
                    p_bounds = decision_engine.get_platform_info(px_w, py_w, danger_margin=d_margin)

                if p_bounds and cur_fh:
                    _, x_min, x_max, safe_left, safe_right, x_mid = p_bounds
                    dl1_m, dly1_m = mp.world_to_minimap(x_min, cur_fh.avg_y, crop_w=cw, crop_h=ch)
                    dl2_m, dly2_m = mp.world_to_minimap(safe_left, cur_fh.avg_y, crop_w=cw, crop_h=ch)
                    dr1_m, dry1_m = mp.world_to_minimap(safe_right, cur_fh.avg_y, crop_w=cw, crop_h=ch)
                    dr2_m, dry2_m = mp.world_to_minimap(x_max, cur_fh.avg_y, crop_w=cw, crop_h=ch)
                    
                    sl_m, sly_m = mp.world_to_minimap(safe_left, cur_fh.avg_y, crop_w=cw, crop_h=ch)
                    sr_m, sry_m = mp.world_to_minimap(safe_right, cur_fh.avg_y, crop_w=cw, crop_h=ch)
                    
                    # 绘制左侧危险区 (亮红 4px)
                    cv2.line(canvas, (dl1_m, dly1_m), (dl2_m, dly2_m), (0, 0, 255), 4, cv2.LINE_AA)
                    cv2.line(canvas, (dl1_m, dly1_m - 4), (dl1_m, dly1_m + 4), (0, 0, 255), 2)
                    
                    # 绘制右侧危险区 (亮红 4px)
                    cv2.line(canvas, (dr1_m, dry1_m), (dr2_m, dry2_m), (0, 0, 255), 4, cv2.LINE_AA)
                    cv2.line(canvas, (dr2_m, dry2_m - 4), (dr2_m, dry2_m + 4), (0, 0, 255), 2)
                    
                    # 绘制中间安全巡逻区 (亮青 4px)
                    cv2.line(canvas, (sl_m, sly_m), (sr_m, sry_m), (255, 255, 0), 4, cv2.LINE_AA)
                    cv2.circle(canvas, (sl_m, sly_m), 4, (0, 255, 255), -1)
                    cv2.circle(canvas, (sr_m, sry_m), 4, (0, 255, 255), -1)
                    
                    # 绘制文字指示
                    cv2.putText(canvas, "SAFE", (sl_m + 2, max(12, sly_m - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                    cv2.putText(canvas, f"DANGER({d_margin})", (dl1_m, max(12, dly1_m - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 255), 1)
                    
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

            # (4.5) 绘制用户录制的巡逻路点 (P1, P2, P3...) 与航点连线
            if route_manager and route_manager.nodes:
                prev_n_pt = None
                first_n_pt = None
                orig_w, orig_h = minimap_bgr.shape[1], minimap_bgr.shape[0]
                for n_idx, node in enumerate(route_manager.nodes):
                    if is_tall_map:
                        n_xw, n_yw = mp.minimap_to_world(node.x, node.y, crop_w=orig_w, crop_h=orig_h)
                        nx_disp, ny_disp = mp.world_to_minimap(n_xw, n_yw, crop_w=cw, crop_h=ch)
                    else:
                        n_xw, n_yw = mp.minimap_to_world(node.x, node.y, crop_w=orig_w, crop_h=orig_h)
                        nx_disp, ny_disp = mp.world_to_minimap(n_xw, n_yw, crop_w=cw, crop_h=ch)
                    
                    # 绘制节点发光圆点与序号 (亮橙色)
                    cv2.circle(canvas, (nx_disp, ny_disp), 5, (0, 140, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (nx_disp, ny_disp), 8, (0, 215, 255), 1, cv2.LINE_AA)
                    cv2.putText(canvas, f"P{node.node_id}", (nx_disp + 7, max(12, ny_disp - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 215, 255), 1)

                    if prev_n_pt is not None:
                        cv2.line(canvas, prev_n_pt, (nx_disp, ny_disp), (0, 165, 255), 1, cv2.LINE_AA)
                    else:
                        first_n_pt = (nx_disp, ny_disp)
                    prev_n_pt = (nx_disp, ny_disp)
                
                # 首尾闭环连线 (录制 >= 2 个节点时)
                if len(route_manager.nodes) >= 2 and first_n_pt and prev_n_pt:
                    cv2.line(canvas, prev_n_pt, first_n_pt, (0, 100, 200), 1, cv2.LINE_AA)

            # (5) 绘制顶部 HUD 决策状态条
            mode_tag = "🌟 全景上帝视角" if is_tall_map else "🌟 A* 寻路"
            rec_tag = f" | 录制路点:{len(route_manager.nodes)}" if (route_manager and route_manager.nodes) else ""
            if active_path and cur_step_idx < len(active_path):
                cur_step = active_path[cur_step_idx]
                step_info = f"[{mode_tag}] {cur_step.action} ({cur_step_idx+1}/{len(active_path)}){rec_tag}"
            else:
                step_info = f"[{mode_tag}] IDLE/巡航{rec_tag}"
                
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

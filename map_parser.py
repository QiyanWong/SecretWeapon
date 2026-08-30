import os
import sys
import xml.etree.ElementTree as ET
import math

# 修复控制台 UTF-8 输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


class Foothold:
    """地图可站立平台线段"""
    def __init__(self, fh_id, x1, y1, x2, y2, layer_id=0, group_id=0):
        self.id = int(fh_id)
        # 统一确保 x1 <= x2
        if x1 <= x2:
            self.x1, self.y1 = x1, y1
            self.x2, self.y2 = x2, y2
        else:
            self.x1, self.y1 = x2, y2
            self.x2, self.y2 = x1, y1
            
        self.layer_id = layer_id
        self.group_id = group_id
        self.is_horizontal = (self.y1 == self.y2)
        self.avg_y = (self.y1 + self.y2) / 2.0

    def contains_x(self, x, margin=15):
        """检查 X 坐标是否落在该平台区间内 (带微小容差)"""
        return (self.x1 - margin) <= x <= (self.x2 + margin)

    def get_y_at_x(self, x):
        """获取指定 X 坐标在平台线段上的插值 Y 坐标"""
        if self.x2 == self.x1:
            return self.y1
        t = (x - self.x1) / float(self.x2 - self.x1)
        t = max(0.0, min(1.0, t))
        return self.y1 + t * (self.y2 - self.y1)

    def __repr__(self):
        return f"Foothold(id={self.id}, ({self.x1}, {self.y1}) -> ({self.x2}, {self.y2}))"


class LadderRope:
    """梯子与绳索"""
    def __init__(self, lr_id, x, y1, y2, l_type):
        self.id = int(lr_id)
        self.x = int(x)
        # 确保 y1 为顶部 (较小值)，y2 为底部 (较大值)
        self.y_top = min(int(y1), int(y2))
        self.y_bottom = max(int(y1), int(y2))
        self.is_ladder = (str(l_type) == "1") # 1 = 梯子, 2 = 绳子
        
        # 关联的上下平台 ID (由 NavGraph 自动吸附绑定)
        self.bottom_foothold = None
        self.top_foothold = None

    def __repr__(self):
        t_str = "Ladder" if self.is_ladder else "Rope"
        return f"LadderRope(id={self.id}, type={t_str}, x={self.x}, y_top={self.y_top}, y_bottom={self.y_bottom})"


class MapParser:
    """
    冒险岛 WZ 地图 XML 核心数据解析器
    """
    def __init__(self, xml_path=None):
        self.xml_path = xml_path
        self.map_id = ""
        self.map_name = ""
        
        # 地图要素集合
        self.footholds = {}      # {fh_id: Foothold}
        self.horizontal_fhs = [] # 纯水平平台列表，方便快速检索
        self.ladder_ropes = []   # [LadderRope, ...]
        self.portals = []        # 传送门列表
        
        # 小地图校准参数
        self.canvas_w = 0        # 小地图真实图像像素宽 (如 259)
        self.canvas_h = 0        # 小地图真实图像像素高 (如 79)
        self.center_x = 0
        self.center_y = 0
        self.mm_width = 0
        self.mm_height = 0
        self.mag = 4             # 默认缩放倍率 4
        
        # 地图世界边界
        self.vr_left = -9999
        self.vr_right = 9999
        self.vr_top = -9999
        self.vr_bottom = 9999

        if xml_path and os.path.exists(xml_path):
            self.parse(xml_path)

    def _get_val(self, elem, name):
        """辅助提取 XML 子节点中指定 name 的 value 属性"""
        for c in elem:
            if c.attrib.get('name') == name:
                return c.attrib.get('value')
        return None

    def parse(self, xml_path):
        """解析地图 XML 文件并构建拓扑结构"""
        self.xml_path = xml_path
        self.footholds.clear()
        self.horizontal_fhs.clear()
        self.ladder_ropes.clear()
        self.portals.clear()

        tree = ET.parse(xml_path)
        root = tree.getroot()

        self.map_id = root.attrib.get("name", "").replace(".img", "")

        # 1. 解析 info 节点 (地图边界与基本信息)
        info_dir = None
        for d in root:
            if d.attrib.get('name') == 'info':
                info_dir = d
                break
        if info_dir is not None:
            self.vr_left = int(self._get_val(info_dir, "VRLeft") or -9999)
            self.vr_right = int(self._get_val(info_dir, "VRRight") or 9999)
            self.vr_top = int(self._get_val(info_dir, "VRTop") or -9999)
            self.vr_bottom = int(self._get_val(info_dir, "VRBottom") or 9999)

        # 2. 解析 miniMap 节点 (小地图线性映射参数及真实 canvas 像素尺寸)
        mm_dir = None
        for d in root:
            if d.attrib.get('name') == 'miniMap':
                mm_dir = d
                break

        if mm_dir is not None:
            canvas = None
            for p in mm_dir:
                if p.tag == 'png' and p.attrib.get('name') == 'canvas':
                    canvas = p
                    break
            if canvas is not None:
                self.canvas_w = int(canvas.attrib.get('width', 0))
                self.canvas_h = int(canvas.attrib.get('height', 0))
            else:
                self.canvas_w = 0
                self.canvas_h = 0

            self.center_x = int(self._get_val(mm_dir, "centerX") or 0)
            self.center_y = int(self._get_val(mm_dir, "centerY") or 0)
            self.mm_width = int(self._get_val(mm_dir, "width") or 0)
            self.mm_height = int(self._get_val(mm_dir, "height") or 0)
            mag_val = self._get_val(mm_dir, "mag")
            self.mag = int(mag_val) if mag_val is not None else 4

        # 3. 解析 foothold 平台节点 (层级: foothold -> layer -> group -> foothold_id)
        fh_root = root.find("./dir[@name='foothold']")
        if fh_root is not None:
            for layer in fh_root:
                layer_id = layer.attrib.get("name", "0")
                for group in layer:
                    group_id = group.attrib.get("name", "0")
                    for fh in group:
                        fh_id = fh.attrib.get("name")
                        x1 = int(self._get_val(fh, "x1") or 0)
                        y1 = int(self._get_val(fh, "y1") or 0)
                        x2 = int(self._get_val(fh, "x2") or 0)
                        y2 = int(self._get_val(fh, "y2") or 0)
                        
                        # 过滤无效单点或垂直死线
                        if x1 == x2 and y1 == y2:
                            continue
                        
                        # 过滤绝对垂直墙壁 (x1 == x2)，只保留可站立/行走的斜坡与平面
                        if x1 != x2:
                            foothold_obj = Foothold(fh_id, x1, y1, x2, y2, layer_id, group_id)
                            self.footholds[foothold_obj.id] = foothold_obj
                            self.horizontal_fhs.append(foothold_obj)

        # 4. 解析 ladderRope 节点 (绳索与梯子)
        lr_root = root.find("./dir[@name='ladderRope']")
        if lr_root is not None:
            for lr in lr_root:
                lr_id = lr.attrib.get("name")
                x = int(self._get_val(lr, "x") or 0)
                y1 = int(self._get_val(lr, "y1") or 0)
                y2 = int(self._get_val(lr, "y2") or 0)
                l_type = self._get_val(lr, "l") or "1"
                
                lr_obj = LadderRope(lr_id, x, y1, y2, l_type)
                self.ladder_ropes.append(lr_obj)

        # 5. 自动关联梯子/绳索与上下平台
        self._bind_ladders_to_footholds()

        print(f"✅ 地图 XML 解析成功: {os.path.basename(xml_path)}")
        print(f"   - 平台 (Footholds): {len(self.footholds)} 处")
        print(f"   - 梯子/绳索 (LadderRopes): {len(self.ladder_ropes)} 条")
        print(f"   - 小地图真实尺寸: {self.canvas_w} x {self.canvas_h} px | centerX={self.center_x}, centerY={self.center_y}, mag={self.mag}")

    def _bind_ladders_to_footholds(self):
        """将梯子与绳索的顶部与底部自动吸附到最近的 Foothold 平台 (支持树洞悬挂梯跳跃抓梯)"""
        import math
        for lr in self.ladder_ropes:
            best_bot_fh = None
            min_bot_dist = 9999
            best_top_fh = None
            min_top_dist = 9999

            for fh in self.horizontal_fhs:
                min_fx, max_fx = min(fh.x1, fh.x2), max(fh.x1, fh.x2)
                dx = max(0, min_fx - lr.x, lr.x - max_fx)
                
                # 底部匹配 (允许从下方平台向上跳跃抓梯: bot_dy在 -30 到 160 像素，横向偏差 <= 80px)
                bot_dy = fh.avg_y - lr.y_bottom
                if -30 <= bot_dy <= 160 and dx <= 80:
                    dist = math.hypot(dx, bot_dy)
                    if dist < min_bot_dist:
                        min_bot_dist = dist
                        best_bot_fh = fh
                        
                # 顶部匹配 (平台在 y_top 附近，容差 80px，横向偏差 <= 60px)
                top_dy = abs(fh.avg_y - lr.y_top)
                if top_dy <= 80 and dx <= 60:
                    dist = math.hypot(dx, top_dy)
                    if dist < min_top_dist:
                        min_top_dist = dist
                        best_top_fh = fh

            lr.bottom_foothold = best_bot_fh
            lr.top_foothold = best_top_fh

    def minimap_to_world(self, x_mm, y_mm, crop_w=None, crop_h=None):
        """
        将屏幕小地图像素坐标转换为游戏世界绝对坐标
        支持自动根据实际框选尺寸 (crop_w, crop_h) 与 XML 基础 canvas 尺寸进行等比自适应拉伸校准
        官方 WZ 映射公式: X_world = (x_mm / ratio_x) * (2^mag) - centerX
        """
        scale = float(2 ** self.mag) # 默认为 16
        rx = (float(crop_w) / float(self.canvas_w)) if (crop_w and self.canvas_w > 0) else 1.0
        ry = (float(crop_h) / float(self.canvas_h)) if (crop_h and self.canvas_h > 0) else 1.0

        raw_x_mm = float(x_mm) / rx if rx > 0 else float(x_mm)
        raw_y_mm = float(y_mm) / ry if ry > 0 else float(y_mm)

        x_world = (raw_x_mm * scale) - float(self.center_x)
        y_world = (raw_y_mm * scale) - float(self.center_y)
        return int(round(x_world)), int(round(y_world))

    def world_to_minimap(self, x_world, y_world, crop_w=None, crop_h=None):
        """
        将游戏世界绝对坐标转换为屏幕小地图显示像素坐标
        支持自动自适应拉伸匹配屏幕实际小地图视窗
        官方 WZ 映射公式: x_mm = ((X_world + centerX) / (2^mag)) * ratio_x
        """
        scale = float(2 ** self.mag) # 默认为 16
        rx = (float(crop_w) / float(self.canvas_w)) if (crop_w and self.canvas_w > 0) else 1.0
        ry = (float(crop_h) / float(self.canvas_h)) if (crop_h and self.canvas_h > 0) else 1.0

        raw_x_mm = (float(x_world) + float(self.center_x)) / scale
        raw_y_mm = (float(y_world) + float(self.center_y)) / scale

        disp_x = raw_x_mm * rx
        disp_y = raw_y_mm * ry
        return int(round(disp_x)), int(round(disp_y))

    def snap_to_foothold(self, x_world, y_world, margin=30):
        """
        根据当前世界坐标吸附到玩家脚下所在的最近平台 Foothold
        采用非对称区间设计：
        - 玩家在平台上/上方 (y_world <= fh_y): 允许 35px 容差 (下落/站立吸附)
        - 玩家在平台下方 (y_world > fh_y): 仅允许 12px 极小容差 (严防爬绳时提前吸附上一层)
        """
        best_fh = None
        min_score = 999999

        for fh in self.horizontal_fhs:
            if fh.contains_x(x_world, margin=margin):
                fh_y = fh.get_y_at_x(x_world)
                dy = y_world - fh_y # 负数代表在平台上方，正数代表在平台下方
                
                # 非对称容差判断
                if dy <= 0 and abs(dy) <= 35:
                    # 站在平台上或从上方落脚中
                    score = abs(dy)
                    if score < min_score:
                        min_score = score
                        best_fh = fh
                elif dy > 0 and dy <= 12:
                    # 仅当身体已完全登顶平台边缘 (<= 12px) 时才允许吸附
                    score = dy * 2.0
                    if score < min_score:
                        min_score = score
                        best_fh = fh

        return best_fh

    def summary(self):
        return (f"MapParser(id={self.map_id}, footholds={len(self.footholds)}, "
                f"ladderRopes={len(self.ladder_ropes)}, mag={self.mag})")

import os
import sys
import xml.etree.ElementTree as ET
import math
import base64
import cv2
import numpy as np

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
        
        # 小地图校准与对齐微调参数
        self.canvas_w = 0        # 小地图真实图像像素宽 (如 259)
        self.canvas_h = 0        # 小地图真实图像像素高 (如 79)
        self.canvas_img = None   # 解码后的完整小地图 BGRA 图像
        self.canvas_bgr = None   # 转换为 BGR (黑色透明背景) 的完整小地图图像
        self.canvas_gray = None  # 灰度图像，用于快速模板匹配视窗偏移
        self.scroll_y = 0        # 纵向滚动地图视窗偏移量 (px)
        self.last_valid_scroll_y = 0
        self.offset_x = 0        # 用户手动调整的 X 偏移对齐量 (px)
        self.offset_y = 0        # 用户手动调整的 Y 偏移对齐量 (px)
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
        self.canvas_img = None
        self.canvas_bgr = None
        self.canvas_gray = None
        self.scroll_y = 0
        self.last_valid_scroll_y = 0

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

        # 2. 解析 miniMap 节点 (小地图线性映射参数、真实 canvas 尺寸及底图图像)
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
                if 'value' in canvas.attrib:
                    try:
                        raw_bytes = base64.b64decode(canvas.attrib['value'])
                        nparr = np.frombuffer(raw_bytes, np.uint8)
                        bgra = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                        if bgra is not None:
                            self.canvas_img = bgra
                            if len(bgra.shape) == 3 and bgra.shape[2] == 4:
                                b, g, r, a = cv2.split(bgra)
                                self.canvas_bgr = np.where(a[..., None] == 0, [0, 0, 0], bgra[..., :3]).astype(np.uint8)
                            else:
                                self.canvas_bgr = bgra.copy()
                            self.canvas_gray = cv2.cvtColor(self.canvas_bgr, cv2.COLOR_BGR2GRAY)
                    except Exception as e:
                        print(f"⚠️ 解码 miniMap canvas 图像异常: {e}")
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

    def set_overlay_offset(self, offset_x, offset_y):
        """设置小地图 Overlay 手动对齐微调偏移量 (像素)"""
        self.offset_x = int(offset_x)
        self.offset_y = int(offset_y)

    def minimap_to_world(self, x_mm, y_mm, crop_w=None, crop_h=None):
        """
        将屏幕小地图像素坐标转换为游戏世界绝对坐标
        支持自动根据实际框选尺寸 (crop_w, crop_h) 与 XML 基础 canvas 尺寸进行缩放对齐，
        并消除用户手动微调的对齐偏移量 (offset_x, offset_y)
        映射公式:
        raw_x = (x_mm - offset_x) / rx
        raw_y = (y_mm - offset_y) / ry
        X_world = raw_x * (2^mag) - centerX
        Y_world = raw_y * (2^mag) - centerY
        """
        scale = float(2 ** self.mag) # 默认为 16
        rx = (float(crop_w) / float(self.canvas_w)) if (crop_w and self.canvas_w > 0) else 1.0
        ry = (float(crop_h) / float(self.canvas_h)) if (crop_h and self.canvas_h > 0) else 1.0

        raw_x_mm = (float(x_mm) - float(self.offset_x)) / rx if rx > 0 else (float(x_mm) - float(self.offset_x))
        raw_y_mm = (float(y_mm) - float(self.offset_y)) / ry if ry > 0 else (float(y_mm) - float(self.offset_y))

        x_world = (raw_x_mm * scale) - float(self.center_x)
        y_world = (raw_y_mm * scale) - float(self.center_y)
        return int(round(x_world)), int(round(y_world))

    def world_to_minimap(self, x_world, y_world, crop_w=None, crop_h=None):
        """
        将游戏世界绝对坐标转换为屏幕小地图显示像素坐标
        支持自动根据实际框选尺寸 (crop_w, crop_h) 与 XML 基础 canvas 尺寸进行缩放对齐，
        并叠加用户手动微调的对齐偏移量 (offset_x, offset_y)
        映射公式:
        raw_x = (X_world + centerX) / (2^mag)
        raw_y = (Y_world + centerY) / (2^mag)
        x_mm = raw_x * rx + offset_x
        y_mm = raw_y * ry + offset_y
        """
        scale = float(2 ** self.mag) # 默认为 16
        rx = (float(crop_w) / float(self.canvas_w)) if (crop_w and self.canvas_w > 0) else 1.0
        ry = (float(crop_h) / float(self.canvas_h)) if (crop_h and self.canvas_h > 0) else 1.0

        raw_x_mm = (float(x_world) + float(self.center_x)) / scale
        raw_y_mm = (float(y_world) + float(self.center_y)) / scale

        disp_x = (raw_x_mm * rx) + float(self.offset_x)
        disp_y = (raw_y_mm * ry) + float(self.offset_y)
        return int(round(disp_x)), int(round(disp_y))

    def snap_to_foothold(self, x_world, y_world, margin=45, margin_y=60):
        """
        根据当前世界坐标吸附到玩家脚下所在的最近平台 Foothold
        支持宽容度自适应与就近兜底，确保玩家在跳跃、蹲下或小地图微小偏移时均能稳定吸附平台
        """
        best_fh = None
        min_score = 999999

        # 1. 优先吸附 X 坐标覆盖且高度差 <= margin_y 的水平平台
        for fh in self.horizontal_fhs:
            if fh.contains_x(x_world, margin=margin):
                fh_y = fh.get_y_at_x(x_world)
                dy = abs(y_world - fh_y)
                if dy <= margin_y:
                    score = dy
                    if score < min_score:
                        min_score = score
                        best_fh = fh

        # 2. 兜底搜索：若未精确命中，寻找水平间距 <= 60 且垂直落差 <= 100 的最近平台
        if best_fh is None:
            for fh in self.horizontal_fhs:
                min_fx, max_fx = min(fh.x1, fh.x2), max(fh.x1, fh.x2)
                dx = max(0, min_fx - x_world, x_world - max_fx)
                dy = abs(y_world - fh.avg_y)
                if dx <= 60 and dy <= 100:
                    dist = math.hypot(dx, dy)
                    if dist < min_score:
                        min_score = dist
                        best_fh = fh

        return best_fh

    def summary(self):
        return (f"MapParser(id={self.map_id}, footholds={len(self.footholds)}, "
                f"ladderRopes={len(self.ladder_ropes)}, mag={self.mag})")

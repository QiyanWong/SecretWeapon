import os
import sys
import glob
import json
import random
import datetime
import cv2
import numpy as np
from PIL import Image, ImageEnhance

# 修复控制台 UTF-8 输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "synthetic_assets")
BG_DIR = os.path.join(ASSET_DIR, "backgrounds")
MONSTER_DIR = os.path.join(ASSET_DIR, "monsters")
RAW_OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "raw_images")

# 引入现有的类别列表与中文/常用名映射
from convert_json_to_yolo import CLASS_LIST, CLASS_TO_ID, LABEL_MAPPING

def load_all_sprite_items():
    """
    扫描并加载所有的怪物与玩家独立素材帧，返回包含每个形态独立图像对象的列表
    """
    import re
    sprite_items = [] # [ {"class_name": ..., "image": ..., "source": ...} ]

    if not os.path.exists(MONSTER_DIR):
        os.makedirs(MONSTER_DIR, exist_ok=True)

    # 1. 扫描子文件夹
    subdirs = [d for d in os.listdir(MONSTER_DIR) if os.path.isdir(os.path.join(MONSTER_DIR, d))]
    for sd in subdirs:
        raw_name = sd.strip()
        target_name = LABEL_MAPPING.get(raw_name, raw_name)
        folder_path = os.path.join(MONSTER_DIR, sd)
        png_files = glob.glob(os.path.join(folder_path, "*.png"))
        for pf in png_files:
            try:
                img = Image.open(pf).convert("RGBA")
                sprite_items.append({
                    "class_name": target_name,
                    "image": img,
                    "source": os.path.basename(pf)
                })
            except Exception as e:
                print(f"警告: 无法加载图片 {pf}: {e}")

    # 2. 扫描根目录下的单个 PNG 文件 (智能解析怪物与玩家名称)
    root_pngs = glob.glob(os.path.join(MONSTER_DIR, "*.png"))
    for pf in root_pngs:
        fname = os.path.basename(pf)
        base_stem = os.path.splitext(fname)[0]

        # 玩家特判：plaer_left / player_left / player_right
        if re.search(r'pla(?:y)?er_left', base_stem, re.IGNORECASE):
            target_name = 'player_left'
        elif re.search(r'pla(?:y)?er_right', base_stem, re.IGNORECASE):
            target_name = 'player_right'
        else:
            # 智能匹配动作状态后缀 (stand, move, hit, die, attack, fly, jump, alert, swing) 并切分怪物原名
            parts = re.split(r'(stand|move|hit|die|attack|fly|jump|alert|swing)', base_stem, flags=re.IGNORECASE)
            raw_name = parts[0].strip(' ._-')
            raw_name = re.sub(r'[\d\.\_\-]+$', '', raw_name).strip()
            target_name = LABEL_MAPPING.get(raw_name, raw_name)
        
        try:
            img = Image.open(pf).convert("RGBA")
            sprite_items.append({
                "class_name": target_name,
                "image": img,
                "source": fname
            })
        except Exception as e:
            print(f"警告: 无法加载图片 {pf}: {e}")

    return sprite_items

def load_monster_sprites():
    """兼容旧接口"""
    items = load_all_sprite_items()
    res = {}
    for it in items:
        cls = it["class_name"]
        if cls not in res:
            res[cls] = []
        res[cls].append(it["image"])
    return res

def get_tight_bbox(sprite):
    """
    根据 RGBA 贴图的非透明像素 (Alpha > 10) 计算精准的紧凑包围框 (x_min, y_min, x_max, y_max)
    """
    alpha = np.array(sprite.getchannel('A'))
    non_zero = np.argwhere(alpha > 10)
    if non_zero.size == 0:
        return 0, 0, sprite.width, sprite.height
    y_min, x_min = non_zero.min(axis=0)
    y_max, x_max = non_zero.max(axis=0)
    return int(x_min), int(y_min), int(x_max + 1), int(y_max + 1)

def get_exact_polygon(sprite, paste_x, paste_y, approx_epsilon=0.003):
    """
    通过 PNG Alpha 通道提取极致完美的怪物外边缘多边形轮廓点集 (Polygon Points)
    完美贴合角色每一个像素细节，消除多余背景！
    """
    alpha = np.array(sprite.getchannel('A'))
    _, thresh = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    
    main_contour = max(contours, key=cv2.contourArea)
    epsilon = max(1.0, approx_epsilon * cv2.arcLength(main_contour, True))
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    
    points = []
    for pt in approx:
        px = round(float(pt[0][0] + paste_x), 1)
        py = round(float(pt[0][1] + paste_y), 1)
        points.append([px, py])
    
    return points

def check_overlap(boxA, boxB, max_ioa=0.0):
    """检测两个矩形框 (x1, y1, x2, y2) 是否存在任何重叠 (严格零重叠)"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    return interArea > 0

def generate_synthetic_images(num_to_generate=30, target_w=1366, target_h=768, scale_mult=1.25):
    """
    自动合成指定数量的图片并导出 YOLO 格式的 .jpg 与 .txt 标注文件。
    升级版：全形态滚动队列无遗漏放置、严格零重叠碰撞检测、1.25 倍尺寸！
    """
    os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
    os.makedirs(BG_DIR, exist_ok=True)

    # 1. 彻底清理之前生成的旧 synth_ 文件
    old_synth_files = glob.glob(os.path.join(RAW_OUTPUT_DIR, "synth_*"))
    for f in old_synth_files:
        try:
            os.remove(f)
        except Exception:
            pass
    if old_synth_files:
        print(f"🧹 已彻底删除上一批旧合成图片与标注文件共 {len(old_synth_files)} 个。")

    bg_files = glob.glob(os.path.join(BG_DIR, "*.jpg")) + glob.glob(os.path.join(BG_DIR, "*.png")) + glob.glob(os.path.join(BG_DIR, "*.jpeg"))
    if not bg_files:
        print("【错误】未在 synthetic_assets/backgrounds 找到任何背景图！")
        return

    all_sprite_items = load_all_sprite_items()
    if not all_sprite_items:
        print("【错误】未在 synthetic_assets/monsters 找到任何透明 PNG 贴图！")
        return

    print("=" * 65)
    print("🎨 冒险岛合成数据集生成器 (全形态无遗漏 & 严格零重叠版)")
    print(f"设定生成分辨率窗口: {target_w} x {target_h}")
    print(f"素材尺寸放大系数: {scale_mult} 倍 (1.25x 黄金比例)")
    print(f"已加载背景图: {len(bg_files)} 张")
    print(f"已加载全部独立素材帧: {len(all_sprite_items)} 帧 (涵盖所有怪物的全部动作与玩家各形态)")
    print("=" * 65)

    generated_count = 0
    # 待放置的滚动队列，确保每个形态被持续轮转放置，放不下的自动留到下一张
    overflow_queue = []

    for i in range(num_to_generate):
        full_bg_path = random.choice(bg_files)
        full_bg = Image.open(full_bg_path).convert("RGB")
        fw, fh = full_bg.size

        # 从全景背景图中随机裁剪出一个 1366x768 视窗
        if fw < target_w or fh < target_h:
            ratio = max(target_w / fw, target_h / fh)
            nw, nh = int(fw * ratio), int(fh * ratio)
            full_bg_scaled = full_bg.resize((nw, nh), Image.Resampling.LANCZOS)
            crop_left = random.randint(0, max(0, nw - target_w))
            crop_top = random.randint(0, max(0, nh - target_h))
            bg_img = full_bg_scaled.crop((crop_left, crop_top, crop_left + target_w, crop_top + target_h))
        else:
            crop_left = random.randint(0, max(0, fw - target_w))
            crop_top = random.randint(0, max(0, fh - target_h))
            bg_img = full_bg.crop((crop_left, crop_top, crop_left + target_w, crop_top + target_h))

        bg_w, bg_h = bg_img.size

        # 构建本张图片尝试放置的素材列表：上一张未放下的优先 + 全部素材随机打乱
        items_to_try = list(overflow_queue)
        overflow_queue.clear()
        
        shuffled_pool = list(all_sprite_items)
        random.shuffle(shuffled_pool)
        for item in shuffled_pool:
            if item not in items_to_try:
                items_to_try.append(item)

        yolo_labels = []
        json_shapes = []
        placed_boxes = []

        placed_in_this_frame = 0
        skipped_in_this_frame = 0

        for item in items_to_try:
            chosen_cls = item["class_name"]
            sprite_raw = item["image"]

            # 数据增强：左右翻转
            is_flipped = (random.random() > 0.5)
            if is_flipped:
                sprite_cur = sprite_raw.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                sprite_cur = sprite_raw.copy()

            # 决定翻转后的最终标注类别 (玩家严格定向映射)
            if chosen_cls == 'player_left':
                cls_name = 'player_right' if is_flipped else 'player_left'
            elif chosen_cls == 'player_right':
                cls_name = 'player_left' if is_flipped else 'player_right'
            else:
                cls_name = chosen_cls

            if cls_name not in CLASS_TO_ID:
                continue
            cls_id = CLASS_TO_ID[cls_name]

            # 缩放系数 (1.25 倍)
            scale_factor = random.uniform(scale_mult * 0.96, scale_mult * 1.04)
            new_w = max(10, int(sprite_cur.width * scale_factor))
            new_h = max(10, int(sprite_cur.height * scale_factor))
            sprite = sprite_cur.resize((new_w, new_h), Image.Resampling.LANCZOS)

            # 微调亮度
            brightness = random.uniform(0.94, 1.06)
            sprite = ImageEnhance.Brightness(sprite).enhance(brightness)

            x1, y1, x2, y2 = get_tight_bbox(sprite)
            crop_w = x2 - x1
            crop_h = y2 - y1

            if crop_w <= 4 or crop_h <= 4:
                continue

            # 碰撞检测：尝试 50 次寻找完全不与其他物体重叠的位置
            valid_spot = False
            final_paste_x, final_paste_y = 0, 0
            final_box = (0, 0, 0, 0)

            # 留出 4px 安全边距以防贴合太近
            margin = 4

            for _attempt in range(50):
                px = random.randint(0, max(1, bg_w - sprite.width))
                py = random.randint(int(bg_h * 0.15), max(int(bg_h * 0.15) + 1, bg_h - sprite.height - 10))
                
                cand_box = (px + x1 - margin, py + y1 - margin, px + x2 + margin, py + y2 + margin)

                # 检查是否与任何已放置的物体碰撞
                if not any(check_overlap(cand_box, pb) for pb in placed_boxes):
                    valid_spot = True
                    final_paste_x, final_paste_y = px, py
                    final_box = (px + x1, py + y1, px + x2, py + y2)
                    break

            if not valid_spot:
                # 本张图片空间不足，自动放入滚动溢出队列，在下一张图片继续放置！
                overflow_queue.append(item)
                skipped_in_this_frame += 1
                continue

            # 找到合法空位，执行 Alpha 混合贴图并记录真实框
            bg_img.paste(sprite, (final_paste_x, final_paste_y), mask=sprite)
            placed_boxes.append((final_box[0] - margin, final_box[1] - margin, final_box[2] + margin, final_box[3] + margin))

            box_x1, box_y1, box_x2, box_y2 = final_box
            box_w = box_x2 - box_x1
            box_h = box_y2 - box_y1
            x_center = box_x1 + box_w / 2.0
            y_center = box_y1 + box_h / 2.0

            norm_xc = max(0.0, min(1.0, x_center / bg_w))
            norm_yc = max(0.0, min(1.0, y_center / bg_h))
            norm_w = max(0.0, min(1.0, box_w / bg_w))
            norm_h = max(0.0, min(1.0, box_h / bg_h))

            # 提取完美贴合角色轮廓的多边形顶点集 (Polygon)
            poly_points = get_exact_polygon(sprite, final_paste_x, final_paste_y)
            if len(poly_points) < 3:
                poly_points = [[float(box_x1), float(box_y1)], [float(box_x2), float(box_y2)]]
                shape_type = "rectangle"
            else:
                shape_type = "polygon"

            yolo_labels.append(f"{cls_id} {norm_xc:.6f} {norm_yc:.6f} {norm_w:.6f} {norm_h:.6f}")
            json_shapes.append({
                "label": cls_name,
                "points": poly_points,
                "group_id": None,
                "shape_type": shape_type,
                "flags": {}
            })
            placed_in_this_frame += 1

        if not yolo_labels:
            continue

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        img_name = f"synth_frame_{timestamp}_{i:03d}.jpg"
        txt_name = f"synth_frame_{timestamp}_{i:03d}.txt"
        json_name = f"synth_frame_{timestamp}_{i:03d}.json"

        img_out_path = os.path.join(RAW_OUTPUT_DIR, img_name)
        txt_out_path = os.path.join(RAW_OUTPUT_DIR, txt_name)
        json_out_path = os.path.join(RAW_OUTPUT_DIR, json_name)

        bg_img.save(img_out_path, quality=95)

        with open(txt_out_path, "w", encoding="utf-8") as tf:
            tf.write("\n".join(yolo_labels) + "\n")

        json_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": json_shapes,
            "imagePath": img_name,
            "imageData": None,
            "imageHeight": bg_h,
            "imageWidth": bg_w
        }
        with open(json_out_path, "w", encoding="utf-8") as jf:
            json.dump(json_data, jf, ensure_ascii=False, indent=2)

        generated_count += 1
        print(f"  [{generated_count:02d}/{num_to_generate}] 生成成功: 放置角色 {placed_in_this_frame} 个 (排队顺延至下张 {len(overflow_queue)} 个)")

    print("=" * 65)
    print(f"✨ 成功合成导出 {generated_count} 张完美多边形标注数据集至: {RAW_OUTPUT_DIR}")
    print("💡 已同时生成 YOLO .txt 与 AnyLabeling .json 格式！")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="冒险岛 2D 透明素材自动合成训练集工具")
    parser.add_argument("--num", type=int, default=30, help="要自动合成的图片数量 (默认 30 张)")
    parser.add_argument("--width", type=int, default=1366, help="合成窗口宽度 (默认 1366)")
    parser.add_argument("--height", type=int, default=768, help="合成窗口高度 (默认 768)")
    parser.add_argument("--scale", type=float, default=1.25, help="怪物放大系数 (默认 1.25)")
    args = parser.parse_args()

    generate_synthetic_images(
        num_to_generate=args.num,
        target_w=args.width,
        target_h=args.height,
        scale_mult=args.scale
    )

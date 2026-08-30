import os
import sys
import json

# 修复控制台 UTF-8 输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw_images")

# 将中文标注名称精准映射到官方《冒险岛》英文名
LABEL_MAPPING = {
    # 玩家状态与地图要素
    '我左': 'player_left',
    '我右': 'player_right',
    '我爬': 'player_climb',
    '绳子': 'rope',
    '光圈': 'portal',

    # 冒险岛经典怪物官方英文对应表
    '花蘑菇': 'orange_mushroom',
    '红蜗牛': 'red_snail',
    '绿水灵': 'slime',
    '蓝水灵': 'bubbling',
    '刺蘑菇': 'horny_mushroom',
    '僵尸蘑菇': 'zombie_mushroom',
    '斧木妖': 'axe_stump',
    '野猪': 'wild_boar',
    'wild_roar': 'wild_boar',
    '猪猪': 'pig',
    '飘飘猪': 'ribbon_pig',
    '火野猪': 'fire_boar',
    'fire_roar': 'fire_boar',
    '小青蛇': 'jr_necki',
    '鳄鱼': 'croco',
    '土龙': 'drake',
    '风独眼': 'evil_eye',
    '冰独眼': 'cold_eye',
    '小幽灵': 'jr_wraith',
    '木面怪人': 'wooden_mask',
    '猴子': 'lupin',
    '石面怪人': 'rocky_mask',
    '红螃蟹': 'crab'
}

# 保持固定的类别顺序 (23 类)
CLASS_LIST = [
    'player_left',
    'player_right',
    'orange_mushroom',
    'red_snail',
    'slime',
    'bubbling',
    'horny_mushroom',
    'zombie_mushroom',
    'axe_stump',
    'wild_boar',
    'pig',
    'ribbon_pig',
    'fire_boar',
    'jr_necki',
    'croco',
    'drake',
    'evil_eye',
    'cold_eye',
    'jr_wraith',
    'wooden_mask',
    'lupin',
    'rocky_mask',
    'crab'
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_LIST)}


def convert_all_jsons():
    files = os.listdir(RAW_DIR)
    json_files = [f for f in files if f.endswith('.json')]
    
    print(f"扫描到 {len(json_files)} 个 JSON 标注文件，准备转换位 YOLO .txt 格式...")
    converted_count = 0

    for jf in json_files:
        json_path = os.path.join(RAW_DIR, jf)
        base_name = os.path.splitext(jf)[0]
        txt_path = os.path.join(RAW_DIR, base_name + ".txt")

        try:
            with open(json_path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)

            img_w = data.get('imageWidth')
            img_h = data.get('imageHeight')

            if not img_w or not img_h:
                continue

            yolo_lines = []
            shapes = data.get('shapes', [])

            for shape in shapes:
                raw_label = shape.get('label', '').strip()
                target_label = LABEL_MAPPING.get(raw_label, raw_label)

                if target_label not in CLASS_TO_ID:
                    print(f"警告: 忽略未定义的标签 '{raw_label}'")
                    continue

                cls_id = CLASS_TO_ID[target_label]
                points = shape.get('points', [])

                if not points:
                    continue

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]

                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                bw = x_max - x_min
                bh = y_max - y_min
                cx = (x_min + x_max) / 2.0 / img_w
                cy = (y_min + y_max) / 2.0 / img_h
                nw = bw / img_w
                nh = bh / img_h

                # 限制坐标在 0~1 范围内
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.0, min(1.0, nw))
                nh = max(0.0, min(1.0, nh))

                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            # 写入对应的 .txt 文件
            if yolo_lines:
                with open(txt_path, 'w', encoding='utf-8') as f_txt:
                    f_txt.write("\n".join(yolo_lines) + "\n")
                converted_count += 1

        except Exception as e:
            print(f"处理 {jf} 异常: {e}")

    print(f"【成功】已将 {converted_count} 个标注文件映射并转换为官方英文名 YOLO .txt 格式！")
    print(f"当前识别类别列表 (共 {len(CLASS_LIST)} 类): {CLASS_LIST}")


if __name__ == "__main__":
    convert_all_jsons()

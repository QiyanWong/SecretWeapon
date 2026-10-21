import os
import sys
import glob
import json
import random
import datetime
import re
import argparse
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image

# 修复控制台 UTF-8 输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAP_XML_DIR = os.path.join(BASE_DIR, "map")
ASSET_DIR = os.path.join(BASE_DIR, "synthetic_assets")
MONSTERS_DIR = os.path.join(ASSET_DIR, "monsters")
PLAYER_DIR = os.path.join(ASSET_DIR, "player")
BG_DIR = os.path.join(ASSET_DIR, "backgrounds")
DROPS_DIR = os.path.join(ASSET_DIR, "drops")
DISTRACTORS_DIR = os.path.join(ASSET_DIR, "distractors")

MAP_RAW_ROOT = os.path.join(BASE_DIR, "dataset", "map_raw_images")
os.makedirs(MAP_RAW_ROOT, exist_ok=True)

# 完整怪物官方 ID 映射表
MOB_ID_TO_KEY = {
    100004: 'orange_mushroom',
    100002: 'red_snail',
    100006: 'slime',
    1210103: 'bubbling',
    2110200: 'horny_mushroom',
    2230101: 'zombie_mushroom',
    1130100: 'axe_stump',
    2230102: 'wild_boar',
    100007: 'pig',
    1210101: 'ribbon_pig',
    3210100: 'fire_boar',
    2130103: 'jr_necki',
    3110100: 'croco',
    4130100: 'drake',
    2230100: 'evil_eye',
    4230100: 'cold_eye',
    3230101: 'jr_wraith',
    2230110: 'wooden_mask',
    3210800: 'lupin',
    2230111: 'rocky_mask',
    3230102: 'crab',
    5130100: 'tauromacis',
    4230101: 'zombie_lupin',
    4230102: 'wraith',
    3230300: 'jr_boogie',
    5150001: 'skeleton_soldier',
    6230602: 'officer_skeleton',
    1120100: 'octopus',
    2220100: 'blue_mushroom',
    5100000: 'jr_yeti',
    5100001: 'jr_yeti',
    5130104: 'hector',
    5140000: 'white_fang',
    5130101: 'dark_stone_golem',
    5130102: 'dark_stone_golem',
    5130103: 'croco',
    130101:  'red_snail',
    1210100: 'pig',
    7130100: 'tauromacis',
    2300100: 'stirge',
    4230400: 'fire_boar',
    3230100: 'cold_eye',
    100101:  'red_snail',
    120100:  'orange_mushroom',
    1140100: 'axe_stump',
    2130100: 'dark_axe_stump',
    3230301: 'jr_boogie',
    2230131: 'zombie_mushroom',
    4090000: 'iron_hog',
    5090000: 'shade',
    6090001: 'snow_witch',
    6230100: 'wild_kargo'
}

CUSTOM_MOBS_FILE = os.path.join(BASE_DIR, "custom_mobs.json")

def load_custom_mobs_config():
    if not os.path.exists(CUSTOM_MOBS_FILE):
        return {}, {}
    try:
        with open(CUSTOM_MOBS_FILE, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            return cfg.get("mob_id_mapping", {}), cfg.get("map_monster_overrides", {})
    except Exception:
        return {}, {}

_custom_ids, CUSTOM_MAP_OVERRIDES = load_custom_mobs_config()
for _sid, _info in _custom_ids.items():
    try:
        _mid = int(_sid)
        _k = _info['key']
        MOB_ID_TO_KEY[_mid] = _k
    except ValueError:
        pass

# 怪物专属掉落物绑定
MONSTER_TO_UNIQUE_DROP = {
    'orange_mushroom': 'orange_mushroom_cap',
    'red_snail':       'red_snail_shell',
    'slime':           'squishy_liquid',
    'bubbling':        'bubbling_bubble',
    'horny_mushroom':  'horny_mushroom_cap',
    'zombie_mushroom': 'charm_of_the_undead',
    'axe_stump':       'firewood',
    'dark_axe_stump':  'firewood',
    'wild_boar':       'wild_boar_tooth',
    'pig':             'pig_head',
    'ribbon_pig':      'pig_ribbon',
    'fire_boar':       'fire_boar_tooth',
    'jr_necki':        'jr_necki_skin',
    'croco':           'ligator_skin',
    'drake':           'drake_skull',
    'evil_eye':        'evil_eye_tail',
    'cold_eye':        'cold_eye_tail',
    'jr_wraith':       'tablecloth',
    'wooden_mask':     'wooden_board',
    'lupin':           'lupin_banana',
    'rocky_mask':      'rocky_mask_doll',
    'crab':            'lorang_claw',
    'zombie_lupin':    'cursed_doll',
    'wraith':          'tablecloth',
    'jr_boogie':       'jr_boogie_horn',
    'skeleton_soldier': 'rib',
    'officer_skeleton': 'pelvic_bone',
    'octopus':          'octopus_leg',
    'blue_mushroom':    'blue_mushroom_cap',
    'jr_yeti':          'jr_yeti_skin',
    'hector':           'hector_tail',
    'white_fang':       'white_fang_tail'
}

# 地图背景映射提示表
MAP_BG_HINTS = {
    '第三军营': '第三军营_wall.png',
    '第二军营': '第三军营_wall.png',
    '危险的峡谷': '勇士部落北部.png',
    '火焰之地1': '勇士部落北部.png',
    '火焰之地2': '勇士部落北部.png',
    '沼泽地1': '沼泽地.png',
    '沼泽地2': '沼泽地.png',
    '结冰平原1': '冰雪山谷.png',
    '结冰平原2': '冰雪山谷.png',
    '冰面1': '冰雪山谷.png',
    '冰面2': '冰雪山谷.png',
    '遗迹发掘地2': '遗迹.png',
    '遗迹发掘地3': '遗迹.png',
    '森林迷宫3': '森林迷宫.png',
    '地铁一号线第2地区': '地铁一号线.png',
    '地铁一号线第3地图': '地铁一号线.png',
    '地铁一号线第4地区': '地铁一号线.png',
    '坠落主义': '地铁一号线.png',
    '怪猫的领土': '勇士部落北部.png',
    '龙族打猎场': '勇士部落北部.png'
}


def get_tight_bbox(sprite):
    """提取透明 PNG 图像真实非透明内容的紧凑边界框 (x1, y1, x2, y2)"""
    if sprite.mode != 'RGBA':
        sprite = sprite.convert('RGBA')
    alpha = np.array(sprite.getchannel('A'))
    non_zero = np.argwhere(alpha > 12)
    if non_zero.size == 0:
        return 0, 0, sprite.width, sprite.height
    y_min, x_min = non_zero.min(axis=0)
    y_max, x_max = non_zero.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def extract_polygon_contour(sprite_img, offset_x=0, offset_y=0, epsilon=0.8):
    """提取透明素材轮廓多边形点集 (用于 LabelMe / AnyLabeling 兼容)"""
    if sprite_img.mode != 'RGBA':
        sprite_img = sprite_img.convert('RGBA')
    alpha = np.array(sprite_img.getchannel('A'))
    mask = (alpha > 12).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        w, h = sprite_img.size
        return [[offset_x, offset_y], [offset_x + w, offset_y],
                [offset_x + w, offset_y + h], [offset_x, offset_y + h]]

    largest_c = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest_c, epsilon, True)
    if len(approx) < 4:
        c = largest_c
        step = max(1, len(c) // 25)
        approx = c[::step]
    points = []
    for pt in approx:
        px = round(float(pt[0][0] + offset_x), 1)
        py = round(float(pt[0][1] + offset_y), 1)
        points.append([px, py])
    return points


def check_box_collision(b1, b2, margin=15):
    """检查两个目标边界框是否发生碰撞或非法重叠"""
    x1_a, y1_a, x2_a, y2_a = b1[:4]
    x1_b, y1_b, x2_b, y2_b = b2[:4]
    return not (x2_a + margin < x1_b or x1_a - margin > x2_b or
                y2_a + margin < y1_b or y1_a - margin > y2_b)


def parse_map_mobs(map_name):
    """从地图 XML 中解析出现的所有怪物名称"""
    xml_path = os.path.join(MAP_XML_DIR, f"{map_name}.xml")
    if not os.path.exists(xml_path):
        xml_candidates = glob.glob(os.path.join(MAP_XML_DIR, f"*{map_name}*.xml"))
        if xml_candidates:
            xml_path = xml_candidates[0]
        else:
            return []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        mobs = set()
        for d in root.findall("dir"):
            if d.attrib.get("name") == "life":
                for life_node in d.findall("dir"):
                    m_type = None
                    m_id = None
                    for prop in life_node:
                        name = prop.attrib.get("name")
                        val = prop.attrib.get("value")
                        if name == "type":
                            m_type = val
                        elif name == "id":
                            m_id = val
                    if m_type == "m" and m_id:
                        try:
                            m_id_int = int(m_id)
                            if m_id_int in MOB_ID_TO_KEY:
                                mobs.add(MOB_ID_TO_KEY[m_id_int])
                        except ValueError:
                            pass
        return sorted(list(mobs))
    except Exception as e:
        print(f"⚠️ 解析地图 XML 失败: {e}")
        return []


def find_map_background(map_name):
    """智能查找最匹配该地图的真实游戏背景图像"""
    # 1. 优先检查 hint
    hint_bg = MAP_BG_HINTS.get(map_name)
    if hint_bg:
        hint_path = os.path.join(BG_DIR, hint_bg)
        if os.path.exists(hint_path):
            return hint_path

    # 2. 检查合成素材背景文件夹中是否有精确或前缀匹配的背景图
    exact = os.path.join(BG_DIR, f"{map_name}_wall.png")
    if os.path.exists(exact):
        return exact
    exact2 = os.path.join(BG_DIR, f"{map_name}.png")
    if os.path.exists(exact2):
        return exact2

    candidates = glob.glob(os.path.join(BG_DIR, f"*{map_name}*.png"))
    if candidates:
        return candidates[0]

    # 3. 兜底返回第一个有效的大尺寸背景
    all_bgs = [os.path.join(BG_DIR, f) for f in os.listdir(BG_DIR) if f.endswith(('.png', '.jpg')) and 'minimap' not in f.lower()]
    return all_bgs[0] if all_bgs else None


def load_player_sprites():
    """加载各玩家角色独立素材池 (player1, player2, player3 等等，平权并支持镜像翻转)"""
    players_data = {}
    if not os.path.exists(PLAYER_DIR):
        return players_data

    for sub_dir in sorted(os.listdir(PLAYER_DIR)):
        sub_path = os.path.join(PLAYER_DIR, sub_dir)
        if not os.path.isdir(sub_path):
            continue
        if 'backup' in sub_dir.lower() or 'old' in sub_dir.lower():
            continue

        p_dict = {'player_left': [], 'player_right': []}
        for root, dirs, files in os.walk(sub_path):
            rel_root = os.path.relpath(root, sub_path).replace('\\', '/').lower()
            for f in files:
                if not f.lower().endswith('.png') or f.lower().startswith('icon'):
                    continue
                full_path = os.path.join(root, f)
                fname = f.lower()

                try:
                    img = Image.open(full_path).convert("RGBA")
                    bx1, by1, bx2, by2 = get_tight_bbox(img)
                    if bx2 > bx1 and by2 > by1:
                        img = img.crop((bx1, by1, bx2 + 1, by2 + 1))

                    if 'left' in fname or 'left' in rel_root:
                        im_l = img
                        im_r = img.transpose(Image.FLIP_LEFT_RIGHT)
                        p_dict['player_left'].append(im_l)
                        p_dict['player_right'].append(im_r)
                    elif 'right' in fname or 'right' in rel_root:
                        im_r = img
                        im_l = img.transpose(Image.FLIP_LEFT_RIGHT)
                        p_dict['player_right'].append(im_r)
                        p_dict['player_left'].append(im_l)
                    else:
                        im_l = img
                        im_r = img.transpose(Image.FLIP_LEFT_RIGHT)
                        p_dict['player_left'].append(im_l)
                        p_dict['player_right'].append(im_r)
                except Exception:
                    pass

        if p_dict['player_left'] and p_dict['player_right']:
            players_data[sub_dir] = p_dict

    return players_data


def load_monster_sprites(monster_classes):
    """加载指定地图涉及怪物的全动作帧素材"""
    mobs_dict = {}
    for cls in monster_classes:
        folder = os.path.join(MONSTERS_DIR, cls)
        if not os.path.exists(folder):
            continue
        mobs_dict[cls] = []
        for p in glob.glob(os.path.join(folder, "*.png")):
            try:
                img = Image.open(p).convert("RGBA")
                bx1, by1, bx2, by2 = get_tight_bbox(img)
                if bx2 > bx1 and by2 > by1:
                    img = img.crop((bx1, by1, bx2 + 1, by2 + 1))
                mobs_dict[cls].append({"class_name": cls, "image": img, "name": os.path.basename(p)})
            except Exception:
                pass
    return mobs_dict


def load_drops_and_distractors(monster_classes):
    """加载该地图怪物的掉落物、金色蘑菇跟宠及通用干扰金币"""
    drops = {}
    for cls in monster_classes:
        drop_name = MONSTER_TO_UNIQUE_DROP.get(cls)
        if drop_name:
            sub = os.path.join(DROPS_DIR, drop_name)
            if os.path.exists(sub):
                pngs = glob.glob(os.path.join(sub, "*.png"))
                if pngs:
                    try:
                        im = Image.open(pngs[0]).convert("RGBA")
                        bx1, by1, bx2, by2 = get_tight_bbox(im)
                        if bx2 > bx1 and by2 > by1:
                            im = im.crop((bx1, by1, bx2 + 1, by2 + 1))
                        drops[drop_name] = im
                    except Exception:
                        pass

    # 载入金币与道具
    for c_name in ['gold_coin', 'bronze_coin', 'meso_bills', 'meso_sack']:
        sub = os.path.join(DROPS_DIR, c_name)
        if os.path.exists(sub):
            pngs = glob.glob(os.path.join(sub, "*.png"))
            if pngs:
                try:
                    drops[c_name] = Image.open(pngs[0]).convert("RGBA")
                except Exception:
                    pass

    # 优先载入金蘑菇宠物素材
    golden_pets = []
    pet_dir = os.path.join(DISTRACTORS_DIR, "pet_golden_mushroom")
    if os.path.exists(pet_dir):
        for f in os.listdir(pet_dir):
            if f.lower().endswith('.png') and not f.lower().startswith('icon'):
                try:
                    im = Image.open(os.path.join(pet_dir, f)).convert("RGBA")
                    bx1, by1, bx2, by2 = get_tight_bbox(im)
                    if bx2 > bx1 and by2 > by1:
                        im = im.crop((bx1, by1, bx2 + 1, by2 + 1))
                    golden_pets.append(im)
                except Exception:
                    pass

    return drops, golden_pets


def generate_map_dataset(map_name, count=60, canvas_size=(1280, 720), min_mobs=6, max_mobs=12):
    """为指定地图连续合成高质量训练素材 (优先使用真实全景与地台吸附流水线)"""
    # 优先尝试基于真实地图全景与地台吸附的高清渲染流水线
    try:
        from generate_map_realistic_dataset import build_realistic_dataset
        xml_file = os.path.join(MAP_XML_DIR, f"{map_name}.xml")
        if os.path.exists(xml_file):
            print(f"✨ 检测到地图 XML 【{map_name}.xml】，优先调用真实游戏地台高密度吸附流水线...")
            success = build_realistic_dataset(map_name, count=count, min_mobs=min_mobs, max_mobs=max_mobs)
            if success:
                return True
    except Exception as e:
        print(f"⚠️ 真实地貌吸附生成器提示: {e}，平滑切换为标准模拟模式。")

    print("=" * 70)
    print(f"🗺️ 开始为地图 【{map_name}】 定制合成 YOLO 专属训练素材...")
    print("=" * 70)

    # 1. 解析该地图的怪物列表
    map_mobs = parse_map_mobs(map_name)
    if not map_mobs:
        print(f"❌ 警告: 在地图 XML 或数据库中未找到地图 【{map_name}】 的怪物！请确认地图名称。")
        return False

    # 过滤出本地已备好贴图的怪物
    valid_mobs = [m for m in map_mobs if os.path.exists(os.path.join(MONSTERS_DIR, m))]
    if not valid_mobs:
        print(f"❌ 错误: 地图 【{map_name}】 所需怪物 {map_mobs} 在 synthetic_assets/monsters/ 中均未找到素材！")
        return False

    print(f"   👾 地图实际会出现的怪物 ({len(valid_mobs)} 种): {valid_mobs}")

    # 2. 匹配专属背景
    bg_path = find_map_background(map_name)
    if not bg_path:
        print(f"❌ 错误: 未找到可用的游戏背景素材！")
        return False
    print(f"   🖼️ 选配游戏背景图: {os.path.basename(bg_path)}")

    # 3. 类别定义：只包含 player_left, player_right 和本图出现的怪物
    class_list = ['player_left', 'player_right'] + sorted(valid_mobs)
    class_to_id = {cls_name: i for i, cls_name in enumerate(class_list)}
    print(f"   🎯 专属模型包含类别 ({len(class_list)} 类): {class_list}")

    # 4. 创建地图独立的素材存储文件夹 dataset/map_raw_images/<map_name>
    target_map_dir = os.path.join(MAP_RAW_ROOT, map_name)
    os.makedirs(target_map_dir, exist_ok=True)

    # 写入类别文件 classes.txt
    classes_file = os.path.join(target_map_dir, "classes.txt")
    with open(classes_file, "w", encoding="utf-8") as f:
        for c in class_list:
            f.write(f"{c}\n")

    # 5. 加载贴图资源
    player_sprites = load_player_sprites()
    mobs_sprites = load_monster_sprites(valid_mobs)
    drops_dict, pets = load_drops_and_distractors(valid_mobs)
    bg_img = Image.open(bg_path).convert("RGBA")
    bg_w, bg_h = bg_img.size

    cw, ch = canvas_size
    max_scroll_x = max(0, bg_w - cw)
    max_scroll_y = max(0, bg_h - ch)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    generated_count = 0

    for i in range(count):
        # 裁剪视角
        scroll_x = random.randint(0, max_scroll_x) if max_scroll_x > 0 else 0
        scroll_y = random.randint(0, max_scroll_y) if max_scroll_y > 0 else 0
        crop_bg = bg_img.crop((scroll_x, scroll_y, min(bg_w, scroll_x + cw), min(bg_h, scroll_y + ch)))

        canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 255))
        canvas.paste(crop_bg, (0, 0), crop_bg)

        placed_boxes = [] # (x1, y1, x2, y2, cls_name)
        labels = []       # (cls_id, norm_cx, norm_cy, norm_w, norm_h)
        json_shapes = []

        # 放置本图怪物 (2 ~ 6 只)
        num_mobs = random.randint(2, 6)
        for _ in range(num_mobs):
            m_cls = random.choice(valid_mobs)
            sprite_list = mobs_sprites.get(m_cls, [])
            if not sprite_list:
                continue

            item = random.choice(sprite_list)
            m_img = item["image"].copy()
            if random.random() < 0.5:
                m_img = m_img.transpose(Image.FLIP_LEFT_RIGHT)

            mw, mh = m_img.size
            for attempt in range(25):
                pos_x = random.randint(30, cw - mw - 30)
                pos_y = random.randint(int(ch * 0.25), ch - mh - 20)
                cand_box = (pos_x, pos_y, pos_x + mw, pos_y + mh, m_cls)

                if not any(check_box_collision(cand_box, pb, margin=15) for pb in placed_boxes):
                    canvas.paste(m_img, (pos_x, pos_y), m_img)
                    placed_boxes.append(cand_box)

                    # 记录 YOLO 坐标
                    norm_cx = ((pos_x + pos_x + mw) / 2.0) / cw
                    norm_cy = ((pos_y + pos_y + mh) / 2.0) / ch
                    norm_w = mw / cw
                    norm_h = mh / ch
                    cls_id = class_to_id[m_cls]
                    labels.append((cls_id, norm_cx, norm_cy, norm_w, norm_h))

                    poly_pts = extract_polygon_contour(m_img, offset_x=pos_x, offset_y=pos_y)
                    json_shapes.append({
                        "label": m_cls, "points": poly_pts, "group_id": None,
                        "description": "", "shape_type": "polygon", "flags": {}
                    })

                    # 伴生怪物掉落物
                    if random.random() < 0.45:
                        drop_k = MONSTER_TO_UNIQUE_DROP.get(m_cls)
                        if drop_k and drop_k in drops_dict:
                            d_img = drops_dict[drop_k]
                            canvas.paste(d_img, (pos_x + random.randint(-5, mw), pos_y + mh - 10), d_img)
                    break

        # 放置玩家角色：每张图中同时放入现有的全部 3 个玩家各一帧，且每个玩家身旁必带一个金蘑菇宠物 (不打标签)
        if player_sprites:
            target_pids = list(player_sprites.keys())
            random.shuffle(target_pids)

            for pid in target_pids:
                p_cls = random.choice(['player_left', 'player_right'])
                p_list = player_sprites[pid].get(p_cls, [])
                if not p_list:
                    continue

                p_img = random.choice(p_list)
                pw, ph = p_img.size
                for attempt in range(35):
                    px = random.randint(40, cw - pw - 40)
                    py = random.randint(int(ch * 0.28), ch - ph - 25)
                    p_box = (px, py, px + pw, py + ph, p_cls)

                    if not any(check_box_collision(p_box, pb, margin=20) for pb in placed_boxes):
                        canvas.paste(p_img, (px, py), p_img)
                        placed_boxes.append(p_box)

                        norm_cx = ((px + px + pw) / 2.0) / cw
                        norm_cy = ((py + py + ph) / 2.0) / ch
                        norm_w = pw / cw
                        norm_h = ph / ch
                        cls_id = class_to_id[p_cls]
                        labels.append((cls_id, norm_cx, norm_cy, norm_w, norm_h))

                        poly_pts = extract_polygon_contour(p_img, offset_x=px, offset_y=py)
                        json_shapes.append({
                            "label": p_cls, "points": poly_pts, "group_id": None,
                            "description": "", "shape_type": "polygon", "flags": {}
                        })

                        # 每个玩家身旁必放置金蘑菇跟宠 (不标注)
                        if pets:
                            pet_im = random.choice(pets).copy()
                            if p_cls == 'player_right':
                                pet_im = pet_im.transpose(Image.FLIP_LEFT_RIGHT)
                            pet_w, pet_h = pet_im.size
                            offset_x = (pw + 10) if random.random() < 0.5 else (-pet_w - 10)
                            pet_x = int(np.clip(px + offset_x, 10, cw - pet_w - 10))
                            pet_y = py + ph - pet_h
                            canvas.paste(pet_im, (pet_x, pet_y), pet_im)
                        break

        # 散落通用金币 (不标注)
        for _ in range(random.randint(1, 2)):
            ck = random.choice(['gold_coin', 'bronze_coin', 'meso_bills', 'meso_sack'])
            if ck in drops_dict:
                canvas.paste(drops_dict[ck], (random.randint(40, cw - 60), random.randint(int(ch * 0.4), ch - 50)), drops_dict[ck])

        # 保存图像及 YOLO TXT 标注
        base_name = f"synth_{map_name}_{timestamp}_{i:03d}"
        jpg_path = os.path.join(target_map_dir, f"{base_name}.jpg")
        txt_path = os.path.join(target_map_dir, f"{base_name}.txt")
        json_path = os.path.join(target_map_dir, f"{base_name}.json")

        canvas.convert("RGB").save(jpg_path, "JPEG", quality=95)

        with open(txt_path, "w", encoding="utf-8") as f:
            for l in labels:
                f.write(f"{l[0]} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")

        # 保存 LabelMe 格式 JSON
        labelme_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": json_shapes,
            "imagePath": f"{base_name}.jpg",
            "imageData": None,
            "imageHeight": ch,
            "imageWidth": cw
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)

        generated_count += 1

    print(f"✅ 地图 【{map_name}】 素材合成完毕！共生成 {generated_count} 组训练图像与标签！")
    print(f"📁 存储位置: {target_map_dir}")
    print(f"📄 类别定义: {classes_file}")
    return True


def list_available_maps():
    """列出当前所有可自动合成专属素材的地图清单"""
    print("=" * 75)
    print("📋 扫描地图与素材完备度状态:")
    print("=" * 75)

    xml_files = glob.glob(os.path.join(MAP_XML_DIR, "*.xml"))
    ready_count = 0

    for xf in xml_files:
        map_base = os.path.splitext(os.path.basename(xf))[0]
        mobs = parse_map_mobs(map_base)
        valid_mobs = [m for m in mobs if os.path.exists(os.path.join(MONSTERS_DIR, m))]
        bg_path = find_map_background(map_base)
        bg_name = os.path.basename(bg_path) if bg_path else "None"

        is_ready = (len(valid_mobs) > 0 and bg_path is not None)
        status_icon = "🟢 可一键合成" if is_ready else "⚪ 缺少贴图"
        if is_ready:
            ready_count += 1

        print(f"[{status_icon}] 地图: {map_base:<12s} | 怪物: {valid_mobs} | 配选背景: {bg_name}")

    print("=" * 75)
    print(f"总计就绪地图: {ready_count} 处。可使用 --map <地图名> 或 --all-available 进行连续合成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MapleStory 地图专属训练素材生成器")
    parser.add_argument("--map", type=str, default="", help="指定目标地图名称 (例如: 第三军营)")
    parser.add_argument("--count", type=int, default=60, help="每张地图生成的合成样本数 (默认 60)")
    parser.add_argument("--min-mobs", type=int, default=6, help="每张图最少怪物数 (默认 6)")
    parser.add_argument("--max-mobs", type=int, default=12, help="每张图最多怪物数 (默认 12)")
    parser.add_argument("--all-available", action="store_true", help="自动为所有素材就绪的地图连续批量合成素材")
    parser.add_argument("--list", action="store_true", help="列出当前所有地图及素材就绪状态")
    args = parser.parse_args()

    if args.list:
        list_available_maps()
    elif args.all_available:
        xml_files = glob.glob(os.path.join(MAP_XML_DIR, "*.xml"))
        success_maps = []
        for xf in xml_files:
            m_name = os.path.splitext(os.path.basename(xf))[0]
            mobs = parse_map_mobs(m_name)
            valid_mobs = [m for m in mobs if os.path.exists(os.path.join(MONSTERS_DIR, m))]
            if valid_mobs and find_map_background(m_name):
                ok = generate_map_dataset(m_name, count=args.count, min_mobs=args.min_mobs, max_mobs=args.max_mobs)
                if ok:
                    success_maps.append(m_name)
        print("\n" + "=" * 70)
        print(f"🎉 批量合成结束！共完成 {len(success_maps)} 张地图的素材制作: {success_maps}")
        print("=" * 70)
    elif args.map:
        generate_map_dataset(args.map, count=args.count, min_mobs=args.min_mobs, max_mobs=args.max_mobs)
    else:
        # 默认示范生成当前玩家最常测试的地图：结冰平原1
        print("【提示】未指定 --map 参数，默认启动【结冰平原1】专属高密度素材生成...")
        generate_map_dataset("结冰平原1", count=args.count, min_mobs=args.min_mobs, max_mobs=args.max_mobs)

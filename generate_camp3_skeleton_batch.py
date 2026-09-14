import os
import sys
import glob
import json
import random
import cv2
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "synthetic_assets")
BG_PATH = os.path.join(ASSET_DIR, "backgrounds", "第三军营_wall.png")
MONSTERS_DIR = os.path.join(ASSET_DIR, "monsters")
PLAYER_DIR = os.path.join(ASSET_DIR, "player")
DROPS_DIR = os.path.join(ASSET_DIR, "drops")

RAW_OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "raw_images")
DEBUG_OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "synthetic_debug")
os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

# 30 类别定义
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
    'crab',
    'tauromacis',
    'dark_stone_golem',
    'zombie_lupin',
    'wraith',
    'jr_boogie',
    'skeleton_soldier',
    'officer_skeleton'
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_LIST)}

# 第三军营的 5 层物理站立平台 Y 坐标 (地图绝对高度)
# 各层左平台范围 X: [35, 375], 右平台范围 X: [455, 795]
MAP_TIERS = [
    {"name": "Tier 1 (顶层)", "y": 304, "left": (35, 375), "right": (455, 795)},
    {"name": "Tier 2 (中高)", "y": 645, "left": (35, 375), "right": (455, 795)},
    {"name": "Tier 3 (中层)", "y": 972, "left": (35, 375), "right": (455, 795)},
    {"name": "Tier 4 (中低)", "y": 1258, "left": (35, 375), "right": (455, 795)},
    {"name": "Tier 5 (底层)", "y": 1593, "left": (35, 375), "right": (455, 795)}
]


def get_tight_bbox(sprite):
    alpha = np.array(sprite.getchannel('A'))
    non_zero = np.argwhere(alpha > 10)
    if non_zero.size == 0:
        return 0, 0, sprite.width, sprite.height
    y_min, x_min = non_zero.min(axis=0)
    y_max, x_max = non_zero.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def extract_polygon_contour(sprite_img, offset_x=0, offset_y=0, epsilon=0.8):
    if sprite_img.mode != 'RGBA':
        sprite_img = sprite_img.convert('RGBA')
    alpha = np.array(sprite_img.getchannel('A'))
    mask = (alpha > 10).astype(np.uint8) * 255
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


def load_all_assets():
    # 骷髅士兵与士官 (裁剪至紧致包围盒，确保脚踏实地)
    mobs = {'skeleton_soldier': [], 'officer_skeleton': []}
    for m in ['skeleton_soldier', 'officer_skeleton']:
        d = os.path.join(MONSTERS_DIR, m)
        for f in sorted(os.listdir(d)):
            if f.endswith('.png'):
                im = Image.open(os.path.join(d, f)).convert('RGBA')
                bx1, by1, bx2, by2 = get_tight_bbox(im)
                if bx2 > bx1 and by2 > by1:
                    im = im.crop((bx1, by1, bx2 + 1, by2 + 1))
                mobs[m].append((f, im))

    # 丰富玩家形态: 裁剪掉多余透明留白，分流为实录真实角色 (虎头帽) 与 辅助动作
    players = {'player_left': {'real': [], 'other': []}, 'player_right': {'real': [], 'other': []}}
    for p_cls in ['player_left', 'player_right']:
        d = os.path.join(PLAYER_DIR, p_cls)
        for f in os.listdir(d):
            if f.endswith('.png'):
                fp = os.path.join(d, f)
                try:
                    im = Image.open(fp).convert('RGBA')
                    # 确保左向朝左，右向朝右
                    if p_cls == 'player_left' and 'right' in f.lower() and 'real_player' not in f:
                        im = im.transpose(Image.FLIP_LEFT_RIGHT)
                    elif p_cls == 'player_right' and 'left' in f.lower() and 'real_player' not in f:
                        im = im.transpose(Image.FLIP_LEFT_RIGHT)

                    bx1, by1, bx2, by2 = get_tight_bbox(im)
                    if bx2 > bx1 and by2 > by1:
                        im = im.crop((bx1, by1, bx2 + 1, by2 + 1))

                    if 'real_player' in f.lower():
                        players[p_cls]['real'].append((f, im))
                    else:
                        players[p_cls]['other'].append((f, im))
                except Exception:
                    pass

    # 掉落物
    drops = {}
    for d_name in ['rib', 'pelvic_bone', 'gold_coin', 'meso_bills', 'meso_sack', 'bronze_coin']:
        sub = os.path.join(DROPS_DIR, d_name)
        if os.path.exists(sub):
            pngs = glob.glob(os.path.join(sub, "*.png"))
            if pngs:
                drops[d_name] = Image.open(pngs[0]).convert('RGBA')

    return mobs, players, drops


def check_overlap(box1, box2, margin=10):
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2
    return not (x2_a + margin < x1_b or x1_a - margin > x2_b or
                y2_a + margin < y1_b or y1_a - margin > y2_b)


def generate_batch(num_images=60):
    random.seed(2026)
    np.random.seed(2026)

    mobs_dict, player_dict, drops_dict = load_all_assets()
    print("=" * 75)
    print(f"🚀 开始在【第3军营】高密度批量合成 {num_images} 张高质量训练图像...")
    print(f"   👾 骷髅士兵: {len(mobs_dict['skeleton_soldier'])} 动作帧")
    print(f"   💀 骷髅士官: {len(mobs_dict['officer_skeleton'])} 动作帧")
    print(f"   🤺 玩家形态库: 左向实机 {len(player_dict['player_left']['real'])}+动作 {len(player_dict['player_left']['other'])}, 右向实机 {len(player_dict['player_right']['real'])}+动作 {len(player_dict['player_right']['other'])}")
    print(f"   💎 专属掉落物: {list(drops_dict.keys())}")
    print("=" * 75)

    camp3_map = Image.open(BG_PATH).convert('RGBA')
    mw, mh = camp3_map.size
    cw, ch = 1280, 720
    map_x = (cw - mw) // 2  # 居中偏移

    max_scroll = max(0, mh - ch)
    base_scroll_positions = np.linspace(0, max_scroll, num_images, dtype=int)

    generated_records = []

    for idx, base_cam_y in enumerate(base_scroll_positions):
        img_id = idx + 1
        base_name = f"synth_skeleton_camp3_{img_id:02d}"

        # 增加轻微视口抖动，使每张图背景都不雷同
        cam_jitter = random.randint(-18, 18)
        cam_y = int(np.clip(base_cam_y + cam_jitter, 0, max_scroll))

        # 1. 裁剪对应摄像机高度的地图并居中放入 1280x720 画布
        crop_map = camp3_map.crop((0, cam_y, mw, cam_y + ch))
        canvas = Image.new('RGBA', (cw, ch), (0, 0, 0, 255))
        canvas.paste(crop_map, (map_x, 0), crop_map)

        # 2. 确定在当前视口内可见的平台
        visible_tiers = []
        for tier in MAP_TIERS:
            screen_floor = tier['y'] - cam_y
            if 80 <= screen_floor <= ch - 15:
                visible_tiers.append((tier, screen_floor))

        if not visible_tiers:
            visible_tiers = [(MAP_TIERS[0], 250)]

        placed_boxes = []  # (x1, y1, x2, y2)
        labels = []
        json_shapes = []

        # 3. 放置掉落物 (底层，每层平台均匀分布金币与肋骨/骨盆)
        for tier_info, s_floor in visible_tiers:
            num_drops = random.randint(2, 4)
            for _ in range(num_drops):
                side = random.choice(['left', 'right'])
                min_x, max_x = tier_info[side]
                drop_x = map_x + random.randint(min_x + 5, max_x - 30)
                drop_y = s_floor - random.randint(8, 22)
                dk = random.choice(['rib', 'pelvic_bone', 'gold_coin', 'meso_bills', 'meso_sack', 'bronze_coin'])
                if dk in drops_dict:
                    d_img = drops_dict[dk]
                    canvas.paste(d_img, (drop_x, drop_y), d_img)

        # 4. 重点放置玩家角色 (每张图 2 ~ 3 个，实机虎头帽形象占 65% 权重，严格紧致贴地)
        num_players = random.choice([2, 2, 3, 3, 4])
        players_placed = 0

        for _ in range(num_players):
            p_cls = random.choice(['player_left', 'player_right'])
            # 优先采用用户当前实机角色形象
            pool = player_dict[p_cls]['real'] if (player_dict[p_cls]['real'] and random.random() < 0.65) else player_dict[p_cls]['other']
            if not pool:
                pool = player_dict[p_cls]['real'] or player_dict[p_cls]['other']
            p_fname, p_img_orig = random.choice(pool)
            p_img = p_img_orig.copy()

            tier_info, s_floor = random.choice(visible_tiers)
            side = random.choice(['left', 'right'])
            min_x, max_x = tier_info[side]

            # 角色已经无透明留白，脚底与平台完美对齐
            p_top_y = s_floor - p_img.height + 4

            for attempt in range(40):
                max_pos_x = max(min_x + 1, max_x - p_img.width)
                pos_x = map_x + random.randint(min_x, max_pos_x)
                cand_box = (pos_x, p_top_y, pos_x + p_img.width, p_top_y + p_img.height)

                if not any(check_overlap(cand_box, pb, margin=12) for pb in placed_boxes):
                    canvas.paste(p_img, (pos_x, p_top_y), p_img)
                    placed_boxes.append(cand_box)
                    players_placed += 1

                    abs_x1, abs_y1, abs_x2, abs_y2 = cand_box
                    cx = ((abs_x1 + abs_x2) / 2.0) / cw
                    cy = ((abs_y1 + abs_y2) / 2.0) / ch
                    nw = (abs_x2 - abs_x1) / cw
                    nh = (abs_y2 - abs_y1) / ch
                    labels.append((CLASS_TO_ID[p_cls], cx, cy, nw, nh))

                    lbl_cn = "我左" if p_cls == "player_left" else "我右"
                    poly_pts = extract_polygon_contour(p_img, offset_x=pos_x, offset_y=p_top_y)
                    json_shapes.append({
                        "label": lbl_cn,
                        "points": poly_pts,
                        "group_id": None,
                        "description": "",
                        "shape_type": "polygon",
                        "flags": {}
                    })
                    break

        # 5. 高密度放置怪物 (每张图 6 ~ 9 只，围绕玩家与各层断桥平台充分分布)
        num_mobs = random.randint(6, 9)
        mobs_placed = 0

        for _ in range(num_mobs):
            mob_cls = random.choice(['skeleton_soldier', 'officer_skeleton'])
            sprite_fname, mob_img = random.choice(mobs_dict[mob_cls])
            m_img = mob_img.copy()

            flip = random.choice([True, False])
            if flip:
                m_img = m_img.transpose(Image.FLIP_LEFT_RIGHT)

            tier_info, s_floor = random.choice(visible_tiers)
            side = random.choice(['left', 'right'])
            min_x, max_x = tier_info[side]

            top_y = s_floor - m_img.height + 4

            for attempt in range(40):
                max_pos_x = max(min_x + 1, max_x - m_img.width)
                pos_x = map_x + random.randint(min_x, max_pos_x)
                cand_box = (pos_x, top_y, pos_x + m_img.width, top_y + m_img.height)

                if not any(check_overlap(cand_box, pb, margin=10) for pb in placed_boxes):
                    canvas.paste(m_img, (pos_x, top_y), m_img)
                    placed_boxes.append(cand_box)
                    mobs_placed += 1

                    abs_x1, abs_y1, abs_x2, abs_y2 = cand_box
                    cx = ((abs_x1 + abs_x2) / 2.0) / cw
                    cy = ((abs_y1 + abs_y2) / 2.0) / ch
                    nw = (abs_x2 - abs_x1) / cw
                    nh = (abs_y2 - abs_y1) / ch
                    labels.append((CLASS_TO_ID[mob_cls], cx, cy, nw, nh))

                    lbl_cn = "骷髅士兵" if mob_cls == "skeleton_soldier" else "骷髅士官"
                    poly_pts = extract_polygon_contour(m_img, offset_x=pos_x, offset_y=top_y)
                    json_shapes.append({
                        "label": lbl_cn,
                        "points": poly_pts,
                        "group_id": None,
                        "description": "",
                        "shape_type": "polygon",
                        "flags": {}
                    })
                    break

        # 6. 保存为 JPG、YOLO TXT、AnyLabeling JSON
        rgb_img = canvas.convert("RGB")
        jpg_path = os.path.join(RAW_OUTPUT_DIR, f"{base_name}.jpg")
        txt_path = os.path.join(RAW_OUTPUT_DIR, f"{base_name}.txt")
        json_path = os.path.join(RAW_OUTPUT_DIR, f"{base_name}.json")

        rgb_img.save(jpg_path, quality=95)

        with open(txt_path, 'w', encoding='utf-8') as f:
            for lbl in labels:
                f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "version": "0.3.3",
                "flags": {},
                "shapes": json_shapes,
                "imagePath": f"{base_name}.jpg",
                "imageData": None,
                "imageHeight": ch,
                "imageWidth": cw
            }, f, indent=2, ensure_ascii=False)

        # 7. 生成 Debug 核验图
        dbg_cv = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
        color_map = {
            "骷髅士兵": (0, 255, 255),   # 黄色
            "骷髅士官": (0, 165, 255),   # 橙色
            "我左": (0, 255, 0),       # 绿色
            "我右": (0, 255, 0)
        }
        for shape in json_shapes:
            s_pts = np.array(shape["points"], dtype=np.int32).reshape((-1, 1, 2))
            s_lbl = shape["label"]
            col = color_map.get(s_lbl, (0, 255, 0))
            cv2.polylines(dbg_cv, [s_pts], isClosed=True, color=col, thickness=2)
            top_left = s_pts.min(axis=0)[0]
            cv2.putText(dbg_cv, s_lbl, (top_left[0], max(16, top_left[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        dbg_path = os.path.join(DEBUG_OUTPUT_DIR, f"debug_{base_name}.jpg")
        cv2.imwrite(dbg_path, dbg_cv)

        print(f"  [{img_id:02d}/{num_images}] 已生成 {base_name}.jpg (视口Y: {cam_y:4d}px, {mobs_placed}只怪 + {players_placed}个玩家, 共 {len(labels)} 个目标)")
        generated_records.append((base_name, mobs_placed, players_placed, len(labels)))

    print(f"\n✨ 全部 {num_images} 张高密度大容量第三军营图像与标注生成完毕！")
    return generated_records


if __name__ == '__main__':
    generate_batch(60)


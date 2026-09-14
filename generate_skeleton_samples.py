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
BG_DIR = os.path.join(ASSET_DIR, "backgrounds")
MONSTERS_DIR = os.path.join(ASSET_DIR, "monsters")
PLAYER_DIR = os.path.join(ASSET_DIR, "player")
DROPS_DIR = os.path.join(ASSET_DIR, "drops")

RAW_OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "raw_images")
DEBUG_OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "synthetic_debug")
os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

# 30 类别定义 (2 玩家朝向 + 28 活体怪物)
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


def load_assets():
    # 骷髅士兵与骷髅士官贴图
    skeleton_sprites = {'skeleton_soldier': [], 'officer_skeleton': []}
    for m in ['skeleton_soldier', 'officer_skeleton']:
        dir_p = os.path.join(MONSTERS_DIR, m)
        if os.path.exists(dir_p):
            for f in os.listdir(dir_p):
                if f.endswith('.png'):
                    fp = os.path.join(dir_p, f)
                    try:
                        img = Image.open(fp).convert('RGBA')
                        skeleton_sprites[m].append((f, img))
                    except Exception:
                        pass

    # 玩家贴图
    player_sprites = {'player_left': [], 'player_right': []}
    for p_cls in ['player_left', 'player_right']:
        folder = os.path.join(PLAYER_DIR, p_cls)
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith('.png'):
                    if p_cls == 'player_left' and 'right' in f.lower() and 'plaer_left' not in f and 'player_left' not in f:
                        continue
                    if p_cls == 'player_right' and 'left' in f.lower():
                        continue
                    fp = os.path.join(folder, f)
                    try:
                        img = Image.open(fp).convert('RGBA')
                        player_sprites[p_cls].append((f, img))
                    except Exception:
                        pass

    # 掉落物
    drops = {}
    for d in ['rib', 'pelvic_bone', 'bronze_coin', 'gold_coin', 'meso_bills', 'meso_sack']:
        sub = os.path.join(DROPS_DIR, d)
        if os.path.exists(sub):
            pngs = glob.glob(os.path.join(sub, "*.png"))
            if pngs:
                drops[d] = Image.open(pngs[0]).convert('RGBA')

    return skeleton_sprites, player_sprites, drops


def generate_samples():
    random.seed(42)
    np.random.seed(42)

    skeleton_sprites, player_sprites, drops = load_assets()
    print(f"加载骷髅士兵: {len(skeleton_sprites['skeleton_soldier'])} 帧, 骷髅士官: {len(skeleton_sprites['officer_skeleton'])} 帧")
    print(f"加载玩家形态: left={len(player_sprites['player_left'])} 帧, right={len(player_sprites['player_right'])} 帧")
    print(f"加载掉落物: {list(drops.keys())}")

    # 3 张精心设计的真实场景配置
    configs = [
        {
            'base_name': 'synth_skeleton_01',
            'bg_name': '遗迹.png',
            'crop_offset': (120, 100),
            'desc': '发掘地遗迹-平层巡逻巡视',
            'players': [('player_right', 'player_right_move1.png', 280, 440)],
            'mobs': [
                ('skeleton_soldier', 'stand_0.png', 520, 425, False),
                ('skeleton_soldier', 'move_2.png', 710, 425, True),
                ('officer_skeleton', 'move_1.png', 920, 418, True),
                ('officer_skeleton', 'stand_3.png', 1090, 418, True),
                ('skeleton_soldier', 'move_0.png', 130, 425, False),
            ],
            'drops': [
                ('rib', 480, 505),
                ('gold_coin', 630, 508),
                ('pelvic_bone', 870, 505),
                ('meso_sack', 1030, 508),
            ]
        },
        {
            'base_name': 'synth_skeleton_02',
            'bg_name': '遗迹.png',
            'crop_offset': (220, 80),
            'desc': '发掘地遗迹-阶梯与多层激战',
            'players': [('player_left', 'player_left_attack1.png', 660, 430)],
            'mobs': [
                ('officer_skeleton', 'stand_1.png', 450, 418, False),
                ('skeleton_soldier', 'move_3.png', 310, 425, False),
                ('skeleton_soldier', 'stand_2.png', 160, 425, False),
                ('officer_skeleton', 'move_2.png', 850, 418, True),
                ('skeleton_soldier', 'move_1.png', 1030, 425, True),
            ],
            'drops': [
                ('pelvic_bone', 420, 505),
                ('rib', 770, 505),
                ('gold_coin', 790, 508),
                ('meso_bills', 960, 508),
                ('rib', 240, 505),
            ]
        },
        {
            'base_name': 'synth_skeleton_03',
            'bg_name': '勇士部落北部.png',
            'crop_offset': (50, 60),
            'desc': '勇士部落岩石高台-骷髅军团伏击',
            'players': [
                ('player_right', 'player_right_stand.png', 220, 450),
            ],
            'mobs': [
                ('skeleton_soldier', 'move_0.png', 440, 440, False),
                ('officer_skeleton', 'move_3.png', 620, 432, True),
                ('skeleton_soldier', 'stand_1.png', 800, 440, True),
                ('officer_skeleton', 'stand_0.png', 980, 432, True),
                ('skeleton_soldier', 'move_2.png', 1120, 440, True),
            ],
            'drops': [
                ('rib', 520, 520),
                ('pelvic_bone', 710, 520),
                ('bronze_coin', 890, 522),
                ('meso_bills', 1050, 522),
            ]
        }
    ]

    tw, th = 1280, 720
    generated_files = []

    for cfg in configs:
        out_basename = cfg['base_name']
        print(f"\n🎨 正在合成 [{out_basename}] ({cfg['desc']})...")

        bg_path = os.path.join(BG_DIR, cfg['bg_name'])
        bg_img = Image.open(bg_path).convert('RGBA')
        bw, bh = bg_img.size

        rx, ry = cfg['crop_offset']
        rx = max(0, min(bw - tw, rx))
        ry = max(0, min(bh - th, ry))
        canvas = bg_img.crop((rx, ry, rx + tw, ry + th))

        labels = []
        json_shapes = []

        # 1. 放置掉落物 (底层)
        for dk, dx, dy in cfg['drops']:
            if dk in drops:
                d_img = drops[dk]
                canvas.paste(d_img, (dx, dy), d_img)

        # 2. 放置怪物
        mob_lookup = {
            'skeleton_soldier': {name: img for name, img in skeleton_sprites['skeleton_soldier']},
            'officer_skeleton': {name: img for name, img in skeleton_sprites['officer_skeleton']}
        }

        for mob_cls, mob_file, mx, my, flip in cfg['mobs']:
            deck = mob_lookup[mob_cls]
            m_img = deck.get(mob_file, list(deck.values())[0]).copy()
            if flip:
                m_img = m_img.transpose(Image.FLIP_LEFT_RIGHT)

            bx1, by1, bx2, by2 = get_tight_bbox(m_img)
            canvas.paste(m_img, (mx, my), m_img)
            cand_box = (mx + bx1, my + by1, mx + bx2, my + by2, mob_cls)

            abs_x1, abs_y1, abs_x2, abs_y2, _ = cand_box
            cx = ((abs_x1 + abs_x2) / 2.0) / tw
            cy = ((abs_y1 + abs_y2) / 2.0) / th
            nw = (abs_x2 - abs_x1) / tw
            nh = (abs_y2 - abs_y1) / th
            labels.append((CLASS_TO_ID[mob_cls], cx, cy, nw, nh))

            lbl_cn = "骷髅士兵" if mob_cls == "skeleton_soldier" else "骷髅士官"
            poly_pts = extract_polygon_contour(m_img, offset_x=mx, offset_y=my)
            json_shapes.append({
                "label": lbl_cn,
                "points": poly_pts,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {}
            })

        # 3. 放置玩家
        for p_cls, p_file, px, py in cfg['players']:
            p_img = None
            for fname, img in player_sprites[p_cls]:
                if fname == p_file:
                    p_img = img.copy()
                    break
            if p_img is None and player_sprites[p_cls]:
                p_img = player_sprites[p_cls][0][1].copy()

            if p_img is not None:
                pbx1, pby1, pbx2, pby2 = get_tight_bbox(p_img)
                canvas.paste(p_img, (px, py), p_img)
                p_box = (px + pbx1, py + pby1, px + pbx2, py + pby2, p_cls)

                abs_x1, abs_y1, abs_x2, abs_y2, _ = p_box
                cx = ((abs_x1 + abs_x2) / 2.0) / tw
                cy = ((abs_y1 + abs_y2) / 2.0) / th
                nw = (abs_x2 - abs_x1) / tw
                nh = (abs_y2 - abs_y1) / th
                labels.append((CLASS_TO_ID[p_cls], cx, cy, nw, nh))

                lbl_cn = "我左" if p_cls == "player_left" else "我右"
                poly_pts = extract_polygon_contour(p_img, offset_x=px, offset_y=py)
                json_shapes.append({
                    "label": lbl_cn,
                    "points": poly_pts,
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {}
                })

        # 保存为 JPG、YOLO TXT、AnyLabeling JSON
        rgb_img = canvas.convert("RGB")
        jpg_path = os.path.join(RAW_OUTPUT_DIR, f"{out_basename}.jpg")
        txt_path = os.path.join(RAW_OUTPUT_DIR, f"{out_basename}.txt")
        json_path = os.path.join(RAW_OUTPUT_DIR, f"{out_basename}.json")

        rgb_img.save(jpg_path, quality=95)
        print(f"  ✓ 图像已保存: {jpg_path}")

        with open(txt_path, 'w', encoding='utf-8') as f:
            for lbl in labels:
                f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")
        print(f"  ✓ YOLO 标注已保存 ({len(labels)} 个目标): {txt_path}")

        labelme_data = {
            "version": "0.3.3",
            "flags": {},
            "shapes": json_shapes,
            "imagePath": f"{out_basename}.jpg",
            "imageData": None,
            "imageHeight": th,
            "imageWidth": tw
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ AnyLabeling 多边形 JSON 已保存: {json_path}")

        # 生成 Debug 可视化核验图
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
            cv2.putText(dbg_cv, s_lbl, (top_left[0], max(18, top_left[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        dbg_path = os.path.join(DEBUG_OUTPUT_DIR, f"debug_{out_basename}.jpg")
        cv2.imwrite(dbg_path, dbg_cv)
        print(f"  ✓ 调试核验图已生成: {dbg_path}")
        generated_files.append((out_basename, dbg_path))

    print("\n✨ 3 张合成场景及标注已全部成功生成！")
    return generated_files


if __name__ == '__main__':
    generate_samples()

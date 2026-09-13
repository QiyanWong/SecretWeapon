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
MONSTER_DIR = os.path.join(ASSET_DIR, "monsters", "jr_boogie")
PLAYER_DIR = os.path.join(ASSET_DIR, "player")
DROPS_DIR = os.path.join(ASSET_DIR, "drops")

RAW_OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "raw_images")
DEBUG_OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "synthetic_debug")
os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

# 28 类别定义 (2 玩家朝向 + 26 活体怪物)
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
    'jr_boogie'
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
    # 幼魔精灵贴图
    jb_sprites = []
    for f in os.listdir(MONSTER_DIR):
        if f.endswith('.png'):
            fp = os.path.join(MONSTER_DIR, f)
            try:
                img = Image.open(fp).convert('RGBA')
                jb_sprites.append((f, img))
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
    for d in ['jr_boogie_horn', 'bronze_coin', 'gold_coin', 'meso_bills', 'meso_sack']:
        sub = os.path.join(DROPS_DIR, d)
        if os.path.exists(sub):
            pngs = glob.glob(os.path.join(sub, "*.png"))
            if pngs:
                drops[d] = Image.open(pngs[0]).convert('RGBA')

    return jb_sprites, player_sprites, drops


def generate_single_sample():
    random.seed(99)
    np.random.seed(99)

    jb_sprites, player_sprites, drops = load_assets()
    print(f"加载幼魔精灵帧: {len(jb_sprites)} 帧")
    print(f"加载玩家形态: left={len(player_sprites['player_left'])} 帧, right={len(player_sprites['player_right'])} 帧")
    print(f"加载掉落物: {list(drops.keys())}")

    cfg = {
        'bg_name': '森林迷宫.png',
        'base_name': 'synth_jr_boogie_01',
        'player_config': [('player_right', 'player_right_move1.png', 340, 465)],
        'mobs_config': [
            ('stand_0.png', 580, 435, False),
            ('move_3.png', 780, 430, True),
            ('skill1_2.png', 990, 420, True),
            ('hit1_0.png', 480, 440, False),
            ('move_1.png', 220, 270, False)
        ],
        'drops_config': [
            ('jr_boogie_horn', 520, 525),
            ('gold_coin', 830, 525),
            ('jr_boogie_horn', 1020, 520),
            ('meso_bills', 690, 525)
        ]
    }

    tw, th = 1280, 720
    bg_path = os.path.join(BG_DIR, cfg['bg_name'])
    bg_img = Image.open(bg_path).convert('RGBA')
    bw, bh = bg_img.size

    if bw > tw and bh > th:
        rx = min(bw - tw, max(0, int((bw - tw) * 0.35)))
        ry = min(bh - th, max(0, int((bh - th) * 0.5)))
        canvas = bg_img.crop((rx, ry, rx + tw, ry + th))
    else:
        canvas = bg_img.resize((tw, th), Image.Resampling.LANCZOS)

    labels = []
    json_shapes = []

    # 放置掉落物 (底层)
    for dk, dx, dy in cfg['drops_config']:
        if dk in drops:
            d_img = drops[dk]
            canvas.paste(d_img, (dx, dy), d_img)

    # 放置幼魔精灵
    jb_map = {name: img for name, img in jb_sprites}
    for mob_file, mx, my, flip in cfg['mobs_config']:
        m_img = jb_map.get(mob_file, jb_sprites[0][1]).copy()
        if flip:
            m_img = m_img.transpose(Image.FLIP_LEFT_RIGHT)

        bx1, by1, bx2, by2 = get_tight_bbox(m_img)
        canvas.paste(m_img, (mx, my), m_img)
        cand_box = (mx + bx1, my + by1, mx + bx2, my + by2, 'jr_boogie')

        abs_x1, abs_y1, abs_x2, abs_y2, _ = cand_box
        cx = ((abs_x1 + abs_x2) / 2.0) / tw
        cy = ((abs_y1 + abs_y2) / 2.0) / th
        nw = (abs_x2 - abs_x1) / tw
        nh = (abs_y2 - abs_y1) / th
        labels.append((CLASS_TO_ID['jr_boogie'], cx, cy, nw, nh))

        poly_pts = extract_polygon_contour(m_img, offset_x=mx, offset_y=my)
        json_shapes.append({
            "label": "幼魔精灵",
            "points": poly_pts,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {}
        })

    # 放置玩家
    for p_cls, p_file, px, py in cfg['player_config']:
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
    out_basename = cfg['base_name']
    rgb_img = canvas.convert("RGB")
    jpg_path = os.path.join(RAW_OUTPUT_DIR, f"{out_basename}.jpg")
    txt_path = os.path.join(RAW_OUTPUT_DIR, f"{out_basename}.txt")
    json_path = os.path.join(RAW_OUTPUT_DIR, f"{out_basename}.json")

    rgb_img.save(jpg_path, quality=95)

    with open(txt_path, 'w', encoding='utf-8') as f:
        for lbl in labels:
            f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "version": "0.3.3",
            "flags": {},
            "shapes": json_shapes,
            "imagePath": f"{out_basename}.jpg",
            "imageData": None,
            "imageHeight": th,
            "imageWidth": tw
        }, f, indent=2, ensure_ascii=False)

    # 生成可视化调试图
    dbg_cv = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
    for shape in json_shapes:
        s_pts = np.array(shape["points"], dtype=np.int32).reshape((-1, 1, 2))
        s_lbl = shape["label"]
        is_player = ('我' in s_lbl or 'player' in s_lbl)
        color = (0, 255, 255) if is_player else (0, 255, 0)
        cv2.polylines(dbg_cv, [s_pts], isClosed=True, color=color, thickness=2)

        top_left = s_pts.min(axis=0)[0]
        bottom_right = s_pts.max(axis=0)[0]
        cv2.rectangle(dbg_cv, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), color, 1)

        display_text = f"{s_lbl}"
        cv2.putText(dbg_cv, display_text, (top_left[0], max(18, top_left[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if is_player else (255, 255, 0), 2)

    dbg_path = os.path.join(DEBUG_OUTPUT_DIR, f"debug_{out_basename}.jpg")
    cv2.imwrite(dbg_path, dbg_cv)

    print(f"✓ 生成完成: {out_basename} (包含 {len(labels)} 个标注目标: 幼魔精灵 + 玩家)")
    return jpg_path, txt_path, json_path, dbg_path


if __name__ == "__main__":
    generate_single_sample()

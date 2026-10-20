"""
MapleStory Map Realistic Synthetic Dataset Generator (真实地图吸附式高密度合成生成器)
1. 从地图 XML 自动解析 Map ID，从 MapleStory 官方 API 下载完整前景渲染 (Tiles+Objs+Ladders) 与背景贴图
2. 依据 XML back 配置合成完整真实游戏全景底图 (Alpha Composite)
3. 自动检测并补齐缺失的怪物动作序列透明帧及战利品掉落物
4. 解析物理站立平台 (Footholds) 与刷怪点 (Life)
5. 模拟游戏 1280x720 视口采样，实现高密度怪物站立/巡逻分布 (默认 6~12 只/图，支持配置)
6. 自动引入纯地貌负样本，输出 YOLO .txt + LabelMe .json + 可视化预览图
"""

import os
import sys
import ssl
import json
import base64
import random
import argparse
import urllib.request
import io
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from PIL import Image, ImageSequence

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
headers = {'User-Agent': 'Mozilla/5.0'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAP_DIR = os.path.join(BASE_DIR, "map")
ASSET_DIR = os.path.join(BASE_DIR, "synthetic_assets")
BG_DIR = os.path.join(ASSET_DIR, "backgrounds")
MONSTERS_DIR = os.path.join(ASSET_DIR, "monsters")
DROPS_DIR = os.path.join(ASSET_DIR, "drops")
PLAYER_DIR = os.path.join(ASSET_DIR, "player")
DISTRACTORS_DIR = os.path.join(ASSET_DIR, "distractors")
MAP_RAW_ROOT = os.path.join(BASE_DIR, "dataset", "map_raw_images")

# 官方怪物 ID 映射
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
    1140100: 'axe_stump'
}

MOB_KEY_TO_ID = {v: k for k, v in MOB_ID_TO_KEY.items()}

MONSTER_TO_UNIQUE_DROP = {
    'orange_mushroom': ('orange_mushroom_cap', 4000016),
    'red_snail':       ('red_snail_shell', 4000019),
    'slime':           ('squishy_liquid', 4000011),
    'bubbling':        ('bubbling_bubble', 4000014),
    'horny_mushroom':  ('horny_mushroom_cap', 4000017),
    'zombie_mushroom': ('charm_of_the_undead', 4000018),
    'axe_stump':       ('firewood', 4000015),
    'wild_boar':       ('wild_boar_tooth', 4000020),
    'pig':             ('pig_head', 4000021),
    'ribbon_pig':      ('pig_ribbon', 4000001),
    'fire_boar':       ('fire_boar_tooth', 4000024),
    'jr_necki':        ('jr_necki_skin', 4000025),
    'croco':           ('ligator_skin', 4000027),
    'drake':           ('drake_skull', 4000028),
    'evil_eye':        ('evil_eye_tail', 4000022),
    'cold_eye':        ('cold_eye_tail', 4000023),
    'jr_wraith':       ('tablecloth', 4000030),
    'wooden_mask':     ('wooden_board', 4000032),
    'lupin':           ('lupin_banana', 4000033),
    'rocky_mask':      ('rocky_mask_doll', 4000034),
    'crab':            ('lorang_claw', 4000035),
    'zombie_lupin':    ('cursed_doll', 4000031),
    'wraith':          ('tablecloth', 4000030),
    'jr_boogie':       ('jr_boogie_horn', 4000036),
    'skeleton_soldier': ('rib', 4000045),
    'officer_skeleton': ('pelvic_bone', 4000046),
    'octopus':          ('octopus_leg', 4000013),
    'blue_mushroom':    ('blue_mushroom_cap', 4000012),
    'jr_yeti':          ('jr_yeti_skin', 4000048),
    'hector':           ('hector_tail', 4000051),
    'white_fang':       ('white_fang_tail', 4000052)
}


def get_tight_bbox(sprite):
    """提取透明 PNG 真实内容紧凑包围盒 (x1, y1, x2, y2)"""
    if sprite.mode != 'RGBA':
        sprite = sprite.convert('RGBA')
    alpha = np.array(sprite.getchannel('A'))
    non_zero = np.argwhere(alpha > 12)
    if non_zero.size == 0:
        return 0, 0, sprite.width, sprite.height
    y_min, x_min = non_zero.min(axis=0)
    y_max, x_max = non_zero.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def extract_polygon(sprite_img, offset_x=0, offset_y=0, epsilon=0.8):
    """提取透明素材轮廓多边形 (LabelMe / AnyLabeling 兼容)"""
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
    return [[round(float(pt[0][0] + offset_x), 1), round(float(pt[0][1] + offset_y), 1)] for pt in approx]


def ensure_monster_sprites(mob_key, mob_id):
    """如果本地缺失怪物贴图，全自动通过官方 API 下载 stand 与 move 动作透明序列"""
    mob_dir = os.path.join(MONSTERS_DIR, mob_key)
    os.makedirs(mob_dir, exist_ok=True)
    existing_pngs = [f for f in os.listdir(mob_dir) if f.endswith('.png')]
    if len(existing_pngs) >= 4:
        return True

    print(f"   📥 正在自动从 API 获取怪物 [{mob_key}] (ID: {mob_id}) 动作贴图...")
    downloaded = 0
    for act in ['stand', 'move', 'hit1']:
        for ver in ['83', '210.1.1']:
            url = f"https://maplestory.io/api/GMS/{ver}/mob/{mob_id}/render/{act}"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
                    gif_bytes = resp.read()
                im = Image.open(io.BytesIO(gif_bytes))
                f_idx = 0
                for frame in ImageSequence.Iterator(im):
                    frame_rgba = frame.convert("RGBA")
                    f_name = os.path.join(mob_dir, f"{act}_{f_idx}.png")
                    frame_rgba.save(f_name, format="PNG")
                    f_idx += 1
                    downloaded += 1
                if f_idx > 0:
                    break
            except Exception:
                pass
    if downloaded > 0:
        print(f"   ✓ 成功为 [{mob_key}] 补齐 {downloaded} 帧动作贴图")
        return True
    return len(existing_pngs) > 0


def ensure_drop_icon(drop_key, item_id):
    """确保专属掉落物图标存在"""
    drop_sub = os.path.join(DROPS_DIR, drop_key)
    os.makedirs(drop_sub, exist_ok=True)
    existing = [f for f in os.listdir(drop_sub) if f.endswith('.png')]
    if existing:
        return
    url = f"https://maplestory.io/api/GMS/83/item/{item_id}/icon"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
            data = resp.read()
        with open(os.path.join(drop_sub, f"{drop_key}.png"), "wb") as f:
            f.write(data)
    except Exception:
        pass


def download_map_assets(map_name, xml_path):
    """从 XML 解析并下载地图前景全景、背景与怪物素材"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    map_id = root.attrib.get('name', '').replace('.img', '')
    if not map_id.isdigit():
        for prop in root.findall('./dir[@name="info"]/*'):
            if prop.attrib.get('name') == 'returnMap':
                map_id = prop.attrib.get('value')
                break

    print(f"🗺️  正在处理地图: 【{map_name}】 | 官方 Map ID: {map_id}")

    # 1. 前景渲染 (Tiles + Objs + Ladders + Portals)
    fg_render_path = os.path.join(BG_DIR, f"{map_id}_render.png")
    if not os.path.exists(fg_render_path):
        url = f"https://maplestory.io/api/GMS/83/map/{map_id}/render"
        print(f"   📥 正在下载地图全景前景渲染 (Tiles+Objs): {url} ...")
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=35) as resp:
            data = resp.read()
        with open(fg_render_path, "wb") as f:
            f.write(data)
        print(f"   ✓ 成功下载地图前景渲染层: {fg_render_path}")

    fg_im = Image.open(fg_render_path).convert("RGBA")
    mw, mh = fg_im.size

    # 2. 远景背景
    backs = []
    for b in root.findall('./dir[@name="back"]/dir'):
        p = {c.attrib.get('name'): c.attrib.get('value') for c in b}
        backs.append(p)

    bg_cache_dir = os.path.join(BG_DIR, "back_tiles")
    os.makedirs(bg_cache_dir, exist_ok=True)

    for b in backs:
        bS = b.get('bS')
        no = b.get('no')
        if bS and no:
            tile_file = os.path.join(bg_cache_dir, f"{bS}_{no}.png")
            if not os.path.exists(tile_file) or os.path.getsize(tile_file) < 100:
                tile_url = f"https://maplestory.io/api/wz/GMS/83/Map/Back/{bS}.img/back/{no}"
                try:
                    req = urllib.request.Request(tile_url, headers=headers)
                    with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
                        resp_bytes = resp.read()
                    if resp_bytes.strip().startswith(b'{'):
                        j_obj = json.loads(resp_bytes.decode('utf-8'))
                        raw_png = base64.b64decode(j_obj.get('value', ''))
                    else:
                        raw_png = resp_bytes
                    with open(tile_file, "wb") as f:
                        f.write(raw_png)
                except Exception as e:
                    print(f"   ⚠️ 背景素材 {bS}_{no} 下载跳过: {e}")

    # 合成背景画布
    bg_canvas = Image.new("RGBA", (mw, mh), (20, 24, 36, 255))
    if backs:
        first_bS = backs[0].get('bS')
        b0_path = os.path.join(bg_cache_dir, f"{first_bS}_0.png")
        if os.path.exists(b0_path):
            try:
                b0 = Image.open(b0_path).convert("RGBA")
                tw, th = b0.size
                for y in range(0, mh, th):
                    for x in range(0, mw, tw):
                        bg_canvas.paste(b0, (x, y), b0)
            except Exception:
                pass

        # 远景雪山/巨石装饰物
        for b in backs[1:]:
            b_path = os.path.join(bg_cache_dir, f"{b.get('bS')}_{b.get('no')}.png")
            if os.path.exists(b_path):
                try:
                    dec_im = Image.open(b_path).convert("RGBA")
                    bg_canvas.paste(dec_im, (mw // 2 - dec_im.width // 2, mh - dec_im.height - 150), dec_im)
                    break
                except Exception:
                    pass

    # 3. 合成全景底图 (Alpha Composite)
    full_map = Image.alpha_composite(bg_canvas, fg_im)
    full_map_path = os.path.join(BG_DIR, f"{map_name}_full.png")
    full_map.save(full_map_path, format="PNG")
    print(f"   ✓ 成功生成 100% 真实全景底图: {full_map_path} ({mw}x{mh})")

    # 4. 解析怪物与物理平台
    mobs = set()
    for d in root.findall('./dir[@name="life"]/dir'):
        p = {c.attrib.get('name'): c.attrib.get('value') for c in d}
        if p.get('type') == 'm':
            try:
                m_id = int(p.get('id', 0))
                if m_id in MOB_ID_TO_KEY:
                    m_key = MOB_ID_TO_KEY[m_id]
                    mobs.add(m_key)
                    # 自动补齐素材
                    ensure_monster_sprites(m_key, m_id)
                    drop_info = MONSTER_TO_UNIQUE_DROP.get(m_key)
                    if drop_info:
                        ensure_drop_icon(drop_info[0], drop_info[1])
            except ValueError:
                pass

    # 物理站立平台
    fh_dir = root.find("./dir[@name='foothold']")
    all_fhs = []
    if fh_dir is not None:
        for l in fh_dir:
            for g in l:
                for fh in g:
                    p = {c.attrib.get('name'): c.attrib.get('value') for c in fh}
                    all_fhs.append((int(p.get('x1', 0)), int(p.get('y1', 0)),
                                    int(p.get('x2', 0)), int(p.get('y2', 0))))

    # 精确对齐世界坐标与渲染图 (使用通用数学模型计算偏移)
    if all_fhs:
        min_fx = min(min(f[0] for f in all_fhs), min(f[2] for f in all_fhs))
        max_fx = max(max(f[0] for f in all_fhs), max(f[2] for f in all_fhs))
        min_fy = min(min(f[1] for f in all_fhs), min(f[3] for f in all_fhs))
        max_fy = max(max(f[1] for f in all_fhs), max(f[3] for f in all_fhs))
        fh_w = max_fx - min_fx
        margin_x = max(0, (mw - fh_w) / 2.0)
        ox = int(round(margin_x - min_fx))
        # 地表一般距渲染画布底部 37-40px
        oy = int(round((mh - 37) - max_fy))
    else:
        ox, oy = 864, 1040

    # 提取水平地台线段并自动融合相邻线段为完整大平台
    raw_fhs = []
    for x1, y1, x2, y2 in all_fhs:
        if abs(y1 - y2) <= 5 and x2 > x1:
            rx1 = x1 + ox
            rx2 = x2 + ox
            ry = y1 + oy
            if 0 <= ry < mh:
                raw_fhs.append((rx1, rx2, ry))

    # 按平台高度 Y 聚类并融合连续/临近的地台 (间隙 <= 8px 视为同一地台)
    by_y = {}
    for rx1, rx2, ry in raw_fhs:
        found_y = None
        for gy in by_y:
            if abs(gy - ry) <= 4:
                found_y = gy
                break
        if found_y is None:
            found_y = ry
            by_y[found_y] = []
        by_y[found_y].append((rx1, rx2))

    platforms = []
    for gy, segs in by_y.items():
        segs.sort(key=lambda s: s[0])
        curr_x1, curr_x2 = segs[0]
        for nxt_x1, nxt_x2 in segs[1:]:
            if nxt_x1 <= curr_x2 + 8:
                curr_x2 = max(curr_x2, nxt_x2)
            else:
                if curr_x2 - curr_x1 >= 60:
                    platforms.append({'x1': curr_x1, 'x2': curr_x2, 'y': gy, 'len': curr_x2 - curr_x1})
                curr_x1, curr_x2 = nxt_x1, nxt_x2
        if curr_x2 - curr_x1 >= 60:
            platforms.append({'x1': curr_x1, 'x2': curr_x2, 'y': gy, 'len': curr_x2 - curr_x1})

    print(f"   ✓ 成功从 XML 融合生成 {len(platforms)} 条物理完整站立平台 (总长度: {sum(p['len'] for p in platforms)}px)")
    return full_map, sorted(list(mobs)), platforms


def build_realistic_dataset(map_name, count=60, min_mobs=6, max_mobs=12, negative_ratio=0.10, player_scale=1.0):
    """
    高密度真实地图合成生成器
    - count: 生成图片组数 (默认 60)
    - min_mobs: 视口内最少怪物数 (默认 6 只)
    - max_mobs: 视口内最多怪物数 (默认 12 只)
    - negative_ratio: 纯雪景地貌抗干扰负样本比例 (默认 10%)
    - player_scale: 玩家角色缩放比例 (默认 1.0 即 100% 原始像素)
    """
    xml_path = os.path.join(MAP_DIR, f"{map_name}.xml")
    if not os.path.exists(xml_path):
        print(f"❌ 找不到地图 XML 文件: {xml_path}")
        return False

    full_map, mobs, platforms = download_map_assets(map_name, xml_path)
    mw, mh = full_map.size

    valid_mobs = [m for m in mobs if os.path.exists(os.path.join(MONSTERS_DIR, m))]
    if not valid_mobs:
        print(f"❌ 怪物素材缺失: {mobs}，请确认对应怪物贴图！")
        return False

    # 专属模型类别 (必须包含玩家左右朝向 + 该地图出现的怪物)
    class_list = ['player_left', 'player_right'] + sorted(valid_mobs)
    class_to_id = {c: i for i, c in enumerate(class_list)}

    target_dir = os.path.join(MAP_RAW_ROOT, map_name)
    debug_dir = os.path.join(target_dir, "debug_previews")
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    with open(os.path.join(target_dir, "classes.txt"), "w", encoding="utf-8") as f:
        for c in class_list:
            f.write(f"{c}\n")

    # 载入怪物动作序列
    mobs_sprites = {}
    for m in valid_mobs:
        mobs_sprites[m] = []
        mdir = os.path.join(MONSTERS_DIR, m)
        for f in os.listdir(mdir):
            if f.endswith('.png'):
                im = Image.open(os.path.join(mdir, f)).convert('RGBA')
                bx1, by1, bx2, by2 = get_tight_bbox(im)
                if bx2 > bx1 and by2 > by1:
                    im = im.crop((bx1, by1, bx2 + 1, by2 + 1))
                mobs_sprites[m].append(im)

    # 载入各玩家角色独立素材池 (player1, player2, player3 等等，权重完全一致)
    players_data = {}  # { 'player1': {'player_left': [...], 'player_right': [...]}, ... }

    if os.path.exists(PLAYER_DIR):
        for sub_dir in sorted(os.listdir(PLAYER_DIR)):
            sub_path = os.path.join(PLAYER_DIR, sub_dir)
            if not os.path.isdir(sub_path):
                continue
            if 'backup' in sub_dir.lower() or 'old' in sub_dir.lower():
                continue

            p_dict = {'player_left': [], 'player_right': []}
            for root_d, dirs, files in os.walk(sub_path):
                rel_root = os.path.relpath(root_d, sub_path).replace('\\', '/').lower()
                for f in files:
                    if not f.lower().endswith('.png') or f.lower().startswith('icon'):
                        continue
                    try:
                        im = Image.open(os.path.join(root_d, f)).convert('RGBA')
                        bx1, by1, bx2, by2 = get_tight_bbox(im)
                        if bx2 > bx1 and by2 > by1:
                            im = im.crop((bx1, by1, bx2 + 1, by2 + 1))
                        if player_scale != 1.0 and player_scale > 0.1:
                            nw = max(10, int(round(im.width * player_scale)))
                            nh = max(10, int(round(im.height * player_scale)))
                            im = im.resize((nw, nh), Image.Resampling.BILINEAR)

                        fname = f.lower()
                        # 若文件名或文件夹包含明确左右标识
                        if 'left' in fname or 'left' in rel_root:
                            im_l = im
                            im_r = im.transpose(Image.FLIP_LEFT_RIGHT)
                            p_dict['player_left'].append(im_l)
                            p_dict['player_right'].append(im_r)
                        elif 'right' in fname or 'right' in rel_root:
                            im_r = im
                            im_l = im.transpose(Image.FLIP_LEFT_RIGHT)
                            p_dict['player_right'].append(im_r)
                            p_dict['player_left'].append(im_l)
                        else:
                            # 纸娃娃无左右标识，默认朝左，镜像生成朝右
                            im_l = im
                            im_r = im.transpose(Image.FLIP_LEFT_RIGHT)
                            p_dict['player_left'].append(im_l)
                            p_dict['player_right'].append(im_r)
                    except Exception:
                        pass

            if p_dict['player_left'] and p_dict['player_right']:
                players_data[sub_dir] = p_dict

    # 载入金色蘑菇宠物素材 (每个玩家旁边必带一只，不打标签作为负样本)
    pet_dir = os.path.join(DISTRACTORS_DIR, "pet_golden_mushroom")
    pet_sprites = []
    if os.path.exists(pet_dir):
        for f in os.listdir(pet_dir):
            if f.lower().endswith('.png') and not f.lower().startswith('icon'):
                try:
                    pim = Image.open(os.path.join(pet_dir, f)).convert('RGBA')
                    pbx1, pby1, pbx2, pby2 = get_tight_bbox(pim)
                    if pbx2 > pbx1 and pby2 > pby1:
                        pim = pim.crop((pbx1, pby1, pbx2 + 1, pby2 + 1))
                    pet_sprites.append(pim)
                except Exception:
                    pass

    # 载入备用通用干扰物素材
    distractor_sprites = []
    if os.path.exists(DISTRACTORS_DIR):
        for root_d, dirs, files in os.walk(DISTRACTORS_DIR):
            if 'pet_golden_mushroom' in root_d.lower():
                continue
            for f in files:
                if f.lower().endswith('.png') and not f.lower().startswith('icon'):
                    try:
                        dim = Image.open(os.path.join(root_d, f)).convert('RGBA')
                        dbx1, dby1, dbx2, dby2 = get_tight_bbox(dim)
                        if dbx2 > dbx1 and dby2 > dby1:
                            dim = dim.crop((dbx1, dby1, dbx2 + 1, dby2 + 1))
                        distractor_sprites.append(dim)
                    except Exception:
                        pass

    # 载入掉落物与金币
    drops = {}
    for m in valid_mobs:
        drop_info = MONSTER_TO_UNIQUE_DROP.get(m)
        if drop_info:
            dk = drop_info[0]
            ddir = os.path.join(DROPS_DIR, dk)
            if os.path.exists(ddir):
                for f in os.listdir(ddir):
                    if f.endswith('.png'):
                        drops[dk] = Image.open(os.path.join(ddir, f)).convert('RGBA')
                        break

    for d_name in ['gold_coin', 'bronze_coin']:
        ddir = os.path.join(DROPS_DIR, d_name)
        if os.path.exists(ddir):
            for f in os.listdir(ddir):
                if f.endswith('.png'):
                    drops[d_name] = Image.open(os.path.join(ddir, f)).convert('RGBA')
                    break

    cw, ch = 1280, 720
    max_vx = max(0, mw - cw)
    max_vy = max(0, mh - ch)

    print(f"🎬 开始合成 {count} 组真实训练数据 (每张图怪物: {min_mobs}~{max_mobs} 只, 负样本比例: {negative_ratio*100:.0f}%)...")

    COLOR_MAP = {
        'player_left': (255, 120, 0),
        'player_right': (255, 180, 0),
        'hector': (0, 165, 255),
        'white_fang': (0, 255, 255)
    }

    neg_step = int(1.0 / negative_ratio) if negative_ratio > 0 else 999999

    for idx in range(count):
        is_neg = (idx % neg_step == (neg_step - 1))
        file_id = f"{map_name}_synth_{idx+1:03d}"

        # 智能玩家相机视角 (以实际物理平台为锚点，完全还原游戏内摄像机跟随玩家的视角)
        if not is_neg and platforms:
            focus_plat = random.choice(platforms)
            focus_x = (focus_plat['x1'] + focus_plat['x2']) // 2
            focus_y = focus_plat['y']
            vx = max(0, min(max_vx, focus_x - cw // 2 + random.randint(-180, 180)))
            vy = max(0, min(max_vy, focus_y - ch // 2 + random.randint(-80, 80)))
        else:
            vx = random.randint(0, max_vx)
            vy = random.randint(0, max_vy)

        crop_bg = full_map.crop((vx, vy, vx + cw, vy + ch))
        canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 255))
        canvas.paste(crop_bg, (0, 0))

        placed_boxes = []
        labels = []
        shapes = []

        # 收集当前视口内的可见平台 (要求至少留有 50px 的上下头部与底部安全空间)
        visible_plats = []
        for plat in platforms:
            rel_y = plat['y'] - vy
            if 50 <= rel_y <= ch - 15:
                ix1 = max(plat['x1'], vx + 15)
                ix2 = min(plat['x2'], vx + cw - 15)
                if ix2 - ix1 >= 50:
                    visible_plats.append({
                        'vx1': ix1 - vx,
                        'vx2': ix2 - vx,
                        'vy': rel_y,
                        'len': ix2 - ix1
                    })

        if not is_neg and visible_plats:
            # 1. 优先放置全部 3 个玩家角色 (player1, player2, player3 各一帧，每个玩家身旁必带一只金蘑菇宠物，且不打标签)
            if players_data:
                target_pids = list(players_data.keys())
                random.shuffle(target_pids)

                for pid in target_pids:
                    p_cls = random.choice(['player_left', 'player_right'])
                    p_pool = players_data[pid].get(p_cls, [])
                    if not p_pool:
                        continue
                    p_im = random.choice(p_pool).copy()
                    psp_w, psp_h = p_im.size

                    placed_this_player = False
                    shuffled_plats = list(visible_plats)
                    random.shuffle(shuffled_plats)

                    for tplat in shuffled_plats:
                        if tplat['vx2'] - tplat['vx1'] < psp_w + 6:
                            continue

                        for p_attempt in range(35):
                            pos_x = random.randint(tplat['vx1'], tplat['vx2'] - psp_w)
                            pos_y = tplat['vy'] - psp_h + 2
                            cand_box = (pos_x, pos_y, pos_x + psp_w, pos_y + psp_h, p_cls)
                            overlap = False
                            for pb in placed_boxes:
                                if abs(cand_box[3] - pb[3]) < 30:
                                    if not (cand_box[2] + 12 < pb[0] or cand_box[0] - 12 > pb[2]):
                                        overlap = True
                                        break
                                else:
                                    if not (cand_box[2] < pb[0] or cand_box[0] > pb[2] or cand_box[3] < pb[1] or cand_box[1] > pb[3]):
                                        overlap = True
                                        break

                            if not overlap:
                                canvas.paste(p_im, (pos_x, pos_y), p_im)
                                placed_boxes.append(cand_box)
                                norm_cx = ((pos_x + pos_x + psp_w) / 2.0) / cw
                                norm_cy = ((pos_y + pos_y + psp_h) / 2.0) / ch
                                labels.append((class_to_id[p_cls], norm_cx, norm_cy, psp_w / cw, psp_h / ch))
                                shapes.append({
                                    "label": p_cls,
                                    "points": extract_polygon(p_im, pos_x, pos_y),
                                    "group_id": None, "shape_type": "polygon", "flags": {}
                                })

                                # 每个玩家身旁必带一只金蘑菇宠物 (不打标签作为负样本)
                                if pet_sprites:
                                    pet_im = random.choice(pet_sprites).copy()
                                    if p_cls == 'player_right':
                                        pet_im = pet_im.transpose(Image.FLIP_LEFT_RIGHT)
                                    pet_w, pet_h = pet_im.size

                                    # 智能计算宠物贴地放置坐标 (优先贴在角色身旁，防掉落平台)
                                    pet_offset = -random.randint(18, 30) if random.random() < 0.5 else (psp_w + random.randint(4, 18))
                                    pet_x = pos_x + pet_offset
                                    if pet_x < tplat['vx1']:
                                        pet_x = pos_x + psp_w + random.randint(4, 18)
                                    elif pet_x + pet_w > tplat['vx2']:
                                        pet_x = pos_x - pet_w - random.randint(4, 18)

                                    pet_x = max(tplat['vx1'], min(tplat['vx2'] - pet_w, pet_x))
                                    pet_y = tplat['vy'] - pet_h + 2
                                    canvas.paste(pet_im, (pet_x, pet_y), pet_im)
                                    # 将宠物占位加入 placed_boxes 防怪物叠在宠物上 (标记为 distractor)
                                    placed_boxes.append((pet_x, pet_y, pet_x + pet_w, pet_y + pet_h, 'distractor'))

                                placed_this_player = True
                                break

                        if placed_this_player:
                            break

            # 2. 动态放置地图怪物 (保证落在用户要求的 min_mobs ~ max_mobs 区间)
            target_mob_count = random.randint(min_mobs, max_mobs)
            plat_weights = [max(10, p['len']) for p in visible_plats]

            # 高效高密度放置循环 (最多 200 次尝试直至达到目标怪物数)
            attempts = 0
            while len([b for b in placed_boxes if b[4] in valid_mobs]) < target_mob_count and attempts < 200:
                attempts += 1
                tplat = random.choices(visible_plats, weights=plat_weights, k=1)[0]
                m_cls = random.choice(valid_mobs)
                s_list = mobs_sprites.get(m_cls, [])
                if not s_list:
                    continue

                m_im = random.choice(s_list).copy()
                if random.random() < 0.5:
                    m_im = m_im.transpose(Image.FLIP_LEFT_RIGHT)

                msp_w, msp_h = m_im.size
                if tplat['vx2'] - tplat['vx1'] < msp_w + 4:
                    continue

                pos_x = random.randint(tplat['vx1'], tplat['vx2'] - msp_w)
                pos_y = tplat['vy'] - msp_h + 2
                cand_box = (pos_x, pos_y, pos_x + msp_w, pos_y + msp_h, m_cls)

                # 智能碰撞检测：同平台需水平留出 8px 间隙；不同高度平台（>30px）允许 X 轴重合（完全符合游戏分层站位）
                overlap = False
                for pb in placed_boxes:
                    if abs(cand_box[3] - pb[3]) < 30:
                        # 处于同层平台，水平留出微小间隙
                        if not (cand_box[2] + 8 < pb[0] or cand_box[0] - 8 > pb[2]):
                            overlap = True
                            break
                    else:
                        # 处于不同平台高度，只需包围盒不直接重叠
                        if not (cand_box[2] < pb[0] or cand_box[0] > pb[2] or cand_box[3] < pb[1] or cand_box[1] > pb[3]):
                            overlap = True
                            break

                if not overlap:
                    canvas.paste(m_im, (pos_x, pos_y), m_im)
                    placed_boxes.append(cand_box)
                    norm_cx = ((pos_x + pos_x + msp_w) / 2.0) / cw
                    norm_cy = ((pos_y + pos_y + msp_h) / 2.0) / ch
                    labels.append((class_to_id[m_cls], norm_cx, norm_cy, msp_w / cw, msp_h / ch))
                    shapes.append({
                        "label": m_cls,
                        "points": extract_polygon(m_im, pos_x, pos_y),
                        "group_id": None, "shape_type": "polygon", "flags": {}
                    })

                    # 30% 几率在脚边伴随掉落物
                    if random.random() < 0.30:
                        drop_info = MONSTER_TO_UNIQUE_DROP.get(m_cls)
                        if drop_info and drop_info[0] in drops:
                            d_im = drops[drop_info[0]]
                            canvas.paste(d_im, (pos_x + random.randint(0, max(1, msp_w - 15)), tplat['vy'] - d_im.height + 2), d_im)

        # 保存图片
        out_rgb = canvas.convert("RGB")
        img_path = os.path.join(target_dir, f"{file_id}.jpg")
        out_rgb.save(img_path, quality=95)

        # 保存 YOLO txt 标注
        with open(os.path.join(target_dir, f"{file_id}.txt"), "w", encoding="utf-8") as f:
            for lb in labels:
                f.write(f"{lb[0]} {lb[1]:.6f} {lb[2]:.6f} {lb[3]:.6f} {lb[4]:.6f}\n")

        # 保存 LabelMe json 标注
        with open(os.path.join(target_dir, f"{file_id}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "version": "5.0.1", "flags": {}, "shapes": shapes,
                "imagePath": f"{file_id}.jpg", "imageData": None,
                "imageHeight": ch, "imageWidth": cw
            }, f, indent=2)

        # 预览图保存 (Windows 兼容中文路径)
        prev_arr = cv2.cvtColor(np.array(out_rgb), cv2.COLOR_RGB2BGR)
        for pb in placed_boxes:
            bx1, by1, bx2, by2, b_cls = pb
            color = COLOR_MAP.get(b_cls, (0, 255, 0))
            cv2.rectangle(prev_arr, (bx1, by1), (bx2, by2), color, 2)
            cv2.putText(prev_arr, b_cls, (bx1 + 3, by1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if is_neg:
            cv2.putText(prev_arr, "[NEGATIVE SAMPLE - PURE MAP/SNOW ELEMENTS]", (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        prev_path = os.path.join(debug_dir, f"{file_id}_preview.jpg")
        cv2.imencode('.jpg', prev_arr)[1].tofile(prev_path)

    print(f"\n🎉 成功为【{map_name}】生成 {count} 组逼真高密度训练图像！")
    print(f"📁 训练数据路径: {target_dir}")
    print(f"📁 可视化验证图: {debug_dir}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MapleStory 真实地图吸附式高密度合成生成器")
    parser.add_argument('--map', type=str, default='结冰平原1', help='地图名称 (对应 map/*.xml)')
    parser.add_argument('--count', type=int, default=60, help='合成样本组数 (默认 60)')
    parser.add_argument('--min-mobs', type=int, default=6, help='视口内最少怪物数 (默认 6)')
    parser.add_argument('--max-mobs', type=int, default=12, help='视口内最多怪物数 (默认 12)')
    parser.add_argument('--negative-ratio', type=float, default=0.10, help='纯地貌抗干扰负样本比例 (默认 0.10)')
    parser.add_argument('--player-scale', type=float, default=1.0, help='玩家角色缩放比例 (默认 1.0 即 100% 原生像素)')
    args = parser.parse_args()

    build_realistic_dataset(args.map, count=args.count, min_mobs=args.min_mobs,
                           max_mobs=args.max_mobs, negative_ratio=args.negative_ratio,
                           player_scale=args.player_scale)

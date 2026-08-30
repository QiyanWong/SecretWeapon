import os
import sys
import glob
import json
import random
import datetime
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# 修复控制台 UTF-8 输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(BASE_DIR, "synthetic_assets")
BG_DIR = os.path.join(ASSET_DIR, "backgrounds")
MONSTER_DIR = os.path.join(ASSET_DIR, "monsters")
PLAYER_DIR = os.path.join(ASSET_DIR, "player")
DROPS_DIR = os.path.join(ASSET_DIR, "drops")
DISTRACTORS_DIR = os.path.join(ASSET_DIR, "distractors")

RAW_OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "raw_images")
DEBUG_OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "synthetic_debug")
os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

# 24 类别定义与映射 (排除 rope 和 portal，仅保留 3 种玩家姿态 + 21 种活体怪物)
CLASS_LIST = [
    'player_left',
    'player_right',
    'player_climb',
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
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_LIST)}

# 怪物专属战利品绑定映射 (同台伴生逻辑: 只有图里有该怪，才会在其身旁撒落该战利品)
MONSTER_TO_UNIQUE_DROP = {
    'orange_mushroom': 'orange_mushroom_cap',
    'red_snail':       'red_snail_shell',
    'slime':           'squishy_liquid',
    'bubbling':        'bubbling_bubble',
    'horny_mushroom':  'horny_mushroom_cap',
    'zombie_mushroom': 'charm_of_the_undead',
    'axe_stump':       'firewood',
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
    'crab':            'lorang_claw'
}

# 地图生态与怪物/掉落物主题绑定
BIOME_MAPPING = {
    '沼泽地': {
        'mobs': ['croco', 'jr_necki'],
        'drops': ['ligator_skin', 'jr_necki_skin'],
        'props': ['swamp_purple_flower', 'sabitrama_herb', 'herb_bunch']
    },
    '北部训练场': {
        'mobs': ['slime', 'horny_mushroom', 'zombie_mushroom', 'orange_mushroom', 'red_snail', 'pig'],
        'drops': ['squishy_liquid', 'horny_mushroom_cap', 'charm_of_the_undead', 'orange_mushroom_cap', 'red_snail_shell', 'pig_head'],
        'props': []
    },
    '地铁一号线': {
        'mobs': ['bubbling', 'jr_wraith'],
        'drops': ['bubbling_bubble', 'tablecloth'],
        'props': []
    },
    '勇士部落北部': {
        'mobs': ['axe_stump', 'wild_boar', 'pig', 'ribbon_pig', 'fire_boar'],
        'drops': ['firewood', 'wild_boar_tooth', 'pig_head', 'pig_ribbon', 'fire_boar_tooth'],
        'props': []
    },
    '森林迷宫': {
        'mobs': ['lupin', 'fire_boar', 'drake', 'evil_eye'],
        'drops': ['lupin_banana', 'fire_boar_tooth', 'drake_skull', 'evil_eye_tail'],
        'props': []
    },
    '石人寺院': {
        'mobs': ['wooden_mask', 'rocky_mask'],
        'drops': ['wooden_board', 'rocky_mask_doll'],
        'props': []
    },
    '遗迹': {
        'mobs': ['wooden_mask', 'rocky_mask'],
        'drops': ['wooden_board', 'rocky_mask_doll'],
        'props': []
    },
    '冰冷的洞穴': {
        'mobs': ['evil_eye', 'cold_eye', 'jr_wraith', 'drake'],
        'drops': ['evil_eye_tail', 'cold_eye_tail', 'tablecloth', 'drake_skull'],
        'props': []
    },
    '黄金海滩': {
        'mobs': ['crab', 'lupin', 'red_snail', 'slime'],
        'drops': ['lorang_claw', 'lupin_banana', 'red_snail_shell', 'squishy_liquid'],
        'props': []
    }
}


def load_all_monster_sprites():
    """
    加载 21 种怪物的全部 183 个独立活体帧，按类别归类并建立全覆盖抽取列表
    """
    all_sprites = {} # { class_name: [ {"image": Image, "name": ...}, ... ] }
    flat_sprite_deck = [] # [ {"class_name": ..., "image": ..., "name": ...} ]

    for cls in CLASS_LIST[3:]: # 排除前3个玩家类别
        folder = os.path.join(MONSTER_DIR, cls)
        if not os.path.exists(folder):
            continue
        all_sprites[cls] = []
        pngs = glob.glob(os.path.join(folder, "*.png"))
        for p in pngs:
            try:
                img = Image.open(p).convert("RGBA")
                item = {"class_name": cls, "image": img, "name": os.path.basename(p)}
                all_sprites[cls].append(item)
                flat_sprite_deck.append(item)
            except Exception:
                pass

    return all_sprites, flat_sprite_deck


def load_player_sprites():
    """加载玩家角色的三种状态素材"""
    player_sprites = {'player_left': [], 'player_right': [], 'player_climb': []}
    for p_cls in player_sprites.keys():
        folder = os.path.join(PLAYER_DIR, p_cls)
        if os.path.exists(folder):
            for p in glob.glob(os.path.join(folder, "*.png")):
                try:
                    img = Image.open(p).convert("RGBA")
                    player_sprites[p_cls].append(img)
                except Exception:
                    pass
    return player_sprites


def load_drop_and_distractor_sprites():
    """加载掉落物（金币、专属战利品）、宠物、植物素材"""
    drops = {} # { drop_key: Image }
    if os.path.exists(DROPS_DIR):
        for d in os.listdir(DROPS_DIR):
            sub = os.path.join(DROPS_DIR, d)
            if os.path.isdir(sub):
                pngs = glob.glob(os.path.join(sub, "*.png"))
                if pngs:
                    try:
                        drops[d] = Image.open(pngs[0]).convert("RGBA")
                    except Exception:
                        pass

    distractors = {}
    if os.path.exists(DISTRACTORS_DIR):
        for root, _, files in os.walk(DISTRACTORS_DIR):
            for f in files:
                if f.endswith(".png"):
                    try:
                        k = os.path.splitext(f)[0]
                        distractors[k] = Image.open(os.path.join(root, f)).convert("RGBA")
                    except Exception:
                        pass
                        
    return drops, distractors


def get_tight_bbox(sprite):
    """根据非透明像素 (Alpha > 10) 计算紧凑包围框"""
    alpha = np.array(sprite.getchannel('A'))
    non_zero = np.argwhere(alpha > 10)
    if non_zero.size == 0:
        return 0, 0, sprite.width, sprite.height
    y_min, x_min = non_zero.min(axis=0)
    y_max, x_max = non_zero.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def extract_polygon_contour(sprite_img, offset_x=0, offset_y=0, epsilon=0.8):
    """
    通过 RGBA 贴图的 Alpha 通道自动提取精细多边形轮廓点集 [[x, y], ...]
    """
    if sprite_img.mode != 'RGBA':
        sprite_img = sprite_img.convert('RGBA')
    alpha = np.array(sprite_img.getchannel('A'))
    mask = (alpha > 10).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    if not contours:
        w, h = sprite_img.size
        return [[offset_x, offset_y], [offset_x + w, offset_y], [offset_x + w, offset_y + h], [offset_x, offset_y + h]]
    c = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, epsilon=epsilon, closed=True)
    if len(approx) < 6 and len(c) >= 6:
        step = max(1, len(c) // 25)
        approx = c[::step]
    points = []
    for pt in approx:
        px = round(float(pt[0][0] + offset_x), 1)
        py = round(float(pt[0][1] + offset_y), 1)
        points.append([px, py])
    return points


def generate_dataset(num_images=100):
    """
    一键生成全覆盖、防幻觉的高质量合成数据集
    """
    bg_files = glob.glob(os.path.join(BG_DIR, "*.png"))
    if not bg_files:
        print(f"❌ 错误: 在 {BG_DIR} 中没有找到任何背景图片！")
        return

    monster_sprites_by_cls, flat_monster_deck = load_all_monster_sprites()
    player_sprites = load_player_sprites()
    drops_dict, distractors_dict = load_drop_and_distractor_sprites()

    print("=" * 75)
    print(f"🚀 开始生成高质量 YOLOv8 训练数据集 (目标生成: {num_images} 张)")
    print(f"   🏞️ 背景图数量: {len(bg_files)} 张")
    print(f"   👾 活体怪物总帧数: {len(flat_monster_deck)} 帧 (保证 100% 全覆盖出场)")
    print(f"   💰 掉落物/战利品总数: {len(drops_dict)} 种 (同台伴生，不标注)")
    print(f"   🐾 宠物/互动植物总数: {len(distractors_dict)} 种 (负样本，不标注)")
    print("=" * 75)

    # 1. 建立怪物全覆盖洗牌队列 (保证每一个形态至少出现一次以上)
    deck_queue = flat_monster_deck.copy()
    random.shuffle(deck_queue)

    generated_count = 0
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. 阶段 A: 为 9 种背景各生成 1 张【纯背景负样本图】(0 怪物 0 标注，仅含自然背景或散落金币/植物)
    print("\n[Phase 1] 正在生成 9 种不同地图的【纯背景负样本】(0 标注框，彻底抑制地图假阳性)...")
    for bg_idx, bg_path in enumerate(bg_files):
        bg_name = os.path.splitext(os.path.basename(bg_path))[0]
        bg_img = Image.open(bg_path).convert("RGBA")
        bw, bh = bg_img.size

        # 随机裁剪 1280x720 视口 (若原图较小则缩放或自适应)
        tw, th = 1280, 720
        if bw > tw and bh > th:
            rx = random.randint(0, bw - tw)
            ry = random.randint(0, bh - th)
            crop_bg = bg_img.crop((rx, ry, rx + tw, ry + th))
        else:
            crop_bg = bg_img.resize((tw, th), Image.Resampling.LANCZOS)

        # 随机在此纯背景图上撒落 2~4 个金币或该地图的专属植物 (无标签)
        canvas = crop_bg.copy()
        coin_keys = ['bronze_coin', 'gold_coin', 'meso_bills', 'meso_sack']
        for _ in range(random.randint(1, 4)):
            ck = random.choice(coin_keys)
            if ck in drops_dict:
                c_img = drops_dict[ck]
                cx = random.randint(50, tw - 100)
                cy = random.randint(int(th * 0.4), th - 80)
                canvas.paste(c_img, (cx, cy), c_img)

        # 若是沼泽地图，添加沼泽紫花 (无标签)
        if '沼泽' in bg_name and 'swamp_purple_flower' in distractors_dict:
            sf_img = distractors_dict['swamp_purple_flower']
            sx = random.randint(100, tw - 150)
            sy = random.randint(int(th * 0.5), th - 120)
            canvas.paste(sf_img, (sx, sy), sf_img)

        # 保存纯背景图片与 0KB 空标注文件 (同时输出 AnyLabeling JSON)
        out_name = f"synth_pure_bg_{now_str}_{bg_idx:02d}"
        rgb_img = canvas.convert("RGB")
        rgb_img.save(os.path.join(RAW_OUTPUT_DIR, f"{out_name}.jpg"), quality=95)
        with open(os.path.join(RAW_OUTPUT_DIR, f"{out_name}.txt"), 'w', encoding='utf-8') as f:
            pass # 纯背景空文件

        # 输出 AnyLabeling 兼容的空 JSON
        with open(os.path.join(RAW_OUTPUT_DIR, f"{out_name}.json"), 'w', encoding='utf-8') as f:
            json.dump({
                "version": "0.3.3",
                "flags": {},
                "shapes": [],
                "imagePath": f"{out_name}.jpg",
                "imageData": None,
                "imageHeight": th,
                "imageWidth": tw
            }, f, indent=2, ensure_ascii=False)

        generated_count += 1
        print(f"   ✓ [纯背景负样本] {out_name}.jpg (地图: {bg_name})")

    # 3. 阶段 B: 合成带怪物与玩家的训练图片 (保证 183 个怪物形态轮流全覆盖登场)
    print(f"\n[Phase 2] 正在合成怪物与玩家战斗场景图片 (目标补齐至 {num_images} 张)...")
    
    img_idx = 0
    while generated_count < num_images:
        bg_path = random.choice(bg_files)
        bg_name = os.path.splitext(os.path.basename(bg_path))[0]
        bg_img = Image.open(bg_path).convert("RGBA")
        bw, bh = bg_img.size

        tw, th = 1280, 720
        if bw > tw and bh > th:
            rx = random.randint(0, bw - tw)
            ry = random.randint(0, bh - th)
            canvas = bg_img.crop((rx, ry, rx + tw, ry + th))
        else:
            canvas = bg_img.resize((tw, th), Image.Resampling.LANCZOS)

        labels = [] # [ (cls_id, x_center, y_center, w, h) ]
        json_shapes = [] # AnyLabeling 多边形轮廓标注
        occupied_boxes = [] # 用于防过度重叠 [ (x1, y1, x2, y2) ]
        present_monster_classes = set()

        # 1. 决定本张图生成的怪物数量 (提升密集度: 3 ~ 7 只)
        num_mobs = random.randint(3, 7)
        
        # 提取怪物 (优先从全覆盖洗牌池提取，池空则重新洗牌循环)
        chosen_mob_items = []
        for _ in range(num_mobs):
            if not deck_queue:
                deck_queue = flat_monster_deck.copy()
                random.shuffle(deck_queue)
            chosen_mob_items.append(deck_queue.pop(0))

        # 放置怪物
        for m_item in chosen_mob_items:
            m_cls = m_item["class_name"]
            m_img = m_item["image"]
            present_monster_classes.add(m_cls)

            # 随机水平镜像翻转 (朝左/朝右)
            if random.random() < 0.5:
                m_img = m_img.transpose(Image.FLIP_LEFT_RIGHT)

            # 随机轻微尺寸缩放 (0.95 ~ 1.05)
            scale = random.uniform(0.95, 1.05)
            sw = max(10, int(m_img.width * scale))
            sh = max(10, int(m_img.height * scale))
            m_img_scaled = m_img.resize((sw, sh), Image.Resampling.LANCZOS)

            # 随机在地面/中下部寻找放置位置
            pos_x = random.randint(40, tw - sw - 40)
            pos_y = random.randint(int(th * 0.25), th - sh - 40)

            # 贴入画布
            canvas.paste(m_img_scaled, (pos_x, pos_y), m_img_scaled)

            # 计算精准 tight bounding box 并转换为 YOLO 归一化格式
            bx1, by1, bx2, by2 = get_tight_bbox(m_img_scaled)
            abs_x1 = pos_x + bx1
            abs_y1 = pos_y + by1
            abs_x2 = pos_x + bx2
            abs_y2 = pos_y + by2

            norm_xc = ((abs_x1 + abs_x2) / 2.0) / tw
            norm_yc = ((abs_y1 + abs_y2) / 2.0) / th
            norm_w = (abs_x2 - abs_x1) / tw
            norm_h = (abs_y2 - abs_y1) / th

            cls_id = CLASS_TO_ID.get(m_cls)
            if cls_id is not None:
                labels.append((cls_id, norm_xc, norm_yc, norm_w, norm_h))
                occupied_boxes.append((abs_x1, abs_y1, abs_x2, abs_y2))

                # 提取平滑多边形轮廓点集 (AnyLabeling 专属 polygon 格式)
                poly_pts = extract_polygon_contour(m_img_scaled, offset_x=pos_x, offset_y=pos_y)
                json_shapes.append({
                    "label": m_cls,
                    "points": poly_pts,
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {}
                })

            # 伴生掉落物逻辑: 在该怪物脚下/旁边概率生成它自己的独有战利品 (不标注)
            if random.random() < 0.45:
                drop_k = MONSTER_TO_UNIQUE_DROP.get(m_cls)
                if drop_k and drop_k in drops_dict:
                    d_img = drops_dict[drop_k]
                    dx = max(10, min(tw - 40, pos_x + random.randint(-25, sw + 10)))
                    dy = max(10, min(th - 40, pos_y + sh - random.randint(5, 20)))
                    canvas.paste(d_img, (dx, dy), d_img)

        # 2. 放置玩家角色 (85% 概率出现玩家)
        if random.random() < 0.85:
            p_cls = random.choice(['player_left', 'player_right', 'player_climb'])
            if player_sprites.get(p_cls):
                p_img = random.choice(player_sprites[p_cls])
                pw, ph = p_img.size
                px = random.randint(60, tw - pw - 60)
                py = random.randint(int(th * 0.3), th - ph - 40)
                canvas.paste(p_img, (px, py), p_img)

                # 标注玩家
                pbx1, pby1, pbx2, pby2 = get_tight_bbox(p_img)
                pabs_x1 = px + pbx1
                pabs_y1 = py + pby1
                pabs_x2 = px + pbx2
                pabs_y2 = py + pby2

                p_norm_xc = ((pabs_x1 + pabs_x2) / 2.0) / tw
                p_norm_yc = ((pabs_y1 + pabs_y2) / 2.0) / th
                p_norm_w = (pabs_x2 - pabs_x1) / tw
                p_norm_h = (pabs_y2 - pabs_y1) / th

                labels.append((CLASS_TO_ID[p_cls], p_norm_xc, p_norm_yc, p_norm_w, p_norm_h))

                # 提取玩家多边形轮廓点集 (AnyLabeling polygon)
                p_poly_pts = extract_polygon_contour(p_img, offset_x=px, offset_y=py)
                json_shapes.append({
                    "label": p_cls,
                    "points": p_poly_pts,
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {}
                })

                # 伴生宠物小白雪人 (35% 概率跟在玩家身边，不标注)
                if random.random() < 0.35 and 'stand0_0' in distractors_dict:
                    yeti_img = distractors_dict['stand0_0']
                    yx = max(10, min(tw - 40, px + (pw + 10 if p_cls == 'player_right' else -30)))
                    yy = py + ph - yeti_img.height
                    canvas.paste(yeti_img, (yx, yy), yeti_img)

        # 3. 散落通用金币货币 (不标注)
        coin_keys = ['bronze_coin', 'gold_coin', 'meso_bills', 'meso_sack']
        for _ in range(random.randint(1, 4)):
            ck = random.choice(coin_keys)
            if ck in drops_dict:
                c_img = drops_dict[ck]
                cx = random.randint(50, tw - 100)
                cy = random.randint(int(th * 0.4), th - 60)
                canvas.paste(c_img, (cx, cy), c_img)

        # 4. 图像微光影与色调扰动 (Domain Randomization)
        if random.random() < 0.5:
            enh_bright = ImageEnhance.Brightness(canvas)
            canvas = enh_bright.enhance(random.uniform(0.90, 1.10))
        if random.random() < 0.4:
            enh_contrast = ImageEnhance.Contrast(canvas)
            canvas = enh_contrast.enhance(random.uniform(0.90, 1.12))

        # 5. 保存生成的 JPG, YOLO TXT 标注与 AnyLabeling 多边形 JSON 标注
        out_basename = f"synth_{now_str}_{img_idx:03d}"
        rgb_img = canvas.convert("RGB")
        rgb_img.save(os.path.join(RAW_OUTPUT_DIR, f"{out_basename}.jpg"), quality=95)

        with open(os.path.join(RAW_OUTPUT_DIR, f"{out_basename}.txt"), 'w', encoding='utf-8') as f:
            for lbl in labels:
                f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

        with open(os.path.join(RAW_OUTPUT_DIR, f"{out_basename}.json"), 'w', encoding='utf-8') as f:
            json.dump({
                "version": "0.3.3",
                "flags": {},
                "shapes": json_shapes,
                "imagePath": f"{out_basename}.jpg",
                "imageData": None,
                "imageHeight": th,
                "imageWidth": tw
            }, f, indent=2, ensure_ascii=False)

        # 6. 生成前 15 张的可视化调试质检图 (绘制多边形轮廓)
        if img_idx < 15:
            dbg_cv = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
            for shape in json_shapes:
                s_pts = np.array(shape["points"], dtype=np.int32).reshape((-1, 1, 2))
                s_lbl = shape["label"]
                cv2.polylines(dbg_cv, [s_pts], isClosed=True, color=(0, 255, 0), thickness=2)
                top_left = s_pts.min(axis=0)[0]
                cv2.putText(dbg_cv, s_lbl, (top_left[0], max(15, top_left[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, f"debug_{out_basename}.jpg"), dbg_cv)

        generated_count += 1
        img_idx += 1

    print("\n" + "=" * 75)
    print(f"🎉 全部合成完成！共生成 {generated_count} 张图像 (已保存至 dataset/raw_images/)")
    print(f"🔍 质检预览图已输出至: {DEBUG_OUTPUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MapleStory Synthetic Dataset Generator")
    parser.add_argument("--num", type=int, default=160, help="Total number of images to synthesize")
    args = parser.parse_args()
    generate_dataset(num_images=args.num)

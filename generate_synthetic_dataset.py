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
PLAYER_DIR = os.path.join(ASSET_DIR, "player")
DROPS_DIR = os.path.join(ASSET_DIR, "drops")
DISTRACTORS_DIR = os.path.join(ASSET_DIR, "distractors")

RAW_OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "raw_images")
DEBUG_OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "synthetic_debug")
os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

# 25 类别定义 (2 玩家朝向 + 23 活体怪物)
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
    'dark_stone_golem'
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_LIST)}
MONSTER_CLASSES = CLASS_LIST[2:]

# 怪物专属战利品绑定映射 (同台伴生逻辑)
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


def load_all_monster_sprites():
    """加载 21 种怪物的全部独立活体帧，按种类归类"""
    all_sprites = {} # { class_name: [ {"image": Image, "name": ...}, ... ] }
    flat_sprite_deck = []

    for cls in MONSTER_CLASSES:
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
    """加载纯净透明的玩家角色形态素材 (player_left 与 player_right)"""
    player_sprites = {'player_left': [], 'player_right': []}
    for p_cls in player_sprites.keys():
        folder = os.path.join(PLAYER_DIR, p_cls)
        if os.path.exists(folder):
            for p in glob.glob(os.path.join(folder, "*.png")):
                try:
                    img = Image.open(p).convert("RGBA")
                    alpha = img.getchannel('A')
                    if alpha.getextrema()[0] < 250:
                        player_sprites[p_cls].append(img)
                except Exception:
                    pass
    return player_sprites


def load_drop_and_distractor_sprites():
    """加载掉落物、宠物与环境干扰物素材"""
    drops = {}
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
    """根据非透明像素 (Alpha > 10) 计算紧凑外接框"""
    alpha = np.array(sprite.getchannel('A'))
    non_zero = np.argwhere(alpha > 10)
    if non_zero.size == 0:
        return 0, 0, sprite.width, sprite.height
    y_min, x_min = non_zero.min(axis=0)
    y_max, x_max = non_zero.max(axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def extract_polygon_contour(sprite_img, offset_x=0, offset_y=0, epsilon=0.8):
    """
    通过 RGBA 贴图的 Alpha 通道提取亚像素级精细多边形轮廓点集 [[x, y], ...]
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


def check_box_collision(b1, b2, allow_same_species_overlap=False, margin=20):
    """
    检查两个目标框是否发生冲突。
    - 若 allow_same_species_overlap=True 且两者为同种怪: 允许 20%~50% 适度重叠;
    - 若 allow_same_species_overlap=False 或两者为不同种类/玩家: 严禁重叠并需保持安全间距 margin.
    b = (x1, y1, x2, y2, cls_name)
    """
    x1_a, y1_a, x2_a, y2_a, cls_a = b1
    x1_b, y1_b, x2_b, y2_b, cls_b = b2
    
    # 同种怪适度重叠模式
    if allow_same_species_overlap and cls_a == cls_b and cls_a not in ['player_left', 'player_right']:
        w_min = min(x2_a - x1_a, x2_b - x1_b)
        dist_x = abs((x1_a + x2_a) / 2 - (x1_b + x2_b) / 2)
        dist_y = abs((y1_a + y2_a) / 2 - (y1_b + y2_b) / 2)
        if dist_x < w_min * 0.20 and dist_y < 15:
            return True # 几乎完全重合，拒绝
        return False # 允许适度同种重叠

    # 常规模式: 严禁任何重叠
    if (x1_a - margin < x2_b and x2_a + margin > x1_b and
        y1_a - margin < y2_b and y2_a + margin > y1_b):
        return True
        
    return False


def save_sample(canvas, labels, json_shapes, out_basename, debug_idx=None):
    """保存 JPG 图像、TXT YOLO 标签、AnyLabeling 多边形 JSON 与可选的调试预览图"""
    tw, th = canvas.size
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

    if debug_idx is not None and debug_idx < 20:
        dbg_cv = cv2.cvtColor(np.array(rgb_img), cv2.COLOR_RGB2BGR)
        for shape in json_shapes:
            s_pts = np.array(shape["points"], dtype=np.int32).reshape((-1, 1, 2))
            s_lbl = shape["label"]
            cv2.polylines(dbg_cv, [s_pts], isClosed=True, color=(0, 255, 0), thickness=2)
            top_left = s_pts.min(axis=0)[0]
            cv2.putText(dbg_cv, s_lbl, (top_left[0], max(15, top_left[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, f"debug_{out_basename}.jpg"), dbg_cv)


def generate_dataset():
    """
    分阶段合成:
    阶段 1: 9 张纯背景负样本
    阶段 2: 清晰【无重叠】基础场景 (保证 183 个怪物活体帧每个至少出现 2 次在不同图片中) + 多玩家形态
    阶段 3: 专属【同种怪适度重叠】进阶场景 (保证 21 种怪物每种都有同种群聚重叠图片) + 多玩家形态
    """
    bg_files = glob.glob(os.path.join(BG_DIR, "*.png"))
    if not bg_files:
        print(f"❌ 错误: 在 {BG_DIR} 中没有找到任何背景图片！")
        return

    monster_sprites_by_cls, flat_monster_deck = load_all_monster_sprites()
    player_sprites = load_player_sprites()
    drops_dict, distractors_dict = load_drop_and_distractor_sprites()

    print("=" * 75)
    print(f"🚀 开始全新分阶段高质量数据集生成流水线 (23 类: 2 玩家 + 21 活体怪)")
    print(f"   🏞️ 背景地图: {len(bg_files)} 张")
    print(f"   👾 活体怪物总帧数: {len(flat_monster_deck)} 帧")
    print(f"   🤺 玩家形态: player_left ({len(player_sprites['player_left'])} 帧), player_right ({len(player_sprites['player_right'])} 帧)")
    print(f"   📋 策略: 阶段1[纯背景] -> 阶段2[清晰无重叠·双重全帧保底+多玩家] -> 阶段3[同种怪适度重叠+多玩家]")
    print("=" * 75)

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    total_generated = 0
    tw, th = 1280, 720

    # ================= 阶段 1: 9 张纯背景负样本 =================
    print("\n[Phase 1] 正在生成 9 张不同地图的【纯背景负样本】...")
    for bg_idx, bg_path in enumerate(bg_files):
        bg_name = os.path.splitext(os.path.basename(bg_path))[0]
        bg_img = Image.open(bg_path).convert("RGBA")
        bw, bh = bg_img.size

        if bw > tw and bh > th:
            rx = random.randint(0, bw - tw)
            ry = random.randint(0, bh - th)
            canvas = bg_img.crop((rx, ry, rx + tw, ry + th))
        else:
            canvas = bg_img.resize((tw, th), Image.Resampling.LANCZOS)

        # 散落金币/植物
        coin_keys = ['bronze_coin', 'gold_coin', 'meso_bills', 'meso_sack']
        for _ in range(random.randint(1, 4)):
            ck = random.choice(coin_keys)
            if ck in drops_dict:
                c_img = drops_dict[ck]
                cx = random.randint(50, tw - 100)
                cy = random.randint(int(th * 0.4), th - 80)
                canvas.paste(c_img, (cx, cy), c_img)

        if '沼泽' in bg_name and 'swamp_purple_flower' in distractors_dict:
            sf_img = distractors_dict['swamp_purple_flower']
            sx = random.randint(100, tw - 150)
            sy = random.randint(int(th * 0.5), th - 120)
            canvas.paste(sf_img, (sx, sy), sf_img)

        out_name = f"synth_pure_bg_{now_str}_{bg_idx:02d}"
        save_sample(canvas, [], [], out_name)
        total_generated += 1
        print(f"   ✓ [纯背景] {out_name}.jpg (地图: {bg_name})")

    # ================= 阶段 2: 纯清晰【无重叠】双重全帧覆盖 + 多玩家形态 =================
    print(f"\n[Phase 2] 正在生成【清晰无重叠】基础场景 (保证 183 帧全部至少出现 2 次于不同图片中)...")
    clean_coverage_deck = flat_monster_deck.copy() + flat_monster_deck.copy()
    random.shuffle(clean_coverage_deck)

    p2_img_idx = 0
    while clean_coverage_deck:
        bg_path = random.choice(bg_files)
        bg_img = Image.open(bg_path).convert("RGBA")
        bw, bh = bg_img.size

        if bw > tw and bh > th:
            rx = random.randint(0, bw - tw)
            ry = random.randint(0, bh - th)
            canvas = bg_img.crop((rx, ry, rx + tw, ry + th))
        else:
            canvas = bg_img.resize((tw, th), Image.Resampling.LANCZOS)

        labels = []
        json_shapes = []
        placed_boxes = [] # (x1, y1, x2, y2, cls_name)

        # 单张图抽取 3 ~ 5 个不同的动作帧
        mobs_to_place = []
        for _ in range(random.randint(3, 5)):
            if not clean_coverage_deck:
                break
            for i, cand in enumerate(clean_coverage_deck):
                if cand["name"] not in [m["name"] for m in mobs_to_place]:
                    mobs_to_place.append(clean_coverage_deck.pop(i))
                    break
            else:
                mobs_to_place.append(clean_coverage_deck.pop(0))

        # 无重叠放置怪物
        for m_item in mobs_to_place:
            m_cls = m_item["class_name"]
            m_img = m_item["image"]
            
            if random.random() < 0.5:
                m_img = m_img.transpose(Image.FLIP_LEFT_RIGHT)

            scale = random.uniform(0.95, 1.05)
            sw = max(10, int(m_img.width * scale))
            sh = max(10, int(m_img.height * scale))
            m_scaled = m_img.resize((sw, sh), Image.Resampling.LANCZOS)

            for attempt in range(40):
                pos_x = random.randint(30, tw - sw - 30)
                pos_y = random.randint(int(th * 0.22), th - sh - 30)

                bx1, by1, bx2, by2 = get_tight_bbox(m_scaled)
                cand_box = (pos_x + bx1, pos_y + by1, pos_x + bx2, pos_y + by2, m_cls)

                collision = any(check_box_collision(cand_box, pb, allow_same_species_overlap=False, margin=25) for pb in placed_boxes)
                if not collision:
                    canvas.paste(m_scaled, (pos_x, pos_y), m_scaled)
                    placed_boxes.append(cand_box)

                    abs_x1, abs_y1, abs_x2, abs_y2, _ = cand_box
                    labels.append((CLASS_TO_ID[m_cls], ((abs_x1 + abs_x2) / 2.0) / tw, ((abs_y1 + abs_y2) / 2.0) / th, (abs_x2 - abs_x1) / tw, (abs_y2 - abs_y1) / th))

                    poly_pts = extract_polygon_contour(m_scaled, offset_x=pos_x, offset_y=pos_y)
                    json_shapes.append({"label": m_cls, "points": poly_pts, "group_id": None, "description": "", "shape_type": "polygon", "flags": {}})

                    if random.random() < 0.40:
                        drop_k = MONSTER_TO_UNIQUE_DROP.get(m_cls)
                        if drop_k and drop_k in drops_dict:
                            d_img = drops_dict[drop_k]
                            dx = max(10, min(tw - 40, pos_x + random.randint(-15, sw + 5)))
                            dy = max(10, min(th - 40, pos_y + sh - random.randint(5, 20)))
                            canvas.paste(d_img, (dx, dy), d_img)
                    break

        # 放置玩家角色 (支持 1~2 个不同姿态形态，严格无重叠)
        num_players = random.choice([1, 1, 2, 2])
        for _ in range(num_players):
            p_cls = random.choice(['player_left', 'player_right'])
            if player_sprites.get(p_cls):
                p_img = random.choice(player_sprites[p_cls])
                pw, ph = p_img.size
                for attempt in range(35):
                    px = random.randint(50, tw - pw - 50)
                    py = random.randint(int(th * 0.28), th - ph - 30)
                    pbx1, pby1, pbx2, pby2 = get_tight_bbox(p_img)
                    p_box = (px + pbx1, py + pby1, px + pbx2, py + pby2, p_cls)

                    collision = any(check_box_collision(p_box, pb, allow_same_species_overlap=False, margin=25) for pb in placed_boxes)
                    if not collision:
                        canvas.paste(p_img, (px, py), p_img)
                        placed_boxes.append(p_box)

                        pabs_x1, pabs_y1, pabs_x2, pabs_y2, _ = p_box
                        labels.append((CLASS_TO_ID[p_cls], ((pabs_x1 + pabs_x2) / 2.0) / tw, ((pabs_y1 + pabs_y2) / 2.0) / th, (pabs_x2 - pabs_x1) / tw, (pabs_y2 - pabs_y1) / th))
                        p_poly_pts = extract_polygon_contour(p_img, offset_x=px, offset_y=py)
                        json_shapes.append({"label": p_cls, "points": p_poly_pts, "group_id": None, "description": "", "shape_type": "polygon", "flags": {}})

                        if random.random() < 0.30 and 'stand0_0' in distractors_dict:
                            yeti_img = distractors_dict['stand0_0']
                            yx = max(10, min(tw - 40, px + (pw + 10 if p_cls == 'player_right' else -30)))
                            yy = py + ph - yeti_img.height
                            canvas.paste(yeti_img, (yx, yy), yeti_img)
                        break

        # 散落通用金币
        for _ in range(random.randint(1, 3)):
            ck = random.choice(['bronze_coin', 'gold_coin', 'meso_bills', 'meso_sack'])
            if ck in drops_dict:
                canvas.paste(drops_dict[ck], (random.randint(50, tw - 100), random.randint(int(th * 0.4), th - 60)), drops_dict[ck])

        # 微调光影
        if random.random() < 0.5:
            canvas = ImageEnhance.Brightness(canvas).enhance(random.uniform(0.92, 1.08))
        if random.random() < 0.4:
            canvas = ImageEnhance.Contrast(canvas).enhance(random.uniform(0.92, 1.08))

        out_name = f"synth_clean_{now_str}_{p2_img_idx:03d}"
        save_sample(canvas, labels, json_shapes, out_name, debug_idx=p2_img_idx)
        total_generated += 1
        p2_img_idx += 1

    print(f"   ✓ 阶段 2 清晰无重叠场景生成完成，共 {p2_img_idx} 张图！")

    # ================= 阶段 3: 专属【同种怪适度重叠】进阶场景 + 多玩家形态 =================
    print(f"\n[Phase 3] 正在生成【同种怪适度重叠】进阶场景 (为 21 种怪逐一生成 2 张重叠图)...")
    overlap_species_deck = MONSTER_CLASSES.copy() + MONSTER_CLASSES.copy() # 42 张
    random.shuffle(overlap_species_deck)

    p3_img_idx = 0
    while overlap_species_deck:
        primary_sp = overlap_species_deck.pop(0)
        bg_path = random.choice(bg_files)
        bg_img = Image.open(bg_path).convert("RGBA")
        bw, bh = bg_img.size

        if bw > tw and bh > th:
            rx = random.randint(0, bw - tw)
            ry = random.randint(0, bh - th)
            canvas = bg_img.crop((rx, ry, rx + tw, ry + th))
        else:
            canvas = bg_img.resize((tw, th), Image.Resampling.LANCZOS)

        labels = []
        json_shapes = []
        placed_boxes = []

        available_frames = monster_sprites_by_cls.get(primary_sp, [])
        if available_frames:
            inst_count = min(len(available_frames), random.choice([2, 2, 3]))
            if len(available_frames) >= inst_count:
                chosen_frames = random.sample(available_frames, k=inst_count)
            else:
                chosen_frames = [random.choice(available_frames) for _ in range(inst_count)]

            base_x = random.randint(80, tw - 260)
            base_y = random.randint(int(th * 0.25), th - 150)

            for f_idx, f_item in enumerate(chosen_frames):
                m_img = f_item["image"]
                if random.random() < 0.5:
                    m_img = m_img.transpose(Image.FLIP_LEFT_RIGHT)
                scale = random.uniform(0.95, 1.05)
                sw = max(10, int(m_img.width * scale))
                sh = max(10, int(m_img.height * scale))
                m_scaled = m_img.resize((sw, sh), Image.Resampling.LANCZOS)

                for attempt in range(25):
                    if f_idx == 0:
                        pos_x = base_x
                        pos_y = base_y
                    else:
                        pos_x = base_x + int(sw * random.uniform(0.25, 0.45) * random.choice([-1, 1]))
                        pos_y = base_y + random.randint(-8, 8)

                    bx1, by1, bx2, by2 = get_tight_bbox(m_scaled)
                    cand_box = (pos_x + bx1, pos_y + by1, pos_x + bx2, pos_y + by2, primary_sp)

                    collision = any(check_box_collision(cand_box, pb, allow_same_species_overlap=True, margin=20) for pb in placed_boxes)
                    if not collision:
                        canvas.paste(m_scaled, (pos_x, pos_y), m_scaled)
                        placed_boxes.append(cand_box)

                        abs_x1, abs_y1, abs_x2, abs_y2, _ = cand_box
                        labels.append((CLASS_TO_ID[primary_sp], ((abs_x1 + abs_x2) / 2.0) / tw, ((abs_y1 + abs_y2) / 2.0) / th, (abs_x2 - abs_x1) / tw, (abs_y2 - abs_y1) / th))
                        poly_pts = extract_polygon_contour(m_scaled, offset_x=pos_x, offset_y=pos_y)
                        json_shapes.append({"label": primary_sp, "points": poly_pts, "group_id": None, "description": "", "shape_type": "polygon", "flags": {}})

                        if random.random() < 0.40:
                            drop_k = MONSTER_TO_UNIQUE_DROP.get(primary_sp)
                            if drop_k and drop_k in drops_dict:
                                d_img = drops_dict[drop_k]
                                canvas.paste(d_img, (max(10, min(tw - 40, pos_x + random.randint(-10, sw))), max(10, min(th - 40, pos_y + sh - 10))), d_img)
                        break

        # 额外添加 1 只异种散落怪
        other_species = [s for s in MONSTER_CLASSES if s != primary_sp]
        extra_sp = random.choice(other_species)
        if monster_sprites_by_cls.get(extra_sp):
            ex_item = random.choice(monster_sprites_by_cls[extra_sp])
            ex_img = ex_item["image"]
            ex_sw = ex_img.width
            ex_sh = ex_img.height
            for attempt in range(25):
                ex_x = random.randint(40, tw - ex_sw - 40)
                ex_y = random.randint(int(th * 0.25), th - ex_sh - 40)
                bx1, by1, bx2, by2 = get_tight_bbox(ex_img)
                cand_box = (ex_x + bx1, ex_y + by1, ex_x + bx2, ex_y + by2, extra_sp)
                if not any(check_box_collision(cand_box, pb, allow_same_species_overlap=False, margin=25) for pb in placed_boxes):
                    canvas.paste(ex_img, (ex_x, ex_y), ex_img)
                    placed_boxes.append(cand_box)
                    abs_x1, abs_y1, abs_x2, abs_y2, _ = cand_box
                    labels.append((CLASS_TO_ID[extra_sp], ((abs_x1 + abs_x2) / 2.0) / tw, ((abs_y1 + abs_y2) / 2.0) / th, (abs_x2 - abs_x1) / tw, (abs_y2 - abs_y1) / th))
                    poly_pts = extract_polygon_contour(ex_img, offset_x=ex_x, offset_y=ex_y)
                    json_shapes.append({"label": extra_sp, "points": poly_pts, "group_id": None, "description": "", "shape_type": "polygon", "flags": {}})
                    break

        # 放置玩家 (1~2 个)
        num_players = random.choice([1, 1, 2])
        for _ in range(num_players):
            p_cls = random.choice(['player_left', 'player_right'])
            if player_sprites.get(p_cls):
                p_img = random.choice(player_sprites[p_cls])
                pw, ph = p_img.size
                for attempt in range(30):
                    px = random.randint(50, tw - pw - 50)
                    py = random.randint(int(th * 0.28), th - ph - 30)
                    pbx1, pby1, pbx2, pby2 = get_tight_bbox(p_img)
                    p_box = (px + pbx1, py + pby1, px + pbx2, py + pby2, p_cls)

                    if not any(check_box_collision(p_box, pb, allow_same_species_overlap=False, margin=25) for pb in placed_boxes):
                        canvas.paste(p_img, (px, py), p_img)
                        placed_boxes.append(p_box)
                        pabs_x1, pabs_y1, pabs_x2, pabs_y2, _ = p_box
                        labels.append((CLASS_TO_ID[p_cls], ((pabs_x1 + pabs_x2) / 2.0) / tw, ((pabs_y1 + pabs_y2) / 2.0) / th, (pabs_x2 - pabs_x1) / tw, (pabs_y2 - pabs_y1) / th))
                        p_poly_pts = extract_polygon_contour(p_img, offset_x=px, offset_y=py)
                        json_shapes.append({"label": p_cls, "points": p_poly_pts, "group_id": None, "description": "", "shape_type": "polygon", "flags": {}})
                        break

        # 散落通用金币
        for _ in range(random.randint(1, 3)):
            ck = random.choice(['bronze_coin', 'gold_coin', 'meso_bills', 'meso_sack'])
            if ck in drops_dict:
                canvas.paste(drops_dict[ck], (random.randint(50, tw - 100), random.randint(int(th * 0.4), th - 60)), drops_dict[ck])

        out_name = f"synth_overlap_{now_str}_{p3_img_idx:03d}"
        save_sample(canvas, labels, json_shapes, out_name, debug_idx=p2_img_idx + p3_img_idx)
        total_generated += 1
        p3_img_idx += 1

    print(f"   ✓ 阶段 3 同种怪重叠场景生成完成，共 {p3_img_idx} 张图！")

    print("\n" + "=" * 75)
    print(f"🎉 全部合成大功告成！总计生成 {total_generated} 张图像 (23 类别对齐):")
    print(f"   - 阶段 1: 9 张纯背景负样本")
    print(f"   - 阶段 2: {p2_img_idx} 张清晰【无重叠】全帧双重覆盖 + 多玩家场景")
    print(f"   - 阶段 3: {p3_img_idx} 张【同种怪适度重叠】+ 多玩家场景")
    print(f"📁 数据集已保存至: {RAW_OUTPUT_DIR}")
    print(f"🔍 质检图已保存至: {DEBUG_OUTPUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    generate_dataset()

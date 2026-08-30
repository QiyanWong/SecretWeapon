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

# 24 类别定义 (3 玩家姿态 + 21 活体怪物)
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
MONSTER_CLASSES = CLASS_LIST[3:]

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
    """加载 21 种怪物的全部独立活体帧，按种类字典归类"""
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
    """加载纯净透明的玩家角色形态素材"""
    player_sprites = {'player_left': [], 'player_right': [], 'player_climb': []}
    for p_cls in player_sprites.keys():
        folder = os.path.join(PLAYER_DIR, p_cls)
        if os.path.exists(folder):
            for p in glob.glob(os.path.join(folder, "*.png")):
                try:
                    img = Image.open(p).convert("RGBA")
                    # 确保是带有效透明通道的贴图
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


def check_box_collision(b1, b2, margin=15):
    """
    检查两个目标框是否发生非法重叠。
    - 同种怪物之间: 允许 20%~60% 重叠 (返回 False / 无冲突);
    - 异种怪物或玩家之间: 严禁重叠并需保持安全间距 (返回 True / 产生冲突).
    b = (x1, y1, x2, y2, cls_name)
    """
    x1_a, y1_a, x2_a, y2_a, cls_a = b1
    x1_b, y1_b, x2_b, y2_b, cls_b = b2
    
    # 同种怪物重叠规则
    if cls_a == cls_b and cls_a not in ['player_left', 'player_right', 'player_climb']:
        # 避免两只怪 100% 几乎完全重叠，要求中心水平或垂直有一定微距
        w_min = min(x2_a - x1_a, x2_b - x1_b)
        dist_x = abs((x1_a + x2_a) / 2 - (x1_b + x2_b) / 2)
        dist_y = abs((y1_a + y2_a) / 2 - (y1_b + y2_b) / 2)
        if dist_x < w_min * 0.15 and dist_y < 15:
            return True # 几乎 100% 完全重合，拒绝
        return False # 合法的同种怪部分重叠

    # 异种怪物之间或与玩家之间: 严禁任何重叠
    if (x1_a - margin < x2_b and x2_a + margin > x1_b and
        y1_a - margin < y2_b and y2_a + margin > y1_b):
        return True # 异种冲突
        
    return False


def generate_dataset(num_images=160):
    """
    生成支持同种怪多形态重叠、全类别保底重叠的高质量合成数据集
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
    print(f"   👾 活体怪物总帧数: {len(flat_monster_deck)} 帧")
    print(f"   🧩 同种怪重叠规则: 仅允许同种怪物多形态重叠 (21 种怪 100% 保底重叠)")
    print(f"   💰 掉落物/战利品: {len(drops_dict)} 种 (同台伴生，不标注)")
    print("=" * 75)

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    generated_count = 0

    # 1. 阶段 A: 为 9 种背景各生成 1 张【纯背景负样本图】(0 标注框)
    print("\n[Phase 1] 正在生成 9 种不同地图的【纯背景负样本】(0 标注框)...")
    for bg_idx, bg_path in enumerate(bg_files):
        bg_name = os.path.splitext(os.path.basename(bg_path))[0]
        bg_img = Image.open(bg_path).convert("RGBA")
        bw, bh = bg_img.size

        tw, th = 1280, 720
        if bw > tw and bh > th:
            rx = random.randint(0, bw - tw)
            ry = random.randint(0, bh - th)
            crop_bg = bg_img.crop((rx, ry, rx + tw, ry + th))
        else:
            crop_bg = bg_img.resize((tw, th), Image.Resampling.LANCZOS)

        canvas = crop_bg.copy()
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
        rgb_img = canvas.convert("RGB")
        rgb_img.save(os.path.join(RAW_OUTPUT_DIR, f"{out_name}.jpg"), quality=95)
        with open(os.path.join(RAW_OUTPUT_DIR, f"{out_name}.txt"), 'w', encoding='utf-8') as f:
            pass
        with open(os.path.join(RAW_OUTPUT_DIR, f"{out_name}.json"), 'w', encoding='utf-8') as f:
            json.dump({"version": "0.3.3", "flags": {}, "shapes": [], "imagePath": f"{out_name}.jpg", "imageData": None, "imageHeight": th, "imageWidth": tw}, f, indent=2, ensure_ascii=False)

        generated_count += 1
        print(f"   ✓ [纯背景负样本] {out_name}.jpg (地图: {bg_name})")

    # 2. 阶段 B: 保底覆盖 21 种怪物的【专属同种重叠场景】(每种怪物至少 2 张同种重叠图)
    print("\n[Phase 2] 正在为 21 种怪物逐一生成【专属同种怪多形态重叠图】(保证 100% 出现同种重叠)...")
    guaranteed_species_queue = []
    for cls in MONSTER_CLASSES:
        guaranteed_species_queue.append(cls)
        guaranteed_species_queue.append(cls) # 每种怪安排 2 张重叠图

    img_idx = 0
    while generated_count < num_images:
        bg_path = random.choice(bg_files)
        bg_img = Image.open(bg_path).convert("RGBA")
        bw, bh = bg_img.size

        tw, th = 1280, 720
        if bw > tw and bh > th:
            rx = random.randint(0, bw - tw)
            ry = random.randint(0, bh - th)
            canvas = bg_img.crop((rx, ry, rx + tw, ry + th))
        else:
            canvas = bg_img.resize((tw, th), Image.Resampling.LANCZOS)

        labels = []
        json_shapes = []
        placed_boxes = [] # [ (x1, y1, x2, y2, cls_name) ]

        # 决定本张图的怪物种类分布
        if guaranteed_species_queue:
            # 优先消耗保底队列中的怪物种属
            primary_species = guaranteed_species_queue.pop(0)
            other_species = [s for s in MONSTER_CLASSES if s != primary_species]
            chosen_species_list = [primary_species]
            # 额外随机引入 1 种异种怪
            if random.random() < 0.6:
                chosen_species_list.append(random.choice(other_species))
            force_overlap_primary = True
        else:
            # 正常成组刷新: 随机选取 1 ~ 3 种怪物
            num_species = random.choice([1, 2, 2, 3])
            chosen_species_list = random.sample(MONSTER_CLASSES, k=num_species)
            force_overlap_primary = (random.random() < 0.7) # 70% 概率触发同种重叠

        # 遍历选中的每一种怪，为其生成 2 ~ 4 个不同形态实例
        for sp_idx, sp_cls in enumerate(chosen_species_list):
            available_frames = monster_sprites_by_cls.get(sp_cls, [])
            if not available_frames:
                continue

            # 本种怪生成的实例数 (若是首要怪物且要求重叠，则生成 2~4 只)
            if sp_idx == 0 and force_overlap_primary:
                inst_count = min(len(available_frames), random.randint(2, 4))
            else:
                inst_count = min(len(available_frames), random.randint(1, 3))

            # 随机挑选本种怪的 distinct 动作形态 (例如 stand, hit1, move)
            if len(available_frames) >= inst_count:
                chosen_frames = random.sample(available_frames, k=inst_count)
            else:
                chosen_frames = [random.choice(available_frames) for _ in range(inst_count)]

            # 确定该种群落的基准锚点 (Anchor Base Position)
            cluster_base_x = random.randint(60, tw - 250)
            cluster_base_y = random.randint(int(th * 0.25), th - 150)

            for f_idx, f_item in enumerate(chosen_frames):
                m_img = f_item["image"]
                
                # 随机镜像朝向
                if random.random() < 0.5:
                    m_img = m_img.transpose(Image.FLIP_LEFT_RIGHT)

                # 随机轻微缩放 (0.95 ~ 1.05)
                scale = random.uniform(0.95, 1.05)
                sw = max(10, int(m_img.width * scale))
                sh = max(10, int(m_img.height * scale))
                m_scaled = m_img.resize((sw, sh), Image.Resampling.LANCZOS)

                # 计算放置坐标: 同种怪第二只及之后，以 30%~55% 重叠偏移放置在群落周围
                placed_success = False
                for attempt in range(25):
                    if f_idx == 0:
                        pos_x = max(20, min(tw - sw - 20, cluster_base_x + random.randint(-20, 20)))
                        pos_y = max(int(th * 0.2), min(th - sh - 20, cluster_base_y + random.randint(-15, 15)))
                    else:
                        # 产生同种重叠: 紧贴上一只怪物水平偏移 (25% ~ 55% 宽度)
                        overlap_offset_x = int(sw * random.uniform(0.28, 0.58) * random.choice([-1, 1]))
                        overlap_offset_y = random.randint(-10, 10)
                        pos_x = max(20, min(tw - sw - 20, cluster_base_x + overlap_offset_x))
                        pos_y = max(int(th * 0.2), min(th - sh - 20, cluster_base_y + overlap_offset_y))

                    # 计算紧凑外接框
                    bx1, by1, bx2, by2 = get_tight_bbox(m_scaled)
                    cand_box = (pos_x + bx1, pos_y + by1, pos_x + bx2, pos_y + by2, sp_cls)

                    # 碰撞检测: 与已放置目标比较 (同种怪允许重叠，异种严禁重叠)
                    collision = False
                    for pb in placed_boxes:
                        if check_box_collision(cand_box, pb):
                            collision = True
                            break

                    if not collision:
                        # 成功放置
                        canvas.paste(m_scaled, (pos_x, pos_y), m_scaled)
                        placed_boxes.append(cand_box)

                        abs_x1, abs_y1, abs_x2, abs_y2, _ = cand_box
                        norm_xc = ((abs_x1 + abs_x2) / 2.0) / tw
                        norm_yc = ((abs_y1 + abs_y2) / 2.0) / th
                        norm_w = (abs_x2 - abs_x1) / tw
                        norm_h = (abs_y2 - abs_y1) / th

                        labels.append((CLASS_TO_ID[sp_cls], norm_xc, norm_yc, norm_w, norm_h))

                        # 提取精细多边形轮廓
                        poly_pts = extract_polygon_contour(m_scaled, offset_x=pos_x, offset_y=pos_y)
                        json_shapes.append({
                            "label": sp_cls,
                            "points": poly_pts,
                            "group_id": None,
                            "description": "",
                            "shape_type": "polygon",
                            "flags": {}
                        })

                        # 伴生战利品 (40% 概率)
                        if random.random() < 0.40:
                            drop_k = MONSTER_TO_UNIQUE_DROP.get(sp_cls)
                            if drop_k and drop_k in drops_dict:
                                d_img = drops_dict[drop_k]
                                dx = max(10, min(tw - 40, pos_x + random.randint(-15, sw + 5)))
                                dy = max(10, min(th - 40, pos_y + sh - random.randint(5, 20)))
                                canvas.paste(d_img, (dx, dy), d_img)

                        placed_success = True
                        break

        # 3. 放置玩家角色 (85% 概率出现，严禁与任何怪物重叠)
        if random.random() < 0.85:
            p_cls = random.choice(['player_left', 'player_right', 'player_climb'])
            if player_sprites.get(p_cls):
                p_img = random.choice(player_sprites[p_cls])
                pw, ph = p_img.size
                
                for attempt in range(25):
                    px = random.randint(50, tw - pw - 50)
                    py = random.randint(int(th * 0.3), th - ph - 30)

                    pbx1, pby1, pbx2, pby2 = get_tight_bbox(p_img)
                    p_box = (px + pbx1, py + pby1, px + pbx2, py + pby2, p_cls)

                    collision = False
                    for pb in placed_boxes:
                        if check_box_collision(p_box, pb):
                            collision = True
                            break

                    if not collision:
                        canvas.paste(p_img, (px, py), p_img)
                        placed_boxes.append(p_box)

                        pabs_x1, pabs_y1, pabs_x2, pabs_y2, _ = p_box
                        p_norm_xc = ((pabs_x1 + pabs_x2) / 2.0) / tw
                        p_norm_yc = ((pabs_y1 + pabs_y2) / 2.0) / th
                        p_norm_w = (pabs_x2 - pabs_x1) / tw
                        p_norm_h = (pabs_y2 - pabs_y1) / th

                        labels.append((CLASS_TO_ID[p_cls], p_norm_xc, p_norm_yc, p_norm_w, p_norm_h))

                        p_poly_pts = extract_polygon_contour(p_img, offset_x=px, offset_y=py)
                        json_shapes.append({
                            "label": p_cls,
                            "points": p_poly_pts,
                            "group_id": None,
                            "description": "",
                            "shape_type": "polygon",
                            "flags": {}
                        })

                        # 伴生宠物雪人
                        if random.random() < 0.35 and 'stand0_0' in distractors_dict:
                            yeti_img = distractors_dict['stand0_0']
                            yx = max(10, min(tw - 40, px + (pw + 10 if p_cls == 'player_right' else -30)))
                            yy = py + ph - yeti_img.height
                            canvas.paste(yeti_img, (yx, yy), yeti_img)
                        break

        # 4. 散落通用金币货币 (不标注)
        coin_keys = ['bronze_coin', 'gold_coin', 'meso_bills', 'meso_sack']
        for _ in range(random.randint(1, 4)):
            ck = random.choice(coin_keys)
            if ck in drops_dict:
                c_img = drops_dict[ck]
                cx = random.randint(50, tw - 100)
                cy = random.randint(int(th * 0.4), th - 60)
                canvas.paste(c_img, (cx, cy), c_img)

        # 5. 图像微光影扰动
        if random.random() < 0.5:
            canvas = ImageEnhance.Brightness(canvas).enhance(random.uniform(0.90, 1.10))
        if random.random() < 0.4:
            canvas = ImageEnhance.Contrast(canvas).enhance(random.uniform(0.90, 1.12))

        # 6. 保存 JPG, TXT 与 AnyLabeling 多边形 JSON
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

        # 7. 导出前 15 张质检调试图
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

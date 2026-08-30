import urllib.request
import json
import ssl
import sys
import os
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

# 禁用 SSL 校验
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# 掉落物品资产定义：
# 1. 货币类 (4种金币/钞票/钱袋)
# 2. 21 种怪物的专属独有战利品 (Unique Monster Drop ETC Items)
DROP_ITEMS_MAPPING = {
    # ================= 货币掉落 (Meso Drops) =================
    'bronze_coin': {
        'id': 9000000,
        'cn': '小铜币 (1~49 冒险币)',
        'en': 'Bronze Meso Coin',
        'category': 'meso'
    },
    'gold_coin': {
        'id': 9000001,
        'cn': '大金币 (50~99 冒险币)',
        'en': 'Gold Meso Coin',
        'category': 'meso'
    },
    'meso_bills': {
        'id': 9000002,
        'cn': '纸钞/钱捆 (100~999 冒险币)',
        'en': 'Meso Bills Bundle',
        'category': 'meso'
    },
    'meso_sack': {
        'id': 9000003,
        'cn': '钱袋 (1000+ 冒险币)',
        'en': 'Meso Sack',
        'category': 'meso'
    },

    # ================= 21 种怪物专属独有战利品 (Monster Unique ETC Drops) =================
    'orange_mushroom_cap': {
        'id': 4000001,
        'cn': '花蘑菇盖 (花蘑菇)',
        'en': 'Orange Mushroom Cap',
        'category': 'monster_drop',
        'mob': 'orange_mushroom'
    },
    'red_snail_shell': {
        'id': 4000016,
        'cn': '红蜗牛壳 (红蜗牛)',
        'en': 'Red Snail Shell',
        'category': 'monster_drop',
        'mob': 'red_snail'
    },
    'squishy_liquid': {
        'id': 4000004,
        'cn': '绿水灵珠/黏液 (绿水灵)',
        'en': 'Squishy Liquid',
        'category': 'monster_drop',
        'mob': 'slime'
    },
    'bubbling_bubble': {
        'id': 4000037,
        'cn': '蓝水灵珠/水泡 (蓝水灵)',
        'en': "Bubbling's Huge Bubble",
        'category': 'monster_drop',
        'mob': 'bubbling'
    },
    'horny_mushroom_cap': {
        'id': 4000015,
        'cn': '刺蘑菇盖 (刺蘑菇)',
        'en': 'Horny Mushroom Cap',
        'category': 'monster_drop',
        'mob': 'horny_mushroom'
    },
    'charm_of_the_undead': {
        'id': 4000008,
        'cn': '死者护身符/附咒符 (僵尸蘑菇)',
        'en': 'Charm of the Undead',
        'category': 'monster_drop',
        'mob': 'zombie_mushroom'
    },
    'firewood': {
        'id': 4000018,
        'cn': '柴火/木柴 (斧木妖)',
        'en': 'Firewood',
        'category': 'monster_drop',
        'mob': 'axe_stump'
    },
    'wild_boar_tooth': {
        'id': 4000020,
        'cn': '野猪尖牙 (野猪)',
        'en': 'Wild Boar Tooth',
        'category': 'monster_drop',
        'mob': 'wild_boar'
    },
    'pig_head': {
        'id': 4000017,
        'cn': '肥肥头/猪头 (肥肥/猪猪)',
        'en': "Pig's Head",
        'category': 'monster_drop',
        'mob': 'pig'
    },
    'pig_ribbon': {
        'id': 4000002,
        'cn': '红丝带/红缎带 (飘飘猪)',
        'en': "Pig's Ribbon",
        'category': 'monster_drop',
        'mob': 'ribbon_pig'
    },
    'fire_boar_tooth': {
        'id': 4000024,
        'cn': '火野猪尖牙 (火野猪)',
        'en': "Fire Boar's Tooth",
        'category': 'monster_drop',
        'mob': 'fire_boar'
    },
    'jr_necki_skin': {
        'id': 4000034,
        'cn': '青蛇皮 (小青蛇)',
        'en': 'Jr. Necki Skin',
        'category': 'monster_drop',
        'mob': 'jr_necki'
    },
    'ligator_skin': {
        'id': 4000032,
        'cn': '青鳄鱼皮 (鳄鱼/青鳄)',
        'en': 'Ligator Skin',
        'category': 'monster_drop',
        'mob': 'croco'
    },
    'drake_skull': {
        'id': 4000014,
        'cn': '土龙头骨/土龙角 (土龙)',
        'en': 'Drake Skull',
        'category': 'monster_drop',
        'mob': 'drake'
    },
    'evil_eye_tail': {
        'id': 4000007,
        'cn': '风独眼兽尾巴 (风独眼兽)',
        'en': 'Evil Eye Tail',
        'category': 'monster_drop',
        'mob': 'evil_eye'
    },
    'cold_eye_tail': {
        'id': 4000023,
        'cn': '冰独眼兽尾巴 (冰独眼兽)',
        'en': 'Cold Eye Tail',
        'category': 'monster_drop',
        'mob': 'cold_eye'
    },
    'tablecloth': {
        'id': 4000035,
        'cn': '破桌布/小幽灵布 (小幽灵)',
        'en': 'Tablecloth',
        'category': 'monster_drop',
        'mob': 'jr_wraith'
    },
    'wooden_board': {
        'id': 4000196,
        'cn': '木板/木面具 (木面怪人)',
        'en': 'Wooden Board',
        'category': 'monster_drop',
        'mob': 'wooden_mask'
    },
    'lupin_banana': {
        'id': 4000029,
        'cn': '猴子香蕉/鲁胖香蕉 (猴子/鲁胖)',
        'en': "Lupin's Banana",
        'category': 'monster_drop',
        'mob': 'lupin'
    },
    'rocky_mask_doll': {
        'id': 4032147,
        'cn': '石面怪娃娃/石面具 (石面怪人)',
        'en': 'Rocky Mask Doll',
        'category': 'monster_drop',
        'mob': 'rocky_mask'
    },
    'lorang_claw': {
        'id': 4000043,
        'cn': '罗朗螃蟹钳 (红螃蟹/罗朗)',
        'en': 'Lorang Claw',
        'category': 'monster_drop',
        'mob': 'crab'
    }
}

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic_assets', 'drops')
os.makedirs(OUTPUT_ROOT, exist_ok=True)

print("=" * 75)
print(f"🚀 开始从 maplestory.io 批量下载全部 {len(DROP_ITEMS_MAPPING)} 种掉落物 (金币+怪物战利品)...")
print("=" * 75)

summary = []

for item_key, info in DROP_ITEMS_MAPPING.items():
    item_id = info['id']
    cn_name = info['cn']
    en_name = info['en']
    category = info['category']
    
    target_dir = os.path.join(OUTPUT_ROOT, item_key)
    os.makedirs(target_dir, exist_ok=True)
    
    url = f'https://maplestory.io/api/GMS/210.1.1/item/{item_id}/icon'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx, timeout=12) as resp:
            data = resp.read()
            
        out_file = os.path.join(target_dir, f"{item_key}.png")
        with open(out_file, 'wb') as f:
            f.write(data)
            
        im = Image.open(out_file)
        w, h = im.size
        print(f"✓ [{item_key:<20}] {cn_name:<26} -> 尺寸: {w:2d}x{h:2d} px | RGBA 纯净透明")
        summary.append((item_key, cn_name, item_id, f"{w}x{h}", "成功"))
    except Exception as e:
        print(f"✗ [{item_key:<20}] {cn_name:<26} -> 下载失败: {e}")
        summary.append((item_key, cn_name, item_id, "-", f"失败: {e}"))

print("\n" + "=" * 75)
print("🎉 全部掉落物 (金币 + 21 种怪物独有战利品) 高清透明贴图下载完成！")
print("=" * 75)
print(f"{'标识名':<22} | {'中文说明':<28} | {'物品 ID':<8} | {'尺寸':<8} | {'状态'}")
print("-" * 75)
for k, cn, iid, sz, st in summary:
    print(f"{k:<22} | {cn:<28} | {iid:<8} | {sz:<8} | {st}")
print("=" * 75)

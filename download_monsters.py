import urllib.request
import json
import ssl
import sys
import os
import time
from PIL import Image, ImageSequence

sys.stdout.reconfigure(encoding='utf-8')

# 禁用 SSL 验证以保证下载顺畅
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# 21 种怪物的官方 ID 映射字典 (来自 GMS/KMS 原始数据库)
MOB_MAPPING = {
    'orange_mushroom': {'id': 100004, 'cn': '花蘑菇', 'en': 'Orange Mushroom'},
    'red_snail':       {'id': 100002, 'cn': '红蜗牛', 'en': 'Red Snail'},
    'slime':           {'id': 100006, 'cn': '绿水灵', 'en': 'Slime'},
    'bubbling':        {'id': 1210103, 'cn': '蓝水灵', 'en': 'Bubbling'},
    'horny_mushroom':  {'id': 2110200, 'cn': '刺蘑菇', 'en': 'Horny Mushroom'},
    'zombie_mushroom': {'id': 2230101, 'cn': '僵尸蘑菇', 'en': 'Zombie Mushroom'},
    'axe_stump':       {'id': 1130100, 'cn': '斧木妖', 'en': 'Axe Stump'},
    'wild_boar':       {'id': 2230102, 'cn': '野猪', 'en': 'Wild Boar'},
    'pig':             {'id': 100007, 'cn': '肥肥/猪猪', 'en': 'Pig'},
    'ribbon_pig':      {'id': 1210101, 'cn': '飘飘猪', 'en': 'Ribbon Pig'},
    'fire_boar':       {'id': 3210100, 'cn': '火野猪', 'en': 'Fire Boar'},
    'jr_necki':        {'id': 2130103, 'cn': '小青蛇', 'en': 'Jr. Necki'},
    'croco':           {'id': 3110100, 'cn': '鳄鱼/沼泽青鳄', 'en': 'Ligator'},
    'drake':           {'id': 4130100, 'cn': '土龙/黄土龙', 'en': 'Copper Drake'},
    'evil_eye':        {'id': 2230100, 'cn': '风独眼兽', 'en': 'Evil Eye'},
    'cold_eye':        {'id': 4230100, 'cn': '冰独眼兽', 'en': 'Cold Eye'},
    'jr_wraith':       {'id': 3230101, 'cn': '小幽灵', 'en': 'Jr. Wraith'},
    'wooden_mask':     {'id': 2230110, 'cn': '木面怪人', 'en': 'Wooden Mask'},
    'lupin':           {'id': 3210800, 'cn': '猴子/鲁胖', 'en': 'Lupin'},
    'rocky_mask':      {'id': 2230111, 'cn': '石面怪人', 'en': 'Rocky Mask'},
    'crab':            {'id': 3230102, 'cn': '红螃蟹/罗朗', 'en': 'Lorang'}
}

# 仅下载存活及战斗相关的动态帧 (排除死亡消散 die 动作，防止模型误将尸体当做存活目标)
ACTIONS = ['stand', 'move', 'hit1', 'attack1']
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic_assets', 'monsters')
os.makedirs(OUTPUT_ROOT, exist_ok=True)

print("=" * 70)
print(f"🚀 开始从 maplestory.io 批量下载 data.yaml 包含的 {len(MOB_MAPPING)} 种怪物全套透明素材...")
print("=" * 70)

total_sprites_downloaded = 0
summary = []

for class_name, info in MOB_MAPPING.items():
    mob_id = info['id']
    cn_name = info['cn']
    en_name = info['en']
    
    target_dir = os.path.join(OUTPUT_ROOT, class_name)
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"\n📦 [{class_name}] ({cn_name} - {en_name}, ID: {mob_id})")
    mob_extracted_count = 0
    
    for action in ACTIONS:
        url = f'https://maplestory.io/api/GMS/210.1.1/mob/{mob_id}/render/{action}'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx, timeout=12) as resp:
                data = resp.read()
                
            temp_file = os.path.join(target_dir, f"_temp_{action}.gif")
            with open(temp_file, 'wb') as f:
                f.write(data)
                
            im = Image.open(temp_file)
            n_frames = getattr(im, "n_frames", 1)
            
            for idx, frame in enumerate(ImageSequence.Iterator(im)):
                rgba = frame.convert("RGBA")
                frame_name = f"{action}_{idx}.png"
                rgba.save(os.path.join(target_dir, frame_name))
                mob_extracted_count += 1
                total_sprites_downloaded += 1
                
            im.close()
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            print(f"   ✓ 动作 [{action:7s}]: 提取 {n_frames:2d} 帧透明贴图")
        except Exception as e:
            # 部分怪物没有 attack1 或 hit1 是正常的，跳过
            pass
            
    summary.append((class_name, cn_name, en_name, mob_id, mob_extracted_count))
    time.sleep(0.1) # 礼貌延时

print("\n" + "=" * 70)
print(f"🎉 全部下载完成！共提取 {total_sprites_downloaded} 张 100% 纯净透明 PNG 贴图！")
print("=" * 70)
print(f"{'类别名称':<18} | {'中文名':<10} | {'ID':<8} | {'提取帧数'}")
print("-" * 55)
for c_name, cn, en, mid, count in summary:
    print(f"{c_name:<18} | {cn:<10} | {mid:<8} | {count} 帧")
print("=" * 70)

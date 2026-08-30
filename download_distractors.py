import urllib.request
import json
import ssl
import sys
import os
from PIL import Image, ImageSequence

sys.stdout.reconfigure(encoding='utf-8')

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic_assets', 'distractors')
os.makedirs(OUTPUT_ROOT, exist_ok=True)

print("=" * 75)
print("🚀 开始下载非标注负样本干扰物素材 (宠物小白雪人 & 沼泽互动紫色药草花)...")
print("=" * 75)

# ================= 1. 宠物 - 小白雪人 (Mini Yeti, ID: 5000020) =================
pet_id = 5000020
pet_dir = os.path.join(OUTPUT_ROOT, 'pet_yeti')
os.makedirs(pet_dir, exist_ok=True)

pet_actions = ['stand0', 'stand1', 'move', 'jump', 'hang', 'rest0']
print(f"\n🐾 [1/2] 正在下载宠物 - 小白雪人 (Mini Yeti, ID: {pet_id}) 全套动作...")

pet_count = 0
for act in pet_actions:
    url = f'https://maplestory.io/api/GMS/210.1.1/pet/{pet_id}/render/{act}'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            data = resp.read()
        
        temp_gif = os.path.join(pet_dir, f"_temp_{act}.gif")
        with open(temp_gif, 'wb') as f:
            f.write(data)
            
        im = Image.open(temp_gif)
        n = getattr(im, 'n_frames', 1)
        for idx, frame in enumerate(ImageSequence.Iterator(im)):
            rgba = frame.convert('RGBA')
            rgba.save(os.path.join(pet_dir, f"{act}_{idx}.png"))
            pet_count += 1
            
        im.close()
        if os.path.exists(temp_gif):
            os.remove(temp_gif)
        print(f"   ✓ 动作 [{act:<6s}]: 提取 {n} 帧透明贴图")
    except Exception as e:
        print(f"   ✗ 动作 [{act:<6s}] 失败: {e}")

# ================= 2. 沼泽地互动紫花 (Strange Flower / 奇怪的药草, ID: 4031396 & 4032464) =================
flower_dir = os.path.join(OUTPUT_ROOT, 'swamp_flower')
os.makedirs(flower_dir, exist_ok=True)

print(f"\n🌺 [2/2] 正在下载沼泽地互动植物 - 奇怪的药草/紫花 (Strange Plant & Herb)...")

flower_items = [
    (4031396, 'swamp_purple_flower', '沼泽互动紫花 (Strange Flower)'),
    (4032464, 'sabitrama_herb', '沙比特拉玛药草 (Sabitrama\'s Herb)'),
    (4000916, 'herb_bunch', '药草捆 (Herb Bunch)')
]

flower_count = 0
for iid, name, desc in flower_items:
    url = f'https://maplestory.io/api/GMS/210.1.1/item/{iid}/icon'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            data = resp.read()
            
        out_path = os.path.join(flower_dir, f"{name}.png")
        with open(out_path, 'wb') as f:
            f.write(data)
        im = Image.open(out_path)
        w, h = im.size
        flower_count += 1
        print(f"   ✓ [{name:<20}] {desc:<28} -> 尺寸: {w:2d}x{h:2d} px | RGBA 纯净透明")
    except Exception as e:
        print(f"   ✗ [{name:<20}] 失败: {e}")

print("\n" + "=" * 75)
print(f"🎉 全部干扰物负样本素材下载完成！(小白雪人: {pet_count} 帧, 沼泽紫花与药草: {flower_count} 张)")
print("=" * 75)

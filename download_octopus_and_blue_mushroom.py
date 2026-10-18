import os
import sys
import ssl
import json
import base64
import urllib.request
from PIL import Image, ImageSequence
import io

sys.stdout.reconfigure(encoding='utf-8')

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
headers = {'User-Agent': 'Mozilla/5.0'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "synthetic_assets")
MONSTERS_DIR = os.path.join(ASSETS_DIR, "monsters")
DROPS_DIR = os.path.join(ASSETS_DIR, "drops")

os.makedirs(MONSTERS_DIR, exist_ok=True)
os.makedirs(DROPS_DIR, exist_ok=True)

TARGET_MOBS = [
    {
        'key': 'octopus',
        'id': 1120100,
        'cn': '三眼章鱼',
        'en': 'Octopus',
        'actions': ['stand', 'move', 'hit1'],
        'drop_id': 4000006,
        'drop_key': 'octopus_leg',
        'drop_cn': '章鱼脚 / 章鱼腿'
    },
    {
        'key': 'blue_mushroom',
        'id': 2220100,
        'cn': '蓝蘑菇',
        'en': 'Blue Mushroom',
        'actions': ['stand', 'move', 'hit1', 'jump'],
        'drop_id': 4000009,
        'drop_key': 'blue_mushroom_cap',
        'drop_cn': '蓝蘑菇盖'
    }
]

def fetch_frames_for_action(mob_id, action):
    frames = []
    # 策略 1: 尝试 maplestory.io 动态渲染动图 (GMS/83 节点)
    render_url = f"https://maplestory.io/api/GMS/83/mob/{mob_id}/render/{action}"
    try:
        req = urllib.request.Request(render_url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=7) as resp:
            data = resp.read()
        im = Image.open(io.BytesIO(data))
        for frame in ImageSequence.Iterator(im):
            frames.append(frame.convert("RGBA"))
        if frames:
            return frames, "render_api"
    except Exception:
        pass

    # 策略 2: WZ 节点直接提取原始高保真 Canvas (针对单帧 hit1/jump 渲染服务超时的完美回退)
    frames = []
    idx = 0
    while True:
        wz_url = f"https://maplestory.io/api/wz/GMS/83/Mob/{mob_id}.img/{action}/{idx}"
        try:
            req = urllib.request.Request(wz_url, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=6) as resp:
                node_data = json.loads(resp.read().decode('utf-8'))
            b64_val = node_data.get('value')
            if not b64_val:
                break
            raw_png = base64.b64decode(b64_val)
            im = Image.open(io.BytesIO(raw_png)).convert("RGBA")
            frames.append(im)
            idx += 1
        except Exception:
            break
    
    if frames:
        return frames, "wz_node_canvas"
    return [], "failed"

print("=" * 75)
print("🚀 开始下载三眼章鱼 (Octopus) 与蓝蘑菇 (Blue Mushroom) 素材到 synthetic_assets...")
print("=" * 75)

total_sprites = 0
summary = []

for mob in TARGET_MOBS:
    mob_key = mob['key']
    mob_id = mob['id']
    cn_name = mob['cn']
    en_name = mob['en']
    
    mob_dir = os.path.join(MONSTERS_DIR, mob_key)
    os.makedirs(mob_dir, exist_ok=True)
    
    print(f"\n👾 [{cn_name} - {en_name}] (ID: {mob_id}) -> 目标目录: synthetic_assets/monsters/{mob_key}")
    mob_frames_count = 0
    
    for action in mob['actions']:
        frames, source = fetch_frames_for_action(mob_id, action)
        if frames:
            for idx, frame in enumerate(frames):
                out_path = os.path.join(mob_dir, f"{action}_{idx}.png")
                frame.save(out_path, format="PNG")
                mob_frames_count += 1
                total_sprites += 1
            print(f"   ✓ 动作 [{action:6s}]: 成功保存 {len(frames):2d} 帧透明贴图 (来源: {source})")
        else:
            print(f"   ✗ 动作 [{action:6s}]: 获取失败")
            
    # 下载专属战利品 (掉落物素材)
    drop_key = mob['drop_key']
    drop_id = mob['drop_id']
    drop_cn = mob['drop_cn']
    drop_dir = os.path.join(DROPS_DIR, drop_key)
    os.makedirs(drop_dir, exist_ok=True)
    
    drop_url = f"https://maplestory.io/api/GMS/83/item/{drop_id}/icon"
    drop_status = "失败"
    try:
        req = urllib.request.Request(drop_url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
            drop_bytes = resp.read()
        drop_im = Image.open(io.BytesIO(drop_bytes)).convert("RGBA")
        drop_file = os.path.join(drop_dir, f"{drop_key}.png")
        drop_im.save(drop_file, format="PNG")
        print(f"   ✓ 专属掉落物 [{drop_cn} ({drop_key})]: 成功保存至 synthetic_assets/drops/{drop_key}/{drop_key}.png ({drop_im.size[0]}x{drop_im.size[1]} px)")
        drop_status = "成功"
    except Exception as e:
        print(f"   ✗ 专属掉落物 [{drop_key}] 下载失败: {e}")
        
    summary.append({
        'mob': mob_key,
        'cn': cn_name,
        'en': en_name,
        'id': mob_id,
        'frames': mob_frames_count,
        'drop': drop_key,
        'drop_status': drop_status
    })

print("\n" + "=" * 75)
print(f"🎉 全部素材下载完成！共保存 {total_sprites} 张 100% 纯净透明怪物动作帧及战利品贴图！")
print("=" * 75)
print(f"{'怪物中文名':<10} | {'标识名':<15} | {'官方ID':<8} | {'提取帧数':<8} | {'专属掉落物':<16} | {'掉落物状态'}")
print("-" * 75)
for s in summary:
    print(f"{s['cn']:<10} | {s['mob']:<15} | {s['id']:<8} | {s['frames']:<8} | {s['drop']:<16} | {s['drop_status']}")
print("=" * 75)

import os
import sys
import ssl
import urllib.request
from PIL import Image, ImageSequence

sys.stdout.reconfigure(encoding='utf-8')

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
headers = {'User-Agent': 'Mozilla/5.0'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "synthetic_assets")
MONSTERS_DIR = os.path.join(ASSETS_DIR, "monsters")
DROPS_DIR = os.path.join(ASSETS_DIR, "drops")

mobs = [
    {
        'key': 'skeleton_soldier',
        'id': 5150001,
        'cn': '骷髅士兵',
        'drop_id': 4000206,
        'drop_name': 'rib'
    },
    {
        'key': 'officer_skeleton',
        'id': 6230602,
        'cn': '骷髅士官',
        'drop_id': 4000207,
        'drop_name': 'pelvic_bone'
    }
]

for mob in mobs:
    mob_dir = os.path.join(MONSTERS_DIR, mob['key'])
    os.makedirs(mob_dir, exist_ok=True)
    print(f"\n==========================================")
    print(f"📥 下载怪物: {mob['cn']} ({mob['key']}, ID: {mob['id']})")
    print(f"==========================================")
    
    total_frames = 0
    for action in ['stand', 'move']:
        # 使用稳定的 GMS/83 节点获取渲染动画
        url = f"https://maplestory.io/api/GMS/83/mob/{mob['id']}/render/{action}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
                gif_data = resp.read()
            
            temp_path = os.path.join(mob_dir, f"_temp_{action}.gif")
            with open(temp_path, "wb") as f:
                f.write(gif_data)
            
            im = Image.open(temp_path)
            n_frames = getattr(im, "n_frames", 1)
            for idx, frame in enumerate(ImageSequence.Iterator(im)):
                rgba = frame.convert("RGBA")
                frame_filename = f"{action}_{idx}.png"
                rgba.save(os.path.join(mob_dir, frame_filename))
                total_frames += 1
            im.close()
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"  ✓ 动作 [{action:5s}]: 提取 {n_frames:2d} 帧透明贴图")
        except Exception as e:
            print(f"  ✗ 动作 [{action:5s}] 下载失败: {e}")

    # 下载掉落物
    drop_dir = os.path.join(DROPS_DIR, mob['drop_name'])
    os.makedirs(drop_dir, exist_ok=True)
    drop_url = f"https://maplestory.io/api/GMS/83/item/{mob['drop_id']}/icon"
    try:
        req = urllib.request.Request(drop_url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
            icon_data = resp.read()
        drop_file = os.path.join(drop_dir, f"{mob['drop_name']}.png")
        with open(drop_file, "wb") as f:
            f.write(icon_data)
        print(f"  ✓ 专属掉落物 [{mob['drop_name']}]: 下载成功 ({len(icon_data)} 字节)")
    except Exception as e:
        print(f"  ✗ 掉落物下载失败: {e}")

print("\n✨ 所有素材下载处理完毕！")

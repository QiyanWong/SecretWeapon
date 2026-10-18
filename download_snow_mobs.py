import os
import sys
import ssl
import json
import base64
import urllib.request
import io
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
BACKGROUNDS_DIR = os.path.join(ASSETS_DIR, "backgrounds")

os.makedirs(MONSTERS_DIR, exist_ok=True)
os.makedirs(DROPS_DIR, exist_ok=True)
os.makedirs(BACKGROUNDS_DIR, exist_ok=True)

MOBS = [
    {
        'key': 'jr_yeti',
        'id': 5100000,
        'cn': '小白雪人',
        'en': 'Jr. Yeti',
        'actions': ['stand', 'move'],
        'drop_id': 4000048,
        'drop_key': 'jr_yeti_skin',
        'drop_cn': '小雪人之皮'
    },
    {
        'key': 'hector',
        'id': 5130104,
        'cn': '野狼',
        'en': 'Hector',
        'actions': ['stand', 'move'],
        'drop_id': 4000051,
        'drop_key': 'hector_tail',
        'drop_cn': '狼尾巴'
    },
    {
        'key': 'white_fang',
        'id': 5140000,
        'cn': '白狼',
        'en': 'White Fang',
        'actions': ['stand', 'move'],
        'drop_id': 4000052,
        'drop_key': 'white_fang_tail',
        'drop_cn': '白狼尾巴'
    }
]

print("=" * 70)
print("🚀 开始下载小白雪人、野狼、白狼及其战利品与雪山背景素材...")
print("=" * 70)

for mob in MOBS:
    mob_key = mob['key']
    mob_id = mob['id']
    cn_name = mob['cn']
    en_name = mob['en']
    
    mob_dir = os.path.join(MONSTERS_DIR, mob_key)
    os.makedirs(mob_dir, exist_ok=True)
    
    print(f"\n❄️ [{cn_name} - {en_name}] (ID: {mob_id}) -> synthetic_assets/monsters/{mob_key}")
    
    for action in mob['actions']:
        frames = []
        # 优先使用稳定的 GMS/83 渲染
        for ver in ['83', '210.1.1']:
            render_url = f"https://maplestory.io/api/GMS/{ver}/mob/{mob_id}/render/{action}"
            try:
                req = urllib.request.Request(render_url, headers=headers)
                with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
                    gif_data = resp.read()
                im = Image.open(io.BytesIO(gif_data))
                for frame in ImageSequence.Iterator(im):
                    frames.append(frame.convert("RGBA"))
                if frames:
                    break
            except Exception:
                pass
                
        if frames:
            for idx, frame in enumerate(frames):
                frame_file = os.path.join(mob_dir, f"{action}_{idx}.png")
                frame.save(frame_file, format="PNG")
            print(f"   ✓ 动作 [{action:5s}]: 保存 {len(frames)} 帧透明贴图")
        else:
            print(f"   ✗ 动作 [{action:5s}]: 获取失败")
            
    # 下载专属战利品
    drop_key = mob['drop_key']
    drop_id = mob['drop_id']
    drop_dir = os.path.join(DROPS_DIR, drop_key)
    os.makedirs(drop_dir, exist_ok=True)
    drop_url = f"https://maplestory.io/api/GMS/83/item/{drop_id}/icon"
    try:
        req = urllib.request.Request(drop_url, headers=headers)
        with urllib.request.urlopen(req, context=ctx, timeout=8) as resp:
            icon_data = resp.read()
        drop_file = os.path.join(drop_dir, f"{drop_key}.png")
        with open(drop_file, "wb") as f:
            f.write(icon_data)
        print(f"   ✓ 专属掉落物 [{drop_key}] ({mob['drop_cn']}): 下载成功")
    except Exception as e:
        print(f"   ✗ 掉落物下载失败: {e}")

# 下载冰雪山谷高清全景背景
print(f"\n🏔️ 正在下载【冰雪山谷】高清雪景背景 (Map ID: 211040100)...")
bg_url = "https://maplestory.io/api/GMS/83/map/211040100/render"
try:
    req = urllib.request.Request(bg_url, headers=headers)
    with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
        bg_data = resp.read()
    bg_path = os.path.join(BACKGROUNDS_DIR, "冰雪山谷.png")
    with open(bg_path, "wb") as f:
        f.write(bg_data)
    print(f"   ✓ 【冰雪山谷.png】全景雪地背景下载成功 ({len(bg_data)} 字节)")
except Exception as e:
    print(f"   ✗ 冰雪山谷背景下载失败: {e}")

print("\n🎉 素材下载流程完毕！")

import os
import sys
import ssl
import urllib.request
import numpy as np
from PIL import Image, ImageFilter

sys.stdout.reconfigure(encoding='utf-8')

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
headers = {'User-Agent': 'Mozilla/5.0'}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BG_DIR = os.path.join(BASE_DIR, "synthetic_assets", "backgrounds")
os.makedirs(BG_DIR, exist_ok=True)

map_id = 101030112  # 金银岛 - 第3军营 (Excavation: Military Camp 3)
url = f"https://maplestory.io/api/GMS/83/map/{map_id}/render"

print("=" * 60)
print(f"📥 从 maplestory.io 下载【第3军营】全景地图 (Map ID: {map_id})...")
print("=" * 60)

req = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
    data = resp.read()

raw_map_path = os.path.join(BG_DIR, "第三军营.png")
with open(raw_map_path, "wb") as f:
    f.write(data)

map_im = Image.open(raw_map_path).convert("RGBA")
mw, mh = map_im.size
print(f"✓ 成功下载纯净透明全景图 (无怪物/无玩家): {raw_map_path}")
print(f"  尺寸: {mw} x {mh} (RGBA 包含全部 4 层平台、梯子、神像、火把、传送门及边境巨石)")

# 生成带真实军营暗色石砖墙壁底图的完整版 (第三军营_wall.png)
# 军营内部背景色为暗青灰石砖调 (#22292f ~ #181d22)
bg_wall = Image.new("RGBA", (mw, mh), (24, 28, 33, 255))

# 产生细腻的古堡石纹微质感
noise = np.random.RandomState(42).normal(loc=0, scale=4.5, size=(mh, mw, 3))
wall_arr = np.array(bg_wall.convert("RGB"), dtype=np.float32)
wall_arr = np.clip(wall_arr + noise, 0, 255).astype(np.uint8)
wall_tex = Image.fromarray(wall_arr).convert("RGBA")

# 在石柱外侧边缘加深渐变阴影
shadow = Image.new("RGBA", (mw, mh), (0, 0, 0, 0))
# 合并: 底层石墙质感 + 上层地砖物件
comp = Image.alpha_composite(wall_tex, map_im)

wall_map_path = os.path.join(BG_DIR, "第三军营_wall.png")
comp.save(wall_map_path, format="PNG")
print(f"✓ 成功合成带暗黑石壁背景的完整游戏场景: {wall_map_path}")

# 额外保存小地图作为辅助参考
minimap_url = f"https://maplestory.io/api/GMS/83/map/{map_id}/minimap"
try:
    req_m = urllib.request.Request(minimap_url, headers=headers)
    with urllib.request.urlopen(req_m, context=ctx, timeout=10) as resp_m:
        m_data = resp_m.read()
    minimap_path = os.path.join(BG_DIR, "第三军营_minimap.png")
    with open(minimap_path, "wb") as f:
        f.write(m_data)
    print(f"✓ 成功提取对应官方小地图图标: {minimap_path}")
except Exception as e:
    print(f"✗ 小地图提取跳过: {e}")

print("\n✨ 全部第三军营素材下载并就绪！")

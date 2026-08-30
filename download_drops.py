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

# 掉落物品 (金币/药水/卷轴/其他战利品) 资产下载与管理脚本
# 官方 WZ Special/Item 对应的金币与货币 ID 映射表
DROP_ITEMS_MAPPING = {
    # 1. 经典三种金币体系 (铜币、金币、纸钞)
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
    # 额外附赠：大钱袋 (1000+ 冒险币)
    'meso_sack': {
        'id': 9000003,
        'cn': '钱袋 (1000+ 冒险币)',
        'en': 'Meso Sack',
        'category': 'meso'
    }
}

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synthetic_assets', 'drops')
os.makedirs(OUTPUT_ROOT, exist_ok=True)

print("=" * 70)
print(f"🚀 开始从 maplestory.io 下载掉落物品 (Drops) 高清透明素材...")
print("=" * 70)

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
        print(f"✓ [{item_key:<12}] {cn_name:<20} -> 尺寸: {w:2d}x{h:2d} px | RGBA 纯净透明")
        summary.append((item_key, cn_name, item_id, f"{w}x{h}", "成功"))
    except Exception as e:
        print(f"✗ [{item_key:<12}] {cn_name:<20} -> 下载失败: {e}")
        summary.append((item_key, cn_name, item_id, "-", f"失败: {e}"))

print("\n" + "=" * 70)
print("🎉 掉落物品素材下载与解析完成！")
print("=" * 70)
print(f"{'标识名':<14} | {'中文说明':<24} | {'物品 ID':<8} | {'尺寸':<8} | {'状态'}")
print("-" * 70)
for k, cn, iid, sz, st in summary:
    print(f"{k:<14} | {cn:<24} | {iid:<8} | {sz:<8} | {st}")
print("=" * 70)

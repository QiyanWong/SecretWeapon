import os
import sys
import glob
import json
import ssl
import urllib.request
import io
import xml.etree.ElementTree as ET
from PIL import Image, ImageSequence

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAP_DIR = os.path.join(BASE_DIR, "map")
ASSET_MONSTERS_DIR = os.path.join(BASE_DIR, "synthetic_assets", "monsters")
CUSTOM_MOBS_FILE = os.path.join(BASE_DIR, "custom_mobs.json")

# 怪物字典对照表 (官方 ID: 英文键名, 中文名)
MOB_CATALOG = {
    100004: {'key': 'orange_mushroom', 'cn': '花蘑菇'},
    100002: {'key': 'red_snail',       'cn': '红蜗牛'},
    100006: {'key': 'slime',           'cn': '绿水灵'},
    1210103: {'key': 'bubbling',       'cn': '蓝水灵'},
    2110200: {'key': 'horny_mushroom', 'cn': '刺蘑菇'},
    2230101: {'key': 'zombie_mushroom','cn': '僵尸蘑菇'},
    1130100: {'key': 'axe_stump',      'cn': '斧木妖'},
    2130100: {'key': 'dark_axe_stump', 'cn': '黑斧木妖'},
    2230102: {'key': 'wild_boar',      'cn': '野猪'},
    100007: {'key': 'pig',             'cn': '肥肥'},
    1210101: {'key': 'ribbon_pig',     'cn': '飘飘猪'},
    3210100: {'key': 'fire_boar',      'cn': '火野猪'},
    2130103: {'key': 'jr_necki',       'cn': '小青蛇'},
    3110100: {'key': 'croco',          'cn': '沼泽青鳄'},
    4130100: {'key': 'drake',          'cn': '土龙'},
    2230100: {'key': 'evil_eye',       'cn': '风独眼兽'},
    4230100: {'key': 'cold_eye',       'cn': '冰独眼兽'},
    3230101: {'key': 'jr_wraith',      'cn': '小幽灵'},
    2230110: {'key': 'wooden_mask',    'cn': '木面怪人'},
    3210800: {'key': 'lupin',          'cn': '猴子/鲁胖'},
    2230111: {'key': 'rocky_mask',     'cn': '石面怪人'},
    3230102: {'key': 'crab',           'cn': '红螃蟹/罗朗'},
    5130100: {'key': 'tauromacis',     'cn': '月牙牛魔王'},
    4230101: {'key': 'zombie_lupin',   'cn': '僵尸猴'},
    4230102: {'key': 'wraith',         'cn': '大幽灵'},
    3230300: {'key': 'jr_boogie',      'cn': '幼魔精灵'},
    3230301: {'key': 'jr_boogie',      'cn': '幼魔精灵(变体)'},
    5150001: {'key': 'skeleton_soldier','cn': '骷髅士兵'},
    6230602: {'key': 'officer_skeleton','cn': '骷髅士官'},
    1120100: {'key': 'octopus',         'cn': '三眼章鱼'},
    2220100: {'key': 'blue_mushroom',   'cn': '蓝蘑菇'},
    5100000: {'key': 'jr_yeti',         'cn': '小白雪人'},
    5100001: {'key': 'jr_yeti',         'cn': '小白雪人(变体)'},
    5130104: {'key': 'hector',          'cn': '野狼'},
    5140000: {'key': 'white_fang',      'cn': '白狼'},
    5130101: {'key': 'dark_stone_golem','cn': '黑石头人'},
    5130102: {'key': 'dark_stone_golem','cn': '黑石头人(变体)'},
    5130103: {'key': 'croco',          'cn': '沼泽青鳄(变体)'},
    2230131: {'key': 'zombie_mushroom','cn': '烦躁的僵尸蘑菇'},
    2300100: {'key': 'stirge',         'cn': '蝙蝠魔/小蝙蝠'},
    4090000: {'key': 'iron_hog',        'cn': '钢铁猪'},
    4230400: {'key': 'fire_boar',      'cn': '火野猪(变体)'},
    5090000: {'key': 'shade',          'cn': '变种幽灵'},
    6090001: {'key': 'snow_witch',     'cn': '雪山魔女'},
    6230100: {'key': 'wild_kargo',     'cn': '狂暴火龙'},
    7130100: {'key': 'tauromacis',     'cn': '月牙牛魔王(变体)'},
    100101:  {'key': 'red_snail',       'cn': '红蜗牛(变体)'},
    120100:  {'key': 'orange_mushroom', 'cn': '花蘑菇(变体)'},
    130101:  {'key': 'red_snail',       'cn': '红蜗牛(变体)'},
    1140100: {'key': 'axe_stump',      'cn': '斧木妖(变体)'},
    1210100: {'key': 'pig',             'cn': '肥肥(变体)'},
    3230100: {'key': 'cold_eye',       'cn': '冰独眼兽(变体)'}
}


def load_custom_configs():
    """读取 custom_mobs.json 中的手动自定义配置"""
    if not os.path.exists(CUSTOM_MOBS_FILE):
        return {}, {}
    try:
        with open(CUSTOM_MOBS_FILE, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
            id_map = cfg.get("mob_id_mapping", {})
            map_overrides = cfg.get("map_monster_overrides", {})
            return id_map, map_overrides
    except Exception as e:
        print(f"⚠️ 读取 custom_mobs.json 失败: {e}")
        return {}, {}


def save_custom_configs(id_map, map_overrides):
    """保存配置到 custom_mobs.json"""
    data = {
        "comment": "本文件用于手动配置、补充或覆盖地图怪物映射。在此添加后，所有合成与训练脚本均会自动生效！",
        "mob_id_mapping": id_map,
        "map_monster_overrides": map_overrides
    }
    with open(CUSTOM_MOBS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fetch_mob_info_from_api(mob_id):
    """当遇到未登记的 ID 时，尝试自动从官方 API 探测怪物名称"""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    for ver in ['83', '210.1.1']:
        url = f"https://maplestory.io/api/GMS/{ver}/mob/{mob_id}"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                name = data.get('name') or data.get('meta', {}).get('name', '')
                if name:
                    return name
        except Exception:
            pass
    return None


def download_mob_sprites(mob_key, mob_id):
    """一键下载怪物的 stand, move, hit1 动作透明贴图"""
    target_dir = os.path.join(ASSET_MONSTERS_DIR, mob_key)
    os.makedirs(target_dir, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    headers = {'User-Agent': 'Mozilla/5.0'}

    downloaded = 0
    print(f"📥 正在为 [{mob_key}] (ID: {mob_id}) 下载官方纯净贴图...")
    for act in ['stand', 'move', 'hit1']:
        for ver in ['83', '210.1.1']:
            url = f"https://maplestory.io/api/GMS/{ver}/mob/{mob_id}/render/{act}"
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, context=ctx, timeout=8) as r:
                    b = r.read()
                im = Image.open(io.BytesIO(b))
                idx = 0
                for frame in ImageSequence.Iterator(im):
                    frame.convert('RGBA').save(os.path.join(target_dir, f"{act}_{idx}.png"))
                    idx += 1
                    downloaded += 1
                if idx > 0:
                    break
            except Exception:
                pass
    print(f"✓ 成功下载 {downloaded} 帧贴图到: {target_dir}")
    return downloaded


def inspect_map(xml_file_path):
    map_name = os.path.splitext(os.path.basename(xml_file_path))[0]
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    mob_counts = {} # {id_str: count}
    life_dir = None
    for d in root.findall("dir"):
        if d.attrib.get("name") == "life":
            life_dir = d
            break

    if life_dir is not None:
        for node in life_dir.findall("dir"):
            m_type = None
            m_id = None
            for p in node:
                if p.attrib.get("name") == "type":
                    m_type = p.attrib.get("value")
                elif p.attrib.get("name") == "id":
                    m_id = p.attrib.get("value")
            if m_type == "m" and m_id:
                mob_counts[m_id] = mob_counts.get(m_id, 0) + 1

    return map_name, mob_counts


def verify_map(target_map=""):
    id_map, map_overrides = load_custom_configs()
    # 合并用户自定义 ID
    active_catalog = dict(MOB_CATALOG)
    for sid, info in id_map.items():
        try:
            mid = int(sid)
            active_catalog[mid] = {'key': info['key'], 'cn': info.get('cn', info['key'])}
        except ValueError:
            pass

    print("=" * 85)
    print("🔍 地图 XML 与对应怪物（刷新点数、中文名、贴图状态）核验工具")
    print(f"📌 自定义配置文件: {CUSTOM_MOBS_FILE} (已加载 {len(id_map)} 个自定义 ID, {len(map_overrides)} 个地图强制绑定)")
    print("=" * 85)

    if target_map:
        matches = glob.glob(os.path.join(MAP_DIR, f"*{target_map}*.xml"))
        if not matches:
            print(f"❌ 未找到匹配的地图 XML: {target_map}")
            return
        files_to_check = matches
    else:
        files_to_check = sorted(glob.glob(os.path.join(MAP_DIR, "*.xml")))

    for xf in files_to_check:
        map_name, mob_counts = inspect_map(xf)
        print(f"\n🗺️  【{map_name}】 (XML 文件: {os.path.basename(xf)})")

        # 检查是否有手动覆盖
        if map_name in map_overrides:
            ov = map_overrides[map_name]
            print(f"   ⚡ [已配置手动覆盖怪物列表]: {ov}")

        if not mob_counts:
            print("   ⚪ [无野怪刷新] (可能为城镇、安全区或纯跳跳图)")
            continue

        for mob_id_str, count in mob_counts.items():
            try:
                mob_id = int(mob_id_str)
            except ValueError:
                mob_id = -1

            info = active_catalog.get(mob_id)
            if info:
                key = info['key']
                cn = info['cn']
                has_asset = os.path.exists(os.path.join(ASSET_MONSTERS_DIR, key))
                asset_status = "✅ 贴图已就绪" if has_asset else "❌ 缺少本地贴图 (将自动下载)"
                print(f"   👾 ID: {mob_id:<8d} | {cn:<10s} ({key:<18s}) | 地图刷新点: {count:2d} 处 | {asset_status}")
            else:
                api_name = fetch_mob_info_from_api(mob_id_str)
                name_hint = f"API 识别为: [{api_name}]" if api_name else "待登记"
                print(f"   ❓ ID: {mob_id_str:<8s} | 未知怪物ID ({name_hint:<20s}) | 地图刷新点: {count:2d} 处 | ⚪ 待确认")
                print(f"      👉 可直接手动注册: python verify_map_mobs.py --add-mob {mob_id_str} <英文key> <中文名>")

    print("\n" + "=" * 85)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="核验地图 XML 中刷新的怪物信息与素材状态，支持一键注册与覆盖")
    parser.add_argument("--map", type=str, default="", help="核验指定的地图名称 (如: 野猪的领土 / 结冰平原1，留空则列出所有地图)")
    parser.add_argument("--add-mob", nargs=3, metavar=('ID', 'KEY', 'CN'), help="一键手动注册新怪物ID并自动下载贴图: --add-mob 2130100 dark_axe_stump 黑斧木妖")
    parser.add_argument("--set-map-mobs", nargs=2, metavar=('MAP', 'MOBS'), help="一键手动为指定地图强制指定怪物(英文逗号分隔): --set-map-mobs 野猪的领土 wild_boar,dark_axe_stump,jr_boogie")
    args = parser.parse_args()

    if args.add_mob:
        sid, m_key, m_cn = args.add_mob
        id_map, map_ov = load_custom_configs()
        id_map[str(sid)] = {"key": m_key, "cn": m_cn, "drop": None}
        save_custom_configs(id_map, map_ov)
        print(f"✅ 成功将 ID {sid} -> {m_cn} ({m_key}) 写入 {CUSTOM_MOBS_FILE}！")
        try:
            download_mob_sprites(m_key, int(sid))
        except Exception as e:
            print(f"⚠️ 自动下载贴图出错: {e}，您可以后续手动放置贴图到 synthetic_assets/monsters/{m_key}/")
        sys.exit(0)

    if args.set_map_mobs:
        target_m, mob_str = args.set_map_mobs
        mobs_list = [m.strip() for m in mob_str.split(',') if m.strip()]
        id_map, map_ov = load_custom_configs()
        map_ov[target_m] = mobs_list
        save_custom_configs(id_map, map_ov)
        print(f"✅ 成功为地图【{target_m}】设置手动怪物列表: {mobs_list}！已写入 {CUSTOM_MOBS_FILE}")
        sys.exit(0)

    verify_map(args.map)

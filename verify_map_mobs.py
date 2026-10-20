import os
import sys
import glob
import xml.etree.ElementTree as ET

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAP_DIR = os.path.join(BASE_DIR, "map")
ASSET_MONSTERS_DIR = os.path.join(BASE_DIR, "synthetic_assets", "monsters")

# 怪物字典对照表 (官方 ID: 英文键名, 中文名)
MOB_CATALOG = {
    100004: {'key': 'orange_mushroom', 'cn': '花蘑菇'},
    100002: {'key': 'red_snail',       'cn': '红蜗牛'},
    100006: {'key': 'slime',           'cn': '绿水灵'},
    1210103: {'key': 'bubbling',       'cn': '蓝水灵'},
    2110200: {'key': 'horny_mushroom', 'cn': '刺蘑菇'},
    2230101: {'key': 'zombie_mushroom','cn': '僵尸蘑菇'},
    1130100: {'key': 'axe_stump',      'cn': '斧木妖'},
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
    5150001: {'key': 'skeleton_soldier','cn': '骷髅士兵'},
    6230602: {'key': 'officer_skeleton','cn': '骷髅士官'},
    1120100: {'key': 'octopus',         'cn': '三眼章鱼'},
    2220100: {'key': 'blue_mushroom',   'cn': '蓝蘑菇'},
    5100000: {'key': 'jr_yeti',         'cn': '小白雪人'},
    5130104: {'key': 'hector',          'cn': '野狼'},
    5140000: {'key': 'white_fang',      'cn': '白狼'},
    5130101: {'key': 'dark_stone_golem','cn': '黑石头人'}
}


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
    print("=" * 80)
    print("🔍 地图 XML 与对应怪物（刷新点数、中文名、贴图状态）核验工具")
    print("=" * 80)

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

        if not mob_counts:
            print("   ⚪ [无野怪刷新] (可能为城镇、安全区或纯跳跳图)")
            continue

        for mob_id_str, count in mob_counts.items():
            try:
                mob_id = int(mob_id_str)
            except ValueError:
                mob_id = -1

            info = MOB_CATALOG.get(mob_id)
            if info:
                key = info['key']
                cn = info['cn']
                has_asset = os.path.exists(os.path.join(ASSET_MONSTERS_DIR, key))
                asset_status = "✅ 贴图已就绪" if has_asset else "❌ 缺少本地贴图"
                print(f"   👾 ID: {mob_id:<8d} | {cn:<8s} ({key:<18s}) | 地图刷新点: {count:2d} 处 | {asset_status}")
            else:
                print(f"   ❓ ID: {mob_id_str:<8s} | 未知怪物ID (未登记)            | 地图刷新点: {count:2d} 处 | ⚪ 待确认")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="核验地图 XML 中刷新的怪物信息与素材状态")
    parser.add_argument("--map", type=str, default="", help="核验指定的地图名称 (如: 第三军营 / 结冰平原1，留空则列出所有地图)")
    args = parser.parse_args()

    verify_map(args.map)

import os
import sys
import glob
import re
import argparse
from datetime import datetime

# 修复控制台 UTF-8 输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAP_DIR = os.path.join(BASE_DIR, "map")
MAP_RAW_ROOT = os.path.join(BASE_DIR, "dataset", "map_raw_images")
MODELS_DIR = os.path.join(BASE_DIR, "models")

from generate_map_synthetic import generate_map_dataset, parse_map_mobs, find_map_background, MONSTERS_DIR
from train_map_models import train_single_map
try:
    from generate_map_realistic_dataset import build_realistic_dataset
except ImportError:
    build_realistic_dataset = None


def get_all_xml_maps():
    """获取 map/ 目录下所有有效的地图名称"""
    xmls = glob.glob(os.path.join(MAP_DIR, "*.xml"))
    maps = []
    for xf in sorted(xmls):
        name = os.path.splitext(os.path.basename(xf))[0]
        maps.append(name)
    return maps


def get_map_material_info(map_name):
    """
    检查指定地图的素材状态:
    返回: (exists: bool, valid_pairs: int, classes: list)
    """
    raw_dir = os.path.join(MAP_RAW_ROOT, map_name)
    if not os.path.exists(raw_dir):
        return False, 0, []

    files = os.listdir(raw_dir)
    imgs = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    valid_pairs = 0
    for img in imgs:
        base = os.path.splitext(img)[0]
        if os.path.exists(os.path.join(raw_dir, base + ".txt")):
            valid_pairs += 1

    classes = []
    classes_file = os.path.join(raw_dir, "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]

    return True, valid_pairs, classes


def get_map_model_info(map_name):
    """
    检查指定地图的模型状态:
    返回: (has_model: bool, latest_model_name: str or None, latest_mtime: str or None)
    """
    if not os.path.exists(MODELS_DIR):
        return False, None, None

    normalized_map = map_name.strip().lower()
    matching_models = []

    for f in os.listdir(MODELS_DIR):
        if not f.endswith('.pt'):
            continue
        stem, _ = os.path.splitext(f)
        # 排除通用统一模型
        lower_stem = stem.lower()
        if lower_stem.startswith("best") or lower_stem.startswith("unified") or lower_stem == "yolov8n":
            continue

        # 匹配地图名: {map_name}_{timestamp}.pt 或 {map_name}.pt
        m = re.match(r"^(.+?)_\d{8}_\d{6}$", stem)
        detected_map = m.group(1).strip() if m else stem.strip()

        if detected_map.lower() == normalized_map:
            full_path = os.path.join(MODELS_DIR, f)
            matching_models.append((f, os.path.getmtime(full_path)))

    if not matching_models:
        return False, None, None

    # 按文件修改时间倒序排列，最新的在前
    matching_models.sort(key=lambda x: x[1], reverse=True)
    latest_name, latest_time = matching_models[0]
    mtime_str = datetime.fromtimestamp(latest_time).strftime("%Y-%m-%d %H:%M")
    return True, latest_name, mtime_str


def get_pipeline_status():
    """汇总并分析所有地图的全流程生命周期状态"""
    xml_maps = get_all_xml_maps()
    status_list = []

    for m_name in xml_maps:
        has_folder, pairs_count, classes = get_map_material_info(m_name)
        has_model, model_name, model_time = get_map_model_info(m_name)

        # 判断生命周期阶段
        # 1. 缺素材
        if not has_folder or pairs_count == 0:
            stage = "NEED_SYNTH" # ⚪ 缺素材
            stage_desc = "⚪ 待合成素材"
        # 2. 有素材但缺模型
        elif not has_model:
            stage = "NEED_TRAIN" # 🟡 待训练
            stage_desc = "🟡 待训练模型"
        # 3. 完备就绪
        else:
            stage = "READY"      # 🟢 已就绪
            stage_desc = "🟢 模型已就绪"

        status_list.append({
            "map_name": m_name,
            "has_folder": has_folder,
            "pairs_count": pairs_count,
            "classes": classes,
            "has_model": has_model,
            "model_name": model_name,
            "model_time": model_time,
            "stage": stage,
            "stage_desc": stage_desc
        })

    return status_list


def show_dashboard():
    """打印格式化全景生命周期看板"""
    statuses = get_pipeline_status()

    print("=" * 95)
    print("📋 MapleStory 地图模型全流程生命周期仪表盘 (XML -> 训练素材 -> YOLO模型)")
    print("=" * 95)
    print(f"{'地图名称':<14s} | {'流程阶段':<12s} | {'素材样本数':<10s} | {'最新模型文件':<30s} | {'更新时间'}")
    print("-" * 95)

    need_synth = []
    need_train = []
    ready = []

    for s in statuses:
        m_name = s['map_name']
        stage_desc = s['stage_desc']
        pairs = f"{s['pairs_count']} 组" if s['pairs_count'] > 0 else "无"
        m_file = s['model_name'] if s['model_name'] else "--"
        m_time = s['model_time'] if s['model_time'] else "--"

        print(f"{m_name:<14s} | {stage_desc:<12s} | {pairs:<10s} | {m_file:<30s} | {m_time}")

        if s['stage'] == "NEED_SYNTH":
            need_synth.append(m_name)
        elif s['stage'] == "NEED_TRAIN":
            need_train.append(m_name)
        else:
            ready.append(m_name)

    print("-" * 95)
    print(f"📊 统计概况: 总计地图 {len(statuses)} 张 | 🟢 已就绪: {len(ready)} | 🟡 待训练: {len(need_train)} | ⚪ 待合成素材: {len(need_synth)}")
    print("=" * 95)

    if need_train:
        print(f"\n💡 【增量训练建议】发现 {len(need_train)} 张地图有素材但尚未训练模型: {need_train}")
        print("   -> 运行选项 2 或命令: python map_pipeline_manager.py --auto-train")

    if need_synth:
        print(f"\n💡 【增量合成建议】发现 {len(need_synth)} 张地图尚未生成素材文件夹/内容为空: {need_synth}")
        print("   -> 运行选项 1 或命令: python map_pipeline_manager.py --auto-synth")


def auto_synth_missing_materials(count=100):
    """自动扫描并只为【没有对应素材文件夹或里面没东西】的地图合成图片"""
    statuses = get_pipeline_status()
    missing_maps = [s['map_name'] for s in statuses if s['stage'] == "NEED_SYNTH"]

    if not missing_maps:
        print("✅ 太棒了！当前没有任何地图缺少素材，全部地图均已有训练数据。")
        return []

    print("=" * 80)
    print(f"🔍 扫描到 {len(missing_maps)} 张地图尚未生成训练素材: {missing_maps}")
    print("🚀 开始自动增量合成素材...")
    print("=" * 80)

    success_list = []
    for idx, m_name in enumerate(missing_maps, 1):
        print(f"\n[{idx}/{len(missing_maps)}] 正在为地图 【{m_name}】 合成素材...")
        xml_file = os.path.join(MAP_DIR, f"{m_name}.xml")
        try:
            if build_realistic_dataset is not None and os.path.exists(xml_file):
                print(f"   🏰 启用 100% 物理真实地貌还原合成...")
                ok = build_realistic_dataset(m_name, count=count)
            else:
                ok = generate_map_dataset(m_name, count=count)
            if ok:
                success_list.append(m_name)
        except Exception as e:
            print(f"   ⚠️ 地图 【{m_name}】 真实合成失败: {e}，尝试使用通用合成兜底...")
            try:
                ok = generate_map_dataset(m_name, count=count)
                if ok:
                    success_list.append(m_name)
            except Exception as e2:
                print(f"   ❌ 地图 【{m_name}】 彻底合成失败: {e2}")

    print("\n" + "=" * 80)
    print(f"🎉 增量合成完成！本次为 {len(success_list)} 张地图成功补充了训练素材: {success_list}")
    print("=" * 80)
    return success_list


def auto_train_missing_models(epochs=50, batch=8, imgsz=800):
    """自动扫描并只为【有训练素材但尚未生成对应模型】的地图训练模型"""
    statuses = get_pipeline_status()
    missing_models = [s['map_name'] for s in statuses if s['stage'] == "NEED_TRAIN"]

    if not missing_models:
        print("✅ 当前所有拥有素材的地图均已完成模型训练，没有滞后待训地图！")
        return []

    print("=" * 80)
    print(f"🔍 扫描到 {len(missing_models)} 张地图具备素材但缺少专属模型: {missing_models}")
    print("🚀 开始连续为这些地图自动增量训练专属模型...")
    print("=" * 80)

    trained_list = []
    for idx, m_name in enumerate(missing_models, 1):
        print(f"\n[{idx}/{len(missing_models)}] 正在训练地图 【{m_name}】 专属模型...")
        model_path = train_single_map(m_name, epochs=epochs, batch=batch, imgsz=imgsz)
        if model_path:
            trained_list.append((m_name, model_path))

    print("\n" + "=" * 80)
    print(f"🎉 增量训练全部完成！本次成功生成 {len(trained_list)} 个专属模型:")
    for m_name, p in trained_list:
        print(f"   ✓ 地图 [{m_name}]: {os.path.basename(p)}")
    print("=" * 80)
    return trained_list


def force_train_all_models(epochs=50, batch=8, imgsz=800):
    """强制为所有拥有素材的地图连续训练专属模型（无论是否已有旧模型）"""
    statuses = get_pipeline_status()
    # 只要素材就绪，无论已有模型状态如何，全部列入训练清单
    trainable_maps = [s['map_name'] for s in statuses if s['has_materials'] and s['valid_pairs'] > 0]

    if not trainable_maps:
        print("❌ 当前没有任何地图具备有效训练素材，请先合成素材！")
        return []

    print("=" * 80)
    print(f"🔥 【强制全量训练模式启动】")
    print(f"📋 共扫描到 {len(trainable_maps)} 张具备素材的地图: {trainable_maps}")
    print(f"⚙️ 训练配置: Epochs={epochs}, Batch={batch}, ImgSz={imgsz}")
    print("=" * 80)

    trained_list = []
    for idx, m_name in enumerate(trainable_maps, 1):
        print(f"\n[{idx}/{len(trainable_maps)}] 正在强制重新训练地图 【{m_name}】 专属模型...")
        try:
            model_path = train_single_map(m_name, epochs=epochs, batch=batch, imgsz=imgsz)
            if model_path:
                trained_list.append((m_name, model_path))
        except Exception as e:
            print(f"❌ 地图 【{m_name}】 训练过程出错: {e}")

    print("\n" + "=" * 80)
    print(f"🎉 强制全量训练全部完成！成功生成 {len(trained_list)}/{len(trainable_maps)} 个最新专属模型:")
    for m_name, p in trained_list:
        print(f"   ✓ 地图 [{m_name}]: {os.path.basename(p)}")
    print("=" * 80)
    return trained_list


def interactive_menu():
    """交互式主菜单循环"""
    while True:
        print("\n" + "=" * 80)
        print("🗺️  MapleStory 地图模型全流程智能管理中枢")
        print("=" * 80)
        print(" [1] 查看全景生命周期看板 (XML -> 训练素材 -> YOLO模型 状态)")
        print(" [2] 增量合成素材 (自动检测缺少素材的地图，一键批量合成)")
        print(" [3] 增量训练模型 (自动检测有素材缺模型的地图，一键批量训练)")
        print(" [4] 强制为所有地图全量训练专属模型 (忽略已有模型，重新全量训练)")
        print(" [5] 单独为指定地图合成素材 (如 结冰平原1 / 第二军营)")
        print(" [6] 单独为指定地图训练模型 (如 结冰平原1 / 第二军营)")
        print(" [7] 退出")
        print("=" * 80)
        try:
            opt = input("请选择操作 [1-7] (默认1): ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not opt or opt == '1':
            show_dashboard()
            try: input("\n按回车键返回主菜单...")
            except Exception: pass
        elif opt == '2':
            auto_synth_missing_materials(count=100)
            try: input("\n按回车键返回主菜单...")
            except Exception: pass
        elif opt == '3':
            auto_train_missing_models(epochs=50, batch=8)
            try: input("\n按回车键返回主菜单...")
            except Exception: pass
        elif opt == '4':
            try:
                confirm = input("⚠️ 该操作将强制为全部地图重新训练专属模型，确定开始？(y/n, 默认y): ").strip().lower()
            except Exception:
                confirm = 'y'
            if confirm in ('', 'y', 'yes'):
                force_train_all_models(epochs=50, batch=8)
            try: input("\n按回车键返回主菜单...")
            except Exception: pass
        elif opt == '5':
            mname = input("请输入地图名称 (例如 结冰平原1): ").strip()
            if mname:
                xml_f = os.path.join(MAP_DIR, f"{mname}.xml")
                if build_realistic_dataset is not None and os.path.exists(xml_f):
                    print(f"🏰 正在启用 100% 物理真实地貌还原合成: 【{mname}】...")
                    build_realistic_dataset(mname, count=100)
                else:
                    generate_map_dataset(mname, count=100)
            try: input("\n按回车键返回主菜单...")
            except Exception: pass
        elif opt == '6':
            mname = input("请输入地图名称 (例如 结冰平原1): ").strip()
            if mname:
                train_single_map(mname, epochs=50, batch=8)
            try: input("\n按回车键返回主菜单...")
            except Exception: pass
        elif opt == '7':
            print("再见！")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MapleStory 地图模型全流程生命周期智能管理与增量同步工具")
    parser.add_argument("--status", action="store_true", help="显示全景生命周期状态仪表盘")
    parser.add_argument("--auto-synth", action="store_true", help="增量合成：仅为缺少素材文件夹或内容为空的地图合成素材")
    parser.add_argument("--auto-train", action="store_true", help="增量训练：仅为有素材但没有对应模型的地图连续训练模型")
    parser.add_argument("--force-train-all", action="store_true", help="强制全量训练：不论是否已有模型，为全部拥有素材的地图重新训练专属模型")
    parser.add_argument("--count", type=int, default=100, help="增量合成张数 (默认 100)")
    parser.add_argument("--epochs", type=int, default=50, help="增量训练轮数 (默认 50)")
    parser.add_argument("--batch", type=int, default=8, help="Batch 大小 (默认 8)")
    args = parser.parse_args()

    if args.auto_synth:
        auto_synth_missing_materials(count=args.count)
    elif args.auto_train:
        auto_train_missing_models(epochs=args.epochs, batch=args.batch)
    elif args.force_train_all:
        force_train_all_models(epochs=args.epochs, batch=args.batch)
    elif args.status:
        show_dashboard()
    else:
        interactive_menu()


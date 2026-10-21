import os
import sys
import shutil
import random
import datetime
import glob
import yaml
import argparse
from ultralytics import YOLO

# 修复控制台 UTF-8 输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MAP_RAW_ROOT = os.path.join(DATASET_DIR, "map_raw_images")
MAP_SPLITS_ROOT = os.path.join(DATASET_DIR, "map_splits")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RUNS_DIR = os.path.join(BASE_DIR, "runs", "detect")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MAP_RAW_ROOT, exist_ok=True)
os.makedirs(MAP_SPLITS_ROOT, exist_ok=True)


def load_map_classes(map_dir):
    """读取指定地图目录下的 classes.txt 类别定义"""
    classes_file = os.path.join(map_dir, "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f if line.strip()]
        if classes:
            return classes

    # 兜底：若未提供 classes.txt，默认使用玩家左右朝向 + 通用标注
    return ['player_left', 'player_right']


def prepare_map_dataset(map_name, split_ratio=0.8):
    """
    检查指定地图素材目录，打乱并划分 train (80%) 与 val (20%) 数据集。
    隔离保存在 dataset/map_splits/<map_name>/ 下，互不污染，也不影响根目录 unified 训练集。
    """
    map_raw_dir = os.path.join(MAP_RAW_ROOT, map_name)
    if not os.path.exists(map_raw_dir):
        print(f"❌ 未找到地图素材目录: {map_raw_dir}")
        return False, None, None

    # 读取类别
    class_names = load_map_classes(map_raw_dir)

    # 目标切分目录
    map_split_dir = os.path.join(MAP_SPLITS_ROOT, map_name)
    train_img_dir = os.path.join(map_split_dir, "images", "train")
    val_img_dir = os.path.join(map_split_dir, "images", "val")
    train_lbl_dir = os.path.join(map_split_dir, "labels", "train")
    val_lbl_dir = os.path.join(map_split_dir, "labels", "val")

    # 清空旧切分目录
    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 收集图片与标注对
    all_files = os.listdir(map_raw_dir)
    img_files = [f for f in all_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    valid_pairs = []
    for img_name in img_files:
        base = os.path.splitext(img_name)[0]
        txt_name = base + ".txt"
        txt_path = os.path.join(map_raw_dir, txt_name)
        if os.path.exists(txt_path):
            valid_pairs.append((img_name, txt_name))

    print(f"   📊 扫描到有效样本对: {len(valid_pairs)} 组 (来自 {map_raw_dir})")
    if len(valid_pairs) == 0:
        print(f"   ⚠️ 【跳过】该地图尚未包含标注数据，无法开始训练！")
        return False, None, None

    random.shuffle(valid_pairs)
    split_idx = int(len(valid_pairs) * split_ratio)
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]

    for img_name, txt_name in train_pairs:
        shutil.copy(os.path.join(map_raw_dir, img_name), os.path.join(train_img_dir, img_name))
        shutil.copy(os.path.join(map_raw_dir, txt_name), os.path.join(train_lbl_dir, txt_name))

    for img_name, txt_name in val_pairs:
        shutil.copy(os.path.join(map_raw_dir, img_name), os.path.join(val_img_dir, img_name))
        shutil.copy(os.path.join(map_raw_dir, txt_name), os.path.join(val_lbl_dir, txt_name))

    # 生成专属 data.yaml
    yaml_path = os.path.join(map_split_dir, "data.yaml")
    # 使用相对于根目录的相对路径以确保 YOLO 识别
    rel_split_dir = os.path.relpath(map_split_dir, BASE_DIR).replace('\\', '/')
    data_config = {
        'path': f"./{rel_split_dir}",
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

    print(f"   ✓ 数据集划分为: 训练集 {len(train_pairs)} 组 | 验证集 {len(val_pairs)} 组")
    print(f"   ✓ 类别配置生成完成: {class_names}")
    return True, yaml_path, class_names


def train_single_map(map_name, epochs=50, batch=8, imgsz=800, pretrained='yolov8n.pt'):
    """连续训练单个地图的专属 YOLOv8 模型"""
    print("\n" + "=" * 70)
    print(f"🚀 开始为地图 【{map_name}】 训练专属 YOLO 检测模型...")
    print("=" * 70)

    ready, yaml_path, class_names = prepare_map_dataset(map_name)
    if not ready:
        return None

    # 初始化 YOLO 模型 (优先加载根目录基础模型权重)
    base_weights = os.path.join(BASE_DIR, pretrained) if os.path.exists(os.path.join(BASE_DIR, pretrained)) else pretrained
    model = YOLO(base_weights)

    run_name = f"{map_name}_run"
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=2,
        degrees=0.0,
        fliplr=0.0,   # 严禁翻转，保证 player_left 与 player_right 左右状态清晰可辨
        mosaic=0.5,
        name=run_name,
        project=RUNS_DIR
    )

    # 提取训练完毕的最优权重
    save_dir = getattr(results, 'save_dir', None)
    best_weights_path = os.path.join(save_dir, 'weights', 'best.pt') if save_dir else os.path.join(RUNS_DIR, run_name, 'weights', 'best.pt')

    if os.path.exists(best_weights_path):
        # 命名规范：地图名 + 时间（版本号）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_filename = f"{map_name}_{timestamp}.pt"
        versioned_path = os.path.join(MODELS_DIR, versioned_filename)
        shutil.copy(best_weights_path, versioned_path)

        print("\n" + "*" * 70)
        print(f"🎉 【训练成功】地图 【{map_name}】 专属模型已保存至:")
        print(f"   📁 {versioned_path}")
        print(f"   🎯 识别类别 ({len(class_names)} 类): {class_names}")
        print(f"   🛡️ 根目录统一模型保持不变，启动 yolo_detector 时选择该地图将自动首选本专属模型！")
        print("*" * 70)
        return versioned_path
    else:
        print(f"❌ 警告: 未找到输出权重文件: {best_weights_path}")
        return None


def run_batch_map_training(target_map="", epochs=50, batch=8, imgsz=800):
    """连续训练全部或指定地图的模型"""
    if target_map:
        trained_path = train_single_map(target_map, epochs=epochs, batch=batch, imgsz=imgsz)
        return [trained_path] if trained_path else []

    # 扫描 dataset/map_raw_images/ 下的所有地图目录
    sub_dirs = [d for d in os.listdir(MAP_RAW_ROOT) if os.path.isdir(os.path.join(MAP_RAW_ROOT, d))]
    if not sub_dirs:
        print(f"❌ 未在 {MAP_RAW_ROOT} 中扫描到任何地图素材文件夹！")
        print("请先运行 generate_map_synthetic.py 为目标地图生成素材，或手动放置素材。")
        return []

    print("=" * 70)
    print(f"📋 扫描到已创建素材的地图 ({len(sub_dirs)} 个): {sub_dirs}")
    print("开始连续训练每个地图的专属模型...")
    print("=" * 70)

    trained_models = []
    for idx, m_name in enumerate(sub_dirs, 1):
        print(f"\n[{idx}/{len(sub_dirs)}] 准备训练: 【{m_name}】")
        model_path = train_single_map(m_name, epochs=epochs, batch=batch, imgsz=imgsz)
        if model_path:
            trained_models.append((m_name, model_path))

    print("\n" + "=" * 70)
    print(f"🏁 全部地图批量连续训练流程完毕！成功生成 {len(trained_models)} 个专属模型:")
    for m_name, p in trained_models:
        print(f"   ✓ 地图 [{m_name}]: {os.path.basename(p)}")
    print("=" * 70)
    return trained_models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MapleStory 地图专属模型连续批量训练脚本")
    parser.add_argument("--map", type=str, default="", help="指定训练单张地图 (默认留空将连续训练所有已备好素材的地图)")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数 (默认 50)")
    parser.add_argument("--batch", type=int, default=8, help="Batch 大小 (默认 8)")
    parser.add_argument("--imgsz", type=int, default=800, help="输入分辨率 (默认 800)")
    args = parser.parse_args()

    run_batch_map_training(target_map=args.map, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)

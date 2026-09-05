import os
import shutil
import random
import yaml
from ultralytics import YOLO

# 项目基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# 数据集子目录定义
RAW_IMAGES_DIR = os.path.join(DATASET_DIR, "raw_images")
TRAIN_IMG_DIR = os.path.join(DATASET_DIR, "images", "train")
VAL_IMG_DIR = os.path.join(DATASET_DIR, "images", "val")
TRAIN_LBL_DIR = os.path.join(DATASET_DIR, "labels", "train")
VAL_LBL_DIR = os.path.join(DATASET_DIR, "labels", "val")
YAML_PATH = os.path.join(BASE_DIR, "data.yaml")

# 默认检测目标类别列表 (官方英文名称映射，共 25 类: 2 种玩家朝向 + 23 种怪物)
CLASS_NAMES = [
    'player_left',
    'player_right',
    'orange_mushroom',
    'red_snail',
    'slime',
    'bubbling',
    'horny_mushroom',
    'zombie_mushroom',
    'axe_stump',
    'wild_boar',
    'pig',
    'ribbon_pig',
    'fire_boar',
    'jr_necki',
    'croco',
    'drake',
    'evil_eye',
    'cold_eye',
    'jr_wraith',
    'wooden_mask',
    'lupin',
    'rocky_mask',
    'crab',
    'tauromacis',
    'dark_stone_golem'
]


def prepare_dataset(split_ratio=0.8):
    """
    检查并自动打乱划分训练集 (80%) 与 验证集 (20%)
    需要确保每张 JPG 图片同名对应一个 YOLO 格式的 TXT 标注文件
    """
    print("=" * 60)
    print("正在准备与检验 YOLO 数据集目录...")

    # 1. 彻底清空旧的 split 训练集与验证集目录，防止废弃文件残留
    for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 收集已完成标注的图片列表
    all_files = os.listdir(RAW_IMAGES_DIR)
    img_files = [f for f in all_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    valid_pairs = []
    for img_name in img_files:
        base_name = os.path.splitext(img_name)[0]
        txt_name = base_name + ".txt"
        txt_path = os.path.join(RAW_IMAGES_DIR, txt_name)

        if os.path.exists(txt_path):
            valid_pairs.append((img_name, txt_name))

    print(f"在 {RAW_IMAGES_DIR} 中共扫描到 {len(img_files)} 张图片，其中已完成标注的配对数据: {len(valid_pairs)} 组")

    if len(valid_pairs) == 0:
        print("\n【提示】未检测到已标注的 .txt 文件！")
        print("请先使用标注工具（如 makesense.ai / AnyLabeling / LabelImg）将导出的 YOLO 格式 .txt 文件放入 dataset/raw_images 文件夹中。")
        return False

    # 随机打乱并切分数据集
    random.shuffle(valid_pairs)
    split_index = int(len(valid_pairs) * split_ratio)
    train_pairs = valid_pairs[:split_index]
    val_pairs = valid_pairs[split_index:]

    # 复制文件到 YOLO 标准结构
    print(f"正在划分为: 训练集 {len(train_pairs)} 组 | 验证集 {len(val_pairs)} 组...")

    for img_name, txt_name in train_pairs:
        shutil.copy(os.path.join(RAW_IMAGES_DIR, img_name), os.path.join(TRAIN_IMG_DIR, img_name))
        shutil.copy(os.path.join(RAW_IMAGES_DIR, txt_name), os.path.join(TRAIN_LBL_DIR, txt_name))

    for img_name, txt_name in val_pairs:
        shutil.copy(os.path.join(RAW_IMAGES_DIR, img_name), os.path.join(VAL_IMG_DIR, img_name))
        shutil.copy(os.path.join(RAW_IMAGES_DIR, txt_name), os.path.join(VAL_LBL_DIR, txt_name))

    print("数据集文件迁移与划分完成！")
    return True


def create_data_yaml(classes=CLASS_NAMES):
    """自动生成 YOLOv8 格式的配置文件 data.yaml"""
    data_config = {
        'path': './dataset',
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(classes)}
    }

    with open(YAML_PATH, 'w', encoding='utf-8') as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)

    print(f"已自动更新 data.yaml 配置文件: {YAML_PATH}")
    print(f"类别定义: {data_config['names']}")


def run_training(epochs=50, imgsz=800, batch=16):
    """使用 Ultralytics YOLOv8 开始训练"""
    print("=" * 60)
    print("开始初始化 YOLOv8 Nano 模型训练...")
    
    # 加载预训练模型 weights (首次运行会自动从 GitHub 下载 yolov8n.pt)
    model = YOLO('yolov8n.pt')

    # 开始训练 (为 2D 横版游戏禁用旋转 degrees=0.0，增强小目标与扁平怪物特征)
    results = model.train(
        data=YAML_PATH,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        workers=2,
        degrees=0.0,
        name='maple_yolo_run',
        project=os.path.join(BASE_DIR, 'runs', 'detect')
    )

    print("=" * 60)
    print("模型训练完毕！")
    
    # 动态查找并备份 best.pt 到根目录
    save_dir = getattr(results, 'save_dir', None)
    best_weights_path = os.path.join(save_dir, 'weights', 'best.pt') if save_dir else os.path.join(BASE_DIR, 'runs', 'detect', 'maple_yolo_run', 'weights', 'best.pt')
    target_weights_path = os.path.join(BASE_DIR, 'best.pt')

    if os.path.exists(best_weights_path):
        shutil.copy(best_weights_path, target_weights_path)
        print(f"【成功】最优模型权重已更新并复制至项目根目录: {target_weights_path}")
        print("现在你可以启动 yolo_detector.py 开始实时识别游戏中的目标！")
    else:
        print(f"警告: 未找到最优权重文件: {best_weights_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="冒险岛 YOLOv8 训练脚本")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数 (默认 50)")
    parser.add_argument("--batch", type=int, default=16, help="Batch 大小 (默认 16)")
    args = parser.parse_args()

    # 1. 准备数据集结构
    ready = prepare_dataset()
    
    if ready:
        # 2. 生成配置
        create_data_yaml()
        # 3. 启动训练
        run_training(epochs=args.epochs, batch=args.batch)

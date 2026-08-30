import os
import sys
import glob
import cv2
import numpy as np

# 修复控制台 UTF-8 输出
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw_images")
DEBUG_DIR = os.path.join(BASE_DIR, "dataset", "synthetic_debug")

from convert_json_to_yolo import CLASS_LIST

def visualize_synthetic_labels(num_samples=15):
    """读取 raw_images 里的最新 synth_ 图像与 .txt 标注，在图片上画框输出至 synthetic_debug 文件夹供可视化检验"""
    if os.path.exists(DEBUG_DIR):
        for f in os.listdir(DEBUG_DIR):
            os.remove(os.path.join(DEBUG_DIR, f))
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    # 搜寻所有合成图片的 txt 文件并按最新修改时间排序
    synth_txts = glob.glob(os.path.join(RAW_DIR, "synth_*.txt"))
    if not synth_txts:
        synth_txts = glob.glob(os.path.join(RAW_DIR, "*.txt"))
        synth_txts = [t for t in synth_txts if os.path.basename(t) != "classes.txt"]

    if not synth_txts:
        print("未在 dataset/raw_images 找到任何 .txt 标注文件！")
        return

    # 优先抽取最新生成的合成图片
    synth_txts.sort(key=os.path.getmtime, reverse=True)
    print(f"扫描到 {len(synth_txts)} 个 .txt 标注文件，准备抽取最新的 {min(num_samples, len(synth_txts))} 张图片绘制画框预览...")

    # 随机或选取最新样本
    selected_txts = synth_txts[:num_samples]

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (128, 255, 0), (0, 128, 255), (255, 128, 0), (128, 0, 255)
    ]

    for txt_path in selected_txts:
        base_stem = os.path.splitext(txt_path)[0]
        img_path = base_stem + ".jpg"
        if not os.path.exists(img_path):
            img_path = base_stem + ".png"
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        json_path = base_stem + ".json"

        # 如果存在 AnyLabeling JSON 文件，优先绘制极致平滑的多边形轮廓
        if os.path.exists(json_path):
            try:
                import json
                with open(json_path, 'r', encoding='utf-8') as jf:
                    jdata = json.load(jf)
                
                overlay = img.copy()
                for idx, shape in enumerate(jdata.get('shapes', [])):
                    pts = np.array(shape.get('points', []), dtype=np.int32)
                    color = colors[idx % len(colors)]
                    label = shape.get('label', '')

                    if shape.get('shape_type') == 'polygon' and len(pts) >= 3:
                        poly_pts = pts.reshape((-1, 1, 2))
                        cv2.fillPoly(overlay, [poly_pts], color)
                        cv2.polylines(img, [poly_pts], isClosed=True, color=color, thickness=2)
                        tx, ty = int(pts[:, 0].mean()), int(pts[:, 1].min() - 5)
                    else:
                        x1, y1 = pts[0]
                        x2, y2 = pts[1]
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        tx, ty = int(x1), int(y1 - 5)

                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(img, (tx, ty - th - 4), (tx + tw, ty), color, -1)
                    cv2.putText(img, label, (tx, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
                out_file = os.path.join(DEBUG_DIR, os.path.basename(img_path))
                cv2.imwrite(out_file, img)
                print(f"【多边形轮廓渲染保存】 -> {os.path.abspath(out_file)}")
                continue
            except Exception as e:
                print(f"JSON 渲染异常: {e}")

        # 回退至读取 txt 矩形框
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id = int(parts[0])
            norm_xc, norm_yc, norm_w, norm_h = map(float, parts[1:])

            w = norm_w * img_w
            h = norm_h * img_h
            xc = norm_xc * img_w
            yc = norm_yc * img_h

            x1 = int(xc - w / 2.0)
            y1 = int(yc - h / 2.0)
            x2 = int(xc + w / 2.0)
            y2 = int(yc + h / 2.0)

            class_name = CLASS_LIST[cls_id] if cls_id < len(CLASS_LIST) else f"cls_{cls_id}"
            color = colors[cls_id % len(colors)]

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_text = f"{class_name} ({cls_id})"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(img, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        out_file = os.path.join(DEBUG_DIR, os.path.basename(img_path))
        cv2.imwrite(out_file, img)
        print(f"【矩形渲染预览保存】 -> {os.path.abspath(out_file)}")

    print(f"\n✨ 可视化渲染完成！请前往文件夹查看画框效果: {DEBUG_DIR}")

if __name__ == "__main__":
    visualize_synthetic_labels()

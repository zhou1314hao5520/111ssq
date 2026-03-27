import os
import numpy as np
import torch
import cv2
import rasterio
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

# 确保当前目录下有 model.py，并且包含 AttentionUNet 类的定义
from model import AttentionUNet

# ==============================
# 模型加载
# ==============================
def load_model(model_path, device):
    print(f"正在加载模型权重: {model_path}")
    model = AttentionUNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    # 兼容直接保存 state_dict 或保存了 epoch/optimizer 的字典格式
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

# ==============================
# 归一化
# ==============================
def normalize_image(img):
    img = img.astype(np.float32)
    if img.max() > 1:
        img = img / 255.0
    return img

# ==============================
# 读取与保存 TIF
# ==============================
def read_tif(path):
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        data = src.read()
    return data, profile, transform, crs

def read_mask_tif(path):
    with rasterio.open(path) as src:
        mask = src.read(1)
    mask = (mask > 0).astype(np.uint8)
    return mask

def save_tif(path, mask, profile):
    profile.update(dtype=rasterio.uint8, count=1, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)

def save_png(path, mask):
    png_mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(path, png_mask)

# ==============================
# GAP统计与科研图
# ==============================
def gap_analysis(pred_mask, rgb, output_dir, pixel_size=0.5):
    print("正在生成林窗空间统计与科研级可视化图表...")
    labeled = label(pred_mask)
    regions = regionprops(labeled)

    areas = []
    centroids = []

    for r in regions:
        area_pixel = r.area
        area_m2 = area_pixel * pixel_size * pixel_size
        areas.append(area_m2)
        centroids.append(r.centroid)

    areas = np.array(areas)
    if len(areas) == 0:
        print("未检测到任何林窗，跳过统计图绘制。")
        return

    # 1. Histogram
    plt.figure()
    plt.hist(areas, bins=30)
    plt.xlabel("Gap Area (m²)")
    plt.ylabel("Frequency")
    plt.title("Gap Size Distribution")
    plt.savefig(os.path.join(output_dir, "gap_histogram.png"), dpi=300)
    plt.close()

    # 2. CDF
    sorted_area = np.sort(areas)
    cdf = np.arange(len(sorted_area)) / len(sorted_area)
    plt.figure()
    plt.plot(sorted_area, cdf)
    plt.xlabel("Gap Area (m²)")
    plt.ylabel("CDF")
    plt.title("Gap Size Cumulative Distribution")
    plt.savefig(os.path.join(output_dir, "gap_cdf.png"), dpi=300)
    plt.close()

    # 3. Gap size class bar
    bins = [0, 10, 50, 100, 500, 10000]
    labels_bar = ["0-10", "10-50", "50-100", "100-500", ">500"]
    counts = np.histogram(areas, bins)[0]
    plt.figure()
    plt.bar(labels_bar, counts)
    plt.xlabel("Gap Size Class (m²)")
    plt.ylabel("Count")
    plt.title("Gap Size Class Distribution")
    plt.savefig(os.path.join(output_dir, "gap_size_class_bar.png"), dpi=300)
    plt.close()

    # 4. Rank plot
    rank = np.arange(1, len(sorted_area) + 1)
    plt.figure()
    plt.loglog(rank, sorted_area)
    plt.xlabel("Rank")
    plt.ylabel("Gap Area (m²)")
    plt.title("Gap Size Rank Plot")
    plt.savefig(os.path.join(output_dir, "gap_area_rank_plot.png"), dpi=300)
    plt.close()

    # 5. Largest gap map
    overlay = rgb.copy()
    sorted_regions = sorted(regions, key=lambda r: r.area, reverse=True)
    top = sorted_regions[:20]

    for r in top:
        y, x = r.centroid
        cv2.circle(overlay, (int(x), int(y)), 10, (255, 0, 0), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Largest Gap Locations")
    plt.savefig(os.path.join(output_dir, "largest_gap_map.png"), dpi=300)
    plt.close()

    # 6. Statistics text
    with open(os.path.join(output_dir, "gap_statistics.txt"), "w", encoding='utf-8') as f:
        f.write(f"总林窗数量 (Gap count): {len(areas)}\n")
        f.write(f"平均面积 (Mean area): {areas.mean():.2f} m²\n")
        f.write(f"最大面积 (Max area): {areas.max():.2f} m²\n")
        f.write(f"最小面积 (Min area): {areas.min():.2f} m²\n")

# ==============================
# Overlay 可视化
# ==============================
def overlay_prediction(rgb, pred, save_path):
    img = rgb.copy()
    img[pred == 1] = [255, 0, 0]
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Prediction Overlay")
    plt.savefig(save_path, dpi=300)
    plt.close()

def overlay_gt(rgb, gt, save_path):
    img = rgb.copy()
    img[gt == 1] = [0, 255, 0]
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title("GT Overlay")
    plt.savefig(save_path, dpi=300)
    plt.close()

def error_map(rgb, pred, gt, save_path):
    img = rgb.copy()
    tp = (pred == 1) & (gt == 1)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)

    img[tp] = [0, 255, 0]  # TP 绿色
    img[fp] = [0, 0, 255]  # FP 蓝色
    img[fn] = [255, 0, 0]  # FN 红色

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Error Map (TP:Green, FP:Blue, FN:Red)")
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==============================
# 大图推理核心函数 (供外部或 GUI 调用)
# ==============================
def predict_large_tif(
        model_path,
        tif_path,
        output_tif,
        gt_path=None,
        tile_size=512,
        stride=384,
        threshold=0.35,
        device=None):
    
    # 初始化设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用计算设备: {device}")
    
    # 路径清理与文件夹创建
    if gt_path == "": 
        gt_path = None
    output_dir = os.path.dirname(output_tif)
    os.makedirs(output_dir, exist_ok=True)
    output_png = output_tif.replace(".tif", ".png")

    # 加载模型
    model = load_model(model_path, device)

    # 读取输入影像
    print(f"读取全景影像: {tif_path}")
    data, profile, transform, crs = read_tif(tif_path)
    img = np.transpose(data[:3], (1, 2, 0))
    img = normalize_image(img)
    h, w, c = img.shape

    # 滑动窗口预测容器
    pred_sum = np.zeros((h, w), dtype=np.float32)
    pred_count = np.zeros((h, w), dtype=np.float32)

    xs = list(range(0, max(w - tile_size, 0) + 1, stride))
    ys = list(range(0, max(h - tile_size, 0) + 1, stride))

    if xs[-1] != w - tile_size:
        xs.append(w - tile_size)
    if ys[-1] != h - tile_size:
        ys.append(h - tile_size)

    print("开始大图滑动窗口推理...")
    for y in tqdm(ys, desc="Processing Rows"):
        for x in xs:
            patch = img[y:y+tile_size, x:x+tile_size]
            ph, pw, _ = patch.shape

            # 边缘 padding
            if ph != tile_size or pw != tile_size:
                pad = np.zeros((tile_size, tile_size, c), dtype=np.float32)
                pad[:ph, :pw] = patch
                patch = pad

            patch_tensor = torch.tensor(
                np.transpose(patch, (2, 0, 1)),
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(patch_tensor)[0, 0].cpu().numpy()

            pred = pred[:ph, :pw]
            pred_sum[y:y+ph, x:x+pw] += pred
            pred_count[y:y+ph, x:x+pw] += 1

    # 平均并二值化
    pred_avg = pred_sum / np.clip(pred_count, 1e-6, None)
    pred_bin = (pred_avg > threshold).astype(np.uint8)

    # 结果保存
    print("保存预测掩膜结果...")
    save_tif(output_tif, pred_bin, profile)
    save_png(output_png, pred_bin)

    # 准备可视化底图
    rgb_vis = (img * 255).astype(np.uint8)

    # 分析与科研图表导出
    gap_analysis(pred_bin, rgb_vis, output_dir)
    overlay_prediction(rgb_vis, pred_bin, os.path.join(output_dir, "overlay_prediction.png"))

    # 如果提供了 GT，计算误差图
    if gt_path is not None and os.path.exists(gt_path):
        print(f"检测到真实值标注，正在生成误差对比图...")
        gt = read_mask_tif(gt_path)
        overlay_gt(rgb_vis, gt, os.path.join(output_dir, "overlay_gt.png"))
        error_map(rgb_vis, pred_bin, gt, os.path.join(output_dir, "overlay_error_map.png"))

    print("\n===============================")
    print("🔥 推理与可视化分析全部完成！")
    print(f"输出目录: {output_dir}")
    print("===============================\n")

# ==============================
# 本地测试入口
# ==============================
if __name__ == "__main__":
    test_model_path = r"K:\ssq\data\train_output4\best_model.pth"
    test_tif_path = r"K:\ssq\data\HSTAC\tiantongshan\Tiantongshan_DOM.tif"
    test_gt_path = r"K:\ssq\data\HSTAC\tiantongshan\Tiantongshan_gap_result_3.tif"
    test_output_tif = r"K:\ssq\data\HSTAC\tiantongshan\paper_outputs\Tiantongshan_predict_result.tif"

    predict_large_tif(
        model_path=test_model_path,
        tif_path=test_tif_path,
        output_tif=test_output_tif,
        gt_path=test_gt_path,
        tile_size=512,
        stride=384,
        threshold=0.35
    )
import rasterio
import numpy as np
import cv2
import os

from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    binary_opening,
    binary_closing,
    disk
)

# =========================
# 辅助函数：归一化
# =========================
def robust_normalize(x, mask, low=2, high=98):
    vals = x[mask]
    if len(vals) == 0:
        return np.zeros_like(x)
    p_low, p_high = np.percentile(vals, [low, high])
    x_norm = (x - p_low) / (p_high - p_low + 1e-6)
    return np.clip(x_norm, 0, 1)


# =========================
# 核心处理函数 (供主程序或 GUI 调用)
# =========================
def run_gap_extraction(
    input_tif, 
    output_tif, 
    output_vis, 
    gt_tif=None, 
    min_gap_area_m2=5.0, 
    open_radius=2, 
    close_radius=4, 
    dark_factor=0.95, 
    vari_max_for_gap=0.05, 
    exg_max_for_gap=0.02
):
    """
    执行无监督林窗提取，并输出预测TIF与可视化覆盖图。
    """
    print(f"开始处理: {input_tif}")
    
    # =========================
    # 1. 读取影像
    # =========================
    with rasterio.open(input_tif) as src:
        r = src.read(1).astype(np.float32)
        g = src.read(2).astype(np.float32)
        b = src.read(3).astype(np.float32)
        profile = src.profile
        transform = src.transform
        nodata = src.nodata

    # =========================
    # 2. 有效像元掩膜
    # =========================
    if nodata is not None:
        valid_mask = (r != nodata) & (g != nodata) & (b != nodata)
    else:
        valid_mask = (r > 0) & (g > 0) & (b > 0)

    r = np.where(valid_mask, r, np.nan)
    g = np.where(valid_mask, g, np.nan)
    b = np.where(valid_mask, b, np.nan)

    # 归一化到 0~1
    r_n = robust_normalize(r, valid_mask)
    g_n = robust_normalize(g, valid_mask)
    b_n = robust_normalize(b, valid_mask)

    # =========================
    # 3. 亮度特征与阈值计算
    # =========================
    gray = 0.299 * r_n + 0.587 * g_n + 0.114 * b_n

    rgb_uint8 = np.dstack([
        (r_n * 255).astype(np.uint8),
        (g_n * 255).astype(np.uint8),
        (b_n * 255).astype(np.uint8)
    ])
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    dark_score = 0.5 * gray + 0.5 * v

    # Otsu 自动阈值
    th_auto = threshold_otsu(dark_score[valid_mask])
    th_dark = th_auto * dark_factor

    # =========================
    # 4. RGB 植被指标与纹理
    # =========================
    exg = 2 * g_n - r_n - b_n
    vari = (g_n - r_n) / (g_n + r_n - b_n + 1e-6)

    gray_u8 = (gray * 255).astype(np.uint8)
    gray_blur = cv2.GaussianBlur(gray_u8, (9, 9), 0)
    gray_sq_blur = cv2.GaussianBlur((gray_u8.astype(np.float32) ** 2), (9, 9), 0)
    local_std = np.sqrt(np.clip(gray_sq_blur - gray_blur.astype(np.float32) ** 2, 0, None))
    local_std = local_std / (np.nanmax(local_std[valid_mask]) + 1e-6)

    # =========================
    # 5. 初始 gap 判定
    # =========================
    gap_init = (
        (dark_score < th_dark) &
        (vari < vari_max_for_gap) &
        (exg < exg_max_for_gap) &
        (local_std < 0.35) &
        valid_mask
    )

    # 面积换算
    pixel_size_x = abs(transform.a)
    pixel_size_y = abs(transform.e)
    pixel_area_m2 = pixel_size_x * pixel_size_y
    min_pixels = max(1, int(round(min_gap_area_m2 / pixel_area_m2)))

    # =========================
    # 6. 形态学处理与连通域过滤
    # =========================
    gap_mask = gap_init.copy()
    gap_mask = binary_opening(gap_mask, disk(open_radius))
    gap_mask = binary_closing(gap_mask, disk(close_radius))
    gap_mask = remove_small_objects(gap_mask, min_size=min_pixels)
    gap_mask = remove_small_holes(gap_mask, area_threshold=min_pixels // 2)

    labels = label(gap_mask)
    filtered = np.zeros_like(gap_mask, dtype=np.uint8)

    for region in regionprops(labels):
        if region.area >= min_pixels:
            coords = region.coords
            filtered[coords[:, 0], coords[:, 1]] = 1

    gap_ratio = filtered.sum() / (valid_mask.sum() + 1e-6)
    print(f"林窗比例 (Gap Ratio): {gap_ratio:.4f}")

    # =========================
    # 7. 保存 GeoTIFF
    # =========================
    profile.update(
        count=1,
        dtype=rasterio.uint8,
        compress='lzw',
        nodata=0
    )
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_tif), exist_ok=True)
    
    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(filtered.astype(np.uint8), 1)
    print(f"掩膜结果已输出至: {output_tif}")

    # =========================
    # 8. 可视化生成
    # =========================
    r_vis = np.nan_to_num(r_n, nan=0.0)
    g_vis = np.nan_to_num(g_n, nan=0.0)
    b_vis = np.nan_to_num(b_n, nan=0.0)

    rgb_vis = np.dstack([
        (r_vis * 255).astype(np.uint8),
        (g_vis * 255).astype(np.uint8),
        (b_vis * 255).astype(np.uint8)
    ])
    overlay = rgb_vis.copy()

    # 处理 GT 的逻辑：如果是空字符串或None，则跳过
    if gt_tif and os.path.exists(gt_tif):
        with rasterio.open(gt_tif) as gt_src:
            gt = gt_src.read(1)
            gt_mask = gt > 0

        pred_mask = filtered.astype(bool)

        tp = pred_mask & gt_mask
        fp = pred_mask & (~gt_mask)
        fn = (~pred_mask) & gt_mask

        # TP: 绿色, FP: 红色, FN: 蓝色
        overlay[tp] = [0, 255, 0]
        overlay[fp] = [255, 0, 0]
        overlay[fn] = [0, 0, 255]

        iou = tp.sum() / (tp.sum() + fp.sum() + fn.sum() + 1e-6)
        precision = tp.sum() / (tp.sum() + fp.sum() + 1e-6)
        recall = tp.sum() / (tp.sum() + fn.sum() + 1e-6)

        print(f"评估指标 -> IoU: {iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    else:
        # 没有 GT 时，仅把预测林窗标红
        overlay[filtered == 1] = [255, 0, 0]

    os.makedirs(os.path.dirname(output_vis), exist_ok=True)
    cv2.imwrite(output_vis, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"可视化叠加图已输出至: {output_vis}")
    
    return {"gap_ratio": gap_ratio, "output_tif": output_tif}


# =========================
# 本地测试入口 (硬编码测试)
# =========================
if __name__ == "__main__":
    test_input = r"K:\ssq\data\HSTAC\Tiantongshan_DOM.tif"
    test_out_tif = r"K:\ssq\data\HSTAC\Tiantongshan_gap_result2.tif"
    test_out_vis = r"K:\ssq\data\HSTAC\Tiantongshan_gap_overlay2.png"
    
    # 你可以在这里独立测试你的脚本
    run_gap_extraction(
        input_tif=test_input,
        output_tif=test_out_tif,
        output_vis=test_out_vis,
        gt_tif=None,
        min_gap_area_m2=5.0
    )
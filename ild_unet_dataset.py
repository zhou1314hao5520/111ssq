import rasterio
import numpy as np
import os
import cv2

def generate_patches(rgb_tif, mask_tif, out_dir, patch_size=256, stride=256, empty_threshold=0.001):
    """
    将大尺寸的遥感影像和林窗掩膜切分为用于 U-Net 训练的 Patch。
    
    参数:
        rgb_tif (str): 输入 RGB 影像路径 (.tif)
        mask_tif (str): 输入二值化掩膜路径 (.tif)
        out_dir (str): 数据集输出根目录
        patch_size (int): 切片大小 (默认 256)
        stride (int): 步长 (默认 256)
        empty_threshold (float): 空白或无林窗背景的过滤阈值 (默认 0.001)
    """
    print(f"开始构建数据集...\nRGB影像: {rgb_tif}\nMask掩膜: {mask_tif}")

    # ==============================
    # 创建目录
    # ==============================
    img_dir = os.path.join(out_dir, "images")
    mask_dir = os.path.join(out_dir, "masks")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    # ==============================
    # 读取影像
    # ==============================
    try:
        with rasterio.open(rgb_tif) as src:
            rgb = src.read([1, 2, 3])
            rgb = np.transpose(rgb, (1, 2, 0))
            
        with rasterio.open(mask_tif) as src:
            mask = src.read(1)
    except Exception as e:
        print(f"读取影像失败，请检查文件格式或路径: {e}")
        raise e

    H, W, _ = rgb.shape
    count = 0

    # ==============================
    # 切 patch
    # ==============================
    print(f"大图尺寸: {W}x{H}, 切片大小: {patch_size}, 滑动步长: {stride}")
    
    # 加上 +1 确保如果整除的话不会漏掉最后一行/列
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):

            img_patch = rgb[i:i+patch_size, j:j+patch_size]
            mask_patch = mask[i:i+patch_size, j:j+patch_size]

            # 去掉空 patch (掩膜中几乎没有林窗的区域)
            if np.mean(mask_patch) < empty_threshold:
                continue

            img_name = f"{count:05d}.png"

            # OpenCV 保存要求 BGR 格式，需要做颜色空间转换
            cv2.imwrite(os.path.join(img_dir, img_name),
                        cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))

            # 掩膜保存 (原掩膜假设为0和1，保存为0和255的灰度图方便直接查看)
            cv2.imwrite(os.path.join(mask_dir, img_name),
                        mask_patch * 255)

            count += 1

    print("-" * 30)
    print(f"数据集构建完成！共生成包含有效目标的 patch 数量: {count}")
    print(f"输出目录: {out_dir}")
    print("-" * 30)
    
    return count

# ==============================
# 本地测试入口 (硬编码测试)
# ==============================
if __name__ == "__main__":
    test_rgb_tif = r"K:\ssq\data\HSTAC\Tiantongshan_DOM.tif"
    test_mask_tif = r"K:\ssq\data\HSTAC\Tiantongshan_gap_result.tif"
    test_out_dir = r"K:\ssq\data\Tiantongshan_dataset"
    
    # 你可以在这里独立测试你的脚本
    generate_patches(
        rgb_tif=test_rgb_tif,
        mask_tif=test_mask_tif,
        out_dir=test_out_dir,
        patch_size=256,
        stride=256
    )
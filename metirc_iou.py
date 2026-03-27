import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.errors import ShapelyDeprecationWarning
from shapely.ops import unary_union

# 忽略常见警告，保持控制台输出干净
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _safe_div(numerator, denominator, eps=1e-8):
    return numerator / (denominator + eps)


def _safe_union(geometries):
    try:
        return geometries.union_all()
    except AttributeError:
        return unary_union(geometries)


def _calculate_vector_metrics(gt_shp, pred_shp):
    gt_gdf = gpd.read_file(gt_shp)
    pred_gdf = gpd.read_file(pred_shp)

    if gt_gdf.empty or pred_gdf.empty:
        raise ValueError("输入矢量为空，无法计算指标。")

    if gt_gdf.crs != pred_gdf.crs:
        print("坐标系不一致，正在转换预测结果坐标系以匹配真实标签...")
        pred_gdf = pred_gdf.to_crs(gt_gdf.crs)

    gt_gdf = gt_gdf[gt_gdf.geometry.notnull()].copy()
    pred_gdf = pred_gdf[pred_gdf.geometry.notnull()].copy()
    if gt_gdf.empty or pred_gdf.empty:
        raise ValueError("输入矢量包含空几何，清理后无可用对象。")

    gt_gdf["gt_id"] = gt_gdf.index
    pred_gdf["pred_id"] = pred_gdf.index

    print("正在进行空间连接 (Spatial Join)...")
    joined = gpd.sjoin(gt_gdf[["gt_id", "geometry"]], pred_gdf[["pred_id", "geometry"]], how="left", predicate="intersects")

    matched_pred_ids = joined["pred_id"].dropna().astype(int).unique()
    filtered_pred_gdf = pred_gdf.loc[matched_pred_ids] if len(matched_pred_ids) > 0 else pred_gdf.iloc[0:0].copy()
    deleted_count = len(pred_gdf) - len(filtered_pred_gdf)
    print(f"已自动删除 {deleted_count} 个完全不相交的预测多余林窗。")

    gt_union = _safe_union(gt_gdf.geometry)
    pred_union = _safe_union(filtered_pred_gdf.geometry) if not filtered_pred_gdf.empty else None

    total_gt_area = gt_union.area
    total_pred_area = 0.0 if pred_union is None else pred_union.area

    if pred_union is None:
        total_tp_area = 0.0
        total_fp_area = 0.0
        total_fn_area = total_gt_area
    else:
        # 用几何集合运算计算 TP/FP/FN，避免逐目标重复累计造成精度偏差。
        total_tp_area = gt_union.intersection(pred_union).area
        total_fp_area = pred_union.difference(gt_union).area
        total_fn_area = gt_union.difference(pred_union).area

    precision = _safe_div(total_tp_area, total_tp_area + total_fp_area)
    recall = _safe_div(total_tp_area, total_tp_area + total_fn_area)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    iou = _safe_div(total_tp_area, total_tp_area + total_fp_area + total_fn_area)

    summary_data = {
        "Metrics (指标)": [
            "Total Ground Truth Area (㎡)",
            "Filtered Predicted Area (过滤后预测面积, ㎡)",
            "True Positive Area (TP, ㎡)",
            "False Positive Area (FP, ㎡)",
            "False Negative Area (FN, ㎡)",
            "Precision (精确率)",
            "Recall (召回率)",
            "F1-Score",
            "Matched Global IoU (匹配对象全局交并比)",
        ],
        "Value (数值)": [
            round(total_gt_area, 2),
            round(total_pred_area, 2),
            round(total_tp_area, 2),
            round(total_fp_area, 2),
            round(total_fn_area, 2),
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            round(iou, 4),
        ],
    }
    return pd.DataFrame(summary_data)


def _calculate_raster_metrics(gt_tif, pred_tif):
    with rasterio.open(gt_tif) as gt_src:
        gt = gt_src.read(1)
    with rasterio.open(pred_tif) as pred_src:
        pred = pred_src.read(1)

    if gt.shape != pred.shape:
        raise ValueError(f"栅格尺寸不一致: GT={gt.shape}, Pred={pred.shape}")

    gt_bin = gt > 0
    pred_bin = pred > 0

    tp = float(np.logical_and(gt_bin, pred_bin).sum())
    fp = float(np.logical_and(~gt_bin, pred_bin).sum())
    fn = float(np.logical_and(gt_bin, ~pred_bin).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    iou = _safe_div(tp, tp + fp + fn)

    summary_data = {
        "Metrics (指标)": [
            "GT Positive Pixels (像素)",
            "Pred Positive Pixels (像素)",
            "True Positive Pixels (TP)",
            "False Positive Pixels (FP)",
            "False Negative Pixels (FN)",
            "Precision (精确率)",
            "Recall (召回率)",
            "F1-Score",
            "Global IoU",
        ],
        "Value (数值)": [
            int(gt_bin.sum()),
            int(pred_bin.sum()),
            int(tp),
            int(fp),
            int(fn),
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            round(iou, 4),
        ],
    }
    return pd.DataFrame(summary_data)

# ==============================
# 核心指标计算函数 (供外部或 GUI 调用)
# ==============================
def calculate_global_summary_matched(gt_shp, pred_shp, output_csv):
    """
    计算基于匹配对象（过滤孤岛预测）的全局交并比和各项评价指标。
    
    参数:
        gt_shp (str): 真实标签 Shapefile 路径
        pred_shp (str): 预测结果 Shapefile 路径
        output_csv (str): 汇总表输出路径 (.csv)
    """
    print(f"正在读取数据，准备计算全局汇总指标...\n真实标签: {gt_shp}\n预测结果: {pred_shp}")

    vector_ext = {".shp", ".gpkg", ".geojson", ".json"}
    raster_ext = {".tif", ".tiff"}
    gt_ext = os.path.splitext(gt_shp)[1].lower()
    pred_ext = os.path.splitext(pred_shp)[1].lower()

    if gt_ext in vector_ext and pred_ext in vector_ext:
        df_summary = _calculate_vector_metrics(gt_shp, pred_shp)
        done_msg = "过滤孤岛后的论文级汇总表计算完成！"
    elif gt_ext in raster_ext and pred_ext in raster_ext:
        print("检测到栅格输入，按像素级 TP/FP/FN 计算全局指标。")
        df_summary = _calculate_raster_metrics(gt_shp, pred_shp)
        done_msg = "栅格全局汇总指标计算完成！"
    else:
        raise ValueError("GT 与预测文件类型不一致或不受支持。请同时使用 shp/gpkg/geojson 或 tif/tiff。")

    df_summary.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 50)
    print(f"🔥 {done_msg}")
    print("=" * 50)
    print(df_summary.to_string(index=False))
    print("=" * 50)
    print(f"指标结果已保存至: {output_csv}")


# ==============================
# 本地测试入口 (硬编码测试)
# ==============================
if __name__ == "__main__":
    # 在这里填写你本地的测试路径
    test_gt_shp = r"J:\ssq\data\HSTAC\tiantongshan\shp\gap_shapes_merged.shp"
    test_pred_shp = r"J:\ssq\data\HSTAC\tiantongshan\shp\gap_pre-shapes_merged.shp"
    test_output_csv = r"J:\ssq\data\HSTAC\tiantongshan\shp\Paper_Global_Summary_Matched_Only4-2.csv"
    
    try:
        calculate_global_summary_matched(test_gt_shp, test_pred_shp, test_output_csv)
    except Exception as e:
        print(f"运行失败: {e}")
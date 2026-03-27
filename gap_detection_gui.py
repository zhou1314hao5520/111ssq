# ForestMetrics Studio v1.0 - 林窗检测模块
import sys
import os
from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QTabWidget,
                             QWidget, QFormLayout, QLineEdit, QPushButton,
                             QHBoxLayout, QFileDialog, QSpinBox, QDoubleSpinBox,
                             QMessageBox, QLabel, QComboBox)
from PyQt5.QtCore import Qt

# 导入所有核心逻辑模块
import gap_label
import ild_unet_dataset
import train_V5
import infer_alltif
import metirc_iou


class GapDetectionModule(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("林窗检测分析模块 (Gap Detection) - ForestMetrics Studio v1.0")
        self.resize(1100, 760)
        self.setMinimumSize(980, 680)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)

        self.tabs.addTab(self.create_label_tab(), "1. 标签制作 (Labeling)")
        self.tabs.addTab(self.create_dataset_tab(), "2. 数据集构建 (Dataset)")
        self.tabs.addTab(self.create_train_tab(), "3. 模型训练 (Training)")
        self.tabs.addTab(self.create_infer_tab(), "4. 推理与可视化 (Inference)")
        self.tabs.addTab(self.create_metric_tab(), "5. 精度评价 (Metrics)")

        layout.addWidget(self.tabs)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        close_btn = QPushButton("关闭 (Close)")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def create_file_picker(self, label_text, is_dir=False, is_save=False):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setFixedWidth(170)
        line_edit = QLineEdit()
        line_edit.setMinimumWidth(560)
        btn = QPushButton("浏览...")
        btn.setFixedWidth(90)

        def pick_path():
            if is_dir:
                path = QFileDialog.getExistingDirectory(self, "选择文件夹")
            elif is_save:
                path, _ = QFileDialog.getSaveFileName(self, "保存文件")
            else:
                path, _ = QFileDialog.getOpenFileName(self, "选择文件")
            if path:
                line_edit.setText(path)

        btn.clicked.connect(pick_path)
        layout.addWidget(label)
        layout.addWidget(line_edit)
        layout.addWidget(btn)
        return layout, line_edit

    # Tab 1: 标签制作
    def create_label_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.lbl_in_tif_layout, self.lbl_in_tif = self.create_file_picker("输入 DOM (.tif):")
        self.lbl_out_tif_layout, self.lbl_out_tif = self.create_file_picker("输出结果 (.tif):", is_save=True)
        self.lbl_gt_tif_layout, self.lbl_gt_tif = self.create_file_picker("参考 GT (.tif, 可选):")

        layout.addLayout(self.lbl_in_tif_layout)
        layout.addLayout(self.lbl_out_tif_layout)
        layout.addLayout(self.lbl_gt_tif_layout)

        form = QFormLayout()
        self.lbl_min_area = QDoubleSpinBox();
        self.lbl_min_area.setValue(5.0);
        self.lbl_min_area.setSuffix(" m²")
        self.lbl_dark_factor = QDoubleSpinBox();
        self.lbl_dark_factor.setValue(0.95);
        self.lbl_dark_factor.setSingleStep(0.01)
        self.lbl_open_radius = QSpinBox();
        self.lbl_open_radius.setValue(2)
        self.lbl_close_radius = QSpinBox();
        self.lbl_close_radius.setValue(4)

        form.addRow("最小林窗面积:", self.lbl_min_area)
        form.addRow("Otsu暗度阈值系数:", self.lbl_dark_factor)
        form.addRow("开运算半径 (Pixel):", self.lbl_open_radius)
        form.addRow("闭运算半径 (Pixel):", self.lbl_close_radius)
        layout.addLayout(form)

        run_btn = QPushButton("执行无监督标签提取")
        run_btn.clicked.connect(self.run_labeling)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        return widget

    # Tab 2: 数据集构建
    def create_dataset_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.ds_rgb_layout, self.ds_rgb = self.create_file_picker("RGB 影像 (.tif):")
        self.ds_mask_layout, self.ds_mask = self.create_file_picker("掩膜 Mask (.tif):")
        self.ds_out_layout, self.ds_out = self.create_file_picker("输出根目录:", is_dir=True)

        layout.addLayout(self.ds_rgb_layout)
        layout.addLayout(self.ds_mask_layout)
        layout.addLayout(self.ds_out_layout)

        form = QFormLayout()
        self.ds_patch = QSpinBox();
        self.ds_patch.setMaximum(2048);
        self.ds_patch.setValue(256)
        self.ds_stride = QSpinBox();
        self.ds_stride.setMaximum(2048);
        self.ds_stride.setValue(256)

        form.addRow("切片大小 (Patch Size):", self.ds_patch)
        form.addRow("滑动步长 (Stride):", self.ds_stride)
        layout.addLayout(form)

        run_btn = QPushButton("生成训练切片")
        run_btn.clicked.connect(self.run_dataset_gen)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        return widget

    # Tab 3: 模型训练
    def create_train_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.tr_img_layout, self.tr_img = self.create_file_picker("训练图像目录:", is_dir=True)
        self.tr_mask_layout, self.tr_mask = self.create_file_picker("训练掩膜目录:", is_dir=True)
        self.tr_out_layout, self.tr_out = self.create_file_picker("模型输出目录:", is_dir=True)

        layout.addLayout(self.tr_img_layout)
        layout.addLayout(self.tr_mask_layout)
        layout.addLayout(self.tr_out_layout)

        form = QFormLayout()
        self.tr_epochs = QSpinBox();
        self.tr_epochs.setMaximum(1000);
        self.tr_epochs.setValue(60)
        self.tr_batch = QSpinBox();
        self.tr_batch.setValue(8)
        self.tr_lr = QDoubleSpinBox();
        self.tr_lr.setDecimals(5);
        self.tr_lr.setValue(0.0002);
        self.tr_lr.setSingleStep(0.0001)

        form.addRow("训练轮数 (Epochs):", self.tr_epochs)
        form.addRow("批次大小 (Batch Size):", self.tr_batch)
        form.addRow("学习率 (Learning Rate):", self.tr_lr)
        layout.addLayout(form)

        run_btn = QPushButton("启动模型训练")
        run_btn.clicked.connect(self.run_training)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        return widget

    # Tab 4: 推理与分析
    def create_infer_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.inf_model_layout, self.inf_model = self.create_file_picker("预训练权重 (.pth):")
        self.inf_tif_layout, self.inf_tif = self.create_file_picker("待预测影像 (.tif):")
        self.inf_gt_layout, self.inf_gt = self.create_file_picker("真实值影像 (跨林型对比可选):")
        self.inf_out_layout, self.inf_out = self.create_file_picker("预测输出 (.tif):", is_save=True)

        layout.addLayout(self.inf_model_layout)
        layout.addLayout(self.inf_tif_layout)
        layout.addLayout(self.inf_gt_layout)
        layout.addLayout(self.inf_out_layout)

        form = QFormLayout()
        self.inf_tile = QSpinBox();
        self.inf_tile.setMaximum(2048);
        self.inf_tile.setValue(512)
        self.inf_stride = QSpinBox();
        self.inf_stride.setMaximum(2048);
        self.inf_stride.setValue(384)
        self.inf_thresh = QDoubleSpinBox();
        self.inf_thresh.setValue(0.35);
        self.inf_thresh.setSingleStep(0.05)

        form.addRow("推理切片大小 (Tile Size):", self.inf_tile)
        form.addRow("推理步长 (Stride):", self.inf_stride)
        form.addRow("二值化阈值 (Threshold):", self.inf_thresh)
        layout.addLayout(form)

        run_btn = QPushButton("执行全图推理与可视化")
        run_btn.clicked.connect(self.run_inference)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        return widget

    # Tab 5: 精度评价
    def create_metric_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        self.met_gt_layout, self.met_gt = self.create_file_picker("真实值 (.shp/.tif):")
        self.met_pred_layout, self.met_pred = self.create_file_picker("预测值 (.shp/.tif):")
        self.met_out_layout, self.met_out = self.create_file_picker("输出统计结果 (.csv):", is_save=True)

        layout.addLayout(self.met_gt_layout)
        layout.addLayout(self.met_pred_layout)
        layout.addLayout(self.met_out_layout)

        run_btn = QPushButton("计算全局匹配指标")
        run_btn.clicked.connect(self.run_metrics)
        layout.addWidget(run_btn)
        layout.addStretch(1)
        return widget

    # ==========================
    # 执行槽函数映射 (核心逻辑已补全)
    # ==========================
    def run_labeling(self):
        input_tif = self.lbl_in_tif.text()
        output_tif = self.lbl_out_tif.text()
        if not input_tif or not output_tif:
            QMessageBox.warning(self, "警告", "请确保已选择输入和输出路径！")
            return

        # 自动生成可视化路径
        out_vis = output_tif.replace('.tif', '_vis.png')
        gt_tif = self.lbl_gt_tif.text() if self.lbl_gt_tif.text() else None

        try:
            print(f"正在启动标签提取... 面积阈值: {self.lbl_min_area.value()}")
            gap_label.run_gap_extraction(
                input_tif=input_tif,
                output_tif=output_tif,
                output_vis=out_vis,
                gt_tif=gt_tif,
                min_gap_area_m2=self.lbl_min_area.value(),
                open_radius=self.lbl_open_radius.value(),
                close_radius=self.lbl_close_radius.value(),
                dark_factor=self.lbl_dark_factor.value()
            )
            QMessageBox.information(self, "成功", "标签制作及可视化生成完成！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"标签提取失败:\n{str(e)}")

    def run_dataset_gen(self):
        if not self.ds_rgb.text() or not self.ds_mask.text() or not self.ds_out.text():
            QMessageBox.warning(self, "警告", "请补全数据集路径配置！")
            return

        try:
            print("正在构建数据集...")
            count = ild_unet_dataset.generate_patches(
                rgb_tif=self.ds_rgb.text(),
                mask_tif=self.ds_mask.text(),
                out_dir=self.ds_out.text(),
                patch_size=self.ds_patch.value(),
                stride=self.ds_stride.value()
            )
            QMessageBox.information(self, "成功", f"数据集切片构建完成！\n共生成 {count} 张有效样本。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"数据集构建失败:\n{str(e)}")

    def run_training(self):
        if not self.tr_img.text() or not self.tr_mask.text() or not self.tr_out.text():
            QMessageBox.warning(self, "警告", "请提供完整的训练数据及输出目录！")
            return

        if os.path.isfile(self.tr_out.text()):
            QMessageBox.warning(self, "警告", f"模型输出路径 '{self.tr_out.text()}' 是一个文件，而非目录。\n请选择或新建一个文件夹作为输出目录！")
            return

        try:
            print("开始训练（单次训练模式）...")
            # 注意：训练可能耗时较长，会导致 GUI 假死。生产环境中建议用 QThread 包装，这里直接调用
            train_V5.main_train(
                img_dir=self.tr_img.text(),
                mask_dir=self.tr_mask.text(),
                output_dir=self.tr_out.text(),
                epochs=self.tr_epochs.value(),
                batch_size=self.tr_batch.value(),
                lr=self.tr_lr.value()
            )
            QMessageBox.information(self, "成功", "模型训练流程已结束，请前往控制台查看详情！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"训练进程发生异常:\n{str(e)}")

    def run_inference(self):
        if not self.inf_model.text() or not self.inf_tif.text() or not self.inf_out.text():
            QMessageBox.warning(self, "警告", "缺少必要的推理文件或输出路径！")
            return

        gt_path = self.inf_gt.text() if self.inf_gt.text() else None

        try:
            print("启动大图推理...")
            infer_alltif.predict_large_tif(
                model_path=self.inf_model.text(),
                tif_path=self.inf_tif.text(),
                output_tif=self.inf_out.text(),
                gt_path=gt_path,
                tile_size=self.inf_tile.value(),
                stride=self.inf_stride.value(),
                threshold=self.inf_thresh.value()
            )
            QMessageBox.information(self, "成功", "全图推理与可视化分析图表生成完成！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"推理过程失败:\n{str(e)}")

    def run_metrics(self):
        if not self.met_gt.text() or not self.met_pred.text() or not self.met_out.text():
            QMessageBox.warning(self, "警告", "需要提供 GT、预测值文件（.shp/.tif）以及 CSV 输出路径！")
            return

        try:
            print("计算全局匹配指标...")
            metirc_iou.calculate_global_summary_matched(
                gt_shp=self.met_gt.text(),
                pred_shp=self.met_pred.text(),
                output_csv=self.met_out.text()
            )
            QMessageBox.information(self, "成功", "过滤孤岛后的论文级汇总表计算完成！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"精度计算失败:\n{str(e)}")
import os
import glob
import random

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from skimage.measure import label, regionprops

# 确保当前目录下有 model.py
from model import AttentionUNet

# =========================
# 随机种子设定
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# Dataset 数据集定义
# =========================
class GapDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        self.img_paths = img_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        img_ori = img.copy()
        mask_ori = mask.copy()

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        img = np.transpose(img, (2, 0, 1))
        mask = np.expand_dims(mask, 0)

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            torch.tensor(img_ori),
            torch.tensor(mask_ori),
            os.path.basename(img_path)
        )

# =========================
# 损失函数 (Loss)
# =========================
def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

class BCEDiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = dice_loss(pred, target)
        pt = pred * target + (1 - pred) * (1 - target)
        focal = self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)
        return bce + dice - focal.mean()

# =========================
# Gap 统计与可视化辅助函数
# =========================
def gap_statistics(pred):
    labels = label(pred)
    areas = [r.area for r in regionprops(labels)]
    gap_ratio = pred.sum() / pred.size
    return gap_ratio, areas

def save_visualization(img, gt, pred, save_path):
    if torch.is_tensor(img): img = img.cpu().numpy()
    if torch.is_tensor(gt): gt = gt.cpu().numpy()
    if img.shape[0] == 3: img = np.transpose(img, (1, 2, 0))

    img = img.astype(np.uint8)
    overlay = img.copy()
    overlay[pred == 1] = [255, 0, 0]
    overlay[gt == 1] = [0, 255, 0]

    fig, ax = plt.subplots(1, 4, figsize=(12, 4))
    ax[0].imshow(img); ax[0].set_title("Image")
    ax[1].imshow(gt, cmap="gray"); ax[1].set_title("GT")
    ax[2].imshow(pred, cmap="gray"); ax[2].set_title("Pred")
    ax[3].imshow(overlay); ax[3].set_title("Overlay")

    for a in ax: a.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_gap_plots(areas, ratios, output_dir, epoch):
    if len(areas) == 0: return
    areas = np.array(areas)

    # Histogram
    plt.hist(areas, bins=30)
    plt.xlabel("Gap Area")
    plt.ylabel("Frequency")
    plt.title("Gap Size Distribution")
    plt.savefig(os.path.join(output_dir, f"gap_hist_epoch{epoch}.png"))
    plt.close()

    # CDF
    sorted_area = np.sort(areas)
    cdf = np.arange(len(areas)) / float(len(areas))
    plt.plot(sorted_area, cdf)
    plt.xlabel("Gap Area")
    plt.ylabel("CDF")
    plt.title("Gap Size CDF")
    plt.savefig(os.path.join(output_dir, f"gap_cdf_epoch{epoch}.png"))
    plt.close()

    # Ratio curve
    plt.plot(ratios)
    plt.xlabel("Epoch")
    plt.ylabel("Gap Ratio")
    plt.savefig(os.path.join(output_dir, "gap_ratio_curve.png"))
    plt.close()

# =========================
# 核心训练控制函数 (供外部或 GUI 调用)
# =========================
def main_train(
    img_dir,
    mask_dir,
    model_save_path,
    epochs=60,
    batch_size=8,
    lr=2e-4
):
    """
    启动模型训练流程。
    model_save_path: 最佳模型的保存路径，必须以 .pth 结尾，例如 D:/output/best_model.pth
    所有其他训练产物（断点、TensorBoard 日志、可视化图等）保存在同一目录下。
    """
    seed_everything(42)

    if not model_save_path.endswith(".pth"):
        raise ValueError(f"模型保存路径 '{model_save_path}' 必须以 .pth 结尾，例如: D:/output/best_model.pth")

    # 以模型文件所在目录作为所有训练产物的工作目录
    exp_dir = os.path.dirname(os.path.abspath(model_save_path))
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print("🚀 启动训练任务 | 单次训练模式")
    print(f"📁 图像目录: {img_dir}")
    print(f"💾 最佳模型将保存至: {model_save_path}")
    print(f"⚙️ 参数: Epochs={epochs}, BatchSize={batch_size}, LR={lr}")
    print(f"{'='*50}\n")

    writer = SummaryWriter(os.path.join(exp_dir, "tensorboard"))
    # 断点续训检查点（与最佳模型文件放在同一目录下）
    model_path = os.path.join(exp_dir, "model_checkpoint.pth")

    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    masks = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    if len(imgs) == 0 or len(imgs) != len(masks):
        raise ValueError(f"数据加载异常: 找到 {len(imgs)} 张图像和 {len(masks)} 张掩膜，请检查路径。")

    data = list(zip(imgs, masks))
    random.shuffle(data)
    imgs, masks = zip(*data)

    split = int(len(imgs) * 0.8)
    train_dataset = GapDataset(imgs[:split], masks[:split])
    val_dataset = GapDataset(imgs[split:], masks[split:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=max(1, batch_size // 2))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ 训练设备: {device}")

    model = AttentionUNet().to(device)
    criterion = BCEDiceFocalLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    start_epoch = 0
    best_loss = float('inf')

    # 支持断点续训
    if os.path.exists(model_path):
        print(f"发现已有检查点，尝试恢复训练...")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        print(f"✅ 从 Epoch {start_epoch} 继续训练")

    # 固定可视化样本
    vis_indices = random.sample(range(len(val_dataset)), min(100, len(val_dataset)))
    ratio_history = []

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0

        # 添加 TQDM 进度条
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train")
        for imgs_b, masks_b, _, _, _ in pbar:
            imgs_b = imgs_b.to(device)
            masks_b = masks_b.to(device)

            optimizer.zero_grad()
            outputs = model(imgs_b)
            loss = criterion(outputs, masks_b)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0
        ratios = []
        areas_all = []

        epoch_vis_dir = os.path.join(exp_dir, f"epoch_{epoch}")
        os.makedirs(epoch_vis_dir, exist_ok=True)

        with torch.no_grad():
            for i, (imgs_b, masks_b, img_ori, mask_ori, names) in enumerate(tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val")):
                imgs_b = imgs_b.to(device)
                masks_b = masks_b.to(device)

                outputs = model(imgs_b)
                loss = criterion(outputs, masks_b)
                val_loss += loss.item()

                for b in range(imgs_b.size(0)):
                    global_idx = i * val_loader.batch_size + b

                    pred = outputs[b, 0].cpu().numpy()
                    pred_bin = (pred > 0.5).astype(np.uint8)

                    gt = mask_ori[b].cpu().numpy()
                    gt = (gt > 127).astype(np.uint8)

                    img_np = img_ori[b].cpu().numpy()

                    ratio, areas = gap_statistics(pred_bin)
                    ratios.append(ratio)
                    areas_all.extend(areas)

                    if global_idx in vis_indices:
                        img_name = names[b]
                        save_visualization(img_np, gt, pred_bin, os.path.join(epoch_vis_dir, img_name))

        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        ratio_history.append(np.mean(ratios))

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metrics/Gap_Ratio", np.mean(ratios), epoch)

        print(f"\n[Epoch {epoch+1}/{epochs} 总结] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Avg Gap Ratio: {np.mean(ratios):.4f}")

        # 保存最优模型（直接写入用户指定的 .pth 文件）
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_loss": best_loss
            }, model_save_path)
            print(f"🌟 发现更优模型，已保存至 {model_save_path}")

        # 实时覆盖保存最新模型
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss
        }, model_path)

        # 保存统计科研图
        save_gap_plots(areas_all, ratio_history, exp_dir, epoch)

    writer.close()
    print(f"\n🎉 训练全部完成！\n最佳模型已保存至: {model_save_path}\n其他产物存放于: {exp_dir}")

# =========================
# 本地测试入口
# =========================
if __name__ == "__main__":
    # 在这里填写你的本地测试路径
    test_img_dir = r"K:\ssq\data\Tiantongshan_datasets_3\images"
    test_mask_dir = r"K:\ssq\data\Tiantongshan_datasets_3\masks"
    test_model_save_path = r"K:\ssq\data\train_output\best_model.pth"

    main_train(
        img_dir=test_img_dir,
        mask_dir=test_mask_dir,
        model_save_path=test_model_save_path,
        epochs=2,         # 测试时设小一点
        batch_size=4,
        lr=2e-4
    )
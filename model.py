import torch
import torch.nn as nn

# =========================
# 双卷积模块 (Double Convolution)
# 作用: U-Net 中每一层特征提取的基础模块
# =========================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


# =========================
# 注意力模块 (Attention Block)
# 作用: 在跳跃连接 (Skip Connection) 处抑制无关背景区域的激活，
#       使模型更聚焦于林窗 (Gap) 的边缘和特征。
# =========================
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        参数:
            F_g: 门控信号特征通道数 (来自解码器的上一层)
            F_l: 局部特征通道数 (来自编码器的跳跃连接)
            F_int: 中间过渡通道数
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True), 
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True), 
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True), 
            nn.BatchNorm2d(1), 
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # 将门控特征与局部特征相加，并通过 ReLU 激活
        psi = self.relu(g1 + x1)
        # 计算注意力权重 (0~1 之间)
        psi_weight = self.psi(psi)
        # 将局部特征与注意力权重相乘，起到特征筛选的作用
        return x * psi_weight


# =========================
# 注意力 U-Net (Attention U-Net)
# =========================
class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        """
        参数:
            in_ch: 输入通道数 (默认为3，对应 RGB。如融合多源数据可修改)
            out_ch: 输出通道数 (默认为1，二分类任务输出林窗概率图)
        """
        super().__init__()
        
        # --- 编码器 (Encoder / Downsampling) ---
        self.dconv_down1 = DoubleConv(in_ch, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- 解码器 (Decoder / Upsampling & Attention) ---
        self.upsample4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=512, F_l=256, F_int=256)
        self.dconv_up3 = DoubleConv(512 + 256, 256)

        self.upsample3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=256, F_l=128, F_int=128)
        self.dconv_up2 = DoubleConv(256 + 128, 128)

        self.upsample2 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=128, F_l=64, F_int=64)
        self.dconv_up1 = DoubleConv(128 + 64, 64)

        # --- 输出层 ---
        # 使用 Sigmoid 将输出映射到 [0, 1] 表示概率
        self.conv_last = nn.Conv2d(64, out_ch, kernel_size=1)

    def forward(self, x):
        # 编码路径
        conv1 = self.dconv_down1(x)
        conv2 = self.dconv_down2(self.maxpool(conv1))
        conv3 = self.dconv_down3(self.maxpool(conv2))
        conv4 = self.dconv_down4(self.maxpool(conv3))

        # 解码路径 4 -> 3
        x4 = self.upsample4(conv4)
        x3 = self.att3(g=x4, x=conv3)
        x3 = self.dconv_up3(torch.cat([x4, x3], dim=1))

        # 解码路径 3 -> 2
        x3_up = self.upsample3(x3)
        x2 = self.att2(g=x3_up, x=conv2)
        x2 = self.dconv_up2(torch.cat([x3_up, x2], dim=1))

        # 解码路径 2 -> 1
        x2_up = self.upsample2(x2)
        x1 = self.att1(g=x2_up, x=conv1)
        x1 = self.dconv_up1(torch.cat([x2_up, x1], dim=1))

        # 最终输出
        out = torch.sigmoid(self.conv_last(x1))
        return out


# ==============================
# 本地验证与张量测试入口
# ==============================
if __name__ == "__main__":
    print("正在实例化 Attention U-Net 模型...")
    # 模拟输入一个 Batch Size 为 2，3通道，256x256 的图像张量
    dummy_input = torch.randn(2, 3, 256, 256)
    
    model = AttentionUNet(in_ch=3, out_ch=1)
    
    # 验证前向传播
    try:
        output = model(dummy_input)
        print(f"✅ 模型前向传播测试通过！")
        print(f"输入张量形状 (B, C, H, W): {dummy_input.shape}")
        print(f"输出张量形状 (B, C, H, W): {output.shape}")
        
        # 计算模型参数总量
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型可训练参数总计: {total_params:,} 个")
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
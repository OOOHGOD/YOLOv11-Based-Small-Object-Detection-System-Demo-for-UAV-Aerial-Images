import torch.nn as nn


# 深度可分离卷积 (Depthwise Separable Convolution)
# 由两步组成：
# 1. 深度卷积 (Depthwise Convolution)：每个输入通道单独进行卷积操作
# 2. 逐点卷积 (Pointwise Convolution)：使用1x1卷积融合通道信息


class DWSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        """
        初始化深度可分离卷积层
        :param in_ch: 输入通道数
        :param out_ch: 输出通道数
        :param kernel_size: 卷积核大小
        :param stride: 步长
        :param padding: 填充大小
        """
        super().__init__()

        # 深度卷积：每个输入通道分别进行卷积，groups=in_channels 表示深度卷积
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch  # 实现深度卷积
        )

        # 逐点卷积：1x1 卷积用于合并通道，提升通道维度到 out_channels
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,  # 使用 1x1 卷积
            stride=1,  # 固定步长 1，只做映射不降维
            padding=0,  # 不需要填充
            groups=1  # 普通的1x1卷积
        )

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入张量 (N, C_in, H, W)
        :return: 输出张量 (N, C_out, H_new, W_new)
        """
        # 先经过深度卷积，保持通道数不变，提取空间特征
        out = self.depthwise_conv(x)

        # 再通过1x1卷积，调整输出通道数量
        out = self.pointwise_conv(out)

        return out

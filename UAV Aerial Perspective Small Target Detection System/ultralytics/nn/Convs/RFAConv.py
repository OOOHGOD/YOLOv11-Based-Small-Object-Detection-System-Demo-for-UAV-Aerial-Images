from torch import nn
from einops import rearrange

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # 计算实际的卷积核大小
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # 自动计算padding值
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)  # 初始化卷积层
        self.bn = nn.BatchNorm2d(c2)  # 初始化批归一化层
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()  # 初始化激活函数

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))  # 前向传播：卷积 -> 批归一化 -> 激活

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))  # 融合后的前向传播：卷积 -> 激活


class RFAConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size  # 卷积核大小

        # 获取权重的网络结构
        self.get_weight = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),  # 平均池化层
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                      groups=in_channel, bias=False)  # 卷积层，用于生成权重
        )

        # 生成特征的网络结构
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size,
                      padding=kernel_size // 2, stride=stride, groups=in_channel, bias=False),  # 卷积层，用于生成特征
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),  # 批归一化层
            nn.ReLU()  # 激活函数
        )

        # 最终的卷积层
        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)

    def forward(self, x):
        b, c = x.shape[0:2]  # 获取输入张量的批量大小和通道数
        weight = self.get_weight(x)  # 生成权重
        h, w = weight.shape[2:]  # 获取权重张量的高度和宽度
        # 对权重进行reshape并应用softmax
        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)
        # 生成特征并进行reshape
        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h, w)
        # 对特征和权重进行乘法操作
        weighted_data = feature * weighted
        # 使用einops库对张量进行重排
        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              n2=self.kernel_size)
        # 应用最终的卷积层
        return self.conv(conv_data)

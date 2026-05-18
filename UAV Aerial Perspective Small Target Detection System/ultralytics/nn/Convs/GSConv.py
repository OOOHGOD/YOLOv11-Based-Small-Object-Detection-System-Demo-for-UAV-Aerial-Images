import torch
from torch import nn

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Mish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class GSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        初始化GSConv模块
        :param c1: 输入通道数
        :param c2: 输出通道数
        :param k: 卷积核大小
        :param s: 步长
        :param g: 分组卷积的组数
        :param act: 是否使用激活函数
        """
        super().__init__()
        # 将输出通道数分成两半
        c_ = c2 // 2

        # 第一个普通卷积层
        self.cv1 = Conv(c1, c_, k, s, None, g, act)

        # 第二个深度可分离卷积层，卷积核大小为5x5
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        # 第一次卷积操作
        x1 = self.cv1(x)

        # 对第一次卷积的结果进行第二次卷积，并将两个结果在通道维度拼接
        x2 = torch.cat((x1, self.cv2(x1)), 1)

        # 获取张量形状：batch_size, channels, height, width
        b, n, h, w = x2.data.size()

        # 将batch和channel维度合并以便后续处理
        b_n = b * n // 2

        # 改变张量形状并进行维度置换
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        # 最终将两个分组在通道维度拼接，恢复通道顺序
        return torch.cat((y[0], y[1]), 1)

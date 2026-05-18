import torch


class GroupNorm(torch.nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        # 添加卷积层
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        # 添加批量归一化层
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        # 初始化批量归一化层的权重和偏置
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        # 融合卷积层和批量归一化层
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class SHSA(torch.nn.Module):
    """Single-Head Self-Attention"""

    def __init__(self, dim, qk_dim=16, pdim=32):
        super().__init__()
        # 计算缩放因子
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        # 添加预归一化层
        self.pre_norm = GroupNorm(pdim)

        # 添加卷积层用于生成查询、键和值
        self.qkv = Conv2d_BN(pdim, qk_dim * 2 + pdim)
        # 添加投影层
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            dim, dim, bn_weight_init=0))

    def forward(self, x):
        B, C, H, W = x.shape
        # 将输入张量按通道维度拆分为两部分
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=1)
        # 对第一部分应用预归一化
        x1 = self.pre_norm(x1)
        # 生成查询、键和值
        qkv = self.qkv(x1)
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=1)
        q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)

        # 计算注意力分数
        attn = (q.transpose(-2, -1) @ k) * self.scale
        # 应用softmax函数
        attn = attn.softmax(dim=-1)
        # 计算加权和
        x1 = (v @ attn.transpose(-2, -1)).reshape(B, self.pdim, H, W)
        # 将加权和与未处理的部分拼接并应用投影层
        x = self.proj(torch.cat([x1, x2], dim=1))

        return x


if __name__ == '__main__':
    # 创建SHSA模块实例
    block = SHSA(64)  # 输入通道数C

    # 创建随机输入张量
    input = torch.randn(1, 64, 32, 32)  # 输入形状为[B, C, H, W]

    # 打印输入张量的形状
    print(input.size())

    # 通过SHSA模块进行前向传播
    output = block(input)

    # 打印输出张量的形状
    print(output.size())


# 论文：Poly Kernel Inception Network for Remote Sensing Detection(CVPR 2024)
# 论文地址：https://arxiv.org/pdf/2403.06258
# 代码地址：https://github.com/NUST-Machine-Intelligence-Laboratory/PKINet
# Context Anchor Attention (CAA) module

from typing import Optional
import torch.nn as nn
import torch

class ConvModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            norm_cfg: Optional[dict] = None,
            act_cfg: Optional[dict] = None):
        super().__init__()
        layers = []
        # 添加卷积层
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=(norm_cfg is None)))
        # 添加归一化层（如果配置了）
        if norm_cfg:
            norm_layer = self._get_norm_layer(out_channels, norm_cfg)
            layers.append(norm_layer)
        # 添加激活层（如果配置了）
        if act_cfg:
            act_layer = self._get_act_layer(act_cfg)
            layers.append(act_layer)
        # 将所有层组合成一个顺序容器
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播
        return self.block(x)

    def _get_norm_layer(self, num_features, norm_cfg):
        # 根据配置获取归一化层
        if norm_cfg['type'] == 'BN':
            return nn.BatchNorm2d(num_features, momentum=norm_cfg.get('momentum', 0.1), eps=norm_cfg.get('eps', 1e-5))
        # 如果需要，可以添加更多归一化类型
        raise NotImplementedError(f"Normalization layer '{norm_cfg['type']}' is not implemented.")

    def _get_act_layer(self, act_cfg):
        # 根据配置获取激活层
        if act_cfg['type'] == 'ReLU':
            return nn.ReLU(inplace=True)
        if act_cfg['type'] == 'SiLU':
            return nn.SiLU(inplace=True)
        # 如果需要，可以添加更多激活类型
        raise NotImplementedError(f"Activation layer '{act_cfg['type']}' is not implemented.")

class CAA(nn.Module):
    """Context Anchor Attention"""
    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[dict] = dict(type='SiLU')):
        super().__init__()
        # 平均池化层，用于提取上下文信息
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        # 第一个卷积模块
        self.conv1 = ConvModule(channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # 水平方向的深度可分离卷积模块
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1, (0, h_kernel_size // 2), groups=channels, norm_cfg=None, act_cfg=None)
        # 垂直方向的深度可分离卷积模块
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), groups=channels, norm_cfg=None, act_cfg=None)
        # 第二个卷积模块
        self.conv2 = ConvModule(channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        # Sigmoid激活函数，用于生成注意力权重
        self.act = nn.Sigmoid()

    def forward(self, x):
        # 前向传播过程
        # 1. 通过平均池化层提取上下文信息
        # 2. 通过conv1卷积模块
        # 3. 通过h_conv进行水平方向的深度可分离卷积
        # 4. 通过v_conv进行垂直方向的深度可分离卷积
        # 5. 通过conv2卷积模块
        # 6. 通过Sigmoid激活函数生成注意力权重
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor

# 示例用法，打印输入和输出的形状
if __name__ == "__main__":
    input = torch.randn(1, 64, 128, 128) # 输入 B C H W
    block = CAA(64)
    output = block(input)
    print(input.size())
    print(output.size())

# --coding:utf-8--

""" 模型定义模块
包含自定义 CNN、MobileNetV2、ResNet18、EfficientNetV2-S 和 ConvNeXt Tiny 模型
v1.04: 新增模型集成支持
v1.05: 使用 weights=DEFAULT API，新增 SE 注意力、SimpleCNNV2、ResNet50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List


class SELayer(nn.Module):
    """v1.05: Squeeze-and-Excitation 通道注意力模块"""

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SimpleCNN(nn.Module):
    """自定义 CNN 模型"""

    def __init__(self, num_classes=6):
        super(SimpleCNN, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第四个卷积块
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool3(x)

        # 第四个卷积块
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.maxpool4(x)

        # 全局平均池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # 全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ConvBlock(nn.Module):
    """v1.05: 卷积块（Conv + BN + ReLU + SE + 残差连接）"""

    def __init__(self, in_channels, out_channels, se_reduction=16):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(out_channels, reduction=se_reduction)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 残差连接：输入通过 1x1 卷积匹配输出通道数
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.se(out)
        out = self.pool(out)
        # 池化后可能尺寸不匹配，对 identity 也池化
        if identity.shape[2:] != out.shape[2:]:
            identity = F.avg_pool2d(identity, kernel_size=2, stride=2)
        out = out + identity
        return out


class SimpleCNNV2(nn.Module):
    """v1.05: 增强版 SimpleCNN — 添加 SE 注意力和残差连接"""

    def __init__(self, num_classes=6):
        super(SimpleCNNV2, self).__init__()
        # 四个卷积块，每块包含 SE + 残差连接
        self.block1 = ConvBlock(3, 32, se_reduction=8)
        self.block2 = ConvBlock(32, 64, se_reduction=8)
        self.block3 = ConvBlock(64, 128, se_reduction=16)
        self.block4 = ConvBlock(128, 256, se_reduction=16)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MobileNetV2Model(nn.Module):
    """基于 MobileNetV2 的转移学习模型"""

    def __init__(self, num_classes=6, pretrained=True):
        super(MobileNetV2Model, self).__init__()
        # 加载预训练的 MobileNetV2
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.mobilenet = models.mobilenet_v2(weights=weights)
        # 替换分类头
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.mobilenet(x)


class MobileNetV2SEModel(nn.Module):
    """v1.05: MobileNetV2 + SE 通道注意力"""

    def __init__(self, num_classes=6, pretrained=True):
        super(MobileNetV2SEModel, self).__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.mobilenet = models.mobilenet_v2(weights=weights)

        # 在 InvertedResidual 块后插入 SE 层
        features = list(self.mobilenet.features)
        self.features_with_se = nn.ModuleList()
        for i, layer in enumerate(features):
            self.features_with_se.append(layer)
            # 在每个 MBConv 块后插入 SE（跳过开头几个低通道层）
            if isinstance(layer, models.mobilenet.InvertedResidual) and layer.expand_ratio > 1:
                in_channels = layer.conv[-1].out_channels
                se_reduction = max(4, in_channels // 8)
                self.features_with_se.append(SELayer(in_channels, reduction=se_reduction))

        self.features = self.features_with_se

        # 替换分类头
        num_features = self.mobilenet.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = x.mean([2, 3])  # 全局平均池化
        x = self.classifier(x)
        return x


class ResNet18Model(nn.Module):
    """基于 ResNet18 的转移学习模型"""

    def __init__(self, num_classes=6, pretrained=True):
        super(ResNet18Model, self).__init__()
        # 加载预训练的 ResNet18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        # 替换为带 Dropout 和 BN 的分类头
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


class ResNet50Model(nn.Module):
    """v1.05: 基于 ResNet50 的转移学习模型（更深版本）"""

    def __init__(self, num_classes=6, pretrained=True):
        super(ResNet50Model, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet50(weights=weights)
        # 替换为带 Dropout 和 BN 的分类头
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


class EfficientNetV2SModel(nn.Module):
    """基于 EfficientNetV2-S 的转移学习模型（v1.03 新增）"""

    def __init__(self, num_classes=6, pretrained=True):
        super(EfficientNetV2SModel, self).__init__()
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.efficientnet = models.efficientnet_v2_s(weights=weights)
        # 替换分类头
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)


class EfficientNetV2MModel(nn.Module):
    """v1.06: 基于 EfficientNetV2-M 的转移学习模型（54M 参数，更强的特征提取能力）"""

    def __init__(self, num_classes=6, pretrained=True):
        super(EfficientNetV2MModel, self).__init__()
        weights = models.EfficientNet_V2_M_Weights.DEFAULT if pretrained else None
        self.efficientnet = models.efficientnet_v2_m(weights=weights)
        # 替换分类头（使用更高 Dropout 防止过拟合）
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)


class ConvNeXtTinyModel(nn.Module):
    """基于 ConvNeXt Tiny 的转移学习模型（v1.03 新增）"""

    def __init__(self, num_classes=6, pretrained=True):
        super(ConvNeXtTinyModel, self).__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.convnext = models.convnext_tiny(weights=weights)
        # 替换分类头
        num_features = self.convnext.classifier[2].in_features
        self.convnext.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(num_features, eps=1e-6),
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.convnext(x)


class ConvNeXtSmallModel(nn.Module):
    """v1.06: 基于 ConvNeXt Small 的转移学习模型（50M 参数）"""

    def __init__(self, num_classes=6, pretrained=True):
        super(ConvNeXtSmallModel, self).__init__()
        weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
        self.convnext = models.convnext_small(weights=weights)
        # 替换分类头
        num_features = self.convnext.classifier[2].in_features
        self.convnext.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.LayerNorm(num_features, eps=1e-6),
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.convnext(x)


def create_model(model_name: str, num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    """
    创建指定的模型

    Args:
        model_name: 模型名称
            'simple_cnn', 'simple_cnn_v2', 'mobilenetv2', 'mobilenetv2_se',
            'resnet18', 'resnet50', 'efficientnetv2s', 'efficientnetv2m',
            'convnexttiny', 'convnextsmall'
        num_classes: 分类数量
        pretrained: 是否使用预训练权重

    Returns:
        模型实例
    """
    if model_name.lower() == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes)
    elif model_name.lower() == 'simple_cnn_v2':
        return SimpleCNNV2(num_classes=num_classes)
    elif model_name.lower() == 'mobilenetv2':
        return MobileNetV2Model(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == 'mobilenetv2_se':
        return MobileNetV2SEModel(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == 'resnet18':
        return ResNet18Model(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == 'resnet50':
        return ResNet50Model(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == 'efficientnetv2s':
        return EfficientNetV2SModel(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == 'efficientnetv2m':
        return EfficientNetV2MModel(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == 'convnexttiny':
        return ConvNeXtTinyModel(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == 'convnextsmall':
        return ConvNeXtSmallModel(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"未知的模型名称：{model_name}")


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

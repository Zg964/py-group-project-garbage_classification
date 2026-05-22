""" 模型定义模块
包含自定义 CNN、MobileNetV2、ResNet18、EfficientNetV2-S 和 ConvNeXt Tiny 模型
v1.04: 新增模型集成支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List


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


class MobileNetV2Model(nn.Module):
    """基于 MobileNetV2 的转移学习模型"""

    def __init__(self, num_classes=6, pretrained=True):
        super(MobileNetV2Model, self).__init__()
        # 加载预训练的 MobileNetV2
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)
        # 替换分类头
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.mobilenet(x)


class ResNet18Model(nn.Module):
    """基于 ResNet18 的转移学习模型"""

    def __init__(self, num_classes=6, pretrained=True):
        super(ResNet18Model, self).__init__()
        # 加载预训练的 ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
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
        self.efficientnet = models.efficientnet_v2_s(pretrained=pretrained)
        # 替换分类头
        num_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)


class ConvNeXtTinyModel(nn.Module):
    """基于 ConvNeXt Tiny 的转移学习模型（v1.03 新增）"""

    def __init__(self, num_classes=6, pretrained=True):
        super(ConvNeXtTinyModel, self).__init__()
        self.convnext = models.convnext_tiny(pretrained=pretrained)
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


def create_model(model_name: str, num_classes: int = 6, pretrained: bool = True) -> nn.Module:
    """
    创建指定的模型

    Args:
        model_name: 模型名称 ('simple_cnn', 'mobilenetv2', 'resnet18', 'efficientnetv2s', 'convnexttiny')
        num_classes: 分类数量
        pretrained: 是否使用预训练权重

    Returns:
        模型实例
    """
    if model_name.lower() == 'simple_cnn':
        return SimpleCNN(num_classes=num_classes)
    elif model_name.lower() == 'mobilenetv2':
        return MobileNetV2Model(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == 'resnet18':
        return ResNet18Model(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == 'efficientnetv2s':
        return EfficientNetV2SModel(num_classes=num_classes, pretrained=pretrained)
    elif model_name.lower() == 'convnexttiny':
        return ConvNeXtTinyModel(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"未知的模型名称：{model_name}")


def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

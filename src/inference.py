"""
推理模块（v1.03 新增）
支持单张图像预测、批量预测和垃圾分类建议
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from typing import List, Tuple, Optional

import torchvision.transforms as transforms

from src.models import create_model
from src.config import GARBAGE_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS

logger = logging.getLogger(__name__)

# 垃圾分类建议（中英文）
SORTING_ADVICE = {
    'cardboard': {
        'category': '可回收物',
        'advice': '纸板类属于可回收物。请保持纸板干燥、清洁，折叠后放入可回收物垃圾桶。'
    },
    'glass': {
        'category': '可回收物',
        'advice': '玻璃类属于可回收物。请清空内容物，冲洗干净后放入可回收物垃圾桶。注意轻拿轻放。'
    },
    'metal': {
        'category': '可回收物',
        'advice': '金属类属于可回收物。请清空内容物，压缩体积后放入可回收物垃圾桶。'
    },
    'paper': {
        'category': '可回收物',
        'advice': '纸质类属于可回收物。请保持纸张干燥、清洁，平整堆放后放入可回收物垃圾桶。'
    },
    'plastic': {
        'category': '可回收物',
        'advice': '塑料类属于可回收物。请清空内容物，压扁后放入可回收物垃圾桶。'
    },
    'trash': {
        'category': '其他垃圾',
        'advice': '其他垃圾（不可回收物）。请放入其他垃圾桶。这些垃圾通常会被填埋或焚烧处理。'
    }
}


class InferenceModel:
    """推理模型封装"""

    def __init__(self, model_name: str, model_path: str,
                 device: Optional[torch.device] = None,
                 num_classes: int = 6):
        """
        Args:
            model_name: 模型名称
            model_path: 模型权重文件路径
            device: 运行设备
            num_classes: 分类数量
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.num_classes = num_classes

        # 创建模型并加载权重
        self.model = create_model(model_name, num_classes=num_classes)
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"模型 {model_name} 已加载，设备：{self.device}")

        # 推理变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def predict(self, image: Image.Image, topk: int = 3) -> dict:
        """对单张图像进行预测

        Args:
            image: PIL 图像
            topk: 返回前 topk 个预测结果

        Returns:
            dict: 包含预测结果和置信度的字典
        """
        # 预处理
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 推理
        outputs = self.model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

        # 获取 top-k 预测
        top_probs, top_indices = torch.topk(probabilities, k=min(topk, self.num_classes))

        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            class_name = IDX_TO_CLASS[idx]
            advice_info = SORTING_ADVICE.get(class_name, {})
            predictions.append({
                'class_name': class_name,
                'class_idx': int(idx),
                'probability': float(prob),
                'category': advice_info.get('category', '未知'),
                'advice': advice_info.get('advice', '')
            })

        result = {
            'predictions': predictions,
            'top_class': predictions[0]['class_name'],
            'top_probability': predictions[0]['probability'],
        }

        return result

    @torch.no_grad()
    def predict_batch(self, images: List[Image.Image], topk: int = 3) -> List[dict]:
        """批量预测

        Args:
            images: PIL 图像列表
            topk: 返回前 topk 个预测结果

        Returns:
            List[dict]: 每个图像的预测结果
        """
        batch_tensors = torch.stack([self.transform(img) for img in images])
        batch_tensors = batch_tensors.to(self.device)

        outputs = self.model(batch_tensors)
        probabilities = torch.softmax(outputs, dim=1)

        results = []
        for probs in probabilities:
            top_probs, top_indices = torch.topk(probs, k=min(topk, self.num_classes))
            predictions = []
            for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                class_name = IDX_TO_CLASS[idx]
                advice_info = SORTING_ADVICE.get(class_name, {})
                predictions.append({
                    'class_name': class_name,
                    'class_idx': int(idx),
                    'probability': float(prob),
                    'category': advice_info.get('category', '未知'),
                    'advice': advice_info.get('advice', '')
                })
            results.append({
                'predictions': predictions,
                'top_class': predictions[0]['class_name'],
                'top_probability': predictions[0]['probability'],
            })

        return results


def load_model(model_name: str = 'efficientnetv2s',
               model_dir: str = 'models') -> InferenceModel:
    """加载训练好的模型进行推理

    Args:
        model_name: 模型名称
        model_dir: 模型目录

    Returns:
        InferenceModel 实例
    """
    model_path = Path(model_dir) / f'{model_name}_best.pth'
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在：{model_path}")

    return InferenceModel(model_name, str(model_path))


def predict_image(model: InferenceModel, image_path: str, topk: int = 3) -> dict:
    """对单张图像文件进行预测

    Args:
        model: InferenceModel 实例
        image_path: 图像文件路径
        topk: 返回前 topk 个预测结果

    Returns:
        dict: 预测结果
    """
    image = Image.open(image_path).convert('RGB')
    return model.predict(image, topk=topk)


def get_sorting_advice(class_name: str) -> dict:
    """获取垃圾分类建议

    Args:
        class_name: 类别名称

    Returns:
        dict: 垃圾分类建议
    """
    return SORTING_ADVICE.get(class_name, {
        'category': '未知',
        'advice': '无法确定该物品的分类，请参考当地垃圾分类指南。'
    })

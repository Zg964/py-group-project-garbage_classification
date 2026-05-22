""" 推理模块（v1.03 新增）
v1.04: 新增 TTA (Test-Time Augmentation) 支持
支持单张图像预测、批量预测和垃圾分类建议
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from typing import List, Tuple, Optional
import torchvision.transforms as transforms
from src.models import create_model
from src.config import GARBAGE_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, TTA_FLIP, TTA_ROTATION_ANGLES

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


class TTAPredictor:
    """v1.04: TTA (Test-Time Augmentation) 预测器"""

    def __init__(self, model, device, use_flip=True, rotation_angles=None):
        """
        Args:
            model: 模型实例
            device: 运行设备
            use_flip: 是否使用水平翻转
            rotation_angles: 旋转角度列表
        """
        self.model = model
        self.device = device
        self.use_flip = use_flip
        self.rotation_angles = rotation_angles or [0]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def get_augmented_tensors(self, image: Image.Image) -> List[torch.Tensor]:
        """获取所有增强版本的图像 tensor

        Args:
            image: PIL Image

        Returns:
            增强后的图像 tensor 列表
        """
        tensors = []

        # 原始图像
        tensors.append(self.transform(image))

        # 水平翻转
        if self.use_flip:
            flip_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            tensors.append(self.transform(flip_image))

        # 旋转
        for angle in self.rotation_angles:
            if angle not in [0, 360]:
                rotated = image.rotate(angle, expand=True)
                tensors.append(self.transform(rotated))

                # 翻转的旋转版本
                if self.use_flip:
                    rotated_flip = rotated.transpose(Image.FLIP_LEFT_RIGHT)
                    tensors.append(self.transform(rotated_flip))

        return tensors

    def predict(self, image: Image.Image) -> tuple:
        """使用 TTA 进行预测

        Args:
            image: PIL Image

        Returns:
            平均后的概率和预测类别
        """
        self.model.eval()
        augmented_tensors = self.get_augmented_tensors(image)

        all_probs = []
        with torch.no_grad():
            for tensor in augmented_tensors:
                tensor = tensor.unsqueeze(0).to(self.device)
                output = self.model(tensor)
                prob = F.softmax(output, dim=1)
                all_probs.append(prob)

        # 平均所有增强版本的预测
        avg_prob = torch.mean(torch.cat(all_probs, dim=0), dim=0, keepdim=True)
        confidence, predicted = torch.max(avg_prob, 1)

        return avg_prob, predicted, confidence.item()


class InferenceModel:
    """推理模型封装"""

    def __init__(self, model_name: str, model_path: str, device: Optional[torch.device] = None,
                 num_classes: int = 6, use_tta: bool = False):
        """
        Args:
            model_name: 模型名称
            model_path: 模型权重文件路径
            device: 运行设备
            num_classes: 分类数量
            use_tta: 是否使用 TTA
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.num_classes = num_classes
        self.use_tta = use_tta

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

        # TTA 预测器
        if use_tta:
            self.tta_predictor = TTAPredictor(
                self.model, self.device,
                use_flip=TTA_FLIP,
                rotation_angles=TTA_ROTATION_ANGLES
            )
        else:
            self.tta_predictor = None

    @torch.no_grad()
    def predict(self, image: Image.Image, topk: int = 3, use_tta: bool = None) -> dict:
        """对单张图像进行预测

        Args:
            image: PIL 图像
            topk: 返回前 topk 个预测结果
            use_tta: 是否使用 TTA（如果为 None，则使用初始化时的设置）

        Returns:
            dict: 包含预测结果和置信度的字典
        """
        # 如果指定了 use_tta，则使用指定的值，否则使用初始化时的设置
        if use_tta is None:
            use_tta = self.use_tta

        if use_tta and self.tta_predictor is not None:
            # 使用 TTA 预测
            probabilities, predicted, confidence = self.tta_predictor.predict(image)
            probabilities = probabilities.cpu().numpy()[0]
        else:
            # 标准预测
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            probabilities = probabilities.cpu().numpy()
            confidence = np.max(probabilities)
            predicted = np.argmax(probabilities)

        # 获取 top-k 预测
        top_indices = np.argsort(probabilities)[::-1][:min(topk, self.num_classes)]
        predictions = []

        for idx in top_indices:
            class_name = IDX_TO_CLASS[idx]
            advice_info = SORTING_ADVICE.get(class_name, {})
            predictions.append({
                'class_name': class_name,
                'class_idx': int(idx),
                'probability': float(probabilities[idx]),
                'category': advice_info.get('category', '未知'),
                'advice': advice_info.get('advice', '')
            })

        result = {
            'predictions': predictions,
            'top_class': predictions[0]['class_name'],
            'top_probability': predictions[0]['probability'],
            'use_tta': use_tta
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
            probs_np = probs.cpu().numpy()
            top_indices = np.argsort(probs_np)[::-1][:min(topk, self.num_classes)]
            predictions = []

            for idx in top_indices:
                class_name = IDX_TO_CLASS[idx]
                advice_info = SORTING_ADVICE.get(class_name, {})
                predictions.append({
                    'class_name': class_name,
                    'class_idx': int(idx),
                    'probability': float(probs_np[idx]),
                    'category': advice_info.get('category', '未知'),
                    'advice': advice_info.get('advice', '')
                })

            results.append({
                'predictions': predictions,
                'top_class': predictions[0]['class_name'],
                'top_probability': predictions[0]['probability'],
            })

        return results


def load_model(model_name: str = 'efficientnetv2s', model_dir: str = 'models',
               use_tta: bool = False) -> InferenceModel:
    """加载训练好的模型进行推理

    Args:
        model_name: 模型名称
        model_dir: 模型目录
        use_tta: 是否使用 TTA

    Returns:
        InferenceModel 实例
    """
    model_path = Path(model_dir) / f'{model_name}_best.pth'
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在：{model_path}")

    return InferenceModel(model_name, str(model_path), use_tta=use_tta)


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

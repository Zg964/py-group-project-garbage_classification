"""
数据加载和预处理模块
处理垃圾分类数据集的加载、清洗和增强
"""

import os
import shutil
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging

from src.config import (GARBAGE_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS,
                        TEST_SIZE, VAL_RATIO, RANDOM_SEED,
                        RANDAUGMENT_NUM_OPS, RANDAUGMENT_MAGNITUDE,
                        MIXUP_ALPHA)

# 向后兼容的导入

logger = logging.getLogger(__name__)


class DataCleaner:
    """数据清洗类"""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.statistics = {}

    def load_dataset_info(self):
        """加载数据集信息，检查缺失和异常值"""
        logger.info("开始加载数据集信息...")

        dataset_info = {
            'total_images': 0,
            'class_distribution': {},
            'missing_files': [],
            'corrupted_files': [],
            'duplicate_files': {}
        }

        for cls in GARBAGE_CLASSES:
            cls_dir = self.data_dir / cls
            if not cls_dir.exists():
                logger.warning(f"类别目录不存在：{cls_dir}")
                dataset_info['class_distribution'][cls] = 0
                continue

            images = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png'))
            dataset_info['class_distribution'][cls] = len(images)
            dataset_info['total_images'] += len(images)

            # 检查损坏的图像
            for img_path in images:
                if not self._is_valid_image(img_path):
                    dataset_info['corrupted_files'].append(str(img_path))

        logger.info(f"数据集统计:")
        for cls, count in dataset_info['class_distribution'].items():
            logger.info(f" {cls}: {count} 张")

        return dataset_info

    @staticmethod
    def _is_valid_image(img_path):
        """检查图像是否有效"""
        try:
            img = Image.open(img_path)
            img.verify()
            return True
        except:
            return False

    def remove_corrupted_files(self, corrupted_files):
        """移除损坏的文件"""
        for file_path in corrupted_files:
            try:
                os.remove(file_path)
                logger.warning(f"已删除损坏的文件：{file_path}")
            except Exception as e:
                logger.error(f"删除文件失败：{file_path}, 错误：{e}")

    def detect_duplicates(self):
        """检测重复图像（基于文件大小和内容哈希）"""
        logger.info("检测重复文件...")
        duplicates = {}

        file_hashes = {}
        for cls in GARBAGE_CLASSES:
            cls_dir = self.data_dir / cls
            if not cls_dir.exists():
                continue

            for img_path in cls_dir.glob('*.*'):
                if img_path.is_file():
                    file_hash = self._compute_image_hash(img_path)

                    if file_hash in file_hashes:
                        if file_hash not in duplicates:
                            duplicates[file_hash] = [file_hashes[file_hash]]
                        duplicates[file_hash].append(str(img_path))
                    else:
                        file_hashes[file_hash] = str(img_path)

        # 只保留有重复的哈希
        duplicates = {k: v for k, v in duplicates.items() if len(v) > 1}
        return duplicates

    @staticmethod
    def _compute_image_hash(img_path, hash_size=8):
        """计算图像的感知哈希"""
        try:
            img = Image.open(img_path).convert('L').resize((hash_size, hash_size))
            pixels = list(img.getdata())
            avg = sum(pixels) / len(pixels)
            hash_bits = ''.join(['1' if p > avg else '0' for p in pixels])
            return hash_bits
        except:
            return None

    def remove_duplicates(self, duplicates, keep_first=True):
        """移除重复文件"""
        removed_count = 0
        for hash_val, file_list in duplicates.items():
            if keep_first:
                files_to_remove = file_list[1:]  # 保留第一个
            else:
                files_to_remove = file_list

            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    removed_count += 1
                    logger.info(f"已删除重复文件：{file_path}")
                except Exception as e:
                    logger.error(f"删除文件失败：{file_path}, 错误：{e}")

        return removed_count

    def validate_labels(self):
        """验证标签正确性"""
        logger.info("验证标签正确性...")
        misclassified = []

        for cls in GARBAGE_CLASSES:
            cls_dir = self.data_dir / cls
            if not cls_dir.exists():
                logger.warning(f"类别目录不存在：{cls_dir}")
                continue

            for img_path in cls_dir.glob('*.*'):
                if img_path.is_file() and not self._is_valid_image(img_path):
                    misclassified.append({
                        'file': str(img_path),
                        'expected_class': cls,
                        'issue': 'damaged_file'
                    })

        return misclassified

    def generate_statistics(self, dataset_info):
        """生成数据清洗统计"""
        stats = {
            'total_images': dataset_info['total_images'],
            'class_distribution': dataset_info['class_distribution'],
            'corrupted_count': len(dataset_info['corrupted_files']),
            'missing_files': len(dataset_info['missing_files'])
        }
        return stats


# 全局缓存数据集划分索引
_dataset_split_cache = None


def _get_or_create_split_indices(image_paths, labels, test_size=TEST_SIZE, val_size=VAL_RATIO, random_seed=RANDOM_SEED):
    """获取或创建数据集划分索引（确保一致性）"""
    global _dataset_split_cache

    if _dataset_split_cache is not None:
        return _dataset_split_cache

    # 先分离出测试集
    temp_indices, test_indices, temp_labels, test_labels = train_test_split(
        list(range(len(image_paths))), labels,
        test_size=test_size, random_state=random_seed, stratify=labels
    )

    # 再从临时集中分离验证集
    val_split = val_size / (1 - test_size)
    train_indices, val_indices = train_test_split(
        temp_indices, test_size=val_split, random_state=random_seed, stratify=temp_labels
    )

    _dataset_split_cache = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices
    }

    return _dataset_split_cache


class GarbageDataset(Dataset):
    """垃圾分类数据集"""

    def __init__(self, data_dir, split='train', transform=None, test_size=TEST_SIZE, val_size=VAL_RATIO, random_seed=RANDOM_SEED):
        """
        初始化数据集

        Args:
            data_dir: 数据根目录
            split: 'train', 'val', 或 'test'
            transform: 图像变换
            test_size: 测试集比例
            val_size: 验证集比例
            random_seed: 随机种子
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.random_seed = random_seed

        # 加载所有图像路径和标签
        self.images = []
        self.labels = []

        for cls in GARBAGE_CLASSES:
            cls_dir = self.data_dir / cls
            if not cls_dir.exists():
                logger.warning(f"类别目录不存在：{cls_dir}")
                continue

            for img_path in cls_dir.glob('*.*'):
                if img_path.is_file() and self._is_valid_image(img_path):
                    self.images.append(str(img_path))
                    self.labels.append(CLASS_TO_IDX[cls])

        if len(self.images) == 0:
            logger.warning(f"在 {data_dir} 中没有找到任何图像")

        # 使用固定随机种子确保划分一致性
        split_indices = _get_or_create_split_indices(
            self.images, self.labels,
            test_size=test_size, val_size=val_size, random_seed=random_seed
        )

        # 根据划分索引获取对应的图像
        if split == 'train':
            indices = split_indices['train']
        elif split == 'val':
            indices = split_indices['val']
        elif split == 'test':
            indices = split_indices['test']
        else:
            raise ValueError(f"未知的划分类型：{split}")

        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

    @staticmethod
    def _is_valid_image(img_path):
        """检查图像是否有效"""
        try:
            img = Image.open(img_path)
            img.verify()
            return True
        except:
            return False

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        # 加载并处理图像
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, img_path


def get_transforms(use_randaugment=False):
    """获取数据增强变换

    Args:
        use_randaugment: 是否使用 RandAugment 增强

    Returns:
        train_transform, val_test_transform
    """

    # 基础增强
    augmentations = [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ]

    # v1.03: RandAugment 自动增强策略
    if use_randaugment:
        try:
            randaugment = transforms.RandAugment(
                num_ops=RANDAUGMENT_NUM_OPS,
                magnitude=RANDAUGMENT_MAGNITUDE
            )
            augmentations.insert(1, randaugment)
        except Exception as e:
            logger.warning(f"RandAugment 不可用，跳过: {e}")

    train_transform = transforms.Compose(augmentations + [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 验证/测试集变换（不做数据增强）
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_test_transform


class MixUpCollate:
    """v1.03: MixUp 批量合成样本"""

    def __init__(self, alpha=MIXUP_ALPHA):
        self.alpha = alpha

    def __call__(self, batch):
        images, labels, paths = zip(*batch)
        images = torch.stack(images, 0)
        labels = torch.tensor(labels)
        paths = list(paths)

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
            batch_size = images.size(0)
            index = torch.randperm(batch_size)

            mixed_images = lam * images + (1 - lam) * images[index, :]
            labels_a, labels_b = labels, labels[index]

            return mixed_images, labels_a, labels_b, lam, paths

        return images, labels, paths


def create_dataloaders(data_dir, batch_size=32, num_workers=4,
                       use_randaugment=False, use_mixup=False):
    """创建数据加载器

    Args:
        data_dir: 数据目录
        batch_size: 批大小
        num_workers: 工作进程数
        use_randaugment: 是否使用 RandAugment
        use_mixup: 是否使用 MixUp

    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform, val_test_transform = get_transforms(use_randaugment=use_randaugment)

    train_dataset = GarbageDataset(data_dir, split='train', transform=train_transform)
    val_dataset = GarbageDataset(data_dir, split='val', transform=val_test_transform)
    test_dataset = GarbageDataset(data_dir, split='test', transform=val_test_transform)

    # v1.03: MixUp collate 函数
    collate_fn = MixUpCollate() if use_mixup else None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

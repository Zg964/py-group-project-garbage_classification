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

logger = logging.getLogger(__name__)

# 定义垃圾分类类别
GARBAGE_CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(GARBAGE_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

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
                logger.warning(f"类别目录不存在: {cls_dir}")
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
            logger.info(f"  {cls}: {count} 张")
        
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
                logger.warning(f"已删除损坏的文件: {file_path}")
            except Exception as e:
                logger.error(f"删除文件失败: {file_path}, 错误: {e}")
    
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
                    logger.info(f"已删除重复文件: {file_path}")
                except Exception as e:
                    logger.error(f"删除文件失败: {file_path}, 错误: {e}")
        
        return removed_count
    
    def validate_labels(self):
        """验证标签正确性"""
        logger.info("验证标签正确性...")
        misclassified = []
        
        for cls in GARBAGE_CLASSES:
            cls_dir = self.data_dir / cls
            if not cls_dir.exists():
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


class GarbageDataset(Dataset):
    """垃圾分类数据集"""
    
    def __init__(self, data_dir, split='train', transform=None, test_size=0.2, val_size=0.1):
        """
        初始化数据集
        
        Args:
            data_dir: 数据根目录
            split: 'train', 'val', 或 'test'
            transform: 图像变换
            test_size: 测试集比例
            val_size: 验证集比例
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # 加载所有图像路径和标签
        self.images = []
        self.labels = []
        
        for cls in GARBAGE_CLASSES:
            cls_dir = self.data_dir / cls
            if not cls_dir.exists():
                continue
            
            for img_path in cls_dir.glob('*.*'):
                if img_path.is_file() and self._is_valid_image(img_path):
                    self.images.append(str(img_path))
                    self.labels.append(CLASS_TO_IDX[cls])
        
        # 划分训练、验证、测试集
        if split == 'train':
            # 先分离出测试集
            temp_images, test_images, temp_labels, test_labels = train_test_split(
                self.images, self.labels, test_size=test_size, random_state=42, stratify=self.labels
            )
            # 再从临时集中分离验证集
            val_split = val_size / (1 - test_size)  # 调整比例
            train_images, val_images, train_labels, val_labels = train_test_split(
                temp_images, temp_labels, test_size=val_split, random_state=42, stratify=temp_labels
            )
            self.images, self.labels = train_images, train_labels
        elif split == 'val':
            temp_images, test_images, temp_labels, test_labels = train_test_split(
                self.images, self.labels, test_size=test_size, random_state=42, stratify=self.labels
            )
            val_split = val_size / (1 - test_size)
            train_images, val_images, train_labels, val_labels = train_test_split(
                temp_images, temp_labels, test_size=val_split, random_state=42, stratify=temp_labels
            )
            self.images, self.labels = val_images, val_labels
        elif split == 'test':
            temp_images, self.images, temp_labels, self.labels = train_test_split(
                self.images, self.labels, test_size=test_size, random_state=42, stratify=self.labels
            )
    
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


def get_transforms():
    """获取数据增强变换"""
    
    # 训练集变换（包含数据增强）
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
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


def create_dataloaders(data_dir, batch_size=32, num_workers=4):
    """创建数据加载器"""
    train_transform, val_test_transform = get_transforms()
    
    train_dataset = GarbageDataset(data_dir, split='train', transform=train_transform)
    val_dataset = GarbageDataset(data_dir, split='val', transform=val_test_transform)
    test_dataset = GarbageDataset(data_dir, split='test', transform=val_test_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader

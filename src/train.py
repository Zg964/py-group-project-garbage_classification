"""
模型训练脚本
v1.03: 新增 FocalLoss、CosineAnnealingWarmRestarts、Label Smoothing、
       梯度裁剪、Early Stopping、MixUp 支持
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

from src.models import create_model, count_parameters
from src.data_loader import create_dataloaders, GARBAGE_CLASSES
from src.config import (FOCAL_LOSS_GAMMA, LABEL_SMOOTHING_EPSILON,
                        COSINE_T_0, COSINE_T_MULT, GRAD_CLIP_MAX_NORM,
                        EARLY_STOPPING_PATIENCE, NUM_WORKERS)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """v1.03: Focal Loss - 降低易分类样本损失，聚焦难分类样本"""

    def __init__(self, gamma=FOCAL_LOSS_GAMMA, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight,
                                   reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """v1.03: 标签平滑交叉熵损失"""

    def __init__(self, epsilon=LABEL_SMOOTHING_EPSILON, weight=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.weight = weight

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)

        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.epsilon / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.epsilon)
            smooth_targets = smooth_targets.detach()

        loss = (-smooth_targets * log_probs).sum(dim=1)

        if self.weight is not None:
            loss = loss * self.weight[targets]

        return loss.mean()


class Trainer:
    """模型训练器"""

    def __init__(self, model, device, model_name='model', use_mixup=False):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.use_mixup = use_mixup
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0
        self.best_epoch = 0
        self.early_stop_counter = 0

    def _compute_mixup_loss(self, criterion, outputs, labels_a, labels_b, lam):
        """计算 MixUp 的混合损失"""
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        return loss

    def train_epoch(self, train_loader, criterion, optimizer, epoch, num_epochs):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Train')
        for batch in pbar:
            if self.use_mixup:
                # MixUp 返回 5 个值: images, labels_a, labels_b, lam, paths
                images, labels_a, labels_b, lam, _ = batch
                images = images.to(self.device)
                labels_a = labels_a.to(self.device)
                labels_b = labels_b.to(self.device)
                lam = lam.to(self.device) if isinstance(lam, torch.Tensor) else lam
            else:
                images, labels, _ = batch
                images, labels = images.to(self.device), labels.to(self.device)

            # 前向传播
            outputs = self.model(images)

            if self.use_mixup:
                loss = self._compute_mixup_loss(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # v1.03: 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_MAX_NORM)

            optimizer.step()

            # 计算准确率
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0) if not self.use_mixup else labels_a.size(0)
            if not self.use_mixup:
                correct += (predicted == labels).sum().item()

            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total if total > 0 else 0

        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)

        return epoch_loss, epoch_acc

    def validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for images, labels, _ in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                pbar.set_postfix({'loss': loss.item()})

        epoch_loss = total_loss / len(val_loader)
        epoch_acc = correct / total

        self.val_losses.append(epoch_loss)
        self.val_accs.append(epoch_acc)

        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, num_epochs=50, lr=0.001, weight_decay=1e-4,
              save_dir='models', use_focal=False, use_label_smoothing=False,
              use_cosine=False):
        """完整训练流程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            save_dir: 模型保存目录
            use_focal: 是否使用 Focal Loss
            use_label_smoothing: 是否使用标签平滑
            use_cosine: 是否使用 CosineAnnealingWarmRestarts 调度器
        """

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 设置优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # v1.03: CosineAnnealingWarmRestarts 调度器
        if use_cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=COSINE_T_0, T_mult=COSINE_T_MULT
            )
            logger.info(f"使用 CosineAnnealingWarmRestarts 调度器 (T_0={COSINE_T_0}, T_mult={COSINE_T_MULT})")
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3
            )

        # 计算类别权重以处理数据不平衡
        labels_arr = train_loader.dataset.labels
        class_counts = torch.bincount(torch.tensor(labels_arr))
        total_samples = len(labels_arr)
        n_classes = len(class_counts)
        class_weights = total_samples / (n_classes * class_counts.float())
        class_weights = class_weights.to(self.device)
        logger.info(f"类别权重：{class_weights.cpu().tolist()}")

        # v1.03: 选择损失函数
        if use_focal:
            criterion = FocalLoss(gamma=FOCAL_LOSS_GAMMA, weight=class_weights)
            logger.info(f"使用 Focal Loss (gamma={FOCAL_LOSS_GAMMA})")
        elif use_label_smoothing:
            criterion = LabelSmoothingCrossEntropy(epsilon=LABEL_SMOOTHING_EPSILON, weight=class_weights)
            logger.info(f"使用 Label Smoothing CrossEntropy (epsilon={LABEL_SMOOTHING_EPSILON})")
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 在 MixUp 模式下，使用 KL 散度损失处理平滑标签
        if self.use_mixup and not use_focal:
            criterion = nn.KLDivLoss(reduction='batchmean') if use_label_smoothing else nn.CrossEntropyLoss(weight=class_weights)

        logger.info(f"开始训练 {self.model_name}")
        logger.info(f"模型参数数量：{count_parameters(self.model):,}")
        logger.info(f"设备：{self.device}")
        logger.info(f"学习率：{lr}, 权重衰减：{weight_decay}")
        logger.info(f"总 Epoch 数：{num_epochs}")
        if self.use_mixup:
            logger.info("MixUp 数据增强：已启用")

        start_time = time.time()
        self.early_stop_counter = 0

        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(
                train_loader, criterion, optimizer, epoch, num_epochs
            )

            # 验证
            val_loss, val_acc = self.validate(val_loader, criterion)

            # 调整学习率
            if use_cosine:
                scheduler.step(epoch + 0.5)  # CosineAnnealingWarmRestarts 在 epoch 中间步进
                current_lr = optimizer.param_groups[0]['lr']
            else:
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']

            logger.info(
                f'Epoch [{epoch+1}/{num_epochs}] - '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | '
                f'LR: {current_lr:.2e}'
            )

            # 保存最好的模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.early_stop_counter = 0
                best_model_path = save_dir / f'{self.model_name}_best.pth'
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'architecture': self.model_name,
                    'num_classes': len(GARBAGE_CLASSES),
                    'class_names': GARBAGE_CLASSES,
                    'val_acc': val_acc,
                    'epoch': epoch,
                    'use_focal': use_focal,
                    'use_label_smoothing': use_label_smoothing,
                    'use_cosine': use_cosine,
                    'use_mixup': self.use_mixup
                }
                torch.save(checkpoint, best_model_path)
                logger.info(f'已保存最好的模型：{best_model_path}')
            else:
                # v1.03: Early Stopping
                self.early_stop_counter += 1
                if self.early_stop_counter >= EARLY_STOPPING_PATIENCE:
                    logger.info(
                        f'Early Stopping: {EARLY_STOPPING_PATIENCE} 个 epoch 未改善，'
                        f'停止训练 (最佳 epoch: {self.best_epoch+1}, 最佳 Val Acc: {self.best_val_acc:.4f})'
                    )
                    break

        training_time = time.time() - start_time
        logger.info(f'训练完成，耗时：{training_time:.2f}s')
        logger.info(f'最好的验证准确率：{self.best_val_acc:.4f} (Epoch {self.best_epoch+1})')

        return {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'training_time': training_time,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }


def train_baseline_models(data_dir, output_dir='models', num_epochs=50, batch_size=32,
                          num_workers=NUM_WORKERS, use_randaugment=False, use_mixup=False,
                          use_focal=False, use_label_smoothing=False, use_cosine=False,
                          models_to_train=None):
    """训练模型

    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批大小
        num_workers: 工作进程数
        use_randaugment: 是否使用 RandAugment
        use_mixup: 是否使用 MixUp
        use_focal: 是否使用 Focal Loss
        use_label_smoothing: 是否使用标签平滑
        use_cosine: 是否使用 CosineAnnealingWarmRestarts
        models_to_train: 要训练的模型列表，默认为所有模型
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备：{device}")

    # 创建数据加载器
    logger.info("加载数据...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir, batch_size=batch_size, num_workers=num_workers,
        use_randaugment=use_randaugment, use_mixup=use_mixup
    )

    logger.info(f"训练集大小：{len(train_loader.dataset)}")
    logger.info(f"验证集大小：{len(val_loader.dataset)}")
    logger.info(f"测试集大小：{len(test_loader.dataset)}")

    # v1.03: 包含 5 个模型
    all_models_config = [
        {'name': 'simple_cnn', 'pretrained': False},
        {'name': 'mobilenetv2', 'pretrained': True},
        {'name': 'resnet18', 'pretrained': True},
        {'name': 'efficientnetv2s', 'pretrained': True},
        {'name': 'convnexttiny', 'pretrained': True},
    ]

    if models_to_train is not None:
        models_config = [c for c in all_models_config if c['name'] in models_to_train]
    else:
        models_config = all_models_config

    results = {}

    for config in models_config:
        model_name = config['name']
        logger.info(f"\n{'='*60}")
        logger.info(f"训练模型：{model_name}")
        logger.info(f"{'='*60}")

        # 创建模型
        model = create_model(model_name, num_classes=len(GARBAGE_CLASSES),
                             pretrained=config['pretrained'])

        # 训练
        trainer = Trainer(model, device, model_name, use_mixup=use_mixup)
        history = trainer.train(
            train_loader, val_loader,
            num_epochs=num_epochs,
            lr=0.001,
            weight_decay=1e-4,
            save_dir=output_dir,
            use_focal=use_focal,
            use_label_smoothing=use_label_smoothing,
            use_cosine=use_cosine
        )

        results[model_name] = history

    # 保存训练历史
    history_path = output_dir / 'training_history.json'
    # 将 history 中的 list 转为可序列化格式
    serializable_results = {}
    for name, hist in results.items():
        serializable_results[name] = {
            'best_val_acc': hist['best_val_acc'],
            'best_epoch': hist['best_epoch'],
            'training_time': hist['training_time'],
            'train_losses': [float(x) for x in hist['train_losses']],
            'val_losses': [float(x) for x in hist['val_losses']],
            'train_accs': [float(x) for x in hist['train_accs']],
            'val_accs': [float(x) for x in hist['val_accs']]
        }

    with open(history_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    logger.info(f"训练历史已保存到：{history_path}")

    return results


if __name__ == '__main__':
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/processed'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'models'
    num_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    train_baseline_models(data_dir, output_dir, num_epochs)

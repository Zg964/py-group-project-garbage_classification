""" 模型训练脚本
v1.03: 新增 FocalLoss、CosineAnnealingWarmRestarts、Label Smoothing、梯度裁剪、Early Stopping、MixUp 支持
v1.04: 新增 One Cycle 学习率调度器、CutMix 支持
v1.05: 模型专属训练配置（优化器、LR、weight_decay、grad_clip），新增 DistillationTrainer、EMA
"""

import os
import json
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

from src.models import create_model, count_parameters
from src.data_loader import create_dataloaders, GARBAGE_CLASSES
from src.config import (
    FOCAL_LOSS_GAMMA, LABEL_SMOOTHING_EPSILON,
    COSINE_T_0, COSINE_T_MULT, GRAD_CLIP_MAX_NORM,
    EARLY_STOPPING_PATIENCE, NUM_WORKERS,
    CUTMIX_ALPHA, ONecycle_PCT_START,
    CONVNEXT_LR, CONVNEXT_WEIGHT_DECAY, CONVNEXT_GRAD_CLIP,
    PRETRAINED_LR, PRETRAINED_WEIGHT_DECAY,
    WARMUP_EPOCHS, COSINE_ETA_MIN,
    SGD_LR, SGD_MOMENTUM, SGD_WEIGHT_DECAY,
    SWA_START_FRACTION
)

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
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
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


class FocalLossWithLabelSmoothing(nn.Module):
    """v1.06: 带标签平滑的 Focal Loss — 两者互补：Focal 聚焦难样本，Label Smoothing 防止过拟合"""

    def __init__(self, gamma=FOCAL_LOSS_GAMMA, epsilon=LABEL_SMOOTHING_EPSILON, weight=None, reduction='mean'):
        super(FocalLossWithLabelSmoothing, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)

        # 标签平滑后的目标分布
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.epsilon / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.epsilon)
            smooth_targets = smooth_targets.detach()

        # Focal Loss 调制：使用 softmax 概率
        probs = torch.exp(log_probs)
        focal_weight = (1 - probs) ** self.gamma

        # 组合：Focal modulation × 标签平滑 CE
        loss = -(smooth_targets * log_probs * focal_weight).sum(dim=1)

        if self.weight is not None:
            loss = loss * self.weight[targets]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class ExponentialMovingAverage:
    """v1.05: 指数移动平均 (EMA) — 验证时使用训练过程中的权重平均"""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化 shadow 权重
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        """更新 EMA 权重"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        """将 EMA 权重应用到模型（验证前调用）"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """恢复原始权重（验证后调用）"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


class Trainer:
    """模型训练器"""

    def __init__(self, model, device, model_name='model', use_mixup=False, use_cutmix=False,
                 grad_clip_max_norm=None, use_ema=False, teacher_model=None):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.grad_clip_max_norm = grad_clip_max_norm if grad_clip_max_norm is not None else GRAD_CLIP_MAX_NORM
        self.use_ema = use_ema
        self.ema = ExponentialMovingAverage(model) if use_ema else None
        self.teacher_model = teacher_model
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0
        self.best_epoch = 0
        self.early_stop_counter = 0
        # v1.06: SWA
        self.swa_model = None
        self.swa_started = False

    def _compute_mixup_loss(self, criterion, outputs, labels_a, labels_b, lam):
        """计算 MixUp 的混合损失"""
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        return loss

    def _compute_cutmix_loss(self, criterion, outputs, labels_a, labels_b, lam):
        """计算 CutMix 的混合损失"""
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        return loss

    def train_epoch(self, train_loader, criterion, optimizer, epoch, num_epochs):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{num_epochs}] Train')
        for batch in pbar:
            if self.use_cutmix or self.use_mixup:
                # CutMix/MixUp 返回 5 个值：images, labels_a, labels_b, lam, paths
                images, labels_a, labels_b, lam, _ = batch
                images = images.to(self.device)
                labels_a = labels_a.to(self.device)
                labels_b = labels_b.to(self.device)
                if isinstance(lam, torch.Tensor):
                    lam = lam.to(self.device)
            else:
                images, labels, _ = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                lam = None

            # 前向传播
            outputs = self.model(images)

            if self.use_cutmix:
                loss = self._compute_cutmix_loss(criterion, outputs, labels_a, labels_b, lam)
            elif self.use_mixup:
                loss = self._compute_mixup_loss(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # v1.03: 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)

            optimizer.step()

            # v1.05: EMA 权重更新
            if self.use_ema and self.ema is not None:
                self.ema.update(self.model)

            # 计算准确率
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            if self.use_cutmix or self.use_mixup:
                # 对于 CutMix/MixUp，计算加权准确率
                combined_labels = labels_a if lam > 0.5 else labels_b
                correct += (predicted == combined_labels).sum().item()
                total += combined_labels.size(0)
            else:
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        epoch_acc = correct / total if total > 0 else 0

        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        """验证模型"""
        # v1.05: 验证前切换到 EMA 权重
        if self.use_ema and self.ema is not None:
            self.ema.apply_shadow(self.model)

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        # 使用不带数据增强的验证损失函数
        criterion = nn.CrossEntropyLoss()

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

        epoch_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        epoch_acc = correct / total if total > 0 else 0

        self.val_losses.append(epoch_loss)
        self.val_accs.append(epoch_acc)

        # v1.05: 验证后恢复原始权重
        if self.use_ema and self.ema is not None:
            self.ema.restore(self.model)

        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, num_epochs=50, lr=0.001, weight_decay=1e-4,
              save_dir='models', use_focal=False, use_label_smoothing=False,
              use_cosine=False, use_onecycle=False, use_warmup=False,
              optimizer_type='adam', grad_clip_max_norm=None, use_swa=False):
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
            use_cosine: 是否使用 CosineAnnealingWarmRestarts 调度器（v1.06 弃用，推荐 use_warmup）
            use_onecycle: 是否使用 One Cycle 调度器
            use_warmup: v1.06: 是否使用 Cosine Warmup + CosineAnnealingLR 调度器
            optimizer_type: 优化器类型 ('adam', 'adamw', 'sgd')
            grad_clip_max_norm: 梯度裁剪最大范数（None 表示使用默认值）
            use_swa: v1.06: 是否使用 Stochastic Weight Averaging
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # v1.05: 根据模型选择优化器
        if grad_clip_max_norm is None:
            grad_clip_max_norm = GRAD_CLIP_MAX_NORM

        # v1.06: SGD with Nesterov momentum
        if optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(), lr=lr, momentum=SGD_MOMENTUM,
                weight_decay=weight_decay, nesterov=True
            )
            logger.info(f"使用 SGD (Nesterov) 优化器 (lr={lr}, momentum={SGD_MOMENTUM}, weight_decay={weight_decay})")
        elif optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            logger.info(f"使用 AdamW 优化器 (lr={lr}, weight_decay={weight_decay})")
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            logger.info(f"使用 Adam 优化器 (lr={lr}, weight_decay={weight_decay})")

        # v1.06: Cosine Warmup + CosineAnnealingLR 调度器（替代 CosineAnnealingWarmRestarts）
        if use_warmup:
            from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
            warmup_epochs = min(WARMUP_EPOCHS, num_epochs // 10)
            warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
            cosine_scheduler = CosineAnnealingLR(
                optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=COSINE_ETA_MIN
            )
            scheduler = SequentialLR(
                optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
            )
            logger.info(f"使用 Cosine Warmup + CosineAnnealingLR 调度器 "
                         f"(warmup={warmup_epochs} epochs, T_max={num_epochs - warmup_epochs}, "
                         f"eta_min={COSINE_ETA_MIN})")
        # v1.04: One Cycle 学习率调度器
        elif use_onecycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                epochs=num_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=ONecycle_PCT_START,
                anneal_strategy='cos'
            )
            logger.info(f"使用 OneCycleLR 调度器 (lr={lr}, pct_start={ONecycle_PCT_START})")
        # v1.03: CosineAnnealingWarmRestarts 调度器（向后兼容）
        elif use_cosine:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=COSINE_T_0, T_mult=COSINE_T_MULT
            )
            logger.info(f"使用 CosineAnnealingWarmRestarts 调度器 (T_0={COSINE_T_0}, T_mult={COSINE_T_MULT})")
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=3
            )

        # v1.06: SWA 初始化
        if use_swa:
            from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
            swa_model = AveragedModel(self.model)
            swa_scheduler = SWALR(
                optimizer,
                swa_lr=lr * 0.1  # SWA 学习率为原 LR 的 0.1 倍
            )
            swa_start_epoch = int(num_epochs * SWA_START_FRACTION)
            logger.info(f"使用 SWA (Stochastic Weight Averaging)，从 epoch {swa_start_epoch + 1} 开始")
        else:
            swa_model = None
            swa_scheduler = None
            swa_start_epoch = num_epochs  # 不启用

        # 计算类别权重以处理数据不平衡
        labels_arr = train_loader.dataset.labels
        class_counts = torch.bincount(torch.tensor(labels_arr))
        total_samples = len(labels_arr)
        n_classes = len(class_counts)
        class_weights = total_samples / (n_classes * class_counts.float())
        class_weights = class_weights.to(self.device)

        logger.info(f"类别权重：{class_weights.cpu().tolist()}")

        # v1.03: 选择损失函数
        # v1.06: 新增 FocalLossWithLabelSmoothing — Focal Loss 和 Label Smoothing 可以同时使用
        if use_focal and use_label_smoothing:
            criterion = FocalLossWithLabelSmoothing(
                gamma=FOCAL_LOSS_GAMMA, epsilon=LABEL_SMOOTHING_EPSILON,
                weight=class_weights
            )
            logger.info(f"使用 FocalLossWithLabelSmoothing (gamma={FOCAL_LOSS_GAMMA}, epsilon={LABEL_SMOOTHING_EPSILON})")
        elif use_focal:
            criterion = FocalLoss(gamma=FOCAL_LOSS_GAMMA, weight=class_weights)
            logger.info(f"使用 Focal Loss (gamma={FOCAL_LOSS_GAMMA})")
        elif use_label_smoothing:
            criterion = LabelSmoothingCrossEntropy(epsilon=LABEL_SMOOTHING_EPSILON, weight=class_weights)
            logger.info(f"使用 Label Smoothing CrossEntropy (epsilon={LABEL_SMOOTHING_EPSILON})")
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        # 在 MixUp/CutMix 模式下，使用 KL 散度损失处理平滑标签
        if (self.use_mixup or self.use_cutmix) and not use_focal and not (use_focal and use_label_smoothing):
            if use_label_smoothing:
                criterion = nn.KLDivLoss(reduction='batchmean')
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)

        logger.info(f"开始训练 {self.model_name}")
        logger.info(f"模型参数数量：{count_parameters(self.model):,}")
        logger.info(f"设备：{self.device}")
        logger.info(f"学习率：{lr}, 权重衰减：{weight_decay}")
        logger.info(f"总 Epoch 数：{num_epochs}")

        if self.use_mixup:
            logger.info("MixUp 数据增强：启用")
        if self.use_cutmix:
            logger.info("CutMix 数据增强：启用")
        if use_swa:
            logger.info("SWA：启用")

        start_time = time.time()
        self.early_stop_counter = 0

        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(
                train_loader, criterion, optimizer, epoch, num_epochs
            )

            # 验证
            val_loss, val_acc = self.validate(val_loader)

            # v1.06: SWA 更新（在最后 25% epoch 启动）
            if use_swa and swa_model is not None and epoch >= swa_start_epoch:
                swa_model.update_parameters(self.model)
                swa_scheduler.step()
                self.swa_started = True
                current_lr = optimizer.param_groups[0]['lr']
            # 调整学习率
            elif use_warmup or use_onecycle:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
            elif use_cosine:
                scheduler.step(epoch + 0.5)  # CosineAnnealingWarmRestarts 在 epoch 中间步进
                current_lr = optimizer.param_groups[0]['lr']
            else:
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']

            logger.info(
                f'Epoch [{epoch + 1}/{num_epochs}] - '
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

                # v1.06: 保存最佳模型时应用 EMA 权重（如有），使验证和测试一致
                model_state_dict = self.model.state_dict()
                ema_state_dict = None
                if self.use_ema and self.ema is not None:
                    ema_state_dict = copy.deepcopy(self.ema.shadow)
                    # 保存 checkpoint 时使用 EMA 权重
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and name in self.ema.shadow:
                            param.data.copy_(self.ema.shadow[name])

                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'architecture': self.model_name,
                    'num_classes': len(GARBAGE_CLASSES),
                    'class_names': GARBAGE_CLASSES,
                    'val_acc': val_acc,
                    'epoch': epoch,
                    'use_focal': use_focal,
                    'use_label_smoothing': use_label_smoothing,
                    'use_cosine': use_cosine or use_warmup,
                    'use_onecycle': use_onecycle,
                    'use_warmup': use_warmup,
                    'use_mixup': self.use_mixup,
                    'use_cutmix': self.use_cutmix,
                    'optimizer_type': optimizer_type,
                    'ema_state_dict': ema_state_dict
                }
                torch.save(checkpoint, best_model_path)
                logger.info(f'已保存最好的模型：{best_model_path}')

                # v1.06: 恢复原始权重（EMA 只是用来保存 checkpoint）
                if self.use_ema and self.ema is not None and ema_state_dict is not None:
                    self.model.load_state_dict(model_state_dict)
            else:
                # v1.03: Early Stopping
                self.early_stop_counter += 1
                if self.early_stop_counter >= EARLY_STOPPING_PATIENCE:
                    logger.info(
                        f'Early Stopping: {EARLY_STOPPING_PATIENCE} 个 epoch 未改善，'
                        f'停止训练 (最佳 epoch: {self.best_epoch + 1}, 最佳 Val Acc: {self.best_val_acc:.4f})'
                    )
                    break

        # v1.06: SWA 结束处理 — 更新 BN 统计量
        if use_swa and swa_model is not None and self.swa_started:
            logger.info("SWA 训练结束，更新 BN 统计量...")
            from torch.optim.swa_utils import update_bn
            swa_model.eval()
            try:
                update_bn(train_loader, swa_model, device=self.device)
                logger.info("SWA BN 统计量更新完成")

                # 将 SWA 权重保存为可选文件
                swa_path = save_dir / f'{self.model_name}_swa.pth'
                torch.save(swa_model.state_dict(), swa_path)
                logger.info(f"SWA 模型已保存到：{swa_path}")
            except Exception as e:
                logger.warning(f"SWA BN 统计量更新失败：{e}")

        training_time = time.time() - start_time
        logger.info(f'训练完成，耗时：{training_time:.2f}s')
        logger.info(f'最好的验证准确率：{self.best_val_acc:.4f} (Epoch {self.best_epoch + 1})')

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
                          use_cutmix=False, use_random_erasing=False,
                          use_focal=False, use_label_smoothing=False,
                          use_cosine=False, use_onecycle=False, use_warmup=False,
                          models_to_train=None, use_ema=False, use_distill=False,
                          use_weighted_sampler=False, use_swa=False):
    """训练模型

    Args:
        data_dir: 数据目录
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批大小
        num_workers: 工作进程数
        use_randaugment: 是否使用 RandAugment
        use_mixup: 是否使用 MixUp
        use_cutmix: 是否使用 CutMix
        use_random_erasing: 是否使用 RandomErasing
        use_focal: 是否使用 Focal Loss
        use_label_smoothing: 是否使用标签平滑
        use_cosine: 是否使用 CosineAnnealingWarmRestarts（v1.06 弃用，推荐 use_warmup）
        use_onecycle: 是否使用 One Cycle 调度器
        use_warmup: v1.06: 是否使用 Cosine Warmup + CosineAnnealingLR
        models_to_train: 要训练的模型列表，默认为所有模型
        use_ema: 是否使用 EMA (指数移动平均)
        use_distill: 是否使用知识蒸馏（Teacher: EfficientNetV2-S）
        use_weighted_sampler: v1.06: 是否使用 WeightedRandomSampler
        use_swa: v1.06: 是否使用 Stochastic Weight Averaging
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备：{device}")

    # 创建数据加载器
    logger.info("加载数据...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        use_randaugment=use_randaugment,
        use_mixup=use_mixup or use_cutmix,  # CutMix 优先级高于 MixUp
        use_cutmix=use_cutmix,
        use_random_erasing=use_random_erasing,
        use_weighted_sampler=use_weighted_sampler
    )

    logger.info(f"训练集大小：{len(train_loader.dataset)}")
    logger.info(f"验证集大小：{len(val_loader.dataset)}")
    logger.info(f"测试集大小：{len(test_loader.dataset)}")

    # v1.06: 每个模型使用专属训练配置（新增 SGD 优化器和 EfficientNetV2-M / ConvNeXt Small）
    all_models_config = [
        {
            'name': 'simple_cnn', 'pretrained': False,
            'lr': 0.001, 'weight_decay': 1e-4,
            'optimizer': 'adam', 'grad_clip': 1.0,
            'use_mixup': use_mixup, 'use_cutmix': use_cutmix,
        },
        {
            'name': 'simple_cnn_v2', 'pretrained': False,
            'lr': 0.001, 'weight_decay': 1e-4,
            'optimizer': 'adam', 'grad_clip': 1.0,
            'use_mixup': use_mixup, 'use_cutmix': use_cutmix,
        },
        {
            'name': 'mobilenetv2', 'pretrained': True,
            'lr': PRETRAINED_LR, 'weight_decay': PRETRAINED_WEIGHT_DECAY,
            'optimizer': 'adamw', 'grad_clip': 1.0,
            'use_mixup': use_mixup, 'use_cutmix': use_cutmix,
        },
        {
            'name': 'mobilenetv2_se', 'pretrained': True,
            'lr': PRETRAINED_LR, 'weight_decay': PRETRAINED_WEIGHT_DECAY,
            'optimizer': 'adamw', 'grad_clip': 1.0,
            'use_mixup': use_mixup, 'use_cutmix': use_cutmix,
        },
        {
            'name': 'resnet18', 'pretrained': True,
            'lr': SGD_LR, 'weight_decay': SGD_WEIGHT_DECAY,
            'optimizer': 'sgd', 'grad_clip': 1.0,
            'use_mixup': use_mixup, 'use_cutmix': use_cutmix,
        },
        {
            'name': 'resnet50', 'pretrained': True,
            'lr': SGD_LR, 'weight_decay': SGD_WEIGHT_DECAY,
            'optimizer': 'sgd', 'grad_clip': 1.0,
            'use_mixup': use_mixup, 'use_cutmix': use_cutmix,
        },
        {
            'name': 'efficientnetv2s', 'pretrained': True,
            'lr': SGD_LR, 'weight_decay': SGD_WEIGHT_DECAY,
            'optimizer': 'sgd', 'grad_clip': 1.0,
            'use_mixup': use_mixup, 'use_cutmix': use_cutmix,
        },
        {
            'name': 'efficientnetv2m', 'pretrained': True,
            'lr': SGD_LR, 'weight_decay': SGD_WEIGHT_DECAY,
            'optimizer': 'sgd', 'grad_clip': 1.0,
            'use_mixup': use_mixup, 'use_cutmix': use_cutmix,
        },
        {
            'name': 'convnexttiny', 'pretrained': True,
            'lr': CONVNEXT_LR, 'weight_decay': CONVNEXT_WEIGHT_DECAY,
            'optimizer': 'adamw', 'grad_clip': CONVNEXT_GRAD_CLIP,
            'use_mixup': False, 'use_cutmix': False,  # ConvNeXt: MixUp + Focal Loss 混合信号有害
        },
        {
            'name': 'convnextsmall', 'pretrained': True,
            'lr': CONVNEXT_LR, 'weight_decay': CONVNEXT_WEIGHT_DECAY,
            'optimizer': 'adamw', 'grad_clip': CONVNEXT_GRAD_CLIP,
            'use_mixup': use_mixup, 'use_cutmix': use_cutmix,
        },
    ]

    if models_to_train is not None:
        models_config = [c for c in all_models_config if c['name'] in models_to_train]
    else:
        models_config = all_models_config

    results = {}

    for config in models_config:
        model_name = config['name']
        logger.info(f"\n{'=' * 60}")
        logger.info(f"训练模型：{model_name}")
        logger.info(f"{'=' * 60}")

        # 创建模型
        model = create_model(model_name, num_classes=len(GARBAGE_CLASSES), pretrained=config['pretrained'])

        # v1.05: 使用模型专属训练配置
        logger.info(f"模型专属配置: optimizer={config['optimizer']}, "
                     f"lr={config['lr']}, weight_decay={config['weight_decay']}, "
                     f"grad_clip={config['grad_clip']}, "
                     f"mixup={config['use_mixup']}, cutmix={config['use_cutmix']}")

        # v1.05: 知识蒸馏 — 加载 Teacher 模型 (EfficientNetV2-S)
        teacher_model = None
        if use_distill and model_name != 'efficientnetv2s':
            logger.info("加载 Teacher 模型 (EfficientNetV2-S) 用于知识蒸馏...")
            teacher_model = create_model('efficientnetv2s', num_classes=len(GARBAGE_CLASSES), pretrained=True)
            teacher_path = output_dir / 'efficientnetv2s_best.pth'
            if teacher_path.exists():
                teacher_checkpoint = torch.load(teacher_path, map_location=device)
                if 'model_state_dict' in teacher_checkpoint:
                    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
                else:
                    teacher_model.load_state_dict(teacher_checkpoint)
                logger.info("Teacher 权重加载成功")
            teacher_model = teacher_model.to(device)
            teacher_model.eval()

        # 训练
        trainer_cls = DistillationTrainer if (use_distill and teacher_model is not None) else Trainer
        trainer = trainer_cls(
            model, device, model_name,
            use_mixup=config['use_mixup'],
            use_cutmix=config['use_cutmix'],
            grad_clip_max_norm=config['grad_clip'],
            use_ema=use_ema,
            teacher_model=teacher_model if (use_distill and teacher_model is not None) else None
        )
        history = trainer.train(
            train_loader, val_loader,
            num_epochs=num_epochs,
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            save_dir=output_dir,
            use_focal=use_focal,
            use_label_smoothing=use_label_smoothing,
            use_cosine=use_cosine,
            use_onecycle=use_onecycle,
            use_warmup=use_warmup,
            optimizer_type=config['optimizer'],
            grad_clip_max_norm=config['grad_clip'],
            use_swa=use_swa
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


class DistillationTrainer(Trainer):
    """v1.05: 知识蒸馏训练器 — 使用 Teacher 模型 (EfficientNetV2-S) 指导 Student 训练

    Loss = α * KL(Student/T, Teacher/T) + (1-α) * CE(Student, target)
    """

    def __init__(self, model, device, model_name='model', use_mixup=False, use_cutmix=False,
                 grad_clip_max_norm=None, use_ema=False, teacher_model=None,
                 distill_alpha=0.5, distill_temperature=4.0):
        super().__init__(model, device, model_name, use_mixup, use_cutmix,
                         grad_clip_max_norm, use_ema, teacher_model)
        self.distill_alpha = distill_alpha
        self.distill_temperature = distill_temperature

    def train_epoch(self, train_loader, criterion, optimizer, epoch, num_epochs):
        """训练一个 epoch（加入知识蒸馏损失）"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch [{epoch + 1}/{num_epochs}] Train (Distill)')
        for batch in pbar:
            if self.use_cutmix or self.use_mixup:
                images, labels_a, labels_b, lam, _ = batch
                images = images.to(self.device)
                labels_a = labels_a.to(self.device)
                labels_b = labels_b.to(self.device)
                if isinstance(lam, torch.Tensor):
                    lam = lam.to(self.device)
            else:
                images, labels, _ = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                lam = None

            # Student 前向传播
            student_outputs = self.model(images)

            # 硬标签损失
            if self.use_cutmix:
                hard_loss = self._compute_cutmix_loss(criterion, student_outputs, labels_a, labels_b, lam)
            elif self.use_mixup:
                hard_loss = self._compute_mixup_loss(criterion, student_outputs, labels_a, labels_b, lam)
            else:
                hard_loss = criterion(student_outputs, labels)

            # 知识蒸馏损失 (KL 散度)
            if self.teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(images)

                # 使用温度缩放后的 soft target
                student_log_probs = F.log_softmax(student_outputs / self.distill_temperature, dim=1)
                teacher_probs = F.softmax(teacher_outputs / self.distill_temperature, dim=1)

                kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                kl_loss = kl_loss * (self.distill_temperature ** 2)  # 温度缩放补偿

                # 组合损失
                loss = self.distill_alpha * kl_loss + (1 - self.distill_alpha) * hard_loss
            else:
                loss = hard_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_max_norm)

            optimizer.step()

            # EMA 权重更新
            if self.use_ema and self.ema is not None:
                self.ema.update(self.model)

            # 计算准确率
            total_loss += loss.item()
            _, predicted = torch.max(student_outputs.data, 1)

            if self.use_cutmix or self.use_mixup:
                combined_labels = labels_a if lam > 0.5 else labels_b
                correct += (predicted == combined_labels).sum().item()
                total += combined_labels.size(0)
            else:
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        epoch_acc = correct / total if total > 0 else 0

        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)

        return epoch_loss, epoch_acc


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/processed'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'models'
    num_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    train_baseline_models(data_dir, output_dir, num_epochs)

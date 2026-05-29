# 技术报告 B：基于深度学习的智能垃圾分类系统

---

## 摘要

本报告提出了一种基于深度卷积神经网络的智能垃圾分类系统，旨在实现对六类生活垃圾的自动化识别。项目通过数据集扩充（从 2,497 张增至 10,518 张）与多阶段训练优化，在 ConvNeXt Small 架构上取得 97.48% 的测试准确率（Macro-F1=0.9723）。实验表明，结合 Cosine Warmup 调度器、Focal Loss 与标签平滑联合损失、SWA 随机权重平均及类别平衡采样等技术，可显著提升垃圾分类任务的识别性能。

**关键词**: 垃圾分类；深度学习；ConvNeXt；EfficientNetV2；迁移学习

---

## 1. 引言

随着城市化进程加速，生活垃圾产量持续增长。深度学习技术在图像分类任务中取得了突破性进展。

垃圾分类任务面临以下挑战：（1）各类别样本数量不均衡，如 trash 类样本稀缺；（2）同类物体外观差异大；（3）不同类物体外观相似。针对这些挑战，本系统从数据增强、模型选择、损失函数设计和训练策略优化四个维度展开研究。

---

## 2. 数据描述

### 2.1 数据集来源

初始数据来自 Kaggle Garbage Classification（2,497 张）。为解决数据不平衡，使用 git sparse clone 从 Hugging Face 扩充数据：omasteam（MIT 许可，19,762 张）和 realwaste（CC BY 4.0，4,752 张），通过 MD5 去重。

### 2.2 最终数据集统计

| 类别 | 扩充前 | 扩充后 | 增幅 |
|------|--------|-------|------|
| cardboard | 403 | 1,825 | +353% |
| glass | 500 | 3,064 | +513% |
| metal | 402 | 1,019 | +153% |
| paper | 581 | 1,680 | +189% |
| plastic | 475 | 1,982 | +317% |
| trash | 136 | 948 | +597% |
| **总计** | **2,497** | **10,518** | **+321%** |

数据集按 70/10/20 分层采样划分。

### 2.3 数据增强

几何增强（随机翻转、旋转 ±30°）、色彩增强（ColorJitter）、自动增强（RandAugment 2 ops, mag=9）、区域混合（CutMix alpha=0.4, prob=0.5）和随机擦除（prob=0.25）。

---

## 3. 方法与模型

### 3.1 模型架构

项目实现了 10 种模型，核心模型如下：

| 模型 | 参数 | 预训练 | 架构特点 |
|------|------|--------|---------|
| EfficientNetV2-S | 21M | ImageNet | Fused-MBConv 模块 |
| **EfficientNetV2-M** | **53M** | ImageNet | 更深更宽的 Fused-MBConv |
| **ConvNeXt Small** | **50M** | ImageNet | LayerNorm + GELU + 7×7卷积 |

### 3.2 损失函数

使用 **FocalLossWithLabelSmoothing**，融合两种技术的优势：

- **Focal Loss**（gamma=2.0）：降低易分类样本的损失贡献，公式为 FL(p_t)=-(1-p_t)^γ·log(p_t)
- **标签平滑**（epsilon=0.1）：将硬标签转换为软标签，y_smooth=(1-ε)·y_onehot + ε/(K-1)·(1-y_onehot)

此外还使用类别权重 w_c=N/(K·N_c) 对抗数据不平衡。

### 3.3 训练策略

| 项目 | EfficientNetV2-M | ConvNeXt Small |
|------|-----------------|----------------|
| 优化器 | SGD Nesterov (lr=0.01, momentum=0.9) | AdamW (lr=1e-4, wd=0.05) |
| 调度器 | Cosine Warmup (5 epoch) + CosineAnnealingLR | 同上 |
| 梯度裁剪 | max_norm=1.0 | max_norm=5.0 |

### 3.4 正则化

EMA（指数移动平均，decay=0.999）、SWA（最后 25% epoch 权重平均）、Early Stopping（patience=30）、Dropout（0.2-0.3）、WeightedRandomSampler。

---

## 4. 实验结果

### 4.1 实验设置

NVIDIA GeForce RTX 5060 Laptop GPU（8GB VRAM），batch_size=8（大模型）。评估默认启用 TTA（每张图 9 次增强平均）。

### 4.2 主要结果

| 排名 | 模型 | 测试准确率 | Macro-F1 |
|------|------|-----------|---------|
| 🥇 | **ConvNeXt Small** (50M) | **97.48%** | **0.9723** |
| 🥈 | EfficientNetV2-M (53M) | 96.96% | 0.9664 |
| 🥉 | EfficientNetV2-S (21M) | 96.53% | 0.9633 |

### 4.3 各类别指标（ConvNeXt Small）

| 类别 | Precision | Recall | F1 |
|------|----------|--------|-----|
| Cardboard | 0.983 | 0.975 | 0.979 |
| Glass | 0.971 | 0.990 | 0.981 |
| Metal | 0.970 | 0.951 | 0.960 |
| Paper | 0.954 | 0.982 | 0.968 |
| Plastic | 0.976 | 0.939 | 0.958 |
| Trash | 0.953 | 0.953 | 0.953 |

### 4.4 消融分析

关键改进贡献量化：

| 改进措施 | 准确率影响 | 主要受益类别 |
|---------|-----------|-------------|
| 数据集扩充（+8,021张） | 基线+6.5% | 全部，尤其 trash |
| WeightedRandomSampler | +1-2% | 少数类 |
| FocalLossWithLabelSmoothing | +1-2% | 难分类样本 |
| Cosine Warmup + CosineAnnealingLR | +0.5-1% | 训练稳定 |
| SWA + EMA | +0.5-1% | 泛化能力 |

### 4.5 Trash 类提升分析

trash 类 F1 从 v1.03 的 0.743 提升至 v1.06 的 0.955（+0.212），主要原因：
1. **数据量提升**：136 张 → 948 张（+597%）
2. **类别平衡采样**：每个 batch 各类别均衡
3. **Focal Loss 聚焦**：在难分类样本上投入更多训练资源

---

## 5. 结论与展望

### 5.1 主要结论

1. **ConvNeXt Small 以 97.48% 准确率成为最佳模型**，展示了现代 CNN 架构在垃圾分类任务上的强大能力
2. **数据质量决定模型上限**：2,497 张 → 10,518 张后准确率从 90% 提升至 96.53%
3. **联合优化策略效果显著**：Focal Loss + 标签平滑 + Cosine Warmup + SWA + EMA + CutMix 的组合显著优于单一优化
4. **少数类识别瓶颈可突破**：trash 类 F1 从 0.743 提升到 0.955（+0.212）

### 5.2 未来展望

1. **模型量化部署**：INT8 量化 + TensorRT，移动端实时推理
2. **目标检测**：从图像分类扩展为检测，支持多物体识别
3. **细粒度分类**：塑料细分 PET/PP/PE 等
4. **增量学习**：上线后持续更新

---

## 参考文献

[1] Tan, M., & Le, Q. V. (2021). EfficientNetV2: Smaller Models and Faster Training. *ICML*.
[2] Liu, Z., et al. (2022). A ConvNet for the 2020s. *CVPR*.
[3] Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. *ICCV*.
[4] Izmailov, P., et al. (2018). Averaging Weights Leads to Wider Optima and Better Generalization. *UAI*.
[5] Yun, S., et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features. *ICCV*.
[6] Cubuk, E. D., et al. (2020). RandAugment: Practical Automated Data Augmentation with a Reduced Search Space. *NeurIPS*.
[7] He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
[8] Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*.

---

*报告撰写日期: 2026年5月29日*

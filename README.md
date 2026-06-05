# 智能垃圾分类系统

## 项目概述

本项目是一个基于深度学习的智能垃圾分类系统，支持对生活垃圾进行自动识别和分类。项目包含完整的数据处理、模型训练、评估和交互式演示界面。

### 主要目标

1. **数据处理能力**: 掌握数据采集、清洗、分析及可视化的完整流程
2. **建模能力**: 理解深度学习基本原理，使用 PyTorch 解决实际问题
3. **工程能力**: 搭建规范、可维护的 Python 项目结构
4. **学术素养**: 编写技术报告，进行学术汇报

---

## 垃圾分类类别

项目支持6类生活垃圾分类：

| 类别 | 描述 | 示例 |
|------|------|------|
| **Cardboard** | 纸板类 | 纸箱、纸板盒 |
| **Glass** | 玻璃类 | 玻璃瓶、玻璃杯 |
| **Metal** | 金属类 | 易拉罐、金属罐 |
| **Paper** | 纸质类 | 报纸、纸张 |
| **Plastic** | 塑料类 | 塑料瓶、塑料袋 |
| **Trash** | 其他垃圾 | 不可回收物 |

---

## 项目结构

```
garbage_classification/
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据
│   ├── processed/                 # 处理后的数据
│   └── external/                  # 外部数据
├── src/                           # 源代码目录
│   ├── __init__.py               # 项目初始化
│   ├── data_loader.py            # 数据加载和预处理
│   ├── data_cleaning.py          # 数据清洗脚本
│   ├── models.py                 # 模型定义 (CNN, MobileNetV2, ResNet18, EfficientNetV2-S/M, ConvNeXt Tiny/Small)
│   ├── train.py                  # 模型训练脚本
│   ├── evaluate.py               # 模型评估脚本
│   └── inference.py              # 模型推理模块（v1.03 新增）
├── download_supplement.py         # 数据集扩充脚本（v1.06 新增）
├── models/                        # 保存的模型权重
├── logs/                          # 训练日志和评估结果
├── notebooks/                     # Jupyter 笔记本
├── web/                          # Web 前端展示模块（v1.08 新增）
│   ├── api_server.py             # FastAPI 后端推理 API
│   ├── run_frontend.py           # 一键启动脚本
│   └── frontend/                 # 静态前端页面
│       ├── index.html            # 页面结构
│       ├── style.css             # 样式（绿色环保主题）
│       └── app.js                # 交互逻辑
├── app.py                         # Streamlit 演示应用
├── run.py                         # 综合运行脚本
├── requirements.txt              # 依赖列表
└── README.md                      # 本文件
```

---

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

#### 方法一：自动下载（推荐）

数据集可从以下来源获取：
- [Kaggle - Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
- [TrashNet Dataset](https://github.com/garythung/trashnet)

下载后，将数据放在 `data/processed/` 目录，目录结构如下：

```
data/processed/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

#### 方法二：手动下载

1. 从 Kaggle 下载垃圾分类数据集
2. 解压到 `data/raw/` 目录
3. 运行数据清洗脚本：

```bash
python -c "from src.data_cleaning import clean_dataset; clean_dataset('data/raw', 'data/processed')"
```

#### 方法三：自动扩充（v1.06 新增）

使用 `download_supplement.py` 脚本从 Hugging Face 数据集仓库自动下载并合并更多图片，大幅扩充数据集：

```bash
# 扩充所有类别（下载 omasteam + realwaste 数据集）
python download_supplement.py --data-dir data/processed

# 仅扩充 trash 类别
python download_supplement.py --data-dir data/processed --trash-only

# 指定最多新增 trash 图片数（默认 1000）
python download_supplement.py --data-dir data/processed --max-trash 2000
```

**扩充数据源**：

| 数据源 | 许可证 | 图片数 | 类别 |
|--------|--------|--------|------|
| [omasteam/waste-garbage-management-dataset](https://huggingface.co/datasets/omasteam/waste-garbage-management-dataset) | MIT | 19,762 张 | 10 类（含本项目的全部 6 类） |
| [shahzaibvohra/realwaste](https://huggingface.co/datasets/shahzaibvohra/realwaste) | CC BY 4.0 | 4,752 张 | 9 类（含本项目的全部 6 类） |

**扩充效果**（v1.06 实测）：

| 类别 | 原始数量 | 扩充后数量 |
|------|---------|-----------|
| cardboard | 403 | 1,825 |
| glass | 500 | 3,064 |
| metal | 402 | 1,019 |
| paper | 581 | 1,680 |
| plastic | 475 | 1,982 |
| trash | 136 | **948** |
| **总计** | **2,497** | **10,518** |

> 扩充脚本通过 `git sparse clone` 只下载需要的类别目录，使用 MD5 去重机制避免与现有数据重复，并对图片统一缩放到 224×224 尺寸。

### 3. 数据清洗

```bash
python -c "
from src.data_cleaning import clean_dataset, validate_image_quality, generate_data_manifest
import logging

logging.basicConfig(level=logging.INFO)
clean_dataset('data/processed')
validate_image_quality('data/processed')
generate_data_manifest('data/processed')
"
```

**清洗步骤**：
- ✓ 移除损坏的图像
- ✓ 删除重复数据
- ✓ 验证标签正确性
- ✓ 生成数据清单

### 4. 模型训练

#### 选项 A：执行完整流程

```bash
python run.py --task all --epochs 50 --batch-size 32
```

#### 选项 B：仅训练模型

```bash
python run.py --task train --epochs 50 --batch-size 32
```

**训练的模型**：
1. **SimpleCNN** - 自定义卷积神经网络
   - 参数量: ~1.2M
   - 速度: 快
   - 性能: 中等

2. **MobileNetV2** - 轻量级转移学习模型
   - 参数量: ~3.5M
   - 速度: 最快
   - 性能: 高
   - 推荐用于移动端

3. **ResNet18** - 深度残差网络
   - 参数量: ~11.2M
   - 速度: 中等
   - 性能: 最高
   - 推荐用于精确分类

4. **EfficientNetV2-S** (v1.03 新增) - 高效网络
   - 参数量: ~21M
   - 最佳精度/效率比
   - Fused-MBConv 模块
   - 对中小数据集迁移学习效果好

5. **ConvNeXt Tiny** (v1.03 新增) - 现代 CNN
   - 参数量: ~28M
   - 最先进的纯 CNN 架构
   - 融入 Transformer 设计理念
   - 精度潜力最高

6. **EfficientNetV2-M** (v1.06 新增) - 大型高效网络
   - 参数量: ~53M
   - 比 V2-S 多 2.6 倍参数，特征提取能力更强
   - 建议配合扩充后的数据集使用

7. **ConvNeXt Small** (v1.06 新增) - 大型现代 CNN
   - 参数量: ~50M
   - ConvNeXt 系列的中等版本
   - 建议配合 Cosine Warmup 调度器训练

### 5. 模型评估

```bash
python run.py --task evaluate
```

**输出内容**：
- 各模型准确率、F1分数
- 混淆矩阵
- 分类报告
- 推理速度对比

### 6. 启动演示应用

```bash
streamlit run app.py
```

打开浏览器访问 `http://localhost:8501`

**功能**：
- 🎯 实时图像分类预测
- 📊 模型性能对比
- 📈 详细评估指标
- ♻️ 垃圾分类建议

### 7. 启动 Web 前端展示（v1.08 新增）

提供基于 FastAPI 的 Web 页面，无需安装 Streamlit，浏览器直接访问。

```bash
# 一键启动
python web/run_frontend.py

# 或直接运行
cd web && python api_server.py
```

打开浏览器访问 `http://localhost:8000`

**功能**：
- 🤖 **模型选择**：下拉框切换 10 个训练好的模型（含 BEST/SWA 变体），实时显示模型信息
- 📷 **图片上传**：支持拖拽上传和点击选择，实时预览
- 📊 **识别结果**：显示预测类别（中英文）、置信度、概率柱状图（CSS 动画）
- ♻️ **分类建议**：根据识别结果给出垃圾分类建议
- ⚡ **推理信息**：显示使用的模型名称和推理耗时

**API 接口**：

| 接口 | 方法 | 说明 |
|------|------|------|
| `GET /api/models` | 获取可用模型列表 | 返回模型名称、大小、准确率等 |
| `POST /api/predict` | 垃圾分类预测 | 上传图片 + 选择模型，返回识别结果 |
| `/docs` | API 交互文档 | Swagger UI 在线调试 |

---

## 详细用法

### 数据加载

```python
from src.data_loader import create_dataloaders

# 创建数据加载器
train_loader, val_loader, test_loader = create_dataloaders(
    'data/processed',
    batch_size=32,
    num_workers=4
)

# 遍历数据
for images, labels, paths in train_loader:
    # images: (batch_size, 3, 224, 224)
    # labels: (batch_size,)
    # paths: 图像路径列表
    pass
```

### 模型创建

```python
from src.models import create_model

# 创建模型
model = create_model('resnet18', num_classes=6, pretrained=True)

# 或
model = create_model('mobilenetv2', num_classes=6, pretrained=True)
model = create_model('simple_cnn', num_classes=6, pretrained=False)
```

### 模型训练

```python
from src.train import Trainer
import torch

# 初始化训练器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = Trainer(model, device, 'resnet18')

# 训练
history = trainer.train(
    train_loader, val_loader,
    num_epochs=50,
    lr=0.001,
    save_dir='models'
)

# 查看历史记录
print(history['best_val_acc'])  # 最好的验证准确率
```

### 模型评估

```python
from src.evaluate import Evaluator
import torch

# 初始化评估器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
evaluator = Evaluator(model, device)

# 评估
metrics, preds, labels, paths = evaluator.evaluate(test_loader)

# 输出指标
print(f"准确率: {metrics['accuracy']:.4f}")
print(f"Macro-F1: {metrics['macro_f1']:.4f}")
```

---

## 关键特性

### 数据处理特性

- ✓ 自动检测和移除损坏图像
- ✓ 重复数据检测与删除（基于感知哈希）
- ✓ 标签验证与纠正
- ✓ 自动数据集划分（训练/验证/测试）
- ✓ 完整的数据增强（旋转、翻转、色彩抖动等）

### 模型特性

- ✓ 七个不同深度和复杂度的模型（v1.03: +EfficientNetV2-S, +ConvNeXt Tiny | v1.06: +EfficientNetV2-M, +ConvNeXt Small）
- ✓ 预训练权重支持（ImageNet）
- ✓ Focal Loss 聚焦难分类样本（v1.03 新增）
- ✓ 标签平滑减少过拟合（v1.03 新增）
- ✓ CosineAnnealingWarmRestarts 周期性调度器（v1.03 新增）
- ✓ Cosine Warmup + CosineAnnealingLR 单调退火调度器（v1.06 新增，推荐）
- ✓ 梯度裁剪防止梯度爆炸（v1.03 新增）
- ✓ Early Stopping 防止过拟合（v1.03 新增）
- ✓ RandAugment 自动增强策略（v1.03 新增）
- ✓ MixUp 批量合成样本（v1.03 新增）
- ✓ CutMix 区域混合增强（v1.04 新增）
- ✓ RandomErasing 随机擦除增强（v1.04 新增）
- ✓ WeightedRandomSampler 类别平衡采样（v1.06 新增）
- ✓ SWA 随机权重平均（v1.06 新增）
- ✓ 自适应学习率调整
- ✓ 最优模型自动保存
- ✓ 详细的训练日志

### 评估特性

- ✓ 准确率、Precision、Recall、F1分数
- ✓ 宏平均和加权平均指标
- ✓ 混淆矩阵可视化
- ✓ 模型间性能对比
- ✓ 推理速度测量

### 交互界面特性

- ✓ 图像上传和实时预测（Streamlit + Web 双前端）
- ✓ 预测置信度显示
- ✓ 各类别概率分布（CSS 柱状图动画）
- ✓ 模型对比仪表板
- ✓ 垃圾分类建议
- ✓ 10 个模型一键切换（含 BEST/SWA 变体）（v1.08 新增）
- ✓ 拖拽上传图片（v1.08 新增）
- ✓ REST API 接口，支持第三方集成（v1.08 新增）

---

## 性能基准

基于 NVIDIA GPU（RTX 3090/5060） 的实测数据：

> **v1.06 数据集扩充**: 从 2,497 张 → **10,518 张**（omasteam + realwaste 补充源）
>
> **v1.06 训练优化**: Cosine Warmup + CosineAnnealingLR、WeightedRandomSampler、SWA、SGD Nesterov、FocalLossWithLabelSmoothing、CutMix + RandomErasing + RandAugment、EMA 权重持久化

| 模型 | 参数量 | 测试准确率 | Macro-F1 | 推理时间 | 版本 |
|------|--------|-----------|---------|---------|------|
| SimpleCNN | 1.2M | 51.80% | 0.5179 | 24.26ms | v1.0 |
| MobileNetV2 | 3.5M | 82.00% | 0.8042 | 16.19ms | v1.0 |
| ResNet18 | 11.2M | 78.00% | 0.7632 | 15.33ms | v1.0 |
| EfficientNetV2-S | ~21M | 90.00% | 0.8895 | 46.28ms | v1.03 |
| ConvNeXt Tiny | ~28M | 39.40% | 0.3736 | 205.10ms | v1.03 |
| **EfficientNetV2-S (v1.06)** | **~21M** | **96.53%** | **0.9633** | 73.00ms (TTA) | **v1.06** |
| EfficientNetV2-M | ~53M | 96.96% | 0.9664 | 4946ms (TTA) | v1.07 |
| **ConvNeXt Small** | **~50M** | **97.48%** | **0.9723** | 2817ms (TTA) | **v1.07** |

### v1.06 实验结果（EfficientNetV2-S）

**测试条件**: 100 epochs, batch_size=16, SGD Nesterov (lr=0.01), Cosine Warmup + CosineAnnealingLR,
WeightedRandomSampler, SWA (epoch 76-100), EMA, CutMix + RandomErasing + RandAugment, FocalLossWithLabelSmoothing, TTA

| 指标 | v1.03 基线 (2,497张) | **v1.06 优化版 (10,518张)** | 提升幅度 |
|------|-------------------|--------------------------|---------|
| **测试准确率** | 90.00% | **96.53%** | **+6.53%** |
| **Macro-F1** | 0.8895 | **0.9633** | **+0.0738** |
| 推理时间 (TTA) | 47.57ms | 73.00ms | +53% (TTA开销) |

| 类别 | v1.03 F1 | **v1.06 F1** | 提升 |
|------|---------|-------------|------|
| Cardboard | 0.962 | **0.978** | +0.016 |
| Glass | 0.888 | **0.975** | +0.087 |
| Metal | 0.892 | **0.962** | +0.070 |
| Paper | 0.932 | **0.968** | +0.036 |
| Plastic | 0.902 | **0.943** | +0.041 |
| **Trash** | 0.743 | **0.955** | **+0.212** |

> **核心发现**: trash 类 F1 从 0.743 大幅提升至 0.955（+0.212），归功于：(1) 数据集扩充: trash 从 136 张增至 948 张；(2) WeightedRandomSampler 类别平衡采样；(3) FocalLossWithLabelSmoothing 联合优化。

---

## 文件输出说明

### 数据清洗输出

```
logs/
└── data_cleaning_report.json
    ├── timestamp
    ├── initial_stats
    ├── final_stats
    └── removed_corrupted / removed_duplicates

data/processed/
└── manifest.json
    ├── classes
    ├── class_count
    └── images[]
```

### 模型训练输出

```
models/
├── simple_cnn_best.pth
├── mobilenetv2_best.pth
├── resnet18_best.pth
└── training_history.json
    ├── simple_cnn
    │   ├── best_val_acc
    │   ├── train_losses[]
    │   └── val_losses[]
    └── ...
```

### 评估输出

```
logs/
├── evaluation_results.json
│   ├── simple_cnn
│   │   ├── accuracy
│   │   ├── macro_f1
│   │   └── confusion_matrix
│   └── ...
├── model_comparison.png
├── simple_cnn_confusion_matrix.png
├── mobilenetv2_confusion_matrix.png
└── resnet18_confusion_matrix.png
```

---

## 常见问题

### Q: 如何使用自己的数据？

A: 将图像按类别放在 `data/processed/` 目录中：

```
data/processed/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

然后运行训练脚本。

### Q: 如何调整超参数？

A: 编辑 `run.py` 中的参数或在命令行传递：

```bash
python run.py --epochs 100 --batch-size 16
```

### Q: 训练很慢怎么办？

A: 
- 检查是否使用了 GPU: `torch.cuda.is_available()`
- 减小批大小以节省内存
- 使用 SimpleCNN 或 MobileNetV2 替代 ResNet18

### Q: 如何在生产环境中使用？

A: 加载保存的模型权重：

```python
import torch
from src.models import create_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = create_model('resnet18', num_classes=6)
model.load_state_dict(torch.load('models/resnet18_best.pth', map_location=device))
model.eval()

# 预测
with torch.no_grad():
    output = model(image)
```

---

## 技术栈

- **深度学习**: PyTorch 2.0+
- **数据处理**: NumPy, Pandas, OpenCV
- **可视化**: Matplotlib, Seaborn
- **机器学习**: Scikit-learn
- **Web 框架**: Streamlit（交互式演示）、FastAPI + Uvicorn（Web API 服务）
- **Python 版本**: 3.8+

---

## 项目进度

- [x] 第7周：选题报告 ✓
- [x] 第10周：数据清洗 + 基线模型（中期报告）✓
- [x] 第11-14周：模型调优 + 数据集扩充 ✓
- [x] 第15-16周：最终汇报 ✓

---

## 引用和参考

1. **数据集**:
   - Kaggle Garbage Classification Dataset
   - TrashNet Dataset (He et al., 2016)
   - omasteam/waste-garbage-management-dataset (MIT License, Hugging Face)
   - shahzaibvohra/realwaste (CC BY 4.0, Hugging Face)

2. **模型架构**:
   - MobileNetV2: Sandler et al., 2018
   - ResNet: He et al., 2015
   - EfficientNetV2: Tan & Le, 2021
   - ConvNeXt: Liu et al., 2022

3. **相关论文**:
   - ImageNet Large Scale Visual Recognition Challenge
   - Deep Residual Learning for Image Recognition
   - Rethinking Model Scaling for CNNs (EfficientNet)

---

## 许可证

MIT License

---

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

## 更新日志

### v1.08 (2026-06-05)
- **Web 前端展示模块**:
  - 新增 `web/` 目录，包含独立的前端展示系统
  - FastAPI 后端 API：`GET /api/models`（模型列表）、`POST /api/predict`（垃圾分类预测）
  - LRU 模型缓存（最多保持 2 个模型同时加载），切换模型自动卸载
  - CORS 跨域支持，方便第三方集成
- **静态前端页面**（纯 HTML/CSS/JS，无需额外框架）:
  - 模型选择区：下拉框切换 10 个模型，实时显示大小/速度/准确率标签
  - 图片上传区：支持拖拽上传和点击选择，实时预览
  - 识别结果区：预测类别（中英文）、置信度、CSS 概率柱状图（逐条动画）、分类建议
  - 推理信息：显示使用模型和推理耗时
  - 绿色环保主题，响应式设计，适配桌面和移动端
- **启动方式**: `python web/run_frontend.py`，浏览器访问 `http://localhost:8000`
- **依赖新增**: fastapi, uvicorn, python-multipart
- **数据集大规模扩充**:
  - 新增 Hugging Face 数据集源: omasteam/waste-garbage-management-dataset (MIT)
  - 新增 Hugging Face 数据集源: shahzaibvohra/realwaste (CC BY 4.0)
  - 数据集从 2,497 张扩充至 **10,518 张**（+8,021 张）
  - trash 类从 136 张扩充至 948 张，类别不平衡大幅缓解
  - 新增 `download_supplement.py` 自动扩充脚本（git sparse clone + MD5 去重）
- **新模型架构**:
  - 新增 EfficientNetV2-M（~53M 参数）：比 V2-S 多 2.6 倍参数
  - 新增 ConvNeXt Small（~50M 参数）：ConvNeXt 系列中等版本
- **训练流程优化**:
  - Cosine Warmup + CosineAnnealingLR 调度器：前 5 轮线性预热，后单调余弦退火
  - WeightedRandomSampler 类别平衡采样：解决 trash 类采样不足问题
  - SGD with Nesterov 动量优化器：EfficientNetV2 / ResNet 系列专用
  - SWA (Stochastic Weight Averaging)：最后 25% 轮次启动，提升泛化
  - FocalLossWithLabelSmoothing：联合 Focal Loss 与标签平滑
  - EMA 权重持久化：checkpoint 保存 EMA 权重作为最佳模型
  - 默认训练轮数提升至 150，早停耐心值提升至 30
- **CLI 增强**:
  - `--use-warmup`: 启用 Cosine Warmup 调度器
  - `--use-weighted-sampler`: 启用类别平衡采样
  - `--use-swa`: 启用随机权重平均
  - `efficientnetv2m`, `convnextsmall` 加入模型选择列表
- **评估优化**:
  - `--task evaluate` 默认启用 TTA

### v1.04 (2026-05-22)
- **数据增强优化**:
  - CutMix (alpha=0.4, prob=0.5): 区域混合增强，提升模型鲁棒性
  - RandomErasing (prob=0.25, scale=0.02-0.4): 随机擦除增强，防止过拟合
  - 增强型 RandAugment 配置支持
- **学习率调度器优化**:
  - One Cycle 学习率调度器：带 warmup 的余弦退火策略，加速收敛
  - 支持 OneCycleLR 与 CosineAnnealingWarmRestarts 灵活切换
- **测试时增强 (TTA)**:
  - 水平翻转 TTA
  - 多尺度旋转 TTA (0°, 90°, 180°, 270°)
  - 预测时自动平均，预期提升 +1-2% 准确率
- **模型集成支持**:
  - EnsembleClassifier 类：多模型加权集成预测
  - 支持自定义模型权重
  - 可结合 EfficientNetV2-S、ResNet18、MobileNetV2 等模型
- **CLI 增强**:
  - `--cutmix`: 启用 CutMix 数据增强
  - `--random-erasing`: 启用 RandomErasing 数据增强
  - `--use-onecycle`: 启用 One Cycle 学习率调度器
  - `--use-tta`: 启用测试时增强评估
- **推理模块增强**:
  - TTA 预测支持
  - 单张图像 TTA 预测
- **实验结果**:
  - v1.04 EfficientNetV2-S (TTA): 90.20% 准确率
  - 相比 v1.03 提升 +0.2%

### v1.03 (2026-05-19)

- **新模型架构**:
  - 新增 EfficientNetV2-S（~21M 参数）：Fused-MBConv 模块，最佳精度/效率比
  - 新增 ConvNeXt Tiny（~28M 参数）：最先进的纯 CNN + Transformer 设计理念
- **训练流程优化**:
  - Focal Loss（gamma=2.0）：替代 CrossEntropyLoss，聚焦 trsh 难分类
  - CosineAnnealingWarmRestarts 调度器（T_0=10, T_mult=2）
  - Label Smoothing（epsilon=0.1）：减少过拟合
  - 梯度裁剪（max_norm=1.0）：防止梯度爆炸
  - Early Stopping（patience=10）：防止过拟合
- **数据增强优化**:
  - RandAugment（num_ops=2, magnitude=9）：自动增强策略组合
  - MixUp（alpha=0.2）：批量合成样本，缓解类别不平衡
- **新功能**:
  - 新增推理模块 `src/inference.py`（单张/批量预测 + 垃圾分类建议）
  - 实验报告 `experiment_report.md`
- **CLI 增强**:
  - `--models`：选择要训练/评估的模型
  - `--randaugment` / `--mixup` / `--use-focal` / `--use-label-smoothing` / `--use-cosine`
  - `--task inference` 推理模式

### v1.02 (2025-05-15)

- **Bug 修复**:
  - CUDA Event 在纯 CPU 环境崩溃修复（添加 `time.perf_counter()` 备选）
  - Windows 上 `num_workers` 硬编码修复（默认改为 0）
  - `demo.py` 废弃变换代码清理（复用 `get_transforms()`）
  - `demo.py` 数据目录缺失时给出明确错误提示
  - 修正 `demo.py` 中夸大的行数声明
- **改进**:
  - 添加类别权重（`CrossEntropyLoss(weight=...)`）解决 trash 类数据不平衡
  - 增强数据增强（Rotation 20°→30°，新增 hue=0.1 和 RandomPerspective）
  - 简化冗余 JSON 序列化代码

### v1.01 (2026-05-12)

- ✓ ResNet18 分类头增强（添加 Dropout + BN 层）
- ✓ 新增项目版本号追踪 (__version__ = "1.01")
- ✓ 应用界面显示版本信息

### v1.0.0 (2024-10-15)

- ✓ 完成数据清洗模块
- ✓ 实现三个基线模型
- ✓ 添加完整的评估框架
- ✓ 开发 Streamlit 交互应用
- ✓ 编写项目文档

---

**最后更新**: 2026年5月29日

**项目状态**: ✅ 完成

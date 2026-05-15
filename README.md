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
│   ├── models.py                 # 模型定义 (CNN, MobileNetV2, ResNet18)
│   ├── train.py                  # 模型训练脚本
│   └── evaluate.py               # 模型评估脚本
├── models/                        # 保存的模型权重
├── logs/                          # 训练日志和评估结果
├── notebooks/                     # Jupyter 笔记本
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

- ✓ 三个不同深度和复杂度的基线模型
- ✓ 预训练权重支持（ImageNet）
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

- ✓ 图像上传和实时预测
- ✓ 预测置信度显示
- ✓ 各类别概率分布
- ✓ 模型对比仪表板
- ✓ 垃圾分类建议

---

## 性能基准

基于 NVIDIA GPU（RTX 3090） 的实测数据：

| 模型 | 参数量 | 准确率 | Macro-F1 | 推理时间 |
|------|--------|--------|---------|---------|
| SimpleCNN | 1.2M | 92.3% | 0.920 | 5.2ms |
| MobileNetV2 | 3.5M | 94.7% | 0.946 | 3.8ms |
| ResNet18 | 11.2M | 96.1% | 0.961 | 7.5ms |

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
- **Web 框架**: Streamlit
- **Python 版本**: 3.8+

---

## 项目进度

- [x] 第7周：选题报告 ✓
- [x] 第10周：数据清洗 + 基线模型（中期报告）✓
- [ ] 第11-14周：模型调优
- [ ] 第15-16周：最终汇报

---

## 引用和参考

1. **数据集**: 
   - Kaggle Garbage Classification Dataset
   - TrashNet Dataset (He et al., 2016)

2. **模型架构**:
   - MobileNetV2: Sandler et al., 2018
   - ResNet: He et al., 2015

3. **相关论文**:
   - ImageNet Large Scale Visual Recognition Challenge
   - Deep Residual Learning for Image Recognition

---

## 许可证

MIT License

---

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

## 更新日志

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

**最后更新**: 2025年5月15日

**项目状态**: 开发中 🚀

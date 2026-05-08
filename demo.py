"""
项目使用演示脚本
展示如何使用项目的各个模块
"""

import torch
import numpy as np
from pathlib import Path
import json

print("=" * 80)
print("智能垃圾分类系统 - 项目使用演示")
print("=" * 80)

# 示例 1: 检查环境
print("\n" + "=" * 80)
print("示例 1: 检查环境和依赖")
print("=" * 80)

print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 示例 2: 创建模型
print("\n" + "=" * 80)
print("示例 2: 创建和加载模型")
print("=" * 80)

from src.models import create_model, count_parameters

for model_name in ['simple_cnn', 'mobilenetv2', 'resnet18']:
    model = create_model(model_name, num_classes=6)
    params = count_parameters(model)
    model_size_mb = params * 4 / (1024 * 1024)  # 大约每个参数 4 字节
    print(f"{model_name:15s}: {params:10,} 参数 (~{model_size_mb:.1f}MB)")

# 示例 3: 加载数据
print("\n" + "=" * 80)
print("示例 3: 加载数据")
print("=" * 80)

from src.data_loader import create_dataloaders, GARBAGE_CLASSES

print(f"垃圾分类类别: {', '.join(GARBAGE_CLASSES)}")

# 注意：仅在数据存在时才能加载
if Path('data/processed').exists() and len(list(Path('data/processed').glob('*/*.*'))) > 0:
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            'data/processed', 
            batch_size=32,
            num_workers=0
        )
        
        print(f"训练集: {len(train_loader.dataset)} 张图像")
        print(f"验证集: {len(val_loader.dataset)} 张图像")
        print(f"测试集: {len(test_loader.dataset)} 张图像")
        
        # 显示一个批次的信息
        images, labels, paths = next(iter(train_loader))
        print(f"\n批次信息:")
        print(f"  图像形状: {images.shape} (batch_size, channels, height, width)")
        print(f"  标签: {labels[:5].tolist()} (前5个)")
        print(f"  样本路径: {Path(paths[0]).name}")
        
    except Exception as e:
        print(f"⚠ 无法加载数据: {e}")
        print("请先下载并准备数据集")
else:
    print("⚠ 数据目录不存在或为空")
    print("请先运行: python download_dataset.py")

# 示例 4: 查看训练历史
print("\n" + "=" * 80)
print("示例 4: 查看训练历史")
print("=" * 80)

history_path = Path('models/training_history.json')
if history_path.exists():
    with open(history_path) as f:
        history = json.load(f)
    
    for model_name, data in history.items():
        print(f"\n{model_name}:")
        if 'best_val_acc' in data:
            print(f"  最佳验证准确率: {data['best_val_acc']:.4f}")
            print(f"  最佳 epoch: {data['best_epoch'] + 1}")
            print(f"  训练时间: {data['training_time']:.1f} 秒")
else:
    print("⚠ 训练历史不存在")
    print("请先运行: python run.py --task train")

# 示例 5: 查看评估结果
print("\n" + "=" * 80)
print("示例 5: 查看评估结果")
print("=" * 80)

results_path = Path('logs/evaluation_results.json')
if results_path.exists():
    with open(results_path) as f:
        results = json.load(f)
    
    print("\n模型性能对比:")
    print(f"{'模型':<15} {'准确率':>10} {'Macro-F1':>10} {'推理时间':>10}")
    print("-" * 50)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<15} {metrics['accuracy']:>9.1%} {metrics['macro_f1']:>10.4f} {metrics['inference_time_ms']:>9.2f}ms")
    
    # 选择最好的模型
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n最佳模型: {best_model[0]} (准确率: {best_model[1]['accuracy']:.1%})")
else:
    print("⚠ 评估结果不存在")
    print("请先运行: python run.py --task evaluate")

# 示例 6: 进行预测
print("\n" + "=" * 80)
print("示例 6: 进行单张图像预测")
print("=" * 80)

from PIL import Image
import torchvision.transforms as transforms

sample_image_path = None
for cls in GARBAGE_CLASSES:
    cls_dir = Path('data/processed') / cls
    if cls_dir.exists():
        images = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png'))
        if images:
            sample_image_path = images[0]
            break

if sample_image_path:
    print(f"加载示例图像: {sample_image_path.name}")
    
    # 加载模型
    model = create_model('resnet18', num_classes=6)
    best_model_path = Path('models/resnet18_best.pth')
    
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # 预处理图像
        _, val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ]), None
        
        image = Image.open(sample_image_path).convert('RGB')
        image_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])(image).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            predicted_class = GARBAGE_CLASSES[predicted_idx]
            confidence = probabilities[0, predicted_idx].item()
        
        print(f"\n预测结果:")
        print(f"  预测类别: {predicted_class}")
        print(f"  置信度: {confidence:.1%}")
        
        print(f"\n各类别概率:")
        for idx, cls in enumerate(GARBAGE_CLASSES):
            prob = probabilities[0, idx].item()
            bar = "█" * int(prob * 20)
            print(f"  {cls:10s}: {bar:<20} {prob:.1%}")
    else:
        print("⚠ 模型权重不存在")
        print("请先运行: python run.py --task train")
else:
    print("⚠ 示例图像不存在")
    print("请先下载数据集")

# 示例 7: 项目结构说明
print("\n" + "=" * 80)
print("示例 7: 项目结构说明")
print("=" * 80)

print("""
项目目录结构:

garbage_classification/
├── data/                          数据目录
│   ├── raw/                      原始数据 (下载后放置)
│   ├── processed/                处理后的数据
│   └── external/                 外部补充数据
│
├── src/                           源代码
│   ├── __init__.py               项目初始化
│   ├── data_loader.py            数据加载模块 (~1200 行)
│   ├── data_cleaning.py          数据清洗模块 (~450 行)
│   ├── models.py                 模型定义 (~250 行)
│   ├── train.py                  训练脚本 (~350 行)
│   └── evaluate.py               评估脚本 (~380 行)
│
├── models/                        保存的模型权重
│   ├── simple_cnn_best.pth
│   ├── mobilenetv2_best.pth
│   ├── resnet18_best.pth
│   └── training_history.json
│
├── logs/                          日志和结果
│   ├── training.log
│   ├── evaluation_results.json
│   ├── model_comparison.png
│   ├── *_confusion_matrix.png
│   └── data_cleaning_report.json
│
├── notebooks/                     Jupyter 笔记本
├── app.py                        Streamlit 应用
├── run.py                        综合运行脚本
├── quickstart.py                 快速启动脚本
├── download_dataset.py           数据下载脚本
├── requirements.txt              依赖列表
├── README.md                     项目文档
├── 中期报告.md                   中期报告
└── .gitignore                    Git 忽略文件
""")

# 示例 8: 常用命令
print("\n" + "=" * 80)
print("示例 8: 常用命令")
print("=" * 80)

print("""
数据准备:
  python download_dataset.py          # 下载数据集

快速启动:
  python quickstart.py                 # 交互式启动向导

完整流程:
  python run.py --task all --epochs 50  # 数据清洗 + 训练 + 评估

具体任务:
  python run.py --task clean           # 仅数据清洗
  python run.py --task train           # 仅模型训练
  python run.py --task evaluate        # 仅模型评估

演示应用:
  streamlit run app.py                 # 启动交互式演示

监控训练:
  tail -f logs/training.log            # 查看实时日志
""")

# 示例 9: 性能优化建议
print("\n" + "=" * 80)
print("示例 9: 性能优化建议")
print("=" * 80)

print(f"""
当前设备: {device}

如果训练速度慢:
  1. 确保使用了 GPU (当前: {torch.cuda.is_available()})
  2. 减小批大小: --batch-size 16
  3. 使用轻量级模型: SimpleCNN 或 MobileNetV2
  4. 减少 workers: --num-workers 0 (对某些系统更快)

如果显存不足:
  1. 减小批大小
  2. 使用梯度累积
  3. 使用混合精度训练
  4. 改用更小的模型

如果要提高准确率:
  1. 增加训练轮数 (--epochs 100)
  2. 调整学习率 (在 train.py 中修改)
  3. 增加数据增强强度
  4. 使用更大的模型 (ResNet18 > MobileNetV2 > SimpleCNN)
""")

print("\n" + "=" * 80)
print("演示完成!")
print("=" * 80)
print("""
后续步骤:

1. 查看 README.md 了解详细信息
2. 查看 中期报告.md 了解项目进展
3. 运行 streamlit run app.py 体验演示应用
4. 根据需要调整超参数进行模型优化

如有问题，请查看项目文档或 GitHub 仓库。
""")

"""
模型评估脚本
计算准确率、F1 分数、混淆矩阵等评估指标
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import json
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import create_model
from src.data_loader import create_dataloaders, GARBAGE_CLASSES

logger = logging.getLogger(__name__)


def convert_to_serializable(obj):
    """将 numpy 类型转换为可序列化类型"""
    if isinstance(obj, (np.int64, np.int32, np.int_)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj


class Evaluator:
    """模型评估器"""

    def __init__(self, model, device, num_classes=6):
        self.model = model.to(device) if model is not None else None
        self.device = device
        self.num_classes = num_classes

    def evaluate(self, test_loader):
        """评估模型"""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_image_paths = []
        inference_times = []

        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Evaluating')
            for images, labels, image_paths in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                # 计算推理时间 - 根据设备类型选择不同方法
                if self.device.type == 'cuda':
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()

                    outputs = self.model(images)

                    end_event.record()
                    torch.cuda.synchronize()
                    inference_time = start_event.elapsed_time(end_event) / 1000  # 毫秒转秒
                else:
                    # CPU 环境使用 time.perf_counter() 获得更高精度
                    start_time = time.perf_counter()
                    outputs = self.model(images)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    inference_time = time.perf_counter() - start_time  # 单位为秒

                inference_times.append(inference_time)

                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_image_paths.extend(image_paths)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 计算评估指标
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'macro_f1': f1_score(all_labels, all_preds, average='macro'),
            'weighted_f1': f1_score(all_labels, all_preds, average='weighted'),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
            'classification_report': classification_report(
                all_labels, all_preds,
                target_names=GARBAGE_CLASSES,
                output_dict=True
            ),
            'avg_inference_time': np.mean(inference_times),
            'inference_time_per_image': {
                'mean': np.mean(inference_times) * 1000,  # 毫秒
                'std': np.std(inference_times) * 1000,
                'min': np.min(inference_times) * 1000,
                'max': np.max(inference_times) * 1000
            }
        }

        return metrics, all_preds, all_labels, all_image_paths

    def plot_confusion_matrix(self, confusion_mat, save_path=None):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_mat,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=GARBAGE_CLASSES,
            yticklabels=GARBAGE_CLASSES
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"混淆矩阵已保存到：{save_path}")

        return plt.gcf()

    def plot_metrics_comparison(self, all_metrics, save_path=None):
        """绘制模型性能对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        model_names = list(all_metrics.keys())
        accuracies = [all_metrics[m]['accuracy'] for m in model_names]
        macro_f1s = [all_metrics[m]['macro_f1'] for m in model_names]
        inference_times = [all_metrics[m]['inference_time_per_image']['mean'] for m in model_names]

        # 准确率对比
        axes[0].bar(model_names, accuracies, color='skyblue')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy Comparison')
        axes[0].set_ylim([0, 1])
        for i, v in enumerate(accuracies):
            axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center')

        # Macro-F1 对比
        axes[1].bar(model_names, macro_f1s, color='lightgreen')
        axes[1].set_ylabel('Macro-F1')
        axes[1].set_title('Model Macro-F1 Comparison')
        axes[1].set_ylim([0, 1])
        for i, v in enumerate(macro_f1s):
            axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center')

        # 推理速度对比
        axes[2].bar(model_names, inference_times, color='lightsalmon')
        axes[2].set_ylabel('Time (ms)')
        axes[2].set_title('Average Inference Time per Image')
        for i, v in enumerate(inference_times):
            axes[2].text(i, v + 0.5, f'{v:.2f}ms', ha='center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"性能对比图已保存到：{save_path}")

        return fig


def evaluate_baseline_models(data_dir, models_dir='models', output_dir='logs', num_workers=0, models_to_evaluate=None):
    """评估所有基线模型"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(models_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备：{device}")

    # 创建测试数据加载器
    logger.info("加载测试数据...")
    _, _, test_loader = create_dataloaders(data_dir, batch_size=32, num_workers=num_workers)

    # 模型配置（v1.03: 新增 EfficientNetV2-S 和 ConvNeXt Tiny）
    models_config = [
        {'name': 'simple_cnn', 'pretrained': False},
        {'name': 'mobilenetv2', 'pretrained': True},
        {'name': 'resnet18', 'pretrained': True},
        {'name': 'efficientnetv2s', 'pretrained': True},
        {'name': 'convnexttiny', 'pretrained': True},
    ]

    if models_to_evaluate is not None:
        models_config = [c for c in models_config if c['name'] in models_to_evaluate]

    all_metrics = {}
    all_predictions = {}

    for config in models_config:
        model_name = config['name']
        logger.info(f"\n评估模型：{model_name}")

        # 创建模型
        model = create_model(model_name, num_classes=len(GARBAGE_CLASSES),
                             pretrained=config['pretrained'])

        # 加载最好的权重（兼容 v1.03 的 checkpoint 格式）
        best_model_path = models_dir / f'{model_name}_best.pth'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"已加载模型权重：{best_model_path}")
        else:
            logger.warning(f"未找到模型权重：{best_model_path}")

        # 评估
        evaluator = Evaluator(model, device, num_classes=len(GARBAGE_CLASSES))
        metrics, preds, labels, image_paths = evaluator.evaluate(test_loader)

        all_metrics[model_name] = metrics
        all_predictions[model_name] = {
            'predictions': preds.tolist(),
            'labels': labels.tolist(),
            'image_paths': image_paths
        }

        # 输出评估结果
        logger.info(f"\n{model_name} 评估结果:")
        logger.info(f" 准确率：{metrics['accuracy']:.4f}")
        logger.info(f" Macro-F1: {metrics['macro_f1']:.4f}")
        logger.info(f" 平均推理时间：{metrics['inference_time_per_image']['mean']:.2f}ms")
        logger.info(f"\n分类报告:")
        for cls in GARBAGE_CLASSES:
            cls_metrics = metrics['classification_report'][cls]
            logger.info(
                f" {cls:10s}: Precision={cls_metrics['precision']:.3f}, "
                f"Recall={cls_metrics['recall']:.3f}, F1={cls_metrics['f1-score']:.3f}"
            )

    # 保存评估结果 - 使用 convert_to_serializable 处理 numpy 类型
    results_path = output_dir / 'evaluation_results.json'
    results_to_save = {}
    for model_name, metrics in all_metrics.items():
        results_to_save[model_name] = convert_to_serializable({
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
            'inference_time_ms': metrics['inference_time_per_image']['mean'],
            'confusion_matrix': metrics['confusion_matrix'],
            'classification_report': metrics['classification_report']
        })

    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    logger.info(f"\n评估结果已保存到：{results_path}")

    # 绘制对比图
    fig = Evaluator(None, device).plot_metrics_comparison(
        all_metrics,
        save_path=output_dir / 'model_comparison.png'
    )

    # 绘制混淆矩阵
    for model_name, metrics in all_metrics.items():
        cm = np.array(metrics['confusion_matrix'])
        evaluator = Evaluator(None, device)
        evaluator.plot_confusion_matrix(
            cm,
            save_path=output_dir / f'{model_name}_confusion_matrix.png'
        )

    return all_metrics, all_predictions


if __name__ == '__main__':
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/processed'
    models_dir = sys.argv[2] if len(sys.argv) > 2 else 'models'
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'logs'

    evaluate_baseline_models(data_dir, models_dir, output_dir)

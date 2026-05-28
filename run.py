""" 综合运行脚本
执行完整的数据清洗、模型训练和评估流程
v1.03: 新增模型选择、RandAugment、MixUp、Focal Loss 等训练选项
v1.04: 新增 CutMix、RandomErasing、One Cycle 调度器、TTA、模型集成支持
v1.06: 新增 Cosine Warmup 调度器、加权采样、SWA、EfficientNetV2-M、ConvNeXt Small
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='智能垃圾分类系统 - 数据清洗和模型训练 (v1.06)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='数据目录'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='输出目录'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=150,
        help='训练轮数'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批大小'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['clean', 'train', 'evaluate', 'inference', 'all'],
        default='all',
        help='执行的任务'
    )
    # v1.03: 模型选择
    # v1.06: 新增 efficientnetv2m, convnextsmall
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        choices=['simple_cnn', 'simple_cnn_v2', 'mobilenetv2', 'mobilenetv2_se',
                 'resnet18', 'resnet50', 'efficientnetv2s', 'efficientnetv2m',
                 'convnexttiny', 'convnextsmall'],
        help='要训练/评估的模型（默认全部）'
    )
    # v1.03: 数据增强选项
    parser.add_argument(
        '--randaugment',
        action='store_true',
        help='启用 RandAugment 自动增强'
    )
    parser.add_argument(
        '--mixup',
        action='store_true',
        help='启用 MixUp 数据增强'
    )
    # v1.04: 新增数据增强选项
    parser.add_argument(
        '--cutmix',
        action='store_true',
        help='启用 CutMix 数据增强'
    )
    parser.add_argument(
        '--random-erasing',
        action='store_true',
        help='启用 RandomErasing 数据增强'
    )
    # v1.03: 训练优化选项
    parser.add_argument(
        '--use-focal',
        action='store_true',
        help='启用 Focal Loss'
    )
    parser.add_argument(
        '--use-label-smoothing',
        action='store_true',
        help='启用标签平滑'
    )
    parser.add_argument(
        '--use-cosine',
        action='store_true',
        help='启用 CosineAnnealingWarmRestarts 调度器（v1.06 弃用，推荐 --use-warmup）'
    )
    # v1.04: One Cycle 调度器
    parser.add_argument(
        '--use-onecycle',
        action='store_true',
        help='启用 One Cycle 学习率调度器'
    )
    # v1.06: Cosine Warmup 调度器
    parser.add_argument(
        '--use-warmup',
        action='store_true',
        help='v1.06: 启用 Cosine Warmup + CosineAnnealingLR 调度器'
    )
    # v1.06: 类别平衡采样
    parser.add_argument(
        '--use-weighted-sampler',
        action='store_true',
        help='v1.06: 启用 WeightedRandomSampler 类别平衡采样'
    )
    # v1.06: SWA
    parser.add_argument(
        '--use-swa',
        action='store_true',
        help='v1.06: 启用 Stochastic Weight Averaging'
    )
    # v1.05: EMA 和知识蒸馏
    parser.add_argument(
        '--use-ema',
        action='store_true',
        help='启用指数移动平均 (EMA)'
    )
    parser.add_argument(
        '--use-distill',
        action='store_true',
        help='启用知识蒸馏（Teacher: EfficientNetV2-S）'
    )
    # v1.04: TTA 选项
    parser.add_argument(
        '--use-tta',
        action='store_true',
        help='启用测试时增强 (TTA)'
    )
    # v1.04: 推理模式
    parser.add_argument(
        '--image-path',
        type=str,
        help='推理模式下的图像路径'
    )
    # v1.06: 新增模型名选项
    parser.add_argument(
        '--model-name',
        type=str,
        choices=['simple_cnn', 'simple_cnn_v2', 'mobilenetv2', 'mobilenetv2_se',
                 'resnet18', 'resnet50', 'efficientnetv2s', 'efficientnetv2m',
                 'convnexttiny', 'convnextsmall'],
        default='efficientnetv2s',
        help='推理使用的模型'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("智能垃圾分类系统 - v1.06 极致精度优化")
    logger.info("=" * 80)
    logger.info(f"启动时间：{datetime.now()}")
    logger.info(f"数据目录：{args.data_dir}")
    logger.info(f"输出目录：{args.output_dir}")
    logger.info(f"执行任务：{args.task}")

    if args.task in ['train', 'all']:
        logger.info(f"RandAugment: {'启用' if args.randaugment else '禁用'}")
        logger.info(f"MixUp: {'启用' if args.mixup else '禁用'}")
        logger.info(f"CutMix: {'启用' if args.cutmix else '禁用'}")
        logger.info(f"RandomErasing: {'启用' if args.random_erasing else '禁用'}")
        logger.info(f"Focal Loss: {'启用' if args.use_focal else '禁用'}")
        logger.info(f"标签平滑：{'启用' if args.use_label_smoothing else '禁用'}")
        logger.info(f"CosineAnnealing: {'启用' if args.use_cosine else '禁用'}")
        logger.info(f"One Cycle: {'启用' if args.use_onecycle else '禁用'}")
        logger.info(f"Warmup+Cosine: {'启用' if args.use_warmup else '禁用'}")
        logger.info(f"加权采样: {'启用' if args.use_weighted_sampler else '禁用'}")
        logger.info(f"SWA: {'启用' if args.use_swa else '禁用'}")
        logger.info(f"EMA: {'启用' if args.use_ema else '禁用'}")
        logger.info(f"知识蒸馏: {'启用' if args.use_distill else '禁用'}")

    try:
        if args.task in ['clean', 'all']:
            logger.info("\n" + "=" * 80)
            logger.info("执行任务：数据清洗")
            logger.info("=" * 80)
            from src.data_cleaning import clean_dataset, validate_image_quality, generate_data_manifest
            clean_dataset(args.data_dir, args.data_dir)
            validate_image_quality(args.data_dir)
            generate_data_manifest(args.data_dir, 'data/processed/manifest.json')
            logger.info("✓ 数据清洗完成")

        if args.task in ['train', 'all']:
            logger.info("\n" + "=" * 80)
            logger.info("执行任务：模型训练")
            logger.info("=" * 80)
            from src.train import train_baseline_models
            results = train_baseline_models(
                args.data_dir,
                output_dir=args.output_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                use_randaugment=args.randaugment,
                use_mixup=args.mixup,
                use_cutmix=args.cutmix,
                use_random_erasing=args.random_erasing,
                use_focal=args.use_focal,
                use_label_smoothing=args.use_label_smoothing,
                use_cosine=args.use_cosine,
                use_onecycle=args.use_onecycle,
                use_warmup=args.use_warmup,
                models_to_train=args.models,
                use_ema=args.use_ema,
                use_distill=args.use_distill,
                use_weighted_sampler=args.use_weighted_sampler,
                use_swa=args.use_swa
            )
            logger.info("✓ 模型训练完成")

        if args.task in ['evaluate', 'all']:
            logger.info("\n" + "=" * 80)
            logger.info("执行任务：模型评估")
            logger.info("=" * 80)
            from src.evaluate import evaluate_baseline_models
            metrics, predictions = evaluate_baseline_models(
                args.data_dir,
                models_dir='models',
                output_dir='logs',
                models_to_evaluate=args.models,
                use_tta=args.use_tta if args.use_tta else True  # v1.06: 默认启用 TTA
            )
            logger.info("✓ 模型评估完成")

        if args.task == 'inference':
            logger.info("\n" + "=" * 80)
            logger.info("执行任务：模型推理")
            logger.info("=" * 80)
            if not args.image_path:
                logger.error("推理模式需要指定 --image-path")
                sys.exit(1)

            from src.inference import load_model, predict_image
            model = load_model(args.model_name, args.output_dir)
            result = predict_image(model, args.image_path)
            logger.info(f"预测结果:")
            logger.info(f"  类别：{result['top_class']}")
            logger.info(f"  置信度：{result['top_probability']:.4f}")
            logger.info(f"\n前 {len(result['predictions'])} 个预测:")
            for p in result['predictions']:
                logger.info(f"  {p['class_name']}: {p['probability']:.4f}")
                if p['advice']:
                    logger.info(f"    垃圾分类建议：{p['advice']}")
            logger.info("✓ 模型推理完成")

        logger.info("\n" + "=" * 80)
        logger.info("所有任务执行完成！")
        logger.info("=" * 80)
        logger.info(f"结束时间：{datetime.now()}")

    except Exception as e:
        logger.error(f"执行出错：{e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

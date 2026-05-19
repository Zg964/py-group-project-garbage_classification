"""
综合运行脚本
执行完整的数据清洗、模型训练和评估流程
v1.03: 新增模型选择、RandAugment、MixUp、Focal Loss 等训练选项
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
        description='智能垃圾分类系统 - 数据清洗和模型训练 (v1.03)'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/processed',
        help='数据目录'
    )
    parser.add_argument(
        '--output-dir', type=str, default='models',
        help='输出目录'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='训练轮数'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='批大小'
    )
    parser.add_argument(
        '--task', type=str,
        choices=['clean', 'train', 'evaluate', 'inference', 'all'],
        default='all',
        help='执行的任务'
    )
    # v1.03: 模型选择
    parser.add_argument(
        '--models', type=str, nargs='+',
        choices=['simple_cnn', 'mobilenetv2', 'resnet18',
                 'efficientnetv2s', 'convnexttiny'],
        help='要训练/评估的模型（默认全部）'
    )
    # v1.03: 数据增强选项
    parser.add_argument(
        '--randaugment', action='store_true',
        help='启用 RandAugment 自动增强'
    )
    parser.add_argument(
        '--mixup', action='store_true',
        help='启用 MixUp 数据增强'
    )
    # v1.03: 训练优化选项
    parser.add_argument(
        '--use-focal', action='store_true',
        help='启用 Focal Loss'
    )
    parser.add_argument(
        '--use-label-smoothing', action='store_true',
        help='启用标签平滑'
    )
    parser.add_argument(
        '--use-cosine', action='store_true',
        help='启用 CosineAnnealingWarmRestarts 调度器'
    )
    # v1.03: 推理模式
    parser.add_argument(
        '--image-path', type=str,
        help='推理模式下的图像路径'
    )
    parser.add_argument(
        '--model-name', type=str,
        choices=['simple_cnn', 'mobilenetv2', 'resnet18',
                 'efficientnetv2s', 'convnexttiny'],
        default='efficientnetv2s',
        help='推理使用的模型'
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("智能垃圾分类系统 - v1.03 优化阶段")
    logger.info("="*80)
    logger.info(f"启动时间: {datetime.now()}")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"执行任务: {args.task}")

    if args.task in ['train', 'all']:
        logger.info(f"RandAugment: {'启用' if args.randaugment else '禁用'}")
        logger.info(f"MixUp: {'启用' if args.mixup else '禁用'}")
        logger.info(f"Focal Loss: {'启用' if args.use_focal else '禁用'}")
        logger.info(f"标签平滑: {'启用' if args.use_label_smoothing else '禁用'}")
        logger.info(f"CosineAnnealing: {'启用' if args.use_cosine else '禁用'}")

    try:
        if args.task in ['clean', 'all']:
            logger.info("\n" + "="*80)
            logger.info("执行任务: 数据清洗")
            logger.info("="*80)

            from src.data_cleaning import clean_dataset, validate_image_quality, generate_data_manifest

            clean_dataset(args.data_dir, args.data_dir)
            validate_image_quality(args.data_dir)
            generate_data_manifest(args.data_dir, 'data/processed/manifest.json')

            logger.info("✓ 数据清洗完成")

        if args.task in ['train', 'all']:
            logger.info("\n" + "="*80)
            logger.info("执行任务: 模型训练")
            logger.info("="*80)

            from src.train import train_baseline_models

            results = train_baseline_models(
                args.data_dir,
                output_dir=args.output_dir,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                use_randaugment=args.randaugment,
                use_mixup=args.mixup,
                use_focal=args.use_focal,
                use_label_smoothing=args.use_label_smoothing,
                use_cosine=args.use_cosine,
                models_to_train=args.models
            )

            logger.info("✓ 模型训练完成")

        if args.task in ['evaluate', 'all']:
            logger.info("\n" + "="*80)
            logger.info("执行任务: 模型评估")
            logger.info("="*80)

            from src.evaluate import evaluate_baseline_models

            metrics, predictions = evaluate_baseline_models(
                args.data_dir,
                models_dir='models',
                output_dir='logs',
                models_to_evaluate=args.models
            )

            logger.info("✓ 模型评估完成")

        if args.task == 'inference':
            logger.info("\n" + "="*80)
            logger.info("执行任务: 模型推理")
            logger.info("="*80)

            if not args.image_path:
                logger.error("推理模式需要指定 --image-path")
                sys.exit(1)

            from src.inference import load_model, predict_image

            model = load_model(args.model_name, args.output_dir)
            result = predict_image(model, args.image_path)

            logger.info(f"预测结果:")
            logger.info(f"  类别: {result['top_class']}")
            logger.info(f"  置信度: {result['top_probability']:.4f}")
            logger.info(f"\n前 {len(result['predictions'])} 个预测:")
            for p in result['predictions']:
                logger.info(f"  {p['class_name']}: {p['probability']:.4f}")
                if p['advice']:
                    logger.info(f"    垃圾分类建议: {p['advice']}")

            logger.info("✓ 模型推理完成")

        logger.info("\n" + "="*80)
        logger.info("所有任务执行完成！")
        logger.info("="*80)
        logger.info(f"结束时间: {datetime.now()}")

    except Exception as e:
        logger.error(f"执行出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

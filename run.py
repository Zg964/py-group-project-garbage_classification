"""
综合运行脚本
执行完整的数据清洗和模型训练流程
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
        description='智能垃圾分类系统 - 数据清洗和模型训练'
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
        default=50,
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
        choices=['clean', 'train', 'evaluate', 'all'],
        default='all',
        help='执行的任务'
    )
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("智能垃圾分类系统 - 中期报告阶段")
    logger.info("="*80)
    logger.info(f"启动时间: {datetime.now()}")
    logger.info(f"数据目录: {args.data_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"执行任务: {args.task}")
    
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
                batch_size=args.batch_size
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
                output_dir='logs'
            )
            
            logger.info("✓ 模型评估完成")
        
        logger.info("\n" + "="*80)
        logger.info("所有任务执行完成！")
        logger.info("="*80)
        logger.info(f"结束时间: {datetime.now()}")
        
    except Exception as e:
        logger.error(f"执行出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

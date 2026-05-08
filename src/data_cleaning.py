"""
数据清洗脚本
处理数据集中的缺失值、异常值、重复数据等
"""

import os
import shutil
import json
import logging
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

from src.data_loader import DataCleaner, GARBAGE_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_and_prepare_dataset(raw_data_dir, processed_data_dir):
    """
    下载并准备 TrashNet 数据集
    
    注意：实际使用时需要从 Kaggle 或其他来源下载数据集
    这里提供数据准备的框架
    """
    raw_data_dir = Path(raw_data_dir)
    processed_data_dir = Path(processed_data_dir)
    
    # 创建处理后的数据目录
    for cls in GARBAGE_CLASSES:
        (processed_data_dir / cls).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"数据准备完成，输出目录: {processed_data_dir}")
    
    return processed_data_dir


def clean_dataset(data_dir, output_dir=None):
    """
    完整的数据清洗流程
    
    Args:
        data_dir: 原始数据目录
        output_dir: 清洗后的输出目录
    """
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = data_dir
    else:
        output_dir = Path(output_dir)
    
    logger.info("开始数据清洗流程")
    logger.info("="*60)
    
    # 初始化数据清洗器
    cleaner = DataCleaner(data_dir)
    
    # 第一步：加载数据集信息
    logger.info("\n[步骤 1] 加载数据集信息...")
    dataset_info = cleaner.load_dataset_info()
    
    # 保存初始统计
    initial_stats = {
        'total_images': dataset_info['total_images'],
        'class_distribution': dataset_info['class_distribution'],
        'corrupted_files': len(dataset_info['corrupted_files']),
        'missing_files': len(dataset_info['missing_files'])
    }
    
    logger.info(f"初始数据集统计:")
    logger.info(f"  总图像数: {dataset_info['total_images']}")
    logger.info(f"  损坏文件数: {len(dataset_info['corrupted_files'])}")
    
    # 第二步：删除损坏的文件
    logger.info("\n[步骤 2] 移除损坏的文件...")
    if dataset_info['corrupted_files']:
        cleaner.remove_corrupted_files(dataset_info['corrupted_files'])
        logger.info(f"已删除 {len(dataset_info['corrupted_files'])} 个损坏的文件")
    else:
        logger.info("未发现损坏的文件")
    
    # 第三步：检测并移除重复文件
    logger.info("\n[步骤 3] 检测并移除重复文件...")
    duplicates = cleaner.detect_duplicates()
    if duplicates:
        removed_count = cleaner.remove_duplicates(duplicates, keep_first=True)
        logger.info(f"已删除 {removed_count} 个重复文件")
    else:
        logger.info("未发现重复文件")
    
    # 第四步：验证标签
    logger.info("\n[步骤 4] 验证标签正确性...")
    misclassified = cleaner.validate_labels()
    if misclassified:
        logger.warning(f"发现 {len(misclassified)} 个标签问题")
        for item in misclassified[:5]:  # 只显示前5个
            logger.warning(f"  {item['file']}: {item['issue']}")
    else:
        logger.info("所有标签验证通过")
    
    # 第五步：数据规范化和统计
    logger.info("\n[步骤 5] 规范化和统计...")
    
    # 重新加载清洗后的数据统计
    cleaner_after = DataCleaner(data_dir)
    dataset_info_after = cleaner_after.load_dataset_info()
    
    # 最终统计
    final_stats = cleaner_after.generate_statistics(dataset_info_after)
    
    logger.info("\n数据清洗完成!")
    logger.info("="*60)
    logger.info(f"清洗后数据集统计:")
    logger.info(f"  总图像数: {final_stats['total_images']}")
    logger.info(f"  各类别分布:")
    for cls, count in final_stats['class_distribution'].items():
        percentage = (count / final_stats['total_images'] * 100) if final_stats['total_images'] > 0 else 0
        logger.info(f"    {cls:12s}: {count:4d} 张 ({percentage:5.1f}%)")
    
    # 保存清洗统计报告
    cleaning_report = {
        'timestamp': str(Path.cwd()),
        'initial_stats': initial_stats,
        'final_stats': final_stats,
        'removed_corrupted': len(dataset_info['corrupted_files']),
        'removed_duplicates': len(duplicates),
        'label_issues': len(misclassified)
    }
    
    report_path = output_dir / 'data_cleaning_report.json'
    with open(report_path, 'w') as f:
        json.dump(cleaning_report, f, indent=2)
    
    logger.info(f"\n清洗报告已保存到: {report_path}")
    
    return final_stats


def validate_image_quality(data_dir, min_width=100, min_height=100):
    """
    验证图像质量
    检查图像尺寸、格式等
    """
    logger.info(f"验证图像质量 (最小尺寸: {min_width}x{min_height})")
    
    quality_issues = []
    
    for cls in GARBAGE_CLASSES:
        cls_dir = Path(data_dir) / cls
        if not cls_dir.exists():
            continue
        
        for img_path in cls_dir.glob('*.*'):
            if img_path.is_file():
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        if width < min_width or height < min_height:
                            quality_issues.append({
                                'file': str(img_path),
                                'issue': f'size_too_small ({width}x{height})'
                            })
                except Exception as e:
                    quality_issues.append({
                        'file': str(img_path),
                        'issue': str(e)
                    })
    
    if quality_issues:
        logger.warning(f"发现 {len(quality_issues)} 个质量问题")
        for issue in quality_issues[:10]:
            logger.warning(f"  {issue['file']}: {issue['issue']}")
    else:
        logger.info("所有图像质量检查通过")
    
    return quality_issues


def generate_data_manifest(data_dir, output_path=None):
    """
    生成数据清单
    记录所有图像的信息
    """
    logger.info("生成数据清单...")
    
    manifest = {
        'classes': GARBAGE_CLASSES,
        'class_count': {},
        'images': []
    }
    
    total_images = 0
    
    for cls in GARBAGE_CLASSES:
        cls_dir = Path(data_dir) / cls
        if not cls_dir.exists():
            manifest['class_count'][cls] = 0
            continue
        
        cls_images = []
        for img_path in cls_dir.glob('*.*'):
            if img_path.is_file():
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        file_size = img_path.stat().st_size
                        
                        cls_images.append({
                            'path': str(img_path.relative_to(data_dir.parent)),
                            'class': cls,
                            'width': width,
                            'height': height,
                            'size_kb': file_size / 1024
                        })
                except Exception as e:
                    logger.warning(f"无法读取: {img_path}, 错误: {e}")
        
        manifest['class_count'][cls] = len(cls_images)
        manifest['images'].extend(cls_images)
        total_images += len(cls_images)
    
    manifest['total_images'] = total_images
    
    if output_path is None:
        output_path = Path('data/processed/manifest.json')
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"数据清单已保存到: {output_path}")
    logger.info(f"总计: {total_images} 张图像")
    
    return manifest


if __name__ == '__main__':
    import sys
    
    # 默认参数
    raw_data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/raw'
    processed_data_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/processed'
    
    # 执行清洗
    clean_dataset(processed_data_dir, processed_data_dir)
    
    # 质量验证
    validate_image_quality(processed_data_dir)
    
    # 生成清单
    generate_data_manifest(processed_data_dir)

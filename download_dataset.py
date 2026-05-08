"""
数据集下载和准备脚本
支持从多个来源下载垃圾分类数据集
"""

import os
import sys
import json
import urllib.request
from pathlib import Path
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """数据集下载器"""
    
    # 数据集来源
    SOURCES = {
        'kaggle': {
            'name': 'Kaggle Garbage Classification',
            'url': 'https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification',
            'requires_auth': True
        },
        'trashnet': {
            'name': 'TrashNet Dataset',
            'url': 'https://github.com/garythung/trashnet',
            'requires_auth': False
        }
    }
    
    def __init__(self, data_dir='data/raw'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def list_sources():
        """列出可用的数据源"""
        print("可用的数据源:")
        for key, info in DatasetDownloader.SOURCES.items():
            auth = "需要认证" if info['requires_auth'] else "无需认证"
            print(f"  {key}: {info['name']} ({auth})")
            print(f"    URL: {info['url']}")
    
    def download_from_kaggle(self, dataset_name='asdasdasasdas/garbage-classification'):
        """从 Kaggle 下载数据集"""
        print(f"从 Kaggle 下载: {dataset_name}")
        print("\n需要 Kaggle API 认证。请按以下步骤操作:")
        print("1. 访问: https://www.kaggle.com/settings/account")
        print("2. 点击 'Create New API Token' 下载 kaggle.json")
        print("3. 将 kaggle.json 放在 ~/.kaggle/ 目录中")
        print("4. 运行: kaggle datasets download -d {dataset_name}")
        
        try:
            import kaggle
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=str(self.data_dir),
                unzip=True
            )
            logger.info(f"下载完成: {self.data_dir}")
        except ImportError:
            logger.error("kaggle 库未安装，请运行: pip install kaggle")
        except Exception as e:
            logger.error(f"下载失败: {e}")
    
    def prepare_directories(self):
        """准备目录结构"""
        classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        
        for cls in classes:
            cls_dir = self.data_dir / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"创建目录: {cls_dir}")
    
    @staticmethod
    def verify_structure(data_dir):
        """验证数据集结构"""
        data_dir = Path(data_dir)
        classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        
        print(f"\n验证数据集结构: {data_dir}")
        print("-" * 60)
        
        all_valid = True
        total_images = 0
        
        for cls in classes:
            cls_dir = data_dir / cls
            if not cls_dir.exists():
                print(f"✗ 缺少类别目录: {cls}")
                all_valid = False
                continue
            
            images = list(cls_dir.glob('*.jpg')) + list(cls_dir.glob('*.png'))
            total_images += len(images)
            
            if len(images) == 0:
                print(f"⚠ {cls}: 0 张图像")
                all_valid = False
            else:
                print(f"✓ {cls}: {len(images)} 张图像")
        
        print("-" * 60)
        print(f"总计: {total_images} 张图像")
        
        if all_valid and total_images > 0:
            print("✓ 数据集结构有效!")
            return True
        else:
            print("✗ 数据集结构有问题，请检查")
            return False


def main():
    """主函数"""
    print("=" * 70)
    print("垃圾分类数据集 - 下载和准备工具")
    print("=" * 70)
    
    # 列出可用来源
    DatasetDownloader.list_sources()
    
    print("\n" + "=" * 70)
    print("快速开始")
    print("=" * 70)
    
    print("""
推荐步骤:

1. 访问 Kaggle 垃圾分类数据集:
   https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
   
2. 下载数据集 (需要 Kaggle 账号)
   
3. 将下载的文件解压到 data/processed/ 目录:
   data/processed/
   ├── cardboard/
   ├── glass/
   ├── metal/
   ├── paper/
   ├── plastic/
   └── trash/
   
4. 验证数据集:
    """)
    
    # 验证当前数据集
    processed_dir = Path('data/processed')
    if processed_dir.exists():
        downloader = DatasetDownloader()
        downloader.verify_structure(processed_dir)
    else:
        print("等待: data/processed/ 目录不存在")
        print("请先下载并解压数据集")
    
    print("\n" + "=" * 70)
    print("可选: 自动下载 (需要 Kaggle API)")
    print("=" * 70)
    
    response = input("\n是否尝试使用 Kaggle API 下载? (y/n) [n]: ").lower().strip()
    if response == 'y':
        downloader = DatasetDownloader()
        downloader.download_from_kaggle()
        downloader.prepare_directories()
        downloader.verify_structure(downloader.data_dir)
    else:
        print("\n请手动下载数据集并放在 data/processed/ 目录中")
    
    print("\n" + "=" * 70)
    print("后续步骤")
    print("=" * 70)
    print("""
1. 执行数据清洗:
   python quickstart.py
   
2. 启动训练和评估:
   python run.py --task all
   
3. 启动演示应用:
   streamlit run app.py
    """)


if __name__ == '__main__':
    main()

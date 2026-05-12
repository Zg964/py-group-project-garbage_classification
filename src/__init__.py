"""
项目初始化模块
"""

import os
import sys
from pathlib import Path

__version__ = "1.01"

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 创建必要的目录结构
def create_directories():
    """创建项目所需的目录"""
    directories = [
        'data/raw',
        'data/processed',
        'data/external',
        'models',
        'logs',
        'notebooks'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def check_dependencies():
    """检查依赖包是否安装"""
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'pandas',
        'sklearn',
        'cv2',
        'PIL',
        'streamlit'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print(f"请运行: pip install -r requirements.txt")
        return False
    
    return True


if __name__ == '__main__':
    print("初始化项目...")
    create_directories()
    
    if check_dependencies():
        print("✓ 所有依赖包已安装")
    else:
        sys.exit(1)
    
    print("✓ 项目初始化完成")

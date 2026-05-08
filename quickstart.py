#!/usr/bin/env python
"""
快速启动脚本
一键启动完整的数据清洗和模型训练流程
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("=" * 80)
    print("智能垃圾分类系统 - 快速启动指南")
    print("=" * 80)
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("\n📋 项目配置:")
    print(f"  项目路径: {project_root}")
    print(f"  Python 版本: {sys.version.split()[0]}")
    
    print("\n" + "=" * 80)
    print("步骤 1: 检查依赖")
    print("=" * 80)
    
    try:
        import torch
        import torchvision
        import numpy
        import pandas
        import sklearn
        import cv2
        from PIL import Image
        
        print("✓ 所有核心依赖已安装")
        print(f"  - PyTorch: {torch.__version__}")
        print(f"  - CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("\n请运行: pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("步骤 2: 创建项目目录")
    print("=" * 80)
    
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
        print(f"  ✓ {directory}")
    
    print("\n" + "=" * 80)
    print("步骤 3: 数据准备")
    print("=" * 80)
    print("""
请按以下步骤准备数据:

1. 从 Kaggle 下载垃圾分类数据集:
   https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

2. 将数据解压到 data/processed/ 目录，结构如下:
   data/processed/
   ├── cardboard/
   ├── glass/
   ├── metal/
   ├── paper/
   ├── plastic/
   └── trash/

3. 每个类别目录中应该包含该类垃圾的图像 (jpg/png)

当数据准备好后，输入 'y' 继续:
    """)
    
    response = input("数据已准备好？ (y/n) [n]: ").lower().strip()
    if response != 'y':
        print("请先准备数据，然后重新运行此脚本")
        sys.exit(0)
    
    print("\n" + "=" * 80)
    print("步骤 4: 选择要执行的任务")
    print("=" * 80)
    print("""
1. 完整流程 (数据清洗 + 模型训练 + 评估) [推荐]
2. 仅数据清洗
3. 仅模型训练
4. 仅模型评估
5. 启动 Streamlit 演示应用
6. 显示评估结果
7. 退出
    """)
    
    choice = input("请选择 (1-7) [1]: ").strip() or '1'
    
    if choice == '1':
        print("\n" + "=" * 80)
        print("执行: 完整流程")
        print("=" * 80)
        
        print("\n[1/3] 数据清洗...")
        subprocess.run([sys.executable, '-m', 'src.data_cleaning'])
        
        print("\n[2/3] 模型训练 (这可能需要 10-20 分钟)...")
        subprocess.run([sys.executable, 'run.py', '--task', 'train', '--epochs', '50'])
        
        print("\n[3/3] 模型评估...")
        subprocess.run([sys.executable, 'run.py', '--task', 'evaluate'])
        
        print("\n✓ 完整流程执行完成！")
        
    elif choice == '2':
        print("\n执行: 数据清洗...")
        subprocess.run([sys.executable, '-m', 'src.data_cleaning'])
        
    elif choice == '3':
        print("\n执行: 模型训练...")
        epochs = input("请输入训练轮数 (默认 50): ").strip() or '50'
        subprocess.run([sys.executable, 'run.py', '--task', 'train', '--epochs', epochs])
        
    elif choice == '4':
        print("\n执行: 模型评估...")
        subprocess.run([sys.executable, 'run.py', '--task', 'evaluate'])
        
    elif choice == '5':
        print("\n启动 Streamlit 演示应用...")
        print("打开浏览器访问: http://localhost:8501")
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])
        
    elif choice == '6':
        print("\n显示评估结果...")
        results_path = Path('logs/evaluation_results.json')
        if results_path.exists():
            import json
            with open(results_path) as f:
                results = json.load(f)
            
            print("\n" + "=" * 80)
            print("模型性能对比")
            print("=" * 80)
            
            for model_name, metrics in results.items():
                print(f"\n{model_name}:")
                print(f"  准确率: {metrics['accuracy']:.1%}")
                print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
                print(f"  推理时间: {metrics['inference_time_ms']:.2f}ms")
                
                print(f"\n  分类详情:")
                for cls in ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']:
                    if cls in metrics['classification_report']:
                        m = metrics['classification_report'][cls]
                        print(f"    {cls:10s}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1-score']:.3f}")
        else:
            print("评估结果不存在，请先运行模型评估")
        
    elif choice == '7':
        print("退出")
        sys.exit(0)
    
    print("\n" + "=" * 80)
    print("后续步骤")
    print("=" * 80)
    print("""
1. 启动 Streamlit 应用进行图像分类演示:
   streamlit run app.py

2. 查看详细报告:
   - 中期报告: 中期报告.md
   - 项目文档: README.md
   - 训练日志: logs/training.log
   - 评估结果: logs/evaluation_results.json

3. 进行模型优化:
   - 调整超参数
   - 尝试数据增强
   - 引入新的网络架构

4. 部署到生产环境:
   - 量化模型
   - 优化推理
   - 打包为应用
    """)
    
    print("\n感谢使用智能垃圾分类系统！")
    print("更多信息请参考: README.md 和 中期报告.md")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

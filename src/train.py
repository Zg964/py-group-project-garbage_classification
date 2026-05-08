"""
模型训练脚本
"""

import os
import json
import time
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

from src.models import create_model, count_parameters
from src.data_loader import create_dataloaders, GARBAGE_CLASSES

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """模型训练器"""
    
    def __init__(self, model, device, model_name='model'):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0
        self.best_epoch = 0
    
    def train_epoch(self, train_loader, criterion, optimizer, epoch, num_epochs):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}] Train')
        for images, labels, _ in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算准确率
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for images, labels, _ in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = total_loss / len(val_loader)
        epoch_acc = correct / total
        
        self.val_losses.append(epoch_loss)
        self.val_accs.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs=50, lr=0.001, weight_decay=1e-4, 
              save_dir='models'):
        """完整训练流程"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置优化器和学习率调度器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        criterion = nn.CrossEntropyLoss()
        
        logger.info(f"开始训练 {self.model_name}")
        logger.info(f"模型参数数量: {count_parameters(self.model):,}")
        logger.info(f"设备: {self.device}")
        logger.info(f"学习率: {lr}, 权重衰减: {weight_decay}")
        logger.info(f"总 Epoch 数: {num_epochs}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(
                train_loader, criterion, optimizer, epoch, num_epochs
            )
            
            # 验证
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # 调整学习率
            scheduler.step(val_loss)
            
            logger.info(
                f'Epoch [{epoch+1}/{num_epochs}] - '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}'
            )
            
            # 保存最好的模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                best_model_path = save_dir / f'{self.model_name}_best.pth'
                torch.save(self.model.state_dict(), best_model_path)
                logger.info(f'已保存最好的模型: {best_model_path}')
        
        training_time = time.time() - start_time
        logger.info(f'训练完成，耗时: {training_time:.2f}s')
        logger.info(f'最好的验证准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch+1})')
        
        return {
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'training_time': training_time,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }


def train_baseline_models(data_dir, output_dir='models', num_epochs=50, batch_size=32):
    """训练所有基线模型"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建数据加载器
    logger.info("加载数据...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir, batch_size=batch_size, num_workers=4
    )
    
    logger.info(f"训练集大小: {len(train_loader.dataset)}")
    logger.info(f"验证集大小: {len(val_loader.dataset)}")
    logger.info(f"测试集大小: {len(test_loader.dataset)}")
    
    # 模型配置
    models_config = [
        {'name': 'simple_cnn', 'pretrained': False},
        {'name': 'mobilenetv2', 'pretrained': True},
        {'name': 'resnet18', 'pretrained': True},
    ]
    
    results = {}
    
    for config in models_config:
        model_name = config['name']
        logger.info(f"\n{'='*60}")
        logger.info(f"训练模型: {model_name}")
        logger.info(f"{'='*60}")
        
        # 创建模型
        model = create_model(model_name, num_classes=len(GARBAGE_CLASSES), 
                            pretrained=config['pretrained'])
        
        # 训练
        trainer = Trainer(model, device, model_name)
        history = trainer.train(
            train_loader, val_loader, 
            num_epochs=num_epochs, 
            lr=0.001,
            weight_decay=1e-4,
            save_dir=output_dir
        )
        
        results[model_name] = history
    
    # 保存训练历史
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(
            {k: {kk: (vv if not isinstance(vv, list) else vv) for kk, vv in v.items()} 
             for k, v in results.items()},
            f, indent=2
        )
    logger.info(f"训练历史已保存到: {history_path}")
    
    return results


if __name__ == '__main__':
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/processed'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'models'
    num_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    
    train_baseline_models(data_dir, output_dir, num_epochs)

"""
智能垃圾分类 - FastAPI 后端推理服务
提供 REST API 供前端页面调用
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from collections import OrderedDict
from typing import Optional

# 将项目根目录加入 sys.path，以便导入 src 模块
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io

from src.inference import InferenceModel, SORTING_ADVICE
from src.models import create_model, count_parameters
from src.config import GARBAGE_CLASSES, IDX_TO_CLASS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------- 配置 ----------
PROJECT_ROOT = _project_root
MODEL_DIR = PROJECT_ROOT / 'models'
FRONTEND_DIR = Path(__file__).resolve().parent / 'frontend'
CACHE_MAX_SIZE = 2  # LRU 缓存最多保持 2 个模型同时加载

# 模型显示名称映射
MODEL_DISPLAY_NAMES = {
    'simple_cnn': 'SimpleCNN',
    'simple_cnn_v2': 'SimpleCNN-V2',
    'mobilenetv2': 'MobileNetV2',
    'mobilenetv2_se': 'MobileNetV2-SE',
    'resnet18': 'ResNet18',
    'resnet50': 'ResNet50',
    'efficientnetv2s': 'EfficientNetV2-S',
    'efficientnetv2m': 'EfficientNetV2-M',
    'convnexttiny': 'ConvNeXt-Tiny',
    'convnextsmall': 'ConvNeXt-Small',
}

# 模型速度标签（基于参数量级大致判断）
MODEL_SPEED_LABELS = {
    'simple_cnn': '极快',
    'simple_cnn_v2': '极快',
    'mobilenetv2': '快速',
    'mobilenetv2_se': '快速',
    'resnet18': '中等',
    'resnet50': '较慢',
    'efficientnetv2s': '中等',
    'efficientnetv2m': '较慢',
    'convnexttiny': '中等',
    'convnextsmall': '较慢',
}

# 中文类别名
CLASS_CN_NAMES = {
    'cardboard': '纸板',
    'glass': '玻璃',
    'metal': '金属',
    'paper': '纸张',
    'plastic': '塑料',
    'trash': '其他垃圾',
}

app = FastAPI(title='智能垃圾分类 API', version='1.0.0')

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ---------- 模型 LRU 缓存 ----------
class ModelCache:
    """LRU 模型缓存"""

    def __init__(self, max_size: int = 2):
        self.max_size = max_size
        self._cache: OrderedDict[str, InferenceModel] = OrderedDict()

    def get(self, key: str) -> Optional[InferenceModel]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, model: InferenceModel):
        if key in self._cache:
            self._cache.move_to_end(key)
            return
        if len(self._cache) >= self.max_size:
            # 移除最久未使用的模型
            oldest_key, oldest_model = self._cache.popitem(last=False)
            logger.info(f"卸载模型: {oldest_key} (释放内存)")
        self._cache[key] = model
        logger.info(f"加载模型: {key} (缓存大小: {len(self._cache)})")

    def clear(self):
        self._cache.clear()

model_cache = ModelCache(max_size=CACHE_MAX_SIZE)


def parse_model_filename(filename: str):
    """解析模型文件名，提取模型名和变体"""
    name = filename.replace('.pth', '')
    variant = 'best'
    if name.endswith('_best'):
        name = name[:-5]
    elif name.endswith('_swa'):
        name = name[:-4]
        variant = 'swa'
    return name, variant


def get_model_info():
    """扫描 models/ 目录，获取所有可用模型信息"""
    training_history = {}
    history_path = MODEL_DIR / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r', encoding='utf-8') as f:
            training_history = json.load(f)

    models_info = {}
    for pth_file in sorted(MODEL_DIR.glob('*.pth')):
        model_name, variant = parse_model_filename(pth_file.name)
        if model_name not in models_info:
            models_info[model_name] = {
                'name': model_name,
                'display_name': MODEL_DISPLAY_NAMES.get(model_name, model_name),
                'speed_label': MODEL_SPEED_LABELS.get(model_name, '未知'),
                'size_mb': round(pth_file.stat().st_size / (1024 * 1024), 1),
                'variants': {},
            }

        # 获取该变体的准确率
        accuracy = None
        if model_name in training_history:
            accuracy = training_history[model_name].get('best_val_acc', None)

        models_info[model_name]['variants'][variant] = {
            'filename': pth_file.name,
            'size_mb': models_info[model_name]['size_mb'],
            'accuracy': round(accuracy * 100, 2) if accuracy else None,
        }

    # 合并：对于有 best + swa 的模型，分别显示
    result = []
    for model_name, info in models_info.items():
        variants = info['variants']
        for variant_name, variant_info in variants.items():
            result.append({
                'name': model_name,
                'variant': variant_name,
                'display_name': info['display_name'],
                'full_name': f"{info['display_name']} ({variant_name.upper()})",
                'speed_label': info['speed_label'],
                'size_mb': variant_info['size_mb'],
                'accuracy': variant_info['accuracy'],
                'filename': variant_info['filename'],
            })

    return result


def load_model_with_cache(model_name: str, variant: str = 'best', use_tta: bool = False) -> InferenceModel:
    """从缓存加载模型，缓存未命中则新建"""
    cache_key = f"{model_name}_{variant}"

    model = model_cache.get(cache_key)
    if model is not None:
        logger.info(f"缓存命中: {cache_key}")
        return model

    # 构建模型文件名
    if variant == 'swa':
        filename = f"{model_name}_swa.pth"
    else:
        filename = f"{model_name}_best.pth"

    model_path = MODEL_DIR / filename
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在：{model_path}")

    # 创建 InferenceModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InferenceModel(model_name, str(model_path), device=device, use_tta=use_tta)

    model_cache.put(cache_key, model)
    return model


# ---------- API 路由 ----------

@app.get('/api/models')
def list_models():
    """获取可用模型列表"""
    models = get_model_info()
    return {
        'models': models,
        'count': len(models),
    }


@app.post('/api/predict')
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form('efficientnetv2s'),
    variant: str = Form('best'),
    topk: int = Form(6),
):
    """上传图片并进行垃圾分类预测"""
    # 验证文件
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='请上传图片文件（JPEG/PNG 等）')

    try:
        # 读取图片
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'图片读取失败：{str(e)}')

    # 加载模型
    try:
        model = load_model_with_cache(model_name, variant)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f'模型未找到：{model_name} ({variant})')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'模型加载失败：{str(e)}')

    # 推理
    try:
        start_time = time.time()
        result = model.predict(image, topk=topk)
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'推理失败：{str(e)}')

    # 构建响应
    predictions = []
    for pred in result['predictions']:
        class_name = pred['class_name']
        predictions.append({
            'class_name': class_name,
            'class_cn': CLASS_CN_NAMES.get(class_name, class_name),
            'probability': pred['probability'],
            'category': pred['category'],
            'advice': pred['advice'],
        })

    top_pred = predictions[0]

    return {
        'top_class': top_pred['class_name'],
        'top_class_cn': top_pred['class_cn'],
        'top_probability': top_pred['probability'],
        'category': top_pred['category'],
        'advice': top_pred['advice'],
        'predictions': predictions,
        'inference_time_ms': round(inference_time, 1),
        'model_used': f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} ({variant.upper()})",
    }


# ---------- 静态文件托管 ----------

# 挂载静态文件
if FRONTEND_DIR.exists():
    app.mount('/static', StaticFiles(directory=str(FRONTEND_DIR)), name='frontend')

    @app.get('/')
    def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / 'index.html'))
else:
    @app.get('/')
    def root():
        return {'message': '智能垃圾分类 API 服务运行中', 'docs': '/docs'}


# ---------- 启动 ----------
if __name__ == '__main__':
    import uvicorn
    print("=" * 50)
    print("  智能垃圾分类 API 服务")
    print("=" * 50)
    print(f"  项目目录: {PROJECT_ROOT}")
    print(f"  模型目录: {MODEL_DIR}")
    print(f"  前端目录: {FRONTEND_DIR}")
    print(f"  设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"  模型缓存容量: {CACHE_MAX_SIZE}")
    print()
    print("  访问地址: http://localhost:8000")
    print("  API 文档: http://localhost:8000/docs")
    print("=" * 50)
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')

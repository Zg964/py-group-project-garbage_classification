"""
配置文件 - 集中化管理魔法数字和常量
"""

# 数据相关
HASH_SIZE = 8
TEST_SIZE = 0.2
VAL_RATIO = 0.1
RANDOM_SEED = 42

# 模型相关
NUM_CLASSES = 6
DEFAULT_IMAGE_SIZE = (224, 224)

# 分类标签
GARBAGE_CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
CLASS_NAMES = GARBAGE_CLASSES  # 别名
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(GARBAGE_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

# 训练相关
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 50
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_WEIGHT_DECAY = 1e-4

# 图像相关
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# 目录配置
DEFAULT_DATA_DIR = 'data/processed'
DEFAULT_MODEL_DIR = 'models'
DEFAULT_LOG_DIR = 'logs'

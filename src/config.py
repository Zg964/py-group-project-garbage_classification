""" 配置文件 - 集中化管理魔法数字和常量 """

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
IDX_TO_CLASS = {idx: cls for idx, cls in CLASS_TO_IDX.items()}

# 训练相关
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_EPOCHS = 50
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0

# v1.03 新增训练超参数
FOCAL_LOSS_GAMMA = 2.0  # Focal Loss 聚焦参数
LABEL_SMOOTHING_EPSILON = 0.1  # 标签平滑系数
COSINE_T_0 = 10  # CosineAnnealingWarmRestarts 初始周期
COSINE_T_MULT = 2  # 周期增长倍数
GRAD_CLIP_MAX_NORM = 1.0  # 梯度裁剪最大范数
EARLY_STOPPING_PATIENCE = 10  # Early Stopping 耐心值

# v1.03 数据增强参数
RANDAUGMENT_NUM_OPS = 2  # RandAugment 操作数量
RANDAUGMENT_MAGNITUDE = 9  # RandAugment 增强强度
MIXUP_ALPHA = 0.2  # MixUp 混合参数

# v1.04 新增数据增强参数
CUTMIX_ALPHA = 0.4  # CutMix 混合参数
CUTMIX_PROB = 0.5  # CutMix 应用概率
RANDOM_ERASING_PROB = 0.25  # RandomErasing 概率
RANDOM_ERASING_SCALE = (0.02, 0.4)  # RandomErasing 擦除区域比例
RANDOM_ERASING_RATIO = (0.3, 3.3)  # RandomErasing 宽高比

# v1.04 学习率调度器参数
ONecycle_TOTAL_STEPS = None  # One Cycle 总步数（训练时动态设置）
ONecycle_PCT_START = 0.3  # One Cycle 上升阶段比例
ONecycle_ANNEAL_STRATEGY = 'cos'  # One Cycle 退火策略

# v1.04 TTA 参数
TTA_ENABLED = False  # 是否启用 TTA
TTA_FLIP = True  # 水平翻转
TTA_ROTATION_ANGLES = [0, 90, 180, 270]  # 旋转角度

# 图像相关
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# 目录配置
DEFAULT_DATA_DIR = 'data/processed'
DEFAULT_MODEL_DIR = 'models'
DEFAULT_LOG_DIR = 'logs'

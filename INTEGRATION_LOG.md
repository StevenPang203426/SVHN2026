# 整合日志 — CausalYOLOSvhn → 主项目

> 记录将 `/new_experiment/CausalYOLOSvhn` 实验深度整合到当前项目的全过程。
> 包含：迁移了什么、保留了什么、为什么保留、以及重构决策。

---

## 一、整合概览

| 维度 | 原实验 (CausalYOLOSvhn) | 整合后 (主项目) |
|------|------------------------|----------------|
| 模型 | YOLO11m | 通过 `configs/causal_yolo.yaml` 配置 |
| 数据增强 | `causal_augment.py` (独立脚本) | `datasets/causal_augment.py` (模块化) |
| 数据下载 | `get_dataset.py` (独立脚本) | `datasets/download.py` (合并升级) |
| 训练入口 | `train.py` (硬编码路径) | `train_yolo.py` (YAML 配置驱动) |
| 推理入口 | `predict.py` (硬编码路径) | `predict_yolo.py` (YAML 配置驱动) |
| 配置方式 | 脚本内常量 | `configs/causal_yolo.yaml` |
| 成绩 | 0.9346 (第 7 名) | 配置可复现 |

---

## 二、文件迁移清单

### 新增文件

| 文件 | 来源 | 说明 |
|------|------|------|
| `datasets/causal_augment.py` | `code/causal_augment.py` | **核心亮点**：因果反事实增强模块 |
| `configs/causal_yolo.yaml` | `code/train.py` 中的常量 | YOLO11m + 因果增强配置 |

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| `datasets/download.py` | 合并 `get_dataset.py` 的安全下载逻辑和 YOLO 格式转换 |
| `datasets/__init__.py` | 导出因果增强模块的公共接口 |
| `train_yolo.py` | 支持因果增强预处理、seed 参数、YOLO11m |
| `predict_yolo.py` | 支持 conf=0.3 阈值、batch_size 配置化 |
| `requirements.txt` | 添加 albumentations、opencv-python 依赖 |

### 未迁移文件

| 文件 | 原因 |
|------|------|
| `code/get_dataset.py` | 功能已合并到 `datasets/download.py` |
| `code/train.py` | 功能已整合到 `train_yolo.py` + YAML 配置 |
| `code/predict.py` | 功能已整合到 `predict_yolo.py` |

---

## 三、保留的亮点设计

### 亮点 1：因果反事实数据增强 (⭐ 核心创新)

**位置**: `datasets/causal_augment.py`

**原理**: 传统数据增强是"随机扰动"——对每张图施加相同概率的变换，不区分图像本身的质量问题。因果增强基于结构因果模型 (SCM) 的 Do-Calculus，分两步走：

1. **检测混杂因子** (`detect_confounds`): 用轻量启发式方法检测图像中可能导致模型学到虚假关联的因素：
   - 亮度异常 (Laplacian 均值) → `lighting`
   - 图像模糊 (Laplacian 方差 < 50) → `sensor_blur`
   - 传感器噪声 (Sobel 高频占比 > 12%) → `sensor_noise`
   - 透视变形 (bbox 宽高比偏离) → `geometry`
   - 复杂背景 (Canny 边缘密度 > 2%) → `background`

2. **施加 Do-干预** (`generate_counterfactual_samples`): 针对检测到的混杂因子，从对应的干预池中抽取变换，生成反事实样本。例如检测到 `lighting` 混杂，就施加极端亮度/阴影/CLAHE 等干预，迫使模型在这些条件下仍能正确识别数字。

**为什么保留**: 这是该实验从 ~0.888 mAP50 (YOLOv8s baseline) 提升到 0.9346 的关键因素。传统增强无法区分"模型已经鲁棒的维度"和"模型脆弱的维度"，而因果增强通过 detect → intervene 的闭环，把增强预算集中在模型最需要的维度上。

**使用方式**: 在 `configs/causal_yolo.yaml` 中设置 `use_causal_augment: true`，训练时自动触发。

### 亮点 2：安全下载 (临时文件 + 中断清理)

**位置**: `datasets/download.py` → `download_file()`

**原理**: 原项目的 `download_and_extract()` 直接往目标路径写入数据。如果下载过程中网络中断，会留下一个不完整的文件，下次运行时 `os.path.exists()` 返回 True 就跳过下载，导致后续解压/加载失败。

新实现使用 `tempfile.NamedTemporaryFile` 先写入临时文件，下载完成后才 `shutil.move` 到目标路径。如果中断，`except BaseException` 会清理临时文件，保证目标路径要么是完整文件、要么不存在。

**为什么保留**: 这是一个低成本但高价值的改进。在服务器训练环境中，网络不稳定是常见问题，一个损坏的数据文件可能浪费数小时排查时间。

### 亮点 3：SVHN → YOLO 格式转换

**位置**: `datasets/download.py` → `convert_svhn_to_yolo()`

**原理**: 原项目的 `datasets/convert_to_yolo.py` 需要手动运行，且路径硬编码。新实现将转换逻辑集成到 `download.py` 中，使用相同的函数签名风格，支持从 `download_and_extract()` 后直接调用。

**为什么保留**: 统一数据准备流程，减少手动步骤。

### 亮点 4：置信度阈值 conf=0.3

**位置**: `configs/causal_yolo.yaml` → `yolo_conf: 0.3`

**原理**: YOLO 默认置信度阈值 0.25 会产生较多低置信度的误检测，但设置过高 (如 0.5) 会漏检模糊/小尺寸数字，导致准确率下降 5-8%。原实验通过实验验证 0.3 是最佳平衡点，在保持召回率的同时减少误检。

**为什么保留**: 这是一个来自实战的经验值，已在 `predict_yolo.py` 的文档字符串中标注。

### 亮点 5：稀有类别额外增强

**位置**: `datasets/causal_augment.py` → `_augment_single_image()`

**原理**: 统计训练集中每个数字类别的出现次数，低于平均值的类别被标记为"稀有"。包含稀有类别的图像会获得额外的增强次数 (`rare_extra_count`)，缓解类别不平衡问题。

**为什么保留**: SVHN 数据集中各数字分布不均匀，数字 0 和 1 明显多于 8 和 9。不处理会导致模型对少数类的识别能力偏弱。

---

## 四、重构决策

### 决策 1：将硬编码常量转为 YAML 配置

**原代码**:
```python
# causal_augment.py 中的全局常量
AUGMENT_COUNT = 4
CF_AUGMENT_COUNT = 2
USE_VAL_FOR_TRAIN = True
```

**重构后**: 所有参数迁移到 `configs/causal_yolo.yaml`，通过 `cfg` 对象传递。这符合项目已有的 YAML 配置体系，可以在不改代码的前提下调整增强策略。

### 决策 2：将独立脚本改为可导入模块

**原代码**: `causal_augment.py` 作为独立脚本运行 (`if __name__ == "__main__"`)，通过全局变量 `SCRIPT_DIR`、`BASE_DIR` 等硬编码路径。

**重构后**: 改为 `datasets/causal_augment.py` 模块，所有函数接受参数而非读取全局变量。入口点从 `train_yolo.py` 调用 `augment_yolo_dataset(cfg, src, dst)`。

### 决策 3：合并下载和格式转换到同一模块

**原代码**: `get_dataset.py` 包含下载、解压、YOLO 格式转换三个功能，且与项目已有的 `datasets/download.py` 功能重叠。

**重构后**: 将 `get_dataset.py` 的优秀实现（安全下载、进度条、YOLO 转换）合并到 `datasets/download.py`，保持单一数据准备入口。新增 `convert_svhn_to_yolo()` 函数导出。

### 决策 4：保持 albumentations 为可选依赖

因果增强需要 `albumentations` 和 `opencv-python`，但分类模型训练（ResNet50/SE-ResNet50）不需要。为避免增加基础依赖负担，`datasets/causal_augment.py` 在顶部使用 `try/except` 导入，`datasets/__init__.py` 同样容错导入。不安装 albumentations 时，分类模型训练完全不受影响。

### 决策 5：预览功能暂不迁移

原 `causal_augment.py` 中有 `preview_augmentation()` 和 `preview_augmentation_batch()` 函数，用 matplotlib 绘制增强前后对比图。这些是调试辅助功能，不影响训练流程，且依赖 matplotlib 的 GUI 后端（服务器上不一定有）。暂不迁移，后续有需要时可通过独立脚本实现。

---

## 五、如何使用

### 复现 CausalYOLOSvhn 实验

```bash
# 1. 安装依赖
pip install albumentations opencv-python ultralytics

# 2. 准备数据 (下载 + YOLO 格式转换)
python -c "from datasets.download import download_and_extract, convert_svhn_to_yolo; download_and_extract(dataset_path='./data'); convert_svhn_to_yolo('./data', './data/yolo')"

# 3. 因果增强 + YOLO11m 训练 (自动触发增强)
python train_yolo.py --config causal_yolo

# 4. 推理 (使用 conf=0.3)
python predict_yolo.py --model checkpoints/causal_yolo/train/weights/best.pt --config causal_yolo
```

### 仅使用因果增强 (不训练)

```python
from datasets.causal_augment import augment_yolo_dataset
from config import load_config

cfg = load_config("causal_yolo")
augment_yolo_dataset(cfg, "./data/yolo", "./data/yolo_enhanced")
```

---

## 六、潜在改进方向

1. **混杂因子检测升级**: 当前使用启发式阈值（亮度 < 60、Laplacian 方差 < 50 等），可以训练一个轻量分类器替代，提高检测精度。
2. **增强策略自适应**: 根据训练过程中模型在不同类型样本上的 loss 分布，动态调整各类干预的采样权重。
3. **TTA 集成**: 在推理阶段使用 Test-Time Augmentation，对同一张图的多个增强版本取平均，进一步提升精度。
4. **分类模型也用 albumentations**: 当前分类模型的增强用 torchvision.transforms，可以统一迁移到 albumentations 以获得更丰富的增强操作。

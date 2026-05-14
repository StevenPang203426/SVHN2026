# 配置文件编写指南与规范

本文档总结了深度学习项目中配置管理的推荐实践，并以本项目（SVHN 街景字符识别）为案例进行详细说明。

---

## 一、为什么需要配置文件？

在深度学习实验中，你通常需要调整大量超参数：学习率、batch size、数据增强策略、损失函数、网络架构……如果把这些值硬编码在 Python 文件中，会带来三个问题：

1. **改参数必须改代码**：每次实验都要修改源文件，容易引入 bug，也不方便版本管理。
2. **实验不可复现**：如果忘记记录某次实验的参数，之后无法重现。
3. **对比困难**：多组实验的参数散落在不同 commit 中，难以一眼对比差异。

配置文件把"做什么"（参数）和"怎么做"（代码）解耦，让你可以在不修改任何 Python 代码的情况下切换实验方案。

---

## 二、YAML vs Python vs JSON vs TOML — 格式选型

| 格式 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **YAML** | 可读性最好，支持注释，层级清晰 | 缩进敏感，大型文件易出错 | 深度学习实验配置（推荐） |
| Python (.py) | 可以写逻辑、引用变量 | 配置与代码耦合，难以序列化 | 快速原型 |
| JSON | 通用，所有语言都支持 | 不支持注释，冗长 | API 交互、Web 配置 |
| TOML | 语法简洁，支持注释 | 嵌套层级不如 YAML 直观 | Rust/Go 生态 |

**推荐：YAML**。在 PyTorch 深度学习社区中，YAML 是事实标准——mmdetection、YOLO、Hydra 等主流框架均采用 YAML。

---

## 三、推荐的配置体系结构

### 3.1 分层覆盖原则

```
default.yaml → 实验.yaml → 命令行参数
   (基础)        (覆盖)       (最高优先级)
```

每一层只需声明**与上一层不同的字段**，其余自动继承。

**default.yaml** 定义全部参数的默认值，充当"文档"角色——任何人打开它就能看到项目支持哪些参数、默认值是什么。

**实验.yaml**（如 `baseline.yaml`、`improved_v1.yaml`）只写需要覆盖的字段。这样做的好处是：打开一个实验 yaml，一眼就能看出它和默认配置的差异——这些差异就是该实验的核心变量。

**命令行参数** 用于临时调试，优先级最高。

### 3.2 文件组织

```
configs/
├── default.yaml              # 全量参数 + 注释（必须有）
├── baseline.yaml             # Baseline 实验
├── improved_v1.yaml          # 改进方案 1
├── improved_v2.yaml          # 改进方案 2
├── ablation_no_bn.yaml       # 消融实验 A
└── ablation_extreme_aug.yaml # 消融实验 B
```

注意 `configs/` 目录（复数，存放 YAML 文件）和 `config/` 目录（单数，存放 Python 加载器代码）是分开的：

```
config/                       # Python 包：配置加载逻辑
├── __init__.py
└── config.py                 # load_config(), build_config(), parse_args()

configs/                      # YAML 文件：纯数据，不含代码
├── default.yaml
├── baseline.yaml
└── ...
```

---

## 四、default.yaml 编写规范

default.yaml 是整个配置体系的基石。以下是编写要点：

### 4.1 分区注释

用注释将参数按功能分组，方便快速定位：

```yaml
# ── 数据集 ──
dataset_path: "./data"

# ── 训练参数 ──
batch_size: 64
lr: 0.001
weight_decay: 0.0001

# ── 模型 ──
model_name: "resnet50"           # resnet50 | se_resnet50 | resnet50_no_bn
pretrained_backbone: true

# ── 损失函数 ──
loss_type: "label_smooth"        # label_smooth | focal | ce
smooth: 0.1

# ── 数据增强 ──
aug_level: "medium"              # light | medium | strong | extreme
use_cutout: false
use_mixup: false
```

### 4.2 行尾注释标注可选值

对于枚举类型的参数，在行尾用 `#` 列出所有合法取值。这相当于在配置文件中内嵌了"文档"：

```yaml
model_name: "resnet50"           # resnet50 | se_resnet50 | resnet50_no_bn
loss_type: "label_smooth"        # label_smooth | focal | ce
scheduler: "cosine_warm"         # cosine_warm | step | one_cycle
aug_level: "medium"              # light | medium | strong | extreme
```

### 4.3 为每个参数提供合理的默认值

default.yaml 中的每个参数都必须有值（不要留空）。默认值应该是"保守但能跑通"的——新手拿到项目后 `python train.py` 直接能跑。

### 4.4 使用 null 而非空字符串

YAML 中用 `null` 表示"未设置"，对应 Python 的 `None`：

```yaml
pretrained: null                 # 载入已训练权重的路径（默认不载入）
```

不要用空字符串 `""` 代替 `null`，因为空字符串在 Python 中是 truthy 的，会导致逻辑错误。

---

## 五、实验 YAML 编写规范

### 5.1 只写差异字段

实验 yaml 应该尽可能短——只覆盖与 default.yaml 不同的字段。这样做的好处：

```yaml
# ============================================================
#  改进方案 1 — SE-ResNet50 + 强增强 + Cutout
#  核心改进: SE 通道注意力 + Cutout 正则化
#  预期 Val Acc: ~90%
# ============================================================

experiment_name: "improved_v1_se_resnet50"
model_name: "se_resnet50"
loss_type: "label_smooth"
aug_level: "strong"
use_cutout: true
lr: 0.0005
epochs: 60
scheduler: "cosine_warm"
```

一眼就能看出：这个实验换了 SE-ResNet50 模型、用了强增强和 Cutout、降低了学习率、延长了训练轮数。

### 5.2 顶部注释块

每个实验 yaml 的开头应包含一个注释块，说明：

1. **实验名称**：一句话概括
2. **核心改进**：与 baseline 相比改了什么
3. **预期效果**：大致预期的 Val Acc（帮助判断实验是否正常）

```yaml
# ============================================================
#  消融实验 A — 去除 BatchNorm 层
#  目的: 验证 BN 对深层网络训练稳定性的关键作用
#  预期 Val Acc: <60% (大幅下降，印证 BN 的必要性)
# ============================================================
```

### 5.3 命名规范

文件名使用 `snake_case`，名称应体现实验的关键变量：

| 文件名 | 含义 |
|--------|------|
| `baseline.yaml` | 基线实验 |
| `improved_v1.yaml` | 改进方案（按版本编号） |
| `ablation_no_bn.yaml` | 消融实验（`ablation_` 前缀 + 消融变量名） |
| `ablation_extreme_aug.yaml` | 消融实验（消融数据增强） |

不推荐的命名：`exp1.yaml`、`test.yaml`、`new.yaml`——这些名称在一个月后你自己都不记得是什么。

---

## 六、Python 加载器设计

### 6.1 核心加载流程

```python
def load_config(config_path=None):
    """
    加载配置: default.yaml -> 指定的实验 yaml
    """
    # 1) 始终加载 default.yaml 作为基础
    cfg_dict = load_yaml("configs/default.yaml")

    # 2) 如果指定了实验 yaml，用它覆盖
    if config_path is not None:
        exp_dict = load_yaml(config_path)
        cfg_dict.update(exp_dict)

    return Config(cfg_dict)
```

### 6.2 命令行覆盖

通过 `argparse` 接收命令行参数，但所有参数的 `default=None`。只有用户显式传入的参数才会覆盖 YAML 值：

```python
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--epochs", type=int, default=None)
```

在 `build_config()` 中，只合并非 None 的命令行值：

```python
def build_config(args):
    cfg = load_config(args.config)
    cli = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}
    cfg.merge(cli)   # merge 只覆盖非 None 值
    return cfg
```

### 6.3 Config 容器类

用一个轻量的容器类将字典转化为属性访问（`cfg.lr` 比 `cfg['lr']` 更简洁）：

```python
class Config:
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            setattr(self, k, v)

    def merge(self, other_dict):
        for k, v in other_dict.items():
            if v is not None:
                setattr(self, k, v)
```

### 6.4 路径解析的灵活性

好的配置加载器应该允许多种传入方式，自动解析路径：

```python
# 以下三种方式等价：
python train.py --config baseline
python train.py --config baseline.yaml
python train.py --config configs/baseline.yaml
```

实现方法是在 `load_config()` 中做路径探测：如果文件不存在，就尝试加 `.yaml` 后缀、拼接 `configs/` 目录前缀。

---

## 七、使用示例

### 7.1 运行实验

```bash
# 使用默认配置（default.yaml）
python train.py

# 使用 baseline 配置
python train.py --config baseline

# 使用改进方案 1 + 临时覆盖学习率
python train.py --config improved_v1 --lr 0.001

# 使用改进方案 2 + 启用 wandb
python train.py --config improved_v2 --use_wandb
```

### 7.2 推理

```bash
# 使用 improved_v1 的模型架构加载权重
python predict.py --model checkpoints/improved_v1.pth --config improved_v1

# 启用 TTA
python predict.py --model checkpoints/best.pth --config improved_v2 --use_tta
```

### 7.3 查看当前配置

在 Python 中可以直接打印 Config 对象：

```python
from config import load_config
cfg = load_config("improved_v1")
print(cfg)
# Config(experiment_name='improved_v1_se_resnet50', model_name='se_resnet50', ...)
```

---

## 八、通用规范总结（适用于任何深度学习项目）

### 8.1 文件结构规范

1. **一个 `default` 配置文件**包含全部参数及其默认值，附带行尾注释说明可选值
2. **每个实验一个 YAML 文件**，只写与默认不同的字段
3. 配置文件（`configs/`）与加载代码（`config/`）分离

### 8.2 参数命名规范

1. 使用 `snake_case`：`learning_rate`、`batch_size`、`use_cutout`
2. 布尔值以 `use_`、`enable_`、`is_` 开头：`use_wandb`、`enable_tta`
3. 路径以 `_path`、`_dir` 结尾：`dataset_path`、`checkpoints`
4. 枚举类型用字符串而非数字：`loss_type: "focal"` 而非 `loss_type: 2`

### 8.3 版本控制规范

1. **所有 YAML 配置文件必须纳入 Git**——它们是实验可复现性的核心
2. **不要把运行时生成的配置放入 Git**（如 wandb 自动保存的 config.yaml）
3. 在 `.gitignore` 中排除：`logs/`、`checkpoints/`、`wandb/`、`results/`

### 8.4 常见反模式（不推荐）

| 反模式 | 问题 | 推荐做法 |
|--------|------|----------|
| 把所有实验参数写在一个大 YAML 里 | 文件膨胀，难以 diff | 一个实验一个文件 |
| 在 Python 代码中硬编码默认值 | 代码和配置耦合 | 统一放在 default.yaml |
| 用 JSON 写配置 | 不支持注释 | 用 YAML |
| 实验文件抄写全部字段 | 看不出和 baseline 的差异 | 只写差异字段 |
| argparse default 设为具体值 | 命令行参数无法区分"用户传的"和"默认的" | default=None |

### 8.5 进阶：更大型项目的选择

当项目规模变大（数十个实验、多级嵌套配置）时，可以考虑以下工具：

1. **YACS**（Facebook/Meta 出品）：支持嵌套配置节点、freeze/defrost 机制，Detectron2 使用
2. **Hydra**（Facebook/Meta 出品）：支持配置组合（composition）、多运行（multirun）、自动创建输出目录
3. **OmegaConf**：Hydra 的底层库，支持变量插值（`${model.name}`）

本项目采用的自研轻量方案适合中小型项目（10 个实验以内）。如果你的项目实验数量超过 20 个，建议迁移到 Hydra。

---

## 九、本项目配置文件一览

| 文件 | 用途 | 关键覆盖参数 |
|------|------|-------------|
| `configs/default.yaml` | 全量默认值 | — |
| `configs/baseline.yaml` | 基线实验 | experiment_name, epochs=50 |
| `configs/improved_v1.yaml` | SE-ResNet50 + Cutout | model_name=se_resnet50, aug=strong, lr=5e-4, epochs=60 |
| `configs/improved_v2.yaml` | Focal + MixUp + OneCycle | loss=focal, use_mixup=true, scheduler=one_cycle, epochs=60 |
| `configs/ablation_no_bn.yaml` | 去除 BN 层 | model=resnet50_no_bn, lr=1e-4, epochs=30 |
| `configs/ablation_extreme_aug.yaml` | 过度增强 | aug=extreme, use_cutout=true, epochs=30 |

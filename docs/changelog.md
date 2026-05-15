# 项目变更日志

> 记录 SVHN 街景字符识别项目从单体脚本到模块化工程的完整演变过程。
> 每条变更包含：做了什么、为什么这么做、背后的原理，以及下次遇到类似问题时的解决方案。

---

## 阶段一：单体脚本拆分为模块化架构

### 变更内容

将 452 行的 `baseline.py` 拆分为 6 个模块：

```
baseline.py (452行) → config/ + datasets/ + models/ + losses/ + engine/ + utils/
```

每个目录职责单一：`config/` 管配置、`datasets/` 管数据加载与增强、`models/` 管网络定义、`losses/` 管损失函数、`engine/` 管训练循环、`utils/` 管推理和可视化。

### 为什么这么做

单体脚本在只有一个实验时可以工作，但一旦需要切换模型（ResNet50 → SE-ResNet50）、更换损失函数（CrossEntropy → Focal Loss）、对比不同增强策略，就会面临三个问题：

1. 每次实验都要复制整个文件，导致大量重复代码
2. 修改某个组件（比如加一种新的 loss）需要在一个大文件里定位和修改，容易引入副作用
3. 无法独立测试某个组件

模块化之后，新增一个模型只需在 `models/` 下加一个类并注册到工厂函数，其他代码完全不动。

### 原理

这是软件工程中的**单一职责原则**（SRP）和**关注点分离**（Separation of Concerns）。深度学习项目虽然不是传统软件，但同样受益于这种结构——事实上 mmdetection、detectron2、ultralytics 等主流框架都采用类似的分层设计。

### 经验总结

**下次遇到类似问题怎么办：** 当一个 Python 文件超过 200 行，且包含"数据处理 + 模型定义 + 训练循环 + 推理"等多种逻辑时，就该拆分了。拆分的粒度以"能独立替换"为标准——你应该能在不改 `trainer.py` 的前提下换一个新模型。

---

## 阶段二：实验预设系统（Python Dict → YAML）

### 变更内容

**第一版：** 在 `config.py` 中用 Python 字典定义实验预设（`EXPERIMENT_PRESETS`），通过 `--experiment baseline` 切换。

**第二版：** 迁移为 YAML 配置文件体系。每个实验一个 `.yaml` 文件，通过 `--config baseline` 加载。加载顺序为 `default.yaml → 实验.yaml → 命令行参数`，后者覆盖前者。

### 为什么从 Python Dict 迁移到 YAML

Python Dict 预设有三个缺陷：

1. **配置和代码耦合**：改一个超参数就要改 `.py` 文件，触发 Git diff，混在代码变更里很难区分
2. **无法添加注释说明可选值**：Python 字典里写注释很不自然
3. **不符合社区惯例**：PyTorch 生态（mmdet、YOLO、Hydra）几乎都用 YAML，团队协作时别人看到 YAML 目录就知道怎么用

YAML 的分层覆盖设计来自 mmdetection 的 `_base_` 机制和 Hydra 的 config composition。核心思想是：`default.yaml` 写全量参数作为文档，实验 YAML 只写差异——打开一个实验文件，一眼就看出它和 baseline 的区别。

### 迁移过程中遇到的问题

迁移后，`config/__init__.py`、`train.py`、`predict.py`、`ensemble.py`、`train.sh` 中大量引用了旧的 `EXPERIMENT_PRESETS` 和 `--experiment` 参数。需要逐一更新为 `load_config()` 和 `--config`。

更严重的是，由于 Git 操作（pull/checkout），修改多次被还原为旧版本。总共修复了 3 轮才彻底清除旧引用。

### 经验总结

**下次遇到类似问题怎么办：**

1. 做大范围重构时，先 `git stash` 或 `git commit` 保存当前状态，避免被 pull 覆盖
2. 重构完成后，立即用 `grep -r "旧关键词"` 全局搜索残留引用。本项目中应搜索 `EXPERIMENT_PRESETS`、`--experiment`、`from config import.*Config,` 等
3. 在 `__init__.py` 中控制导出——如果 `EXPERIMENT_PRESETS` 不在 `__init__.py` 的导出列表中，其他文件的 `from config import EXPERIMENT_PRESETS` 会立即报 ImportError，而不是运行到一半才出错

---

## 阶段三：项目结构整理（scripts/ + docs/ + archive/）

### 变更内容

```
根目录（整理前）                    根目录（整理后）
├── train.sh                       ├── train.py          # 入口文件
├── predict.sh                     ├── predict.py
├── setup.sh                       ├── train_yolo.py
├── fix_labels.py                  ├── predict_yolo.py
├── baseline.py (旧)               ├── ensemble.py
├── baseline.ipynb (旧)            ├── evaluate.py
├── report.md                      ├── requirements.txt
├── 人工智能-随堂练习3.pdf           ├── .gitignore
├── svhn_full_architecture_v2.svg  ├── scripts/          # 自动化脚本
└── ...                            ├── docs/             # 文档
                                   ├── configs/          # YAML 配置
                                   └── archive/          # 旧代码归档
```

### 为什么这么做

根目录是项目的"门面"。一个新人打开项目，应该在 3 秒内看懂项目结构。整理前的根目录混杂了入口脚本、shell 工具、旧代码、文档、PDF、SVG，认知负担很高。

整理原则：根目录只保留 Python 入口文件（`train.py`、`predict.py` 等）和项目配置（`requirements.txt`、`.gitignore`），其他一切按类型归入子目录。

### 遇到的问题

沙盒环境的挂载文件系统不允许 `rm` 删除原始文件（`Operation not permitted`）。解决方案是把旧文件内容替换为"转发桩"——比如根目录的 `train.sh` 变成 `exec bash scripts/train.sh "$@"`，实际逻辑在 `scripts/train.sh` 中。

移入 `scripts/` 后，shell 脚本中的 `cd "$(dirname "$0")"` 需要改为 `cd "$(dirname "$0")/.."` 以正确定位到项目根目录。

### 经验总结

**下次遇到类似问题怎么办：**

1. 移动脚本后，一定要更新脚本内部的相对路径引用（`cd` 目标、文件路径）
2. 如果无法删除文件（权限问题），转发桩是一种兼容性方案——旧路径仍可使用，但实际逻辑集中管理
3. 项目根目录的整洁程度直接影响协作效率，值得花时间维护

---

## 阶段四：Checkpoint 分目录 + 统一评估 + 混合集成

### 变更内容

1. **Checkpoint 按实验名存子目录**：`checkpoints/baseline/ep50-acc86.00.pth`，而非全部平铺在 `checkpoints/` 下
2. **创建 `evaluate.py`**：统一评估分类模型和 YOLO 的整串准确率（Full-String Accuracy）
3. **重写 `ensemble.py`**：支持分类模型 + YOLO 混合投票
4. **YOLO 使用 YAML 配置**：`configs/yolo.yaml`，和分类模型用法统一

### 为什么这么做

**Checkpoint 分目录：** 5 个实验各训练 50 轮，每次保存最优模型，全部平铺在一个目录下会有数十个 `.pth` 文件混在一起，根本分不清哪个属于哪个实验。按实验名分子目录后一目了然。

**统一评估（evaluate.py）：** 分类模型的 Trainer 在训练过程中已经输出了 Val Acc，但训练完成后想重新评估（比如换一组数据）需要重跑训练——这不合理。更重要的是，YOLO 的 mAP50 和分类模型的 Acc 是两种不同的指标，无法直接对比。`evaluate.py` 统一用"整串准确率"（4 位数字全部正确才算对）评估所有模型，让结果可以直接比较。

**YOLO 参与集成：** YOLO 是检测模型，输出的是 bounding box + 类别，而分类模型输出的是 logits。两者的预测格式完全不同。解决方案是在"字符串预测"层面做集成——每个模型独立输出预测字符串（如 "1234"），然后对多个字符串做多数投票。这样 YOLO 和分类模型就可以共存于同一个集成框架中。

### 原理

**整串准确率 vs mAP50：** mAP50 衡量的是每个单独数字的检测精度（IoU≥0.5 下的 precision-recall 曲线面积），即使某张图检测对了 3 个数字、漏了 1 个，mAP50 也会给予 75% 的"功劳"。而整串准确率要求全部正确，更贴近实际应用场景。因此需要一个统一的评估标准。

**投票集成 vs Logits 平均：** Logits 平均理论上更优（保留了模型的不确定性信息），但它要求所有模型的输出格式相同（4 个 11 类 logits）。YOLO 不产生这种格式的 logits，所以只能在"预测结果"层面做投票。投票的好处是对异构模型完全兼容——只要模型能输出一个字符串预测，就能参与投票。

### 经验总结

**下次遇到类似问题怎么办：**

1. 训练产出（checkpoint、日志、结果）应该从一开始就按实验名分目录，而不是等文件多了再整理
2. 不同类型模型的对比需要统一指标。如果一个模型输出 mAP、另一个输出 Acc，要先定义一个公共指标，再分别实现
3. 异构模型集成的关键是找到公共接口——在本项目中，公共接口是"字符串预测"。先把每个模型的输出统一转换为同一种格式，再做集成

---

## 阶段五：数据路径问题排查

### 变更内容

`DigitsDataset` 初始化时报 `num_samples=0`，原因是 `get_data_dir()` 硬编码了双层嵌套路径 `data/mchar_train/mchar_train/`，但实际服务器上解压后只有一层 `data/mchar_train/`。

修复方案：新增 `_resolve_img_dir()` 函数自动探测目录结构——优先检查双层路径，不存在则回退到单层。同时在 `DigitsDataset` 中加入诊断打印，帮助定位路径问题。

### 原理

数据集路径问题是深度学习项目最常见的"启动失败"原因。不同的数据来源（手动下载 vs 脚本下载 vs 云端预处理）可能产生不同的目录结构。硬编码路径在"只在我的机器上跑"时没问题，但换一台机器就可能炸。

### 经验总结

**下次遇到类似问题怎么办：**

1. 数据路径永远不要硬编码。至少做两层探测（常见的两种目录结构），找不到时给出清晰的错误信息（打印实际路径、打印目录内容）
2. 数据加载器在初始化时应打印加载了多少样本。`0 samples loaded` 加上路径信息，比 `num_samples should be positive` 的 ValueError 有用得多
3. 数据目录结构应该写在 README 或文档中，告诉使用者期望的格式

---

## 附录：文件被 Git 还原的问题

### 问题描述

在本项目的开发过程中，`config/config.py`、`predict.py`、`ensemble.py`、`train.py` 等文件被多次还原为旧版本。原因是远程仓库和本地修改存在分歧，`git pull` 时合并策略导致本地修改被覆盖。

### 根因分析

1. 修改后没有立即 `git commit`，导致本地修改处于"未提交"状态
2. 执行 `git pull` 时，远程的旧版本覆盖了本地的新修改
3. 多个文件的修改不是一次 commit 的，导致部分文件被还原但发现不及时

### 经验总结

1. **每完成一个逻辑单元的修改就 commit 一次**，不要攒一大堆修改再提交
2. **pull 之前先 commit 或 stash**：`git stash` → `git pull` → `git stash pop`
3. **pull 之后用 grep 检查关键标记**：`grep -r "EXPERIMENT_PRESETS" --include="*.py"` 如果有结果说明被还原了
4. **使用 rebase 而非 merge**：`git pull --rebase origin main` 能更清晰地看到冲突
5. **冲突解决时选正确的版本**：`git checkout --theirs <file>`（用远程版本）或 `git checkout --ours <file>`（用本地版本），搞清楚哪边是新的再选

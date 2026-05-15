"""
因果反事实数据增强 (Causal Counterfactual Augmentation)

核心思路:
    传统增强是"随机扰动"，而因果增强是"有的放矢"——先检测图像中可能导致
    模型学到虚假关联的混杂因子 (confound)，再针对性地施加 Do-干预，生成
    反事实样本，迫使模型学习真正的因果特征而非捷径特征。

流程:
    1. detect_confounds()  — 启发式检测混杂因子 (lighting/sensor/geometry/background)
    2. get_counterfactual_augmentation() — 按 Do-类别组织干预集合
    3. generate_counterfactual_samples() — 生成反事实样本 (targeted/mixed/random)
    4. augment_yolo_dataset() — 完整增强流水线 (常规增强 + 因果增强)

亮点保留说明:
    本模块源自 CausalYOLOSvhn 实验 (0.9346, 第 7 名), 其核心创新点是
    "基于因果推断的数据增强"。相比传统随机增强, 该方法:
    - 能自适应检测图像质量问题 (过暗/过曝/模糊/噪声/透视变形/复杂背景)
    - 针对检测到的问题施加对应的 Do-干预, 而非盲目增强
    - 生成的反事实样本能有效打破虚假关联, 提升模型鲁棒性

依赖: albumentations >= 2.0, opencv-python
"""

import math
import os
import random
import concurrent.futures

import cv2
import numpy as np
from tqdm import tqdm

try:
    import albumentations as A
except ImportError:
    A = None
    print("[WARN] albumentations 未安装, 因果增强功能不可用")
    print("  安装: pip install albumentations")


# ─────────────────────────────────────────────
#  混杂因子检测 (启发式, 轻量高效)
# ─────────────────────────────────────────────

def detect_confounds(image, bboxes=None):
    """
    启发式检测图像中可能的混杂因子

    检测 4 类混杂因子:
        - lighting:      亮度异常 (过暗/过曝) 或对比度过低
        - sensor_blur:   拉普拉斯方差过低 → 图像模糊
        - sensor_noise:  高频梯度占比过高 → 传感器噪声
        - geometry:      bbox 宽高比异常 → 透视/倾斜变形
        - background:    背景区域边缘密度过高 → 复杂背景干扰

    Args:
        image: BGR 图像 (numpy array)
        bboxes: YOLO 格式边界框列表 [[xc, yc, w, h], ...], 可为 None

    Returns:
        list[str]: 检测到的混杂因子名称
    """
    confounds = set()
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape[:2]

    # ── lighting: 亮度和对比度检测 ──
    mean_lum = float(np.mean(img_gray))
    std_lum = float(np.std(img_gray))
    if mean_lum < 60 or mean_lum > 200 or std_lum < 20:
        confounds.add("lighting")

    # ── sensor: 模糊检测 (Laplacian 方差) ──
    lap_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
    if lap_var < 50:
        confounds.add("sensor_blur")

    # ── sensor: 噪声检测 (Sobel 高频能量占比) ──
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    high_freq_ratio = (grad_mag > 20).sum() / (h * w)
    if high_freq_ratio > 0.12:
        confounds.add("sensor_noise")

    # ── geometry: bbox 宽高比异常检测 ──
    if bboxes is not None and len(bboxes) > 0:
        ratios = []
        for bb in bboxes:
            _, _, bw, bh = bb
            if bh > 0:
                ratios.append(bw / bh)
        if ratios:
            ratios = np.array(ratios)
            if ratios.mean() < 0.7 or ratios.mean() > 1.4 or ratios.std() > 0.3:
                confounds.add("geometry")

    # ── background: 背景边缘密度检测 ──
    mask = np.ones_like(img_gray, dtype=np.uint8)
    if bboxes is not None:
        for bb in bboxes:
            xc, yc, bw, bh = bb
            x1 = max(0, int((xc - bw / 2) * w))
            y1 = max(0, int((yc - bh / 2) * h))
            x2 = min(w - 1, int((xc + bw / 2) * w))
            y2 = min(h - 1, int((yc + bh / 2) * h))
            mask[y1:y2, x1:x2] = 0
    edges = cv2.Canny(img_gray, 100, 200)
    bg_pixels = max(1, (mask == 1).sum())
    edge_density = (edges[mask == 1] > 0).sum() / bg_pixels
    if edge_density > 0.02:
        confounds.add("background")

    return list(confounds)


# ─────────────────────────────────────────────
#  常规增强管线 (Albumentations SOTA)
# ─────────────────────────────────────────────

def build_yolo_augmentation(extra=False, is_small=False):
    """
    构建 YOLO 训练用的 albumentations 增强管线

    Args:
        extra: True 时施加更强增强 (用于稀有类别)
        is_small: True 时跳过噪声/模糊 (避免小图过度降质)

    Returns:
        A.Compose 增强流水线 (支持 YOLO bbox)
    """
    if A is None:
        raise ImportError("albumentations is required for YOLO augmentation")

    return A.Compose(
        [
            # 亮度 / 对比度 / 色相 / 饱和度
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.4, p=1),
                A.HueSaturationValue(
                    hue_shift_limit=30, sat_shift_limit=40,
                    val_shift_limit=30, p=1),
                A.RGBShift(
                    r_shift_limit=20, g_shift_limit=20,
                    b_shift_limit=20, p=1),
            ], p=1.0 if extra else 0.8),

            # 几何变换 (仿射)
            A.Affine(
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                rotate=(-15, 15),
                shear=(-15, 15),
                fit_output=False,
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0 if extra else 0.7,
            ),

            # 弹性 / 网格 / 光学扭曲
            A.OneOf([
                A.ElasticTransform(
                    alpha=1, sigma=50,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REPLICATE, p=1),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
                A.OpticalDistortion(
                    distort_limit=(-0.2, 0.2),
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT, p=1),
            ], p=0.8 if extra else 0.5),

            # 噪声 / 压缩 / 下采样 (小图跳过)
            A.OneOf([
                A.GaussNoise(
                    std_range=(0.1, 0.3), mean_range=(0, 0),
                    per_channel=True, p=1),
                A.ImageCompression(quality_range=(30, 100), p=1),
                A.Downscale(scale_range=(0.5, 0.75), p=1),
            ], p=0.6 if (extra and not is_small) else 0.0),

            # 模糊 (小图跳过)
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1),
                A.MedianBlur(blur_limit=3, p=1),
                A.GaussianBlur(blur_limit=2, p=1),
            ], p=0.5 if (extra and not is_small) else 0.0),

            # 天气效果
            A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.3 if extra else 0.1),

            # 颜色通道打乱
            A.OneOf([
                A.ChannelShuffle(p=1),
            ], p=0.6 if extra else 0.3),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.0,
            clip=True,
            filter_invalid_bboxes=True,
        ),
    )


# ─────────────────────────────────────────────
#  因果反事实干预集合 (Do-Calculus)
# ─────────────────────────────────────────────

def get_counterfactual_interventions():
    """
    返回按混杂因子类别组织的 Do-干预集合

    干预分为 4 类, 对应 detect_confounds 检测到的 4 类混杂因子:
        - illumination: 亮度/对比度/阴影/CLAHE
        - quality:      噪声/模糊/降采样/压缩
        - geometry:     透视/仿射/旋转/弹性
        - context:      遮挡/通道打乱/背景干扰

    Returns:
        dict[str, list[A.Compose]]: 因子名 → 干预变换列表
    """
    if A is None:
        raise ImportError("albumentations is required")

    bbox_params = A.BboxParams(
        format="yolo", label_fields=["class_labels"],
        min_visibility=0.0, clip=True, filter_invalid_bboxes=True)

    interventions = {
        "illumination": [
            A.Compose([A.RandomBrightnessContrast(
                brightness_limit=0.9, contrast_limit=0.0, p=1.0)],
                bbox_params=bbox_params),
            A.Compose([A.RandomBrightnessContrast(
                brightness_limit=(-0.9, -0.4), contrast_limit=0.0, p=1.0)],
                bbox_params=bbox_params),
            A.Compose([A.RandomShadow(
                shadow_roi=(0.0, 0.0, 1.0, 0.5), p=1.0)],
                bbox_params=bbox_params),
            A.Compose([A.CLAHE(
                clip_limit=8, tile_grid_size=(8, 8), p=1.0)],
                bbox_params=bbox_params),
        ],

        "quality": [
            A.Compose([A.GaussNoise(p=1.0)], bbox_params=bbox_params),
            A.Compose([A.MotionBlur(blur_limit=(3, 9), p=1.0)],
                       bbox_params=bbox_params),
            A.Compose([A.Downscale(scale_range=(0.3, 0.8), p=1.0)]),
            A.Compose([A.ImageCompression(quality=(10, 50), p=1.0)]),
        ],

        "geometry": [
            A.Compose([A.Perspective(scale=(0.05, 0.25), p=1.0)],
                       bbox_params=bbox_params),
            A.Compose([A.Affine(
                rotate=(-45, 45),
                translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
                shear=(-20, 20), p=1.0)],
                bbox_params=bbox_params),
            A.Compose([A.Rotate(limit=(-30, 30), p=1.0)],
                       bbox_params=bbox_params),
            A.Compose([A.ElasticTransform(
                alpha=1, sigma=50, alpha_affine=10, p=1.0)],
                bbox_params=bbox_params),
        ],

        "context": [
            A.Compose([A.CoarseDropout(
                max_holes=3, max_height=30, max_width=30,
                min_holes=1, min_height=8, min_width=8, p=1.0)],
                bbox_params=bbox_params),
            A.Compose([A.CoarseDropout(
                max_holes=2, max_height=20, max_width=20,
                fill_value=0, p=1.0)],
                bbox_params=bbox_params),
            A.Compose([A.ChannelShuffle(p=1.0), A.CoarseDropout(
                max_holes=2, max_height=20, max_width=20, p=1.0)],
                bbox_params=bbox_params),
        ],

        "random": [
            A.Compose([A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=1.0)],
                bbox_params=bbox_params),
            A.Compose([A.RGBShift(
                r_shift_limit=25, g_shift_limit=25,
                b_shift_limit=25, p=1.0)],
                bbox_params=bbox_params),
            A.Compose([A.Blur(blur_limit=5, p=1.0)], bbox_params=bbox_params),
        ],
    }
    return interventions


def _map_confound_to_factor(confound_name):
    """将 detect_confounds 返回的名称映射到 Do-干预类别"""
    name = confound_name.lower()
    if "light" in name or "illum" in name:
        return "illumination"
    elif "sensor" in name or "blur" in name or "noise" in name or "quality" in name:
        return "quality"
    elif "geom" in name or "perspect" in name:
        return "geometry"
    elif "back" in name or "context" in name:
        return "context"
    return None


def generate_counterfactual_samples(image, bboxes, class_labels,
                                     num_samples=2, strategy="targeted"):
    """
    生成反事实样本 (Do-干预策略)

    原理: 基于 SCM (结构因果模型) 的 Do-Calculus, 对检测到的混杂因子
    施加干预 do(X=x'), 生成反事实图像 Y' = f(do(X=x'), Z), 其中 Z 是
    保持不变的因果特征 (数字形状)。

    Args:
        image:        BGR 图像 (numpy array)
        bboxes:       YOLO 格式边界框 [[xc, yc, w, h], ...]
        class_labels: 对应类别列表
        num_samples:  生成的反事实样本数
        strategy:     干预策略
            - targeted: 基于 detect_confounds 检测结果选择对应干预 (推荐)
            - random:   从所有干预池随机抽取
            - mixed:    先 targeted, 不足时用 random 补齐

    Returns:
        list[dict]: 每个元素包含 image, bboxes, class_labels, intervention_type
    """
    cf_samples = []
    interventions_map = get_counterfactual_interventions()

    # 检测混杂因子并映射到 Do-类别
    detected = detect_confounds(image, bboxes)
    detected_factors = set()
    for d in detected:
        factor = _map_confound_to_factor(d)
        if factor:
            detected_factors.add(factor)
    detected_list = list(detected_factors)

    # 备用池 (flatten)
    all_pools = []
    for k, v in interventions_map.items():
        all_pools.extend([(k, aug) for aug in v])

    # 1) targeted: 为每个检测到的因子生成干预样本
    if strategy in ("targeted", "mixed"):
        for factor in detected_list:
            if len(cf_samples) >= num_samples:
                break
            pool = interventions_map.get(factor, [])
            if not pool:
                continue
            aug = random.choice(pool)
            try:
                augmented = aug(
                    image=image, bboxes=bboxes, class_labels=class_labels)
                if augmented and "bboxes" in augmented:
                    cf_samples.append({
                        "image": augmented["image"],
                        "bboxes": augmented["bboxes"],
                        "class_labels": augmented["class_labels"],
                        "intervention_type": "%s_targeted" % factor,
                    })
            except Exception as e:
                # 某些变换可能因图像尺寸等原因失败, 跳过即可
                continue

    # 2) mixed/random: 补齐至 num_samples
    while len(cf_samples) < num_samples:
        if strategy in ("random", "mixed"):
            k, aug = random.choice(all_pools)
            try:
                augmented = aug(
                    image=image, bboxes=bboxes, class_labels=class_labels)
                if augmented and "bboxes" in augmented:
                    cf_samples.append({
                        "image": augmented["image"],
                        "bboxes": augmented["bboxes"],
                        "class_labels": augmented["class_labels"],
                        "intervention_type": "%s_random" % k,
                    })
            except Exception:
                continue
        else:
            break

    return cf_samples


# ─────────────────────────────────────────────
#  YOLO 标签文件 I/O
# ─────────────────────────────────────────────

def read_yolo_label(label_path):
    """读取 YOLO 格式标签文件, 返回 (bboxes, class_labels)"""
    bboxes, class_labels = [], []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:5]])
    return bboxes, class_labels


def write_yolo_label(label_path, bboxes, class_labels):
    """写入 YOLO 格式标签文件"""
    with open(label_path, "w") as f:
        for i in range(len(bboxes)):
            bb = bboxes[i]
            f.write("%d %.6f %.6f %.6f %.6f\n" % (
                int(class_labels[i]), bb[0], bb[1], bb[2], bb[3]))


# ─────────────────────────────────────────────
#  单张图像增强 (供多进程调用)
# ─────────────────────────────────────────────

def _augment_single_image(args):
    """
    对单张图像执行常规增强 + 因果反事实增强

    Args:
        args: tuple (img_file, src_img_dir, src_label_dir,
                     dst_img_dir, dst_label_dir, rare_classes, aug_cfg)
    """
    (img_file, src_img_dir, src_label_dir,
     dst_img_dir, dst_label_dir, rare_classes, aug_cfg) = args

    img_path = os.path.join(src_img_dir, img_file)
    label_file = os.path.splitext(img_file)[0] + ".txt"
    label_path = os.path.join(src_label_dir, label_file)

    image = cv2.imread(img_path)
    if image is None:
        return

    bboxes, class_labels = read_yolo_label(label_path)
    if len(bboxes) > 0:
        bboxes = np.array(bboxes, dtype=np.float32)
    else:
        bboxes = np.zeros((0, 4), dtype=np.float32)

    aug_count = aug_cfg.get("augment_count", 4)
    cf_count = aug_cfg.get("cf_augment_count", 2)
    rare_extra = aug_cfg.get("rare_extra_count", 2)
    cf_strategy = aug_cfg.get("causal_strategy", "targeted")
    use_cf = aug_cfg.get("use_causal_augment", True)

    contains_rare = any(c in rare_classes for c in class_labels)
    num_aug = aug_count + (rare_extra if contains_rare else 0)
    is_small = image.shape[0] * image.shape[1] < 80 * 80

    ext = os.path.splitext(img_file)[1]
    base = os.path.splitext(img_file)[0]

    # ── 常规增强 ──
    for i in range(num_aug):
        augmentation = build_yolo_augmentation(
            extra=contains_rare, is_small=is_small)
        try:
            augmented = augmentation(
                image=image, bboxes=bboxes, class_labels=class_labels)
            aug_img = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_cls = augmented["class_labels"]

            new_img = "%s_aug%d%s" % (base, i + 1, ext)
            new_lbl = "%s_aug%d.txt" % (base, i + 1)
            cv2.imwrite(os.path.join(dst_img_dir, new_img), aug_img)
            write_yolo_label(
                os.path.join(dst_label_dir, new_lbl), aug_bboxes, aug_cls)
        except Exception:
            continue

    # ── 因果反事实增强 ──
    if use_cf and len(bboxes) > 0:
        cf_samples = generate_counterfactual_samples(
            image=image, bboxes=bboxes, class_labels=class_labels,
            num_samples=cf_count, strategy=cf_strategy)

        for i, cf in enumerate(cf_samples):
            new_img = "%s_cf%d%s" % (base, i + 1, ext)
            new_lbl = "%s_cf%d.txt" % (base, i + 1)
            cv2.imwrite(os.path.join(dst_img_dir, new_img), cf["image"])
            write_yolo_label(
                os.path.join(dst_label_dir, new_lbl),
                cf["bboxes"], cf["class_labels"])


# ─────────────────────────────────────────────
#  完整增强流水线
# ─────────────────────────────────────────────

def augment_yolo_dataset(cfg, src_dataset_dir, dst_dataset_dir):
    """
    对 YOLO 数据集执行完整的因果增强流水线

    流程:
        1. 复制原始数据集 (val/test 不增强)
        2. 可选: 将部分验证集加入训练集
        3. 常规 albumentations 增强
        4. 因果反事实增强 (detect_confounds → Do-干预)
        5. 生成 YOLO 数据集配置文件

    Args:
        cfg:             Config 实例 (含因果增强参数)
        src_dataset_dir: 原始 YOLO 数据集目录
        dst_dataset_dir: 增强后的输出目录
    """
    import shutil

    # 目录结构
    train_img = os.path.join(dst_dataset_dir, "train", "images")
    train_lbl = os.path.join(dst_dataset_dir, "train", "labels")
    val_img = os.path.join(dst_dataset_dir, "val", "images")
    val_lbl = os.path.join(dst_dataset_dir, "val", "labels")
    test_img = os.path.join(dst_dataset_dir, "test", "images")

    for d in [train_img, train_lbl, val_img, val_lbl, test_img]:
        os.makedirs(d, exist_ok=True)

    # 1) 复制验证集和测试集 (不增强)
    print("[CausalAug] 复制验证集和测试集...")
    _copy_dir(os.path.join(src_dataset_dir, "val", "images"), val_img)
    _copy_dir(os.path.join(src_dataset_dir, "val", "labels"), val_lbl)
    src_test = os.path.join(src_dataset_dir, "test", "images")
    if os.path.isdir(src_test):
        _copy_dir(src_test, test_img)

    # 2) 复制训练集
    print("[CausalAug] 复制原始训练集...")
    _copy_dir(os.path.join(src_dataset_dir, "train", "images"), train_img)
    _copy_dir(os.path.join(src_dataset_dir, "train", "labels"), train_lbl)

    # 3) 可选: 将部分验证集加入训练集
    use_val = getattr(cfg, "use_val_for_train", False)
    if use_val:
        ratio = getattr(cfg, "val_train_ratio", 0.8)
        _move_partial_val_to_train(
            val_img, val_lbl, train_img, train_lbl, ratio)

    # 4) 增强训练集
    aug_cfg = {
        "augment_count": getattr(cfg, "augment_count", 4),
        "cf_augment_count": getattr(cfg, "cf_augment_count", 2),
        "rare_extra_count": getattr(cfg, "rare_extra_count", 2),
        "causal_strategy": getattr(cfg, "causal_strategy", "targeted"),
        "use_causal_augment": getattr(cfg, "use_causal_augment", True),
    }
    _augment_train_set(train_img, train_lbl, aug_cfg)

    # 5) 生成 YOLO 配置
    _create_yolo_config(dst_dataset_dir)

    # 6) 打印统计
    _print_dataset_info(dst_dataset_dir)

    print("[CausalAug] 增强完成, 输出目录: %s" % dst_dataset_dir)


def _copy_dir(src, dst):
    """多线程复制目录中的文件"""
    import shutil
    if not os.path.isdir(src):
        return
    files = os.listdir(src)
    num_workers = os.cpu_count() or 4

    def _copy(f):
        shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
        list(tqdm(ex.map(_copy, files), total=len(files),
                  desc="  复制 %s" % os.path.basename(src)))


def _move_partial_val_to_train(val_img_dir, val_lbl_dir,
                                train_img_dir, train_lbl_dir, ratio=0.8):
    """将一部分验证集移动到训练集 (添加 val_ 前缀避免冲突)"""
    import shutil
    val_files = [f for f in os.listdir(val_img_dir)
                 if f.endswith((".png", ".jpg", ".jpeg"))]
    random.shuffle(val_files)
    n = int(len(val_files) * ratio)
    selected = val_files[:n]
    print("[CausalAug] 将 %d/%d 验证集图像加入训练集" % (n, len(val_files)))

    for img_file in tqdm(selected, desc="  迁移验证集"):
        new_name = "val_" + img_file
        lbl_name = os.path.splitext(img_file)[0] + ".txt"
        new_lbl = "val_" + lbl_name

        shutil.move(
            os.path.join(val_img_dir, img_file),
            os.path.join(train_img_dir, new_name))
        lbl_path = os.path.join(val_lbl_dir, lbl_name)
        if os.path.exists(lbl_path):
            shutil.move(lbl_path, os.path.join(train_lbl_dir, new_lbl))


def _augment_train_set(img_dir, lbl_dir, aug_cfg):
    """对训练集执行增强 (多进程加速)"""
    img_files = [f for f in os.listdir(img_dir)
                 if f.endswith((".png", ".jpg", ".jpeg"))]

    # 统计类别分布, 找出稀有类别
    class_counts = {i: 0 for i in range(10)}
    for f in img_files:
        lbl_path = os.path.join(lbl_dir, os.path.splitext(f)[0] + ".txt")
        if os.path.exists(lbl_path):
            _, cls = read_yolo_label(lbl_path)
            for c in cls:
                class_counts[c] = class_counts.get(c, 0) + 1

    avg = sum(class_counts.values()) / max(len(class_counts), 1)
    rare_classes = [c for c, n in class_counts.items() if n < avg]
    print("[CausalAug] 稀有类别 (将获得额外增强): %s" % rare_classes)

    args_list = [
        (f, img_dir, lbl_dir, img_dir, lbl_dir, rare_classes, aug_cfg)
        for f in img_files
    ]

    num_workers = os.cpu_count() or 4
    if num_workers <= 1:
        for args in tqdm(args_list, desc="  增强训练图像"):
            _augment_single_image(args)
    else:
        print("[CausalAug] 使用 %d 进程并行增强..." % num_workers)
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=num_workers) as ex:
            list(tqdm(ex.map(_augment_single_image, args_list),
                      total=len(args_list), desc="  增强训练图像"))


def _create_yolo_config(dataset_dir):
    """生成 YOLO 数据集配置文件 svhn.yaml"""
    content = """path: %s
train: train/images
val: val/images
test: test/images

nc: 10
names:
- '0'
- '1'
- '2'
- '3'
- '4'
- '5'
- '6'
- '7'
- '8'
- '9'
""" % os.path.abspath(dataset_dir)

    config_path = os.path.join(dataset_dir, "svhn.yaml")
    with open(config_path, "w") as f:
        f.write(content)
    print("[CausalAug] YOLO 配置已生成: %s" % config_path)


def _print_dataset_info(dataset_dir):
    """打印数据集统计信息"""
    splits = {"train": "train", "val": "val", "test": "test"}
    for name, folder in splits.items():
        img_dir = os.path.join(dataset_dir, folder, "images")
        if os.path.isdir(img_dir):
            count = len([f for f in os.listdir(img_dir)
                         if f.endswith((".png", ".jpg", ".jpeg"))])
            print("  %s: %d 图像" % (name, count))

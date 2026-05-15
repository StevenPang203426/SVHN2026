from .dataset import DigitsDataset, mixup_data, build_transforms, Cutout
from .download import download_and_extract, convert_svhn_to_yolo

# 因果增强模块 (可选依赖: albumentations)
try:
    from .causal_augment import (
        augment_yolo_dataset,
        detect_confounds,
        generate_counterfactual_samples,
        build_yolo_augmentation,
    )
except ImportError:
    pass

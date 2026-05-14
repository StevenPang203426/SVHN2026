"""
YAML 配置加载器
加载顺序: default.yaml -> 实验 yaml -> 命令行参数覆盖
"""
import os
import argparse
import yaml


class Config:
    """配置容器, 支持属性访问"""

    def __init__(self, cfg_dict=None):
        if cfg_dict:
            for k, v in cfg_dict.items():
                setattr(self, k, v)

    def merge(self, other_dict):
        """用字典覆盖当前配置 (仅覆盖非 None 值)"""
        for k, v in other_dict.items():
            if v is not None:
                setattr(self, k, v)

    def to_dict(self):
        return {k: v for k, v in vars(self).items() if not k.startswith('_')}

    def __repr__(self):
        items = ', '.join('%s=%r' % (k, v) for k, v in self.to_dict().items())
        return 'Config(%s)' % items


def _find_configs_dir():
    """自动定位 configs/ 目录"""
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    return os.path.join(project_root, 'configs')


def load_yaml(path):
    """加载单个 YAML 文件"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_config(config_path=None):
    """
    加载配置: default.yaml -> 指定的实验 yaml

    Args:
        config_path: 实验 yaml 路径, 可以是:
            - 完整路径: "configs/baseline.yaml"
            - 仅文件名: "baseline.yaml"
            - 仅名称:   "baseline"
    Returns:
        Config 实例
    """
    configs_dir = _find_configs_dir()

    # 1) 加载 default.yaml
    default_path = os.path.join(configs_dir, 'default.yaml')
    if os.path.exists(default_path):
        cfg_dict = load_yaml(default_path)
    else:
        cfg_dict = {}

    # 2) 加载实验 yaml (覆盖 default)
    if config_path is not None:
        # 支持多种传入方式
        if not os.path.exists(config_path):
            # 尝试在 configs/ 下查找
            if not config_path.endswith('.yaml'):
                config_path = config_path + '.yaml'
            config_path = os.path.join(configs_dir, config_path)

        if os.path.exists(config_path):
            exp_dict = load_yaml(config_path)
            cfg_dict.update(exp_dict)
        else:
            raise FileNotFoundError("Config not found: %s" % config_path)

    return Config(cfg_dict)


def get_data_dir(cfg):
    """根据 config 构建数据路径索引"""
    p = cfg.dataset_path
    return {
        'train_data': os.path.join(p, 'mchar_train', 'mchar_train') + os.sep,
        'val_data': os.path.join(p, 'mchar_val', 'mchar_val') + os.sep,
        'test_data': os.path.join(p, 'mchar_test_a', 'mchar_test_a') + os.sep,
        'train_label': os.path.join(p, 'mchar_train.json'),
        'val_label': os.path.join(p, 'mchar_val.json'),
        'submit_file': os.path.join(p, 'mchar_sample_submit_A.csv'),
    }


def parse_args():
    """命令行参数 (优先级最高, 覆盖 yaml)"""
    parser = argparse.ArgumentParser(description="SVHN Street Character Recognition")
    parser.add_argument("--config", type=str, default=None,
                        help="实验配置 yaml (如 baseline, improved_v1)")
    parser.add_argument("--model", type=str, default=None, dest="model_name")
    parser.add_argument("--loss", type=str, default=None, dest="loss_type")
    parser.add_argument("--aug", type=str, default=None, dest="aug_level")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true", default=None)
    parser.add_argument("--use_cutout", action="store_true", default=None)
    parser.add_argument("--use_mixup", action="store_true", default=None)
    parser.add_argument("--use_tta", action="store_true", default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None, dest="dataset_path")
    parser.add_argument("--name", type=str, default=None, dest="experiment_name")
    parser.add_argument("--num_workers", type=int, default=None)
    return parser.parse_args()


def build_config(args=None):
    """
    构建最终配置:
        default.yaml -> 实验 yaml -> 命令行参数

    Args:
        args: parse_args() 返回值
    Returns:
        Config 实例
    """
    config_name = None
    if args is not None:
        config_name = getattr(args, 'config', None)

    # 1) 加载 yaml
    cfg = load_config(config_name)

    # 2) 命令行覆盖
    if args is not None:
        cli = {k: v for k, v in vars(args).items()
               if v is not None and k != 'config'}
        cfg.merge(cli)

    return cfg


# ── 兼容旧代码: 直接 import 时提供默认配置 ──
config = load_config()
data_dir = get_data_dir(config)

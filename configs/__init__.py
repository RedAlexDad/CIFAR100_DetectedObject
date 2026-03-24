# Конфигурации для обучения CNN

from .config import *
from .training_config import get_config, print_config, DEFAULT_HYPERPARAMS, DEFAULT_AUGMENTATION, CONFIG_PRESETS
from .yaml_config import load_config, get_model_from_config, get_variants, print_config_summary

__all__ = [
    # Из config.py
    'CLASSES',
    'CLASS_NAMES',
    'CLASS_NAMES_RU',
    'MODEL_CONFIG',
    'TRAIN_CONFIG',
    'DEVICE',
    'CHECKPOINT_DIR',
    'ONNX_DIR',
    'OUTPUT_DIR',
    
    # Из training_config.py
    'get_config',
    'print_config',
    'DEFAULT_HYPERPARAMS',
    'DEFAULT_AUGMENTATION',
    'CONFIG_PRESETS',
    
    # Из yaml_config.py
    'load_config',
    'get_model_from_config',
    'get_variants',
    'print_config_summary',
]

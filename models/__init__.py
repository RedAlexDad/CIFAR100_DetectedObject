# Модели нейронных сетей для ЛР2

from .cnn_models import (
    Normalize,
    Cifar100_CNN_Base,
    Cifar100_CNN_Medium,
    Cifar100_CNN_Deep,
    Cifar100_CNN_Optimized,
)

# Модели для Transfer Learning (ЛР4)
from .transfer_models import (
    ResNet20CIFAR100,
    MobileNetV2CIFAR100,
    get_transfer_model,
    count_trainable_params,
    count_total_params,
    print_model_summary,
)

__all__ = [
    'Normalize',
    'Cifar100_CNN_Base',
    'Cifar100_CNN_Medium',
    'Cifar100_CNN_Deep',
    'Cifar100_CNN_Optimized',
    # Transfer Learning
    'ResNet20CIFAR100',
    'MobileNetV2CIFAR100',
    'get_transfer_model',
    'count_trainable_params',
    'count_total_params',
    'print_model_summary',
]

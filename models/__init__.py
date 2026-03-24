# Модели нейронных сетей для ЛР2

from .cnn_models import (
    Normalize,
    Cifar100_CNN_Base,
    Cifar100_CNN_Medium,
    Cifar100_CNN_Deep,
    Cifar100_CNN_Optimized,
)

__all__ = [
    'Normalize',
    'Cifar100_CNN_Base',
    'Cifar100_CNN_Medium',
    'Cifar100_CNN_Deep',
    'Cifar100_CNN_Optimized',
]

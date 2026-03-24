#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модели сверточных нейронных сетей для классификации CIFAR-100
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalize(nn.Module):
    """Слой нормализации изображений"""
    def __init__(self, mean=None, std=None):
        super(Normalize, self).__init__()
        if mean is None:
            mean = [0.5074, 0.4867, 0.4411]
        if std is None:
            std = [0.2011, 0.1987, 0.2025]
        # Используем register_buffer для правильного переноса на GPU/CPU
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
    
    def forward(self, x):
        # Поддержка обоих форматов: NHWC и NCHW
        if x.dim() == 4 and x.shape[1] > 10:  # NHWC формат (batch, height, width, channels)
            x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        # Иначе уже NCHW формат
        x = x / 255.0
        return (x - self.mean) / self.std


class Cifar100_CNN_Base(nn.Module):
    """
    Базовая модель CNN (2 сверточных слоя)
    Архитектура из примера
    """
    def __init__(self, num_classes=3, hidden_size=32):
        super(Cifar100_CNN_Base, self).__init__()
        self.normalize = Normalize()
        self.seq = nn.Sequential(
            nn.Conv2d(3, hidden_size, 5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_size * 2, num_classes)
        )
    
    def forward(self, x):
        x = self.normalize(x)
        x = self.seq(x)
        return x


class Cifar100_CNN_Medium(nn.Module):
    """
    Средняя модель CNN (3 сверточных слоя)
    Улучшенная архитектура с MaxPool
    """
    def __init__(self, num_classes=3):
        super(Cifar100_CNN_Medium, self).__init__()
        self.normalize = Normalize()
        self.seq = nn.Sequential(
            # Блок 1: 32x32 -> 16x16 (2 conv слоя)
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            # Блок 2: 16x16 -> 8x8 (1 conv слой)
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),

            # Выход
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.normalize(x)
        x = self.seq(x)
        return x


class Cifar100_CNN_Deep(nn.Module):
    """
    Глубокая модель CNN (6 сверточных слоёв)
    """
    def __init__(self, num_classes=3):
        super(Cifar100_CNN_Deep, self).__init__()
        self.normalize = Normalize()
        self.seq = nn.Sequential(
            # Блок 1: 32x32 -> 16x16 (2 conv)
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            # Блок 2: 16x16 -> 8x8 (2 conv)
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            # Блок 3: 8x8 -> 4x4 (2 conv)
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            
            # Выход
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.normalize(x)
        x = self.seq(x)
        return x


class Cifar100_CNN_Optimized(nn.Module):
    """
    ОПТИМИЗИРОВАННАЯ модель CNN (6 сверточных слоёв + BatchNorm + Dropout)
    """
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super(Cifar100_CNN_Optimized, self).__init__()
        self.normalize = Normalize()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.normalize(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

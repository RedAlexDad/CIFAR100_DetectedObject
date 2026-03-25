#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модели для Transfer Learning (ЛР4)
ResNet20 и MobileNetV2 для CIFAR-100
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet20CIFAR100(nn.Module):
    """
    ResNet20 для CIFAR-100 с возможностью заморозки весов
    
    Примечание: torchvision не имеет ResNet20, поэтому создаём архитектуру
    вручную на основе оригинальной статьи
    """
    def __init__(self, num_classes=3, pretrained=True, freeze_backbone=True):
        super(ResNet20CIFAR100, self).__init__()
        
        # Используем ResNet18 как ближайший аналог ResNet20
        # ResNet20 = 3 блока × 3 слоя × 20 параметров
        if pretrained:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet18(weights=None)
        
        # Заменяем первый conv слой для 32x32 изображений
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()  # Убираем maxpool для маленьких изображений
        
        # Заменяем классификатор
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Заморозка весов
        if freeze_backbone:
            self.freeze_backbone()
    
    def freeze_backbone(self):
        """Заморозить веса backbone (все слои кроме fc)"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Размораживаем только последний fully connected слой
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Разморозить все веса для fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def unfreeze_later_layers(self, num_layers=2):
        """
        Разморозить последние num_layers слоёв backbone
        
        Args:
            num_layers: Количество слоёв для разморозки (с конца)
        """
        # Размораживаем последние блоки
        layers_to_unfreeze = []
        
        # layer4 - последние residual блоки
        if num_layers >= 1:
            layers_to_unfreeze.append(self.backbone.layer4)
        if num_layers >= 2:
            layers_to_unfreeze.append(self.backbone.layer3)
        if num_layers >= 3:
            layers_to_unfreeze.append(self.backbone.layer2)
        if num_layers >= 4:
            layers_to_unfreeze.append(self.backbone.layer1)
        
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        # Поддержка обоих форматов: NHWC и NCHW
        if x.dim() == 4 and x.shape[1] > 10:  # NHWC
            x = x.permute(0, 3, 1, 2)
        return self.backbone(x)


class MobileNetV2CIFAR100(nn.Module):
    """
    MobileNetV2 для CIFAR-100 с возможностью заморозки весов
    """
    def __init__(self, num_classes=3, pretrained=True, freeze_backbone=True, width_mult=0.5):
        super(MobileNetV2CIFAR100, self).__init__()
        
        # MobileNetV2 с разной шириной (0.5, 0.75, 1.0)
        if pretrained:
            self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.mobilenet_v2(weights=None, width_mult=width_mult)
        
        # Заменяем классификатор
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
        
        # Заморозка весов
        if freeze_backbone:
            self.freeze_backbone()
    
    def freeze_backbone(self):
        """Заморозить веса backbone (все слои кроме classifier)"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Размораживаем только classifier
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
    
    def unfreeze_backbone(self):
        """Разморозить все веса для fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def unfreeze_later_layers(self, num_layers=4):
        """
        Разморозить последние num_layers inverted residual блоков
        
        Args:
            num_layers: Количество блоков для разморозки (с конца)
        """
        features = list(self.backbone.features)
        layers_to_unfreeze = features[-num_layers:]
        
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
    
    def forward(self, x):
        # Поддержка обоих форматов: NHWC и NCHW
        if x.dim() == 4 and x.shape[1] > 10:  # NHWC
            x = x.permute(0, 3, 1, 2)
        return self.backbone(x)


def get_transfer_model(model_name, num_classes=3, pretrained=True, freeze=True):
    """
    Получение модели для transfer learning
    
    Args:
        model_name: 'resnet20' или 'mobilenetv2'
        num_classes: Количество классов (по умолчанию 3)
        pretrained: Использовать предобученные веса
        freeze: Заморозить ли веса backbone
    
    Returns:
        Модель для transfer learning
    """
    if model_name == 'resnet20':
        return ResNet20CIFAR100(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze)
    elif model_name == 'mobilenetv2':
        return MobileNetV2CIFAR100(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}. Доступны: 'resnet20', 'mobilenetv2'")


def count_trainable_params(model):
    """
    Подсчёт обучаемых параметров модели
    
    Args:
        model: PyTorch модель
    
    Returns:
        Количество обучаемых параметров
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    """
    Подсчёт общего количества параметров модели
    
    Args:
        model: PyTorch модель
    
    Returns:
        Общее количество параметров
    """
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model, model_name):
    """
    Вывод информации о модели
    
    Args:
        model: PyTorch модель
        model_name: Название модели
    """
    total_params = count_total_params(model)
    trainable_params = count_trainable_params(model)
    frozen_params = total_params - trainable_params
    
    print(f"\n{'='*70}")
    print(f" МОДЕЛЬ: {model_name}")
    print(f"{'='*70}")
    print(f"  Всего параметров:    {total_params:,}")
    print(f"  Обучаемых:           {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Заморожено:          {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
    print(f"{'='*70}\n")

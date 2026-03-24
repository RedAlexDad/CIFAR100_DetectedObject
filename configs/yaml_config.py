#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Загрузчик конфигураций из YAML файлов
"""

import os
import yaml
from typing import Dict, List, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Загрузка конфигурации из YAML файла.
    
    Args:
        config_path: Путь к YAML файлу
    
    Returns:
        dict: Конфигурация
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Конфигурация не найдена: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_model_from_config(architecture: List[Dict]) -> 'nn.Sequential':
    """
    Создание модели из конфигурации.
    
    Args:
        architecture: Список слоев из конфигурации
    
    Returns:
        nn.Sequential модель
    """
    import torch.nn as nn
    
    layers = []
    
    for layer_config in architecture:
        layer_type = layer_config['type']
        args = layer_config.get('args', [])
        kwargs = layer_config.get('kwargs', {})
        
        if layer_type == 'Conv2d':
            layers.append(nn.Conv2d(*args, **kwargs))
        elif layer_type == 'ReLU':
            layers.append(nn.ReLU())
        elif layer_type == 'Dropout2d':
            layers.append(nn.Dropout2d(*args, **kwargs))
        elif layer_type == 'Dropout':
            layers.append(nn.Dropout(*args, **kwargs))
        elif layer_type == 'MaxPool2d':
            layers.append(nn.MaxPool2d(*args, **kwargs))
        elif layer_type == 'AvgPool2d':
            layers.append(nn.AvgPool2d(*args, **kwargs))
        elif layer_type == 'AdaptiveAvgPool2d':
            layers.append(nn.AdaptiveAvgPool2d(*args, **kwargs))
        elif layer_type == 'Flatten':
            layers.append(nn.Flatten())
        elif layer_type == 'Linear':
            layers.append(nn.Linear(*args, **kwargs))
        elif layer_type == 'BatchNorm2d':
            layers.append(nn.BatchNorm2d(*args, **kwargs))
        else:
            print(f"⚠️  Неизвестный тип слоя: {layer_type}")
    
    return nn.Sequential(*layers)


def get_variants(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Получение списка вариантов конфигураций.
    
    Args:
        config: Загруженная конфигурация
    
    Returns:
        Список вариантов для перебора
    """
    variants = config.get('variants', [])
    
    if not variants:
        # Если нет вариантов, используем основную конфигурацию
        return [{
            'name': config.get('name', 'default'),
            'model_architecture': config.get('model_architecture', []),
            'hyperparameters': config.get('hyperparameters', {}),
            'augmentation': config.get('augmentation', {})
        }]
    
    # Добавляем недостающие параметры из основной конфигурации
    base_config = {
        'hyperparameters': config.get('hyperparameters', {}),
        'augmentation': config.get('augmentation', {})
    }
    
    result = []
    for variant in variants:
        merged = base_config.copy()
        merged.update(variant)
        result.append(merged)
    
    return result


def print_config_summary(config: Dict[str, Any]):
    """Вывод краткой информации о конфигурации"""
    print("\n" + "="*70)
    print(" КОНФИГУРАЦИЯ ИЗ YAML ")
    print("="*70)
    
    print(f"\n📛 Название: {config.get('name', 'default')}")
    print(f"📝 Описание: {config.get('description', '')}")
    
    # Архитектура
    print("\n🏗️  АРХИТЕКТУРА МОДЕЛИ:")
    arch = config.get('model_architecture', [])
    for i, layer in enumerate(arch):
        layer_type = layer.get('type', 'Unknown')
        args = layer.get('args', [])
        print(f"  {i+1}. {layer_type}{tuple(args) if args else ''}")
    
    # Гиперпараметры
    print("\n📊 ГИПЕРПАРАМЕТРЫ:")
    hp = config.get('hyperparameters', {})
    print(f"  Learning rate:     {hp.get('learning_rate', 'N/A')}")
    print(f"  Momentum:          {hp.get('momentum', 'N/A')}")
    print(f"  Weight decay:      {hp.get('weight_decay', 'N/A')}")
    print(f"  Batch size:        {hp.get('batch_size', 'N/A')}")
    print(f"  Epochs:            {hp.get('epochs', 'N/A')}")
    print(f"  Label smoothing:   {hp.get('label_smoothing', 'N/A')}")
    
    scheduler = hp.get('scheduler', {})
    if scheduler:
        print(f"  Scheduler:         {scheduler.get('type', 'N/A')}")
        print(f"    step_size:       {scheduler.get('step_size', 'N/A')}")
        print(f"    gamma:           {scheduler.get('gamma', 'N/A')}")
    
    # Аугментация
    print("\n🎨 АУГМЕНТАЦИЯ:")
    aug = config.get('augmentation', {})
    print(f"  Brightness:        {aug.get('brightness', 'N/A')}")
    print(f"  Contrast:          {aug.get('contrast', 'N/A')}")
    print(f"  Saturation:        {aug.get('saturation', 'N/A')}")
    print(f"  Hue:               {aug.get('hue', 'N/A')}")
    print(f"  Rotation:          {aug.get('rotation', 'N/A')}")
    print(f"  Translate:         {aug.get('translate', 'N/A')}")
    print(f"  Scale:             {aug.get('scale', 'N/A')}")
    print(f"  Shear:             {aug.get('shear', 'N/A')}")
    
    # Варианты
    variants = config.get('variants', [])
    if variants:
        print(f"\n🔄 ВАРИАНТЫ ДЛЯ ПЕРЕБОРА: {len(variants)}")
        for v in variants:
            print(f"  - {v.get('name', 'unnamed')}")
    
    print("="*70 + "\n")


# Пример использования
if __name__ == '__main__':
    import sys
    
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'configs/config.yaml'
    
    config = load_config(config_file)
    print_config_summary(config)
    
    variants = get_variants(config)
    print(f"Найдено вариантов: {len(variants)}")

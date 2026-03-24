#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Запуск трех независимых экспериментов:
1. Перебор Dropout
2. Перебор Weight Decay
3. Перебор Аугментации
"""

import argparse
import yaml
import itertools
import random
from typing import List, Dict, Any


def load_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def generate_dropout_combinations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Генерация комбинаций для эксперимента 1 (Dropout)"""
    exp = config.get('experiment_1_dropout', {})
    base_arch = config.get('base_architecture', [])
    base_aug = config.get('base_augmentation', {})
    base_hp = config.get('hyperparameters', {})
    
    # dropout_1 может быть фиксированным числом или списком
    dropout_1_val = exp.get('dropout_1', 0.2)
    dropout_1_list = dropout_1_val if isinstance(dropout_1_val, list) else [dropout_1_val]
    dropout_2_list = exp.get('dropout_2', [0.3])
    
    combinations = []
    for d1, d2 in itertools.product(dropout_1_list, dropout_2_list):
        # Создаем архитектуру с нужным dropout
        arch = []
        dropout_count = 0
        for layer in base_arch:
            if layer['type'] == 'Dropout2d':
                args = layer['args'].copy()
                args[0] = d1 if dropout_count == 0 else d2
                arch.append({'type': 'Dropout2d', 'args': args})
                dropout_count += 1
            else:
                arch.append(layer.copy())
        
        combinations.append({
            'name': f"dropout_d1_{d1:.2f}_d2_{d2:.2f}",
            'model_architecture': arch,
            'hyperparameters': base_hp.copy(),
            'augmentation': base_aug.copy()
        })
    
    return combinations


def generate_weight_decay_combinations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Генерация комбинаций для эксперимента 2 (Weight Decay)"""
    exp = config.get('experiment_2_weight_decay', {})
    base_arch = config.get('base_architecture', [])
    base_aug = config.get('base_augmentation', {})
    base_hp = config.get('hyperparameters', {})
    
    wd_list = exp.get('weight_decay', [1e-5])
    d1 = exp.get('dropout_1', 0.2)
    d2 = exp.get('dropout_2', 0.3)
    
    combinations = []
    for wd in wd_list:
        # Создаем архитектуру с фиксированным dropout
        arch = []
        dropout_count = 0
        for layer in base_arch:
            if layer['type'] == 'Dropout2d':
                args = layer['args'].copy()
                args[0] = d1 if dropout_count == 0 else d2
                arch.append({'type': 'Dropout2d', 'args': args})
                dropout_count += 1
            else:
                arch.append(layer.copy())
        
        hp = base_hp.copy()
        hp['weight_decay'] = wd
        
        combinations.append({
            'name': f"wd_{wd:.0e}",
            'model_architecture': arch,
            'hyperparameters': hp,
            'augmentation': base_aug.copy()
        })
    
    return combinations


def generate_augmentation_combinations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Генерация комбинаций для эксперимента 3 (Аугментация)"""
    exp = config.get('experiment_3_augmentation', {})
    base_arch = config.get('base_architecture', [])
    base_hp = config.get('hyperparameters', {})
    
    d1 = exp.get('dropout_1', 0.2)
    d2 = exp.get('dropout_2', 0.3)
    wd = exp.get('weight_decay', 1e-5)
    
    # Получаем предопределенные уровни аугментации
    aug_levels = exp.get('augmentation_levels', [])
    
    combinations = []
    
    for i, aug in enumerate(aug_levels):
        # Создаем архитектуру с фиксированным dropout
        arch = []
        dropout_count = 0
        for layer in base_arch:
            if layer['type'] == 'Dropout2d':
                args = layer['args'].copy()
                args[0] = d1 if dropout_count == 0 else d2
                arch.append({'type': 'Dropout2d', 'args': args})
                dropout_count += 1
            else:
                arch.append(layer.copy())
        
        hp = base_hp.copy()
        hp['weight_decay'] = wd
        
        combinations.append({
            'name': f"aug_level_{i}",
            'model_architecture': arch,
            'hyperparameters': hp,
            'augmentation': aug,
            'level': i
        })
    
    return combinations


def print_experiment_summary(config: Dict[str, Any], exp_num: int):
    """Вывод информации об эксперименте"""
    exp_keys = {
        1: 'experiment_1_dropout',
        2: 'experiment_2_weight_decay',
        3: 'experiment_3_augmentation'
    }
    exp_names = {
        1: 'DROPOUT',
        2: 'WEIGHT DECAY',
        3: 'AUGMENTATION'
    }
    
    exp = config.get(exp_keys[exp_num], {})
    
    print(f"\n{'='*70}")
    print(f" ЭКСПЕРИМЕНТ {exp_num}: ПЕРЕБОР {exp_names[exp_num]}")
    print(f"{'='*70}")
    print(f"Описание: {exp.get('description', '')}")
    
    if exp_num == 1:
        d1 = exp.get('dropout_1', 0.2)
        d2 = exp.get('dropout_2', [])
        
        # d1 может быть числом или списком
        if isinstance(d1, list):
            d1_count = len(d1)
            d1_str = f"{d1_count} значений"
        else:
            d1_count = 1
            d1_str = f"{d1:.2f} (фиксирован)"
        
        d2_count = len(d2) if isinstance(d2, list) else 0
        
        print(f"\nПараметры для перебора:")
        print(f"  Dropout 1: {d1_str}")
        print(f"  Dropout 2: {d2_count} значений")
        print(f"  Всего комбинаций: {d1_count * d2_count}")
        print(f"\nФиксировано:")
        print(f"  Weight decay: {exp.get('weight_decay', 'N/A')}")
        print(f"  Аугментация: base")
    
    elif exp_num == 2:
        wd = exp.get('weight_decay', [])
        print(f"\nПараметры для перебора:")
        print(f"  Weight decay: {len(wd)} значений")
        print(f"  Всего комбинаций: {len(wd)}")
        print(f"\nФиксировано:")
        print(f"  Dropout 1: {exp.get('dropout_1', 'N/A')}")
        print(f"  Dropout 2: {exp.get('dropout_2', 'N/A')}")
        print(f"  Аугментация: base")
    
    elif exp_num == 3:
        aug_levels = exp.get('augmentation_levels', [])
        total = len(aug_levels)
        
        print(f"\nПараметры для перебора:")
        print(f"  Уровни аугментации: {total} (синхронизированные)")
        print(f"  Всего комбинаций: {total}")
        
        # Показываем несколько примеров
        if len(aug_levels) > 0:
            print(f"\nПримеры уровней:")
            for i in [0, len(aug_levels)//2, -1]:
                if 0 <= i < len(aug_levels):
                    aug = aug_levels[i]
                    br = aug.get('brightness', [0,0])
                    rot = aug.get('rotation', [0,0])
                    print(f"  Уровень {i}: brightness={br}, rotation={rot}")
        
        print(f"\nФиксировано:")
        print(f"  Dropout 1: {exp.get('dropout_1', 'N/A')}")
        print(f"  Dropout 2: {exp.get('dropout_2', 'N/A')}")
        print(f"  Weight decay: {exp.get('weight_decay', 'N/A')}")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Генерация комбинаций для независимых экспериментов')
    parser.add_argument('config', help='Путь к YAML конфигурации')
    parser.add_argument('--exp', type=int, choices=[1, 2, 3], help='Номер эксперимента (1, 2, или 3)')
    parser.add_argument('--max', type=int, default=50, help='Максимум комбинаций для показа')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.exp:
        print_experiment_summary(config, args.exp)
    else:
        # Показать все эксперименты
        for i in [1, 2, 3]:
            print_experiment_summary(config, i)

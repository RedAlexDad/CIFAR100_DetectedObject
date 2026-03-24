#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор комбинаций для Grid Search
"""

import itertools
import random
from typing import List, Dict, Any


def generate_augmentation_combinations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Генерация комбинаций аугментации из параметров.
    
    Args:
        config: Конфигурация grid_search
    
    Returns:
        Список комбинаций аугментации
    """
    gs = config.get('grid_search', {})
    
    # Получаем параметры
    brightness_min = gs.get('brightness_min', [0.9])
    brightness_max = gs.get('brightness_max', [1.1])
    contrast_min = gs.get('contrast_min', [0.9])
    contrast_max = gs.get('contrast_max', [1.1])
    saturation_min = gs.get('saturation_min', [0.8])
    saturation_max = gs.get('saturation_max', [1.2])
    rotation_min = gs.get('rotation_min', [-15.0])
    rotation_max = gs.get('rotation_max', [15.0])
    translate = gs.get('translate', [0.1])
    scale_delta = gs.get('scale_delta', [0.2])
    shear_min = gs.get('shear_min', [-5.0])
    shear_max = gs.get('shear_max', [5.0])
    
    # Генерируем все комбинации
    combinations = []
    for b_min, b_max, c_min, c_max, s_min, s_max, r_min, r_max, t, sd, sh_min, sh_max in \
        itertools.product(
            brightness_min, brightness_max,
            contrast_min, contrast_max,
            saturation_min, saturation_max,
            rotation_min, rotation_max,
            translate,
            scale_delta,
            shear_min, shear_max
        ):
        combinations.append({
            'brightness': [b_min, b_max],
            'contrast': [c_min, c_max],
            'saturation': [s_min, s_max],
            'hue': None,
            'rotation': [r_min, r_max],
            'translate': [t, t],
            'scale': [1.0 - sd, 1.0 + sd],
            'shear': [sh_min, sh_max]
        })
    
    return combinations


def generate_all_combinations(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Генерация всех комбинаций гиперпараметров.
    
    Args:
        config: Конфигурация grid_search
    
    Returns:
        Список всех комбинаций
    """
    gs = config.get('grid_search', {})
    
    # Получаем параметры для перебора
    dropout_1_list = gs.get('dropout_1', [0.2])
    dropout_2_list = gs.get('dropout_2', [0.3])
    weight_decay_list = gs.get('weight_decay', [1e-5])
    
    # Генерируем комбинации аугментации
    aug_combinations = generate_augmentation_combinations(config)
    
    # Генерируем все комбинации
    all_combinations = []
    
    for d1, d2, wd, aug in itertools.product(
        dropout_1_list,
        dropout_2_list,
        weight_decay_list,
        aug_combinations
    ):
        all_combinations.append({
            'dropout_1': d1,
            'dropout_2': d2,
            'weight_decay': wd,
            'augmentation': aug
        })
    
    return all_combinations


def generate_random_combinations(config: Dict[str, Any], n: int) -> List[Dict[str, Any]]:
    """
    Генерация случайных комбинаций.
    
    Args:
        config: Конфигурация grid_search
        n: Количество комбинаций
    
    Returns:
        Список случайных комбинаций
    """
    all_combs = generate_all_combinations(config)
    
    if n >= len(all_combs):
        return all_combs
    
    return random.sample(all_combs, n)


def get_combination_name(combo: Dict[str, Any], idx: int) -> str:
    """
    Генерация имени для комбинации.
    
    Args:
        combo: Комбинация параметров
        idx: Индекс комбинации
    
    Returns:
        Имя комбинации
    """
    d1 = combo.get('dropout_1', 0)
    d2 = combo.get('dropout_2', 0)
    wd = combo.get('weight_decay', 0)
    aug = combo.get('augmentation', {})
    
    # Уровень аугментации
    brightness_range = aug.get('brightness', [1, 1])
    br_diff = brightness_range[1] - brightness_range[0]
    
    if br_diff < 0.15:
        aug_level = "weak"
    elif br_diff < 0.35:
        aug_level = "base"
    else:
        aug_level = "strong"
    
    return f"d1_{d1:.2f}_d2_{d2:.2f}_wd_{wd:.0e}_{aug_level}"


def print_grid_search_summary(config: Dict[str, Any]):
    """Вывод информации о grid search"""
    gs = config.get('grid_search', {})
    settings = config.get('grid_settings', {})
    
    print("\n" + "="*70)
    print(" GRID SEARCH КОНФИГУРАЦИЯ ")
    print("="*70)
    
    # Количество параметров
    d1_count = len(gs.get('dropout_1', []))
    d2_count = len(gs.get('dropout_2', []))
    wd_count = len(gs.get('weight_decay', []))
    
    aug_count = (
        len(gs.get('brightness_min', [])) * len(gs.get('brightness_max', [])) *
        len(gs.get('contrast_min', [])) * len(gs.get('contrast_max', [])) *
        len(gs.get('saturation_min', [])) * len(gs.get('saturation_max', [])) *
        len(gs.get('rotation_min', [])) * len(gs.get('rotation_max', [])) *
        len(gs.get('translate', [])) * len(gs.get('scale_delta', [])) *
        len(gs.get('shear_min', [])) * len(gs.get('shear_max', []))
    )
    
    total = d1_count * d2_count * wd_count * aug_count
    
    print(f"\n📊 ПАРАМЕТРЫ ДЛЯ ПЕРЕБОРА:")
    print(f"  Dropout 1:       {d1_count} значений ({gs.get('dropout_1', [])[:3]}...)")
    print(f"  Dropout 2:       {d2_count} значений ({gs.get('dropout_2', [])[:3]}...)")
    print(f"  Weight decay:    {wd_count} значений ({gs.get('weight_decay', [])[:3]}...)")
    print(f"  Аугментация:     {aug_count} комбинаций")
    print(f"\n📈 ВСЕГО КОМБИНАЦИЙ: {total:,}")
    
    max_combs = settings.get('max_combinations', total)
    if settings.get('random_sample', False):
        print(f"  Случайная выборка: {max_combs} комбинаций")
    else:
        print(f"  Полный перебор: {min(total, max_combs)} комбинаций")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    import yaml
    import sys
    
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'configs/grid_search_config.yaml'
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print_grid_search_summary(config)
    
    # Генерация тестовых комбинаций
    combs = generate_random_combinations(config, 5)
    print(f"\nПримеры комбинаций (5 из {len(generate_all_combinations(config))}):")
    for i, c in enumerate(combs):
        name = get_combination_name(c, i)
        print(f"  {i+1}. {name}")
        print(f"      Dropout: {c['dropout_1']}, {c['dropout_2']}")
        print(f"      Weight decay: {c['weight_decay']}")
        print(f"      Aug: brightness={c['augmentation']['brightness']}")

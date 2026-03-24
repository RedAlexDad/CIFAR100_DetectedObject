#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Конфигурация для обучения CNN
Гиперпараметры и аугментация данных
"""

# ============================================================================
# ГИПЕРПАРАМЕТРЫ ПО УМОЛЧАНИЮ
# ============================================================================

DEFAULT_HYPERPARAMS = {
    # Оптимизатор
    'learning_rate': 0.005,
    'momentum': 0.9,
    'weight_decay': 1e-5,
    
    # Обучение
    'batch_size': 128,
    'epochs': 500,
    
    # Функция потерь
    'label_smoothing': 0.1,
    
    # Планировщик (StepLR)
    'step_size': 240,
    'gamma': 0.5,
}

# ============================================================================
# АУГМЕНТАЦИЯ ДАННЫХ
# ============================================================================

DEFAULT_AUGMENTATION = {
    # Цветовые искажения
    'brightness': (0.9, 1.1),    # Яркость
    'contrast': (0.9, 1.1),       # Контраст
    'saturation': (0.8, 1.2),     # Насыщенность
    'hue': None,                  # Оттенок (None = не менять)
    
    # Геометрические искажения
    'degrees': (-15.0, 15.0),     # Поворот (градусы)
    'translate': (0.1, 0.1),      # Сдвиг (доля от размера)
    'scale': (0.8, 1.2),          # Масштаб
    'shear': (-5.0, 5.0),         # Наклон (градусы)
}

# ============================================================================
# ПРЕСЕТЫ КОНФИГУРАЦИЙ
# ============================================================================

CONFIG_PRESETS = {
    'base': {
        'hyperparams': DEFAULT_HYPERPARAMS.copy(),
        'augmentation': DEFAULT_AUGMENTATION.copy(),
        'description': 'Базовая конфигурация'
    },
    
    'fast': {
        'hyperparams': {
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'batch_size': 256,
            'epochs': 100,
            'label_smoothing': 0.0,
            'step_size': 50,
            'gamma': 0.5,
        },
        'augmentation': DEFAULT_AUGMENTATION.copy(),
        'description': 'Быстрое обучение (для тестов)'
    },
    
    'accurate': {
        'hyperparams': {
            'learning_rate': 0.001,
            'momentum': 0.9,
            'weight_decay': 5e-5,
            'batch_size': 64,
            'epochs': 800,
            'label_smoothing': 0.15,
            'step_size': 300,
            'gamma': 0.5,
        },
        'augmentation': {
            'brightness': (0.8, 1.2),
            'contrast': (0.8, 1.2),
            'saturation': (0.7, 1.3),
            'hue': (-0.1, 0.1),
            'degrees': (-30.0, 30.0),
            'translate': (0.15, 0.15),
            'scale': (0.7, 1.3),
            'shear': (-10.0, 10.0),
        },
        'description': 'Точная модель (долгое обучение)'
    },
}


def get_config(preset='base'):
    """
    Получение конфигурации по названию пресета.
    
    Args:
        preset: Название пресета ('base', 'fast', 'accurate')
    
    Returns:
        dict: Конфигурация
    """
    if preset not in CONFIG_PRESETS:
        print(f"⚠️  Пресет '{preset}' не найден, используем 'base'")
        preset = 'base'
    
    config = CONFIG_PRESETS[preset].copy()
    config['preset_name'] = preset
    return config


def print_config(config):
    """Вывод конфигурации"""
    print("\n" + "="*70)
    print(" КОНФИГУРАЦИЯ ОБУЧЕНИЯ ")
    print("="*70)
    
    if 'preset_name' in config:
        print(f"Пресет: {config['preset_name']} ({config.get('description', '')})")
    
    print("\n📊 ГИПЕРПАРАМЕТРЫ:")
    hp = config['hyperparams']
    print(f"  Learning rate:     {hp['learning_rate']}")
    print(f"  Momentum:          {hp['momentum']}")
    print(f"  Weight decay:      {hp['weight_decay']}")
    print(f"  Batch size:        {hp['batch_size']}")
    print(f"  Epochs:            {hp['epochs']}")
    print(f"  Label smoothing:   {hp['label_smoothing']}")
    print(f"  Step size:         {hp['step_size']}")
    print(f"  Gamma:             {hp['gamma']}")
    
    print("\n🎨 АУГМЕНТАЦИЯ:")
    aug = config['augmentation']
    print(f"  Brightness:        {aug['brightness']}")
    print(f"  Contrast:          {aug['contrast']}")
    print(f"  Saturation:        {aug['saturation']}")
    print(f"  Hue:               {aug['hue']}")
    print(f"  Rotation:          {aug['degrees']}°")
    print(f"  Translate:         {aug['translate']}")
    print(f"  Scale:             {aug['scale']}")
    print(f"  Shear:             {aug['shear']}°")
    
    print("="*70 + "\n")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Утилиты для работы с данными
Загрузка, предобработка, визуализация CIFAR-100
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_cifar100_data(data_dir, classes):
    """
    Загрузка и предобработка данных CIFAR-100
    
    Args:
        data_dir: Путь к папке с датасетом
        classes: Список классов для выбора
    
    Returns:
        train_X, train_y, test_X, test_y
    """
    print("=" * 70)
    print("ЗАГРУЗКА ДАННЫХ CIFAR-100")
    print("=" * 70)
    
    with open(os.path.join(data_dir, 'train'), 'rb') as f:
        data_train = pickle.load(f, encoding='latin1')
    with open(os.path.join(data_dir, 'test'), 'rb') as f:
        data_test = pickle.load(f, encoding='latin1')
    
    # Предобработка тренировочных данных
    train_X = data_train['data'].reshape(-1, 3, 32, 32)
    train_X = np.transpose(train_X, [0, 2, 3, 1])  # NCHW -> NHWC
    train_y = np.array(data_train['fine_labels'])
    mask = np.isin(train_y, classes)
    train_X = train_X[mask].copy()
    train_y = train_y[mask].copy()
    
    # Предобработка тестовых данных
    test_X = data_test['data'].reshape(-1, 3, 32, 32)
    test_X = np.transpose(test_X, [0, 2, 3, 1])
    test_y = np.array(data_test['fine_labels'])
    mask = np.isin(test_y, classes)
    test_X = test_X[mask].copy()
    test_y = test_y[mask].copy()
    
    # Перекодировка меток в 0, 1, 2
    sorted_classes = sorted(classes)
    mapping = {cls: i for i, cls in enumerate(sorted_classes)}
    train_y = np.array([mapping[cls] for cls in train_y])
    test_y = np.array([mapping[cls] for cls in test_y])
    
    del data_train, data_test
    
    print(f"Классы: {classes}")
    print(f"Тренировочная выборка: {len(train_X)} изображений")
    print(f"Тестовая выборка: {len(test_X)} изображений")
    print()
    
    return train_X, train_y, test_X, test_y


def visualize_data(train_X, train_y, class_names, output_path=None):
    """
    Визуализация примеров изображений по классам
    
    Args:
        train_X: Изображения
        train_y: Метки
        class_names: Названия классов
        output_path: Путь для сохранения (опционально)
    """
    print("=" * 70)
    print("ВИЗУАЛИЗАЦИЯ ДАННЫХ")
    print("=" * 70)
    
    fig, axes = plt.subplots(3, 5, figsize=(12, 6))
    
    for i, class_name in enumerate(class_names):
        indices = np.where(train_y == i)[0]
        selected_indices = np.random.choice(indices, 5, replace=False)
        
        for j, img_idx in enumerate(selected_indices):
            axes[i, j].imshow(train_X[img_idx])
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(class_name, fontsize=10)
    
    plt.suptitle('Примеры изображений из тренировочной выборки', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Визуализация сохранена в {output_path}")
    
    plt.show()


def get_data_dir():
    """
    Получение пути к датасету CIFAR-100.
    Если датасет не найден - автоматически скачивается.
    """
    import urllib.request
    import tarfile
    
    # Пробуем несколько возможных путей
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    possible_paths = [
        os.path.join(current_dir, 'cifar-100-python'),  # ./cifar-100-python
        os.path.join(os.path.dirname(current_dir), 'cifar-100-python'),  # ../cifar-100-python
    ]
    
    # Проверяем существующие пути
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'train')):
            print(f"✅ Датасет найден: {path}")
            return path
    
    # Если не нашли - скачиваем
    print("⏳ Датасет CIFAR-100 не найден. Начинаю загрузку...")
    
    download_dir = os.path.join(current_dir, 'cifar-100-python')
    tar_gz_path = os.path.join(current_dir, 'cifar-100-python.tar.gz')
    
    # URL для загрузки
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    
    try:
        # Скачиваем
        print(f"📥 Загрузка из {url}...")
        urllib.request.urlretrieve(url, tar_gz_path)
        
        # Распаковываем
        print("📦 Распаковка...")
        with tarfile.open(tar_gz_path, 'r:gz') as tar:
            tar.extractall(path=current_dir)
        
        # Удаляем архив
        if os.path.exists(tar_gz_path):
            os.remove(tar_gz_path)
        
        print(f"✅ Датасет загружен и распакован: {download_dir}")
        return download_dir
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке: {e}")
        print("\nПопробуйте скачать вручную:")
        print("  wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz")
        print("  tar -xvzf cifar-100-python.tar.gz")
        raise

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Функции для оценки и визуализации результатов
"""

import os
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, test_X, test_y, class_names_ru, device='cuda'):
    """
    Оценка модели на тестовой выборке
    
    Args:
        model: Нейронная сеть
        test_X, test_y: Тестовые данные
        class_names_ru: Названия классов на русском
        device: Устройство
    
    Returns:
        report, cm_normalized
    """
    print("\n" + "=" * 70)
    print("ОЦЕНКА МОДЕЛИ НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 70)
    
    model.eval()
    
    batch_size = 128
    y_pred = []
    
    with torch.no_grad():
        for i in range(0, len(test_X), batch_size):
            batch_x = torch.Tensor(test_X[i:i+batch_size]).to(device)
            outputs = model(batch_x)
            predicted = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(predicted)
    
    y_true = test_y.copy()
    y_pred = np.array(y_pred)
    
    # Метрики
    print("\n=== МЕТРИКИ ПО КЛАССАМ (Test) ===\n")
    print(f"{'Класс':<15} {'Precision':<12} {'Recall':<12} {'F1-score':<12}")
    print("-" * 51)
    
    report = classification_report(y_true, y_pred, 
                                   target_names=class_names_ru,
                                   digits=4, output_dict=True, zero_division=0)
    
    for class_name in class_names_ru:
        p = report[class_name]['precision']
        r = report[class_name]['recall']
        f1 = report[class_name]['f1-score']
        print(f"{class_name:<15} {p:<12.4f} {r:<12.4f} {f1:<12.4f}")
    
    print("-" * 51)
    print(f"{'Accuracy':<15} {report['accuracy']:<12.4f}\n")
    
    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    print("=== МАТРИЦА ОШИБОК (в %) ===\n")
    header = f"{'':<12}" + "".join([f"{name:<10}" for name in class_names_ru])
    print(header)
    for i, class_name in enumerate(class_names_ru):
        row = f"{class_name:<12}" + "".join([f"{cm_normalized[i][j]:<10.1f}" for j in range(len(class_names_ru))])
        print(row)
    print()
    
    return report, cm_normalized


def save_confusion_matrix(cm_normalized, class_names_ru, output_path, show_plot=True):
    """
    Сохранение матрицы ошибок

    Args:
        cm_normalized: Нормализованная матрица ошибок
        class_names_ru: Названия классов
        output_path: Путь для сохранения
        show_plot: Показывать график (False = только сохранение)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names_ru, yticklabels=class_names_ru)
    plt.title('Матрица ошибок (в %)')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Матрица ошибок сохранена в {output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def export_to_onnx(model, model_name, output_dir):
    """
    Экспорт модели в ONNX
    
    Args:
        model: Нейронная сеть
        model_name: Имя модели
        output_dir: Папка для сохранения
    
    Returns:
        onnx_filename: Путь к файлу
    """
    import torch.onnx
    import onnx
    
    print("=" * 70)
    print(" ЭКСПОРТ МОДЕЛИ В ONNX ")
    print("=" * 70)
    
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 32, 32, 3).to(device)
    
    os.makedirs(output_dir, exist_ok=True)
    onnx_filename = os.path.join(output_dir, f'cnn_{model_name}.onnx')
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input_image'],
        output_names=['class_scores'],
        dynamic_axes={'input_image': {0: 'batch_size'}, 'class_scores': {0: 'batch_size'}}
    )
    
    print(f"✅ Модель экспортирована в {onnx_filename}")
    
    # Проверка
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)
    print("Проверка ONNX модели: ✅ УСПЕШНО")
    
    return onnx_filename

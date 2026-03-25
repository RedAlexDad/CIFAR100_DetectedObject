#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Сравнение результатов всех лабораторных работ (ЛР1-4)
Создание итоговой таблицы для отчёта

Использование:
    python compare_labs.py
    
    # С выводом в файл
    python compare_labs.py --output outputs/lab_comparison.md
    
    # С конкретными путями к результатам
    python compare_labs.py --lab1-dir runs/lab1 --lab2-dir runs/lab2 --lab3-dir runs/lab3 --lab4-dir runs/lab4
"""

import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd


# ============================================================================
# КОНФИГУРАЦИЯ ПО УМОЛЧАНИЮ
# ============================================================================

DEFAULT_DIRS = {
    'lab1': 'outputs',  # ЛР1 - Метод k ближайших соседей
    'lab2': 'runs',     # ЛР2 - CNN с нуля
    'lab3': 'runs',     # ЛР3 - CNN с аугментацией
    'lab4': 'runs/lab4', # ЛР4 - Transfer Learning
}

CLASS_NAMES_RU = ['Велосипед', 'Камбала', 'Поезд']


def find_latest_experiment(dir_path: str, pattern: str = None) -> Optional[str]:
    """
    Поиск последнего эксперимента в директории
    
    Args:
        dir_path: Путь к директории
        pattern: Шаблон для поиска (опционально)
    
    Returns:
        Путь к последнему эксперименту или None
    """
    if not os.path.exists(dir_path):
        return None
    
    # Получаем все поддиректории
    dirs = [d for d in os.listdir(dir_path) 
            if os.path.isdir(os.path.join(dir_path, d))]
    
    if not dirs:
        return None
    
    # Сортируем по времени модификации (последние первыми)
    dirs.sort(key=lambda d: os.path.getmtime(os.path.join(dir_path, d)), reverse=True)
    
    return os.path.join(dir_path, dirs[0])


def load_metrics_from_json(file_path: str) -> Optional[Dict]:
    """Загрузка метрик из JSON файла"""
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_metrics_from_csv(file_path: str) -> Optional[pd.DataFrame]:
    """Загрузка метрик из CSV файла"""
    if not os.path.exists(file_path):
        return None
    
    return pd.read_csv(file_path)


def extract_lab2_metrics(dir_path: str) -> Optional[Dict]:
    """
    Извлечение метрик для ЛР2 (CNN с нуля)
    
    Ищет файлы: metrics.json, history.json, learning_history.png
    """
    metrics = {
        'name': 'ЛР2: CNN с нуля',
        'model': None,
        'best_accuracy': None,
        'final_train_loss': None,
        'final_test_loss': None,
        'final_train_acc': None,
        'final_test_acc': None,
        'training_time': None,
        'epochs': None,
        'hyperparameters': {}
    }
    
    # Пытаемся загрузить metrics.json
    metrics_file = os.path.join(dir_path, 'metrics.json')
    data = load_metrics_from_json(metrics_file)
    
    if data:
        metrics['best_accuracy'] = data.get('best_accuracy')
        metrics['training_time'] = data.get('training_time')
        metrics['hyperparameters'] = data.get('hyperparameters', {})
        metrics['epochs'] = data.get('hyperparameters', {}).get('epochs')
        metrics['model'] = data.get('model')
        
        # История обучения
        history = data.get('history', {})
        if history:
            metrics['final_train_acc'] = history.get('train_acc', [None])[-1]
            metrics['final_test_acc'] = history.get('test_acc', [None])[-1]
            metrics['final_train_loss'] = history.get('train_loss', [None])[-1]
            metrics['final_test_loss'] = history.get('test_loss', [None])[-1]
    
    return metrics


def extract_lab3_metrics(dir_path: str) -> Optional[Dict]:
    """
    Извлечение метрик для ЛР3 (CNN с аугментацией)
    
    Ищет файлы: metrics.json, history.png
    """
    metrics = {
        'name': 'ЛР3: CNN + Аугментация',
        'model': None,
        'best_accuracy': None,
        'final_train_loss': None,
        'final_test_loss': None,
        'final_train_acc': None,
        'final_test_acc': None,
        'training_time': None,
        'epochs': None,
        'hyperparameters': {},
        'augmentation': {}
    }
    
    # Пытаемся загрузить metrics.json
    metrics_file = os.path.join(dir_path, 'metrics.json')
    data = load_metrics_from_json(metrics_file)
    
    if data:
        metrics['best_accuracy'] = data.get('best_accuracy')
        metrics['training_time'] = data.get('training_time')
        metrics['hyperparameters'] = data.get('hyperparams', data.get('hyperparameters', {}))
        metrics['epochs'] = metrics['hyperparameters'].get('epochs')
        metrics['model'] = data.get('model')
        metrics['augmentation'] = data.get('augmentation', {})
        
        # История обучения
        history = data.get('history', {})
        if history:
            metrics['final_train_acc'] = history.get('train_acc', [None])[-1]
            metrics['final_test_acc'] = history.get('test_acc', [None])[-1]
            metrics['final_train_loss'] = history.get('train_loss', [None])[-1]
            metrics['final_test_loss'] = history.get('test_loss', [None])[-1]
    
    return metrics


def extract_lab4_metrics(dir_path: str) -> Optional[Dict]:
    """
    Извлечение метрик для ЛР4 (Transfer Learning)
    
    Ищет файлы: history.json, comparison_results.json
    """
    metrics = {
        'name': 'ЛР4: Transfer Learning',
        'model': None,
        'best_accuracy': None,
        'frozen_accuracy': None,
        'finetuned_accuracy': None,
        'final_train_loss': None,
        'final_test_loss': None,
        'final_train_acc': None,
        'final_test_acc': None,
        'training_time': None,
        'epochs': None,
        'hyperparameters': {},
        'comparison': {}
    }
    
    # Пытаемся загрузить history.json
    history_file = os.path.join(dir_path, 'history.json')
    data = load_metrics_from_json(history_file)
    
    if data:
        metrics['best_accuracy'] = data.get('best_accuracy')
        metrics['training_time'] = data.get('training_time')
        metrics['hyperparameters'] = data.get('hyperparameters', {})
        metrics['epochs'] = metrics['hyperparameters'].get('epochs')
        metrics['model'] = data.get('model')
        
        # Точности по фазам
        metrics['frozen_accuracy'] = data.get('phase_accuracies', {}).get('frozen')
        metrics['finetuned_accuracy'] = data.get('phase_accuracies', {}).get('fine_tuned')
        
        # История обучения
        history = data.get('history', {})
        if history:
            metrics['final_train_acc'] = history.get('train_acc', [None])[-1]
            metrics['final_test_acc'] = history.get('test_acc', [None])[-1]
            metrics['final_train_loss'] = history.get('train_loss', [None])[-1]
            metrics['final_test_loss'] = history.get('test_loss', [None])[-1]
    
    # Пытаемся загрузить comparison_results.json
    comparison_file = os.path.join(dir_path, 'comparison_results.json')
    comparison_data = load_metrics_from_json(comparison_file)
    
    if comparison_data:
        metrics['comparison'] = comparison_data
        metrics['frozen_accuracy'] = comparison_data.get('frozen', {}).get('accuracy')
        metrics['finetuned_accuracy'] = comparison_data.get('fine_tuning', {}).get('accuracy')
    
    return metrics


def extract_lab1_metrics(manual_input: Dict = None) -> Optional[Dict]:
    """
    Извлечение метрик для ЛР1 (k-NN)
    
    ЛР1 обычно не имеет автоматических логов, поэтому используем ручной ввод
    """
    metrics = {
        'name': 'ЛР1: k-NN',
        'model': 'k-Nearest Neighbors',
        'best_accuracy': None,
        'k_value': None,
        'distance_metric': None,
        'training_time': 0,  # k-NN не требует обучения
        'hyperparameters': {}
    }
    
    if manual_input:
        metrics['best_accuracy'] = manual_input.get('accuracy')
        metrics['k_value'] = manual_input.get('k')
        metrics['distance_metric'] = manual_input.get('metric')
        metrics['hyperparameters'] = {
            'k': manual_input.get('k'),
            'metric': manual_input.get('metric')
        }
    
    return metrics


def create_comparison_table(all_metrics: Dict[str, Dict]) -> pd.DataFrame:
    """
    Создание сравнительной таблицы
    
    Args:
        all_metrics: Словарь с метриками по лабораторным работам
    
    Returns:
        DataFrame с сравнительной таблицей
    """
    rows = []
    
    for lab_name, metrics in all_metrics.items():
        if not metrics:
            continue
        
        row = {
            'Лабораторная работа': metrics.get('name', lab_name),
            'Модель': metrics.get('model', 'N/A'),
            'Лучшая точность (%)': f"{metrics.get('best_accuracy', 0):.2f}" if metrics.get('best_accuracy') else 'N/A',
            'Train Loss': f"{metrics.get('final_train_loss', 0):.4f}" if metrics.get('final_train_loss') else 'N/A',
            'Test Loss': f"{metrics.get('final_test_loss', 0):.4f}" if metrics.get('final_test_loss') else 'N/A',
            'Train Acc (%)': f"{metrics.get('final_train_acc', 0):.2f}" if metrics.get('final_train_acc') else 'N/A',
            'Test Acc (%)': f"{metrics.get('final_test_acc', 0):.2f}" if metrics.get('final_test_acc') else 'N/A',
            'Время обучения (сек)': f"{metrics.get('training_time', 0):.1f}" if metrics.get('training_time') else 'N/A',
            'Эпохи': metrics.get('epochs', 'N/A'),
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def create_detailed_comparison(all_metrics: Dict[str, Dict]) -> str:
    """
    Создание подробного отчёта в Markdown формате
    
    Args:
        all_metrics: Словарь с метриками по лабораторным работам
    
    Returns:
        Markdown строка с отчётом
    """
    lines = []
    lines.append("# Сравнение результатов лабораторных работ\n")
    lines.append(f"**Дата генерации:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("**Классы CIFAR-100:** Велосипед (8), Камбала (32), Поезд (90)\n")
    lines.append("")
    
    # ========================================================================
    # Сводная таблица
    # ========================================================================
    lines.append("## 📊 Сводная таблица результатов\n")
    
    df = create_comparison_table(all_metrics)
    lines.append(df.to_markdown(index=False))
    lines.append("")
    
    # ========================================================================
    # Подробно по каждой ЛР
    # ========================================================================
    lines.append("## 📋 Подробные результаты\n")
    
    for lab_name, metrics in all_metrics.items():
        if not metrics:
            continue
        
        lines.append(f"### {metrics.get('name', lab_name)}\n")
        
        # Основная информация
        lines.append(f"**Модель:** {metrics.get('model', 'N/A')}\n")
        lines.append(f"**Лучшая точность:** {metrics.get('best_accuracy', 0):.2f}%\n" if metrics.get('best_accuracy') else "**Лучшая точность:** N/A\n")
        
        # Гиперпараметры
        hp = metrics.get('hyperparameters', {})
        if hp:
            lines.append("\n**Гиперпараметры:**\n")
            for key, value in hp.items():
                if not isinstance(value, (dict, list)):
                    lines.append(f"- {key}: {value}")
            lines.append("")
        
        # Для ЛР4 - сравнение frozen vs fine-tuning
        if lab_name == 'lab4':
            frozen_acc = metrics.get('frozen_accuracy')
            ft_acc = metrics.get('finetuned_accuracy')
            
            if frozen_acc and ft_acc:
                lines.append("\n**Сравнение Frozen vs Fine-tuning:**\n")
                lines.append(f"- Замороженная модель: {frozen_acc:.2f}%")
                lines.append(f"- Fine-tuning: {ft_acc:.2f}%")
                lines.append(f"- Улучшение: {ft_acc - frozen_acc:+.2f}%")
                lines.append("")
        
        # Для ЛР3 - аугментация
        if lab_name == 'lab3':
            aug = metrics.get('augmentation', {})
            if aug:
                lines.append("\n**Аугментация:**\n")
                for key, value in aug.items():
                    if value:
                        lines.append(f"- {key}: {value}")
                lines.append("")
        
        lines.append("---\n")
    
    # ========================================================================
    # Анализ тенденций
    # ========================================================================
    lines.append("## 📈 Анализ тенденций\n")
    
    # Извлечение точностей
    accuracies = {}
    for lab_name, metrics in all_metrics.items():
        if metrics and metrics.get('best_accuracy'):
            accuracies[lab_name] = metrics['best_accuracy']
    
    if len(accuracies) >= 2:
        lines.append("\n### Динамика точности\n")
        
        lab_order = ['lab1', 'lab2', 'lab3', 'lab4']
        prev_acc = None
        
        for lab in lab_order:
            if lab in accuracies:
                acc = accuracies[lab]
                if prev_acc is not None:
                    diff = acc - prev_acc
                    trend = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                    lines.append(f"- {lab.upper()}: {acc:.2f}% ({trend} {diff:+.2f}%)")
                else:
                    lines.append(f"- {lab.upper()}: {acc:.2f}%")
                prev_acc = acc
        
        lines.append("")
        
        # Лучшая модель
        if accuracies:
            best_lab = max(accuracies, key=accuracies.get)
            best_acc = accuracies[best_lab]
            lines.append(f"\n**🏆 Лучшая модель:** {best_lab.upper()} ({best_acc:.2f}%)\n")
    
    # ========================================================================
    # Выводы
    # ========================================================================
    lines.append("\n## 📝 Выводы\n")
    lines.append("\n1. **Метод k-ближайших соседей (ЛР1):** Базовый подход без обучения модели.\n")
    lines.append("2. **CNN с нуля (ЛР2):** Собственная архитектура нейронной сети.\n")
    lines.append("3. **CNN + Аугментация (ЛР3):** Улучшение результатов за счёт аугментации данных.\n")
    lines.append("4. **Transfer Learning (ЛР4):** Использование предобученных моделей для повышения точности.\n")
    lines.append("\n### Ключевые наблюдения:\n")
    lines.append("- Transfer Learning обычно даёт наилучшие результаты на малых данных\n")
    lines.append("- Аугментация данных помогает улучшить обобщающую способность модели\n")
    lines.append("- Fine-tuning предобученных моделей эффективнее обучения с нуля\n")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Сравнение результатов ЛР1-4')
    parser.add_argument('--lab1-dir', type=str, default=None,
                        help='Путь к результатам ЛР1')
    parser.add_argument('--lab2-dir', type=str, default=None,
                        help='Путь к результатам ЛР2')
    parser.add_argument('--lab3-dir', type=str, default=None,
                        help='Путь к результатам ЛР3')
    parser.add_argument('--lab4-dir', type=str, default=None,
                        help='Путь к результатам ЛР4')
    parser.add_argument('--output', type=str, default=None,
                        help='Путь для сохранения отчёта (Markdown)')
    parser.add_argument('--lab1-accuracy', type=float, default=None,
                        help='Точность ЛР1 (для ручного ввода)')
    parser.add_argument('--lab1-k', type=int, default=5,
                        help='Значение k для ЛР1')
    parser.add_argument('--lab1-metric', type=str, default='euclidean',
                        choices=['euclidean', 'manhattan', 'cosine'],
                        help='Метрика расстояния для ЛР1')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(" СРАВНЕНИЕ РЕЗУЛЬТАТОВ ЛАБОРАТОРНЫХ РАБОТ №1-4 ")
    print("=" * 70)
    print()
    
    # Сбор метрик по каждой ЛР
    all_metrics = {}
    
    # ЛР1: k-NN (ручной ввод)
    lab1_input = {
        'accuracy': args.lab1_accuracy,
        'k': args.lab1_k,
        'metric': args.lab1_metric
    }
    all_metrics['lab1'] = extract_lab1_metrics(lab1_input)
    print(f"ЛР1: k-NN - точность = {args.lab1_accuracy}% (k={args.lab1_k})" if args.lab1_accuracy else "ЛР1: k-NN - нет данных")
    
    # ЛР2: CNN с нуля
    lab2_dir = args.lab2_dir or DEFAULT_DIRS['lab2']
    lab2_latest = find_latest_experiment(lab2_dir)
    all_metrics['lab2'] = extract_lab2_metrics(lab2_latest) if lab2_latest else None
    print(f"ЛР2: CNN с нуля - найдено в {lab2_latest}" if lab2_latest else "ЛР2: CNN с нуля - нет данных")
    
    # ЛР3: CNN + Аугментация
    lab3_dir = args.lab3_dir or DEFAULT_DIRS['lab3']
    lab3_latest = find_latest_experiment(lab3_dir)
    all_metrics['lab3'] = extract_lab3_metrics(lab3_latest) if lab3_latest else None
    print(f"ЛР3: CNN + Аугментация - найдено в {lab3_latest}" if lab3_latest else "ЛР3: CNN + Аугментация - нет данных")
    
    # ЛР4: Transfer Learning
    lab4_dir = args.lab4_dir or DEFAULT_DIRS['lab4']
    lab4_latest = find_latest_experiment(lab4_dir)
    all_metrics['lab4'] = extract_lab4_metrics(lab4_latest) if lab4_latest else None
    print(f"ЛР4: Transfer Learning - найдено в {lab4_latest}" if lab4_latest else "ЛР4: Transfer Learning - нет данных")
    
    print()
    
    # Создание отчёта
    report = create_detailed_comparison(all_metrics)
    
    # Вывод или сохранение
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ Отчёт сохранён в {args.output}")
    else:
        print(report)
    
    # Создание CSV таблицы
    df = create_comparison_table(all_metrics)
    csv_path = args.output.replace('.md', '.csv') if args.output else 'outputs/lab_comparison.csv'
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✅ CSV таблица сохранена в {csv_path}")
    
    print()
    print("=" * 70)
    print(" ЗАВЕРШЕНО ")
    print("=" * 70)


if __name__ == '__main__':
    main()

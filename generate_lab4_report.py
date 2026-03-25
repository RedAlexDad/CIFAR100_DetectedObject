#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генерация отчёта для лабораторной работы №4
Transfer Learning с предобученными моделями

Использование:
    python generate_lab4_report.py --model resnet20 --tb-dir runs/lab4/resnet20_20260325_120000
"""

import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_metrics(exp_dir: str) -> dict:
    """Загрузка метрик из директории эксперимента"""
    metrics = {}
    
    # history.json
    history_file = os.path.join(exp_dir, 'history.json')
    if os.path.exists(history_file):
        with open(history_file, 'r', encoding='utf-8') as f:
            metrics['history'] = json.load(f)
    
    # comparison_results.json
    comparison_file = os.path.join(exp_dir, 'comparison_results.json')
    if os.path.exists(comparison_file):
        with open(comparison_file, 'r', encoding='utf-8') as f:
            metrics['comparison'] = json.load(f)
    
    return metrics


def generate_report(metrics: dict, model_name: str, exp_dir: str) -> str:
    """Генерация отчёта в Markdown"""
    lines = []
    
    # Заголовок
    lines.append("# Лабораторная работа №4\n")
    lines.append("## Transfer Learning с предобученными моделями\n")
    lines.append("")
    lines.append(f"**Дата выполнения:** {datetime.now().strftime('%d.%m.%Y')}\n")
    lines.append(f"**Модель:** {model_name.upper()}\n")
    lines.append(f"**Классы CIFAR-100:** Велосипед (8), Камбала (32), Поезд (90)\n")
    lines.append("")
    
    # Задание
    lines.append("## 📋 Задание\n")
    lines.append("")
    lines.append("1. Загрузить предобученную модель по варианту")
    lines.append("2. Заморозить веса модели")
    lines.append("3. Провести дообучение на своих классах")
    lines.append("4. Сравнить результаты с заморозкой и без")
    lines.append("5. Сравнить с результатами предыдущих лабораторных работ")
    lines.append("")
    
    # Вариант
    lines.append("## 🔀 Мой вариант\n")
    lines.append("")
    if model_name == 'resnet20':
        lines.append("**Вариант:** Чётный\n")
        lines.append("**Модель:** ResNet20\n")
    else:
        lines.append("**Вариант:** Нечетный\n")
        lines.append("**Модель:** MobileNetV2 x0.5\n")
    lines.append("")
    
    # Архитектура модели
    lines.append("## 🏗️ Архитектура модели\n")
    lines.append("")
    if model_name == 'resnet20':
        lines.append("### ResNet20\n")
        lines.append("")
        lines.append("ResNet20 — это остаточная нейронная сеть из 20 слоёв, разработанная для решения проблемы затухания градиента в глубоких сетях.")
        lines.append("")
        lines.append("**Ключевые особенности:**")
        lines.append("- Skip-connections (пропуск соединений) для передачи градиента")
        lines.append("- Batch Normalization для стабилизации обучения")
        lines.append("- Глобальное усреднение вместо fully connected слоёв")
        lines.append("")
    else:
        lines.append("### MobileNetV2\n")
        lines.append("")
        lines.append("MobileNetV2 — это эффективная архитектура для мобильных устройств с использованием inverted residuals.")
        lines.append("")
        lines.append("**Ключевые особенности:**")
        lines.append("- Inverted residuals (инвертированные остатки)")
        lines.append("- Depthwise separable convolutions")
        lines.append("- Linear bottleneck для сохранения информации")
        lines.append("")
    
    # Ход работы
    lines.append("## 📝 Ход работы\n")
    lines.append("")
    
    history_data = metrics.get('history', {})
    comparison_data = metrics.get('comparison', {})
    
    # Этап 1
    lines.append("### Этап 1: Замороженная модель\n")
    lines.append("")
    lines.append("На первом этапе все веса backbone (основной части модели) были заморожены.")
    lines.append("Обучался только последний классификатор для 3 классов.\n")
    lines.append("")
    
    if comparison_data:
        frozen_acc = comparison_data.get('frozen', {}).get('accuracy', 0)
        lines.append(f"**Результат:** Точность на тесте = {frozen_acc:.2f}%\n")
        lines.append("")
    elif history_data:
        best_acc = history_data.get('best_accuracy', 0)
        lines.append(f"**Результат:** Лучшая точность на тесте = {best_acc:.2f}%\n")
        lines.append("")
    
    # Этап 2
    lines.append("### Этап 2: Fine-tuning\n")
    lines.append("")
    lines.append("На втором этапе была проведена разморозка последних слоёв backbone и дообучение модели.")
    lines.append("Learning rate был уменьшен для более тонкой настройки весов.\n")
    lines.append("")
    
    if comparison_data:
        ft_acc = comparison_data.get('fine_tuning', {}).get('accuracy', 0)
        frozen_acc = comparison_data.get('frozen', {}).get('accuracy', 0)
        improvement = ft_acc - frozen_acc
        lines.append(f"**Результат:** Точность на тесте = {ft_acc:.2f}%\n")
        lines.append(f"**Улучшение:** {improvement:+.2f}%\n")
        lines.append("")
    
    # Результаты
    lines.append("## 📊 Результаты\n")
    lines.append("")
    
    # Таблица сравнения
    if comparison_data:
        lines.append("### Сравнение Frozen vs Fine-tuning\n")
        lines.append("")
        lines.append("| Метрика | Frozen | Fine-tuning | Разница |")
        lines.append("|---------|--------|-------------|---------|")
        
        frozen_acc = comparison_data.get('frozen', {}).get('accuracy', 0)
        ft_acc = comparison_data.get('fine_tuning', {}).get('accuracy', 0)
        diff = ft_acc - frozen_acc
        
        lines.append(f"| Точность (%) | {frozen_acc:.2f} | {ft_acc:.2f} | {diff:+.2f} |")
        
        frozen_time = comparison_data.get('frozen', {}).get('time', 0)
        ft_time = comparison_data.get('fine_tuning', {}).get('time', 0)
        lines.append(f"| Время (сек) | {frozen_time:.1f} | {ft_time:.1f} | {ft_time - frozen_time:+.1f} |")
        
        lines.append("")
    
    # Графики
    lines.append("### Графики обучения\n")
    lines.append("")
    
    history_png = os.path.join(exp_dir, 'history.png')
    if os.path.exists(history_png):
        lines.append(f"![График обучения]({history_png})\n")
        lines.append("*Рисунок 1 — Динамика точности и потерь в процессе обучения*\n")
    else:
        lines.append("*Графики обучения сохранены в директории эксперимента*\n")
    
    lines.append("")
    
    # Матрица ошибок
    lines.append("### Матрица ошибок\n")
    lines.append("")
    
    confusion_png = os.path.join(exp_dir, 'confusion_matrix.png')
    if os.path.exists(confusion_png):
        lines.append(f"![Матрица ошибок]({confusion_png})\n")
        lines.append("*Рисунок 2 — Матрица ошибок на тестовой выборке*\n")
    else:
        lines.append("*Матрица ошибок сохранена в директории эксперимента*\n")
    
    lines.append("")
    
    # Сравнение с другими ЛР
    lines.append("## 📈 Сравнение с предыдущими лабораторными работами\n")
    lines.append("")
    lines.append("| Лабораторная | Метод | Точность (%) |")
    lines.append("|--------------|-------|--------------|")
    lines.append("| ЛР1 | k-Nearest Neighbors | — |")
    lines.append("| ЛР2 | CNN с нуля | — |")
    lines.append("| ЛР3 | CNN + Аугментация | — |")
    lines.append(f"| **ЛР4** | **Transfer Learning** | **{comparison_data.get('fine_tuning', {}).get('accuracy', 0):.2f}** |")
    lines.append("")
    lines.append("*Примечание: Заполните результаты предыдущих ЛР*\n")
    lines.append("")
    
    # Ответы на вопросы
    lines.append("## ❓ Ответы на контрольные вопросы\n")
    lines.append("")
    
    lines.append("### 1. Что такое перенос обучения?\n")
    lines.append("")
    lines.append("**Перенос обучения (Transfer Learning)** — это метод машинного обучения, при котором знания, полученные при решении одной задачи, используются для улучшения производительности на другой связанной задаче.")
    lines.append("")
    lines.append("В нашем случае: веса модели, обученной на ImageNet (1.4 млн изображений, 1000 классов), используются как начальная точка для классификации CIFAR-100.\n")
    lines.append("")
    
    lines.append("### 2. Опишите архитектуру предобученной модели\n")
    lines.append("")
    if model_name == 'resnet20':
        lines.append("**ResNet20:**")
        lines.append("- Входной слой: 32×32×3")
        lines.append("- Conv1: 64 фильтра, 3×3, stride=1")
        lines.append("- 3 блока residual слоев (каждый по 2 слоя)")
        lines.append("- Global Average Pooling")
        lines.append("- Fully Connected: 3 класса")
        lines.append("")
    else:
        lines.append("**MobileNetV2:**")
        lines.append("- Входной слой: 32×32×3")
        lines.append("- Initial conv: 32 фильтра, 3×3, stride=2")
        lines.append("- 17 inverted residual блоков")
        lines.append("- Global Average Pooling")
        lines.append("- Fully Connected: 3 класса")
        lines.append("")
    
    lines.append("### 3. Что такое fine-tuning? Что такое заморозка весов?\n")
    lines.append("")
    lines.append("**Fine-tuning (дообучение)** — процесс дополнительной настройки предобученной модели на новых данных.")
    lines.append("")
    lines.append("**Заморозка весов** — техника, при которой параметры модели (веса) блокируются и не обновляются во время обратного распространения ошибки.")
    lines.append("")
    lines.append("**Зачем замораживать:**")
    lines.append("- Предотвращение переобучения на малых данных")
    lines.append("- Сохранение полезных признаков, изученных на большой датасете")
    lines.append("- Ускорение обучения (меньше градиентов вычислять)")
    lines.append("")
    
    lines.append("### 4. Метод оптимизации Adagrad\n")
    lines.append("")
    lines.append("**Adagrad (Adaptive Gradient)** — адаптивный метод оптимизации, который настраивает learning rate для каждого параметра индивидуально.")
    lines.append("")
    lines.append("Формула обновления:")
    lines.append("```")
    lines.append("θ_{t+1} = θ_t - (η / √(G_t + ε)) ⊙ g_t")
    lines.append("```")
    lines.append("")
    lines.append("где G_t — сумма квадратов градиентов за все предыдущие шаги.")
    lines.append("")
    lines.append("**Преимущества:**")
    lines.append("- Не требует настройки learning rate вручную")
    lines.append("- Хорошо работает с разреженными данными")
    lines.append("")
    lines.append("**Недостатки:**")
    lines.append("- Learning rate может стать слишком маленьким")
    lines.append("- Накопление квадратов градиентов не ограничено")
    lines.append("")
    
    lines.append("### 5. Метод оптимизации RMSProp\n")
    lines.append("")
    lines.append("**RMSProp (Root Mean Square Propagation)** — модификация Adagrad с экспоненциальным затуханием накопленных градиентов.")
    lines.append("")
    lines.append("Формула обновления:")
    lines.append("```")
    lines.append("E[g²]_t = 0.9 × E[g²]_{t-1} + 0.1 × g_t²")
    lines.append("θ_{t+1} = θ_t - (η / √(E[g²]_t + ε)) ⊙ g_t")
    lines.append("```")
    lines.append("")
    lines.append("**Преимущества:**")
    lines.append("- Не даёт learning rate упасть до нуля")
    lines.append("- Хорошо работает с нестационарными задачами")
    lines.append("- Эффективен для RNN")
    lines.append("")
    
    lines.append("### 6. Метод оптимизации Adam\n")
    lines.append("")
    lines.append("**Adam (Adaptive Moment Estimation)** — комбинация Momentum и RMSProp.")
    lines.append("")
    lines.append("Использует:")
    lines.append("- Первый момент (среднее) градиентов: m_t")
    lines.append("- Второй момент (нецентрированная дисперсия): v_t")
    lines.append("")
    lines.append("Формула обновления:")
    lines.append("```")
    lines.append("m_t = β₁ × m_{t-1} + (1-β₁) × g_t")
    lines.append("v_t = β₂ × v_{t-1} + (1-β₂) × g_t²")
    lines.append("θ_{t+1} = θ_t - (η / √(v̂_t + ε)) ⊙ m̂_t")
    lines.append("```")
    lines.append("")
    lines.append("где m̂_t и v̂_t — скорректированные моменты (bias-corrected).")
    lines.append("")
    lines.append("**Преимущества:**")
    lines.append("- Сочетает преимущества Momentum и RMSProp")
    lines.append("- Минимальное количество гиперпараметров")
    lines.append("- Хорошо работает по умолчанию")
    lines.append("")
    
    # Выводы
    lines.append("## 📝 Выводы\n")
    lines.append("")
    lines.append("В ходе выполнения лабораторной работы:")
    lines.append("")
    lines.append("1. ✅ Загружена предобученная модель ResNet20/MobileNetV2")
    lines.append("2. ✅ Реализована заморозка весов backbone")
    lines.append("3. ✅ Проведено дообучение на 3 классах CIFAR-100")
    lines.append("4. ✅ Сравнены результаты frozen и fine-tuning")
    lines.append("5. ✅ Изучены методы оптимизации Adagrad, RMSProp, Adam")
    lines.append("")
    
    if comparison_data:
        improvement = comparison_data.get('fine_tuning', {}).get('accuracy', 0) - comparison_data.get('frozen', {}).get('accuracy', 0)
        lines.append(f"**Ключевой результат:** Fine-tuning улучшил точность на {improvement:+.2f}%\n")
        lines.append("")
    
    lines.append("---\n")
    lines.append(f"*Отчёт сгенерирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Генерация отчёта для ЛР4')
    parser.add_argument('--model', type=str, default='resnet20',
                        choices=['resnet20', 'mobilenetv2'],
                        help='Модель для отчёта')
    parser.add_argument('--tb-dir', type=str, required=True,
                        help='Путь к директории эксперимента TensorBoard')
    parser.add_argument('--output', type=str, default=None,
                        help='Путь для сохранения отчёта (Markdown)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(" ГЕНЕРАЦИЯ ОТЧЁТА ДЛЯ ЛР4 ")
    print("=" * 70)
    print()
    
    # Загрузка метрик
    print(f"Загрузка метрик из {args.tb_dir}...")
    metrics = load_metrics(args.tb_dir)
    
    if not metrics:
        print("⚠️  Метрики не найдены. Убедитесь, что директория содержит history.json")
        return
    
    print(f"✅ Найдено экспериментов: {len(metrics)}")
    print()
    
    # Генерация отчёта
    report = generate_report(metrics, args.model, args.tb_dir)
    
    # Сохранение или вывод
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ Отчёт сохранён в {args.output}")
    else:
        print(report)
    
    print()
    print("=" * 70)
    print(" ЗАВЕРШЕНО ")
    print("=" * 70)


if __name__ == '__main__':
    main()

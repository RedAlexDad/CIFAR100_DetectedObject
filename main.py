#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный скрипт для обучения модели CNN на CIFAR-100 с TensorBoard
Лабораторная работа №2/№3

Использование:
    # Базовое обучение
    python main.py --model medium --train
    
    # С TensorBoard логированием
    python main.py --model medium --train --tb-logs
    
    # Просмотр логов в Jupyter
    %load_ext tensorboard
    %tensorboard --logdir runs/exp1
    
    # Просмотр в терминале
    tensorboard --logdir runs/exp1
"""

import argparse
import sys
import os
from datetime import datetime

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Устанавливаем неинтерактивный режим для matplotlib
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from configs.config import (
    CLASSES, CLASS_NAMES, CLASS_NAMES_RU,
    MODEL_CONFIG, TRAIN_CONFIG, DEVICE,
    CHECKPOINT_DIR, ONNX_DIR, OUTPUT_DIR
)
from models import Cifar100_CNN_Base, Cifar100_CNN_Medium, Cifar100_CNN_Deep, Cifar100_CNN_Optimized
from scripts import (
    load_cifar100_data, get_data_dir,
    train_model_with_tensorboard, plot_learning_history,
    evaluate_model, save_confusion_matrix
)
from scripts.data_utils import visualize_data
from scripts.eval_utils import export_to_onnx


def get_model(model_name, device):
    """Получение модели по имени"""
    if model_name == 'base':
        model = Cifar100_CNN_Base(**MODEL_CONFIG['base'])
    elif model_name == 'medium':
        model = Cifar100_CNN_Medium(**MODEL_CONFIG['medium'])
    elif model_name == 'deep':
        model = Cifar100_CNN_Deep(**MODEL_CONFIG['deep'])
    elif model_name == 'optimized':
        model = Cifar100_CNN_Optimized(**MODEL_CONFIG['optimized'])
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")
    
    model = model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description='Обучение CNN для CIFAR-100 с TensorBoard')
    parser.add_argument('--model', type=str, default='medium', choices=['base', 'medium', 'deep', 'optimized'],
                        help='Модель для обучения (base, medium, deep или optimized)')
    parser.add_argument('--train', action='store_true', help='Обучить модель')
    parser.add_argument('--evaluate', action='store_true', help='Оценить модель')
    parser.add_argument('--export', action='store_true', help='Экспорт в ONNX')
    parser.add_argument('--checkpoint', type=str, help='Путь к чекпоинту для оценки')
    parser.add_argument('--visualize', action='store_true', help='Визуализировать данные')
    parser.add_argument('--save-only', action='store_true', 
                        help='Сохранять графики без показа окон')
    parser.add_argument('--tb-logs', action='store_true',
                        help='Включить логирование в TensorBoard')
    parser.add_argument('--tb-dir', type=str, default='runs/exp1',
                        help='Папка для логов TensorBoard (по умолчанию: runs/exp1)')
    
    # Гиперпараметры обучения
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (по умолчанию из config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Размер батча (по умолчанию из config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Количество эпох (по умолчанию из config)')

    args = parser.parse_args()

    if not any([args.train, args.evaluate, args.export, args.visualize]):
        parser.print_help()
        return

    print("=" * 70)
    print(" Сверточная нейронная сеть для классификации CIFAR-100 ")
    print(f" Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()

    # TensorBoard информация
    if args.tb_logs:
        print(f"📊 TensorBoard логи: {args.tb_dir}")
        print(f"🚀 Для просмотра:")
        print(f"   В Jupyter: %load_ext tensorboard")
        print(f"              %tensorboard --logdir {args.tb_dir}")
        print(f"   В терминале: tensorboard --logdir={args.tb_dir}")
        print()

    # Устройство
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Загрузка данных
    data_dir = get_data_dir()
    train_X, train_y, test_X, test_y = load_cifar100_data(data_dir, CLASSES)

    # Визуализация
    if args.visualize:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        visualize_data(train_X, train_y, CLASS_NAMES_RU, 
                      os.path.join(OUTPUT_DIR, 'data_visualization.png'))

    # Получение модели
    model = get_model(args.model, device)
    print(f"Модель: {args.model}")
    print(model)
    print()

    # TensorBoard Writer
    writer = None
    if args.tb_logs:
        os.makedirs(args.tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.tb_dir)
        print(f"✅ TensorBoard writer создан: {args.tb_dir}")
        
        # Логирование архитектуры модели
        writer.add_text('Model_Architecture', str(model), 0)
        
        # Логирование гиперпараметров (будет заполнено после обучения)
        hparams = {
            'model': args.model,
            'learning_rate': args.lr if args.lr else TRAIN_CONFIG[args.model]['lr'],
            'batch_size': args.batch_size if args.batch_size else TRAIN_CONFIG[args.model]['batch_size'],
        }
        writer.add_text('Hyperparameters', 
                       '\n'.join([f"{k}: {v}" for k, v in hparams.items()]), 0)
        print()

    # Обучение
    if args.train:
        # Получаем конфигурацию для модели
        config = TRAIN_CONFIG[args.model].copy()

        # Переопределяем гиперпараметры если указаны
        if args.lr is not None:
            config['lr'] = args.lr
        if args.batch_size is not None:
            config['batch_size'] = args.batch_size
        if args.epochs is not None:
            config['epochs'] = args.epochs

        # Обучение с TensorBoard
        model, history, time_sec, acc = train_model_with_tensorboard(
            model, train_X, train_y, test_X, test_y, config, device,
            writer=writer  # Передаем writer
        )

        # Сохранение чекпоинта
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'cnn_{args.model}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"\nМодель сохранена в {checkpoint_path}")

        # Сохранение графиков
        plot_learning_history(
            history, 
            os.path.join(OUTPUT_DIR, 'learning_history.png'),
            show_plot=not args.save_only
        )

    # Оценка
    if args.evaluate or args.train:
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

        report, cm = evaluate_model(model, test_X, test_y, CLASS_NAMES_RU, device)

        # Сохранение матрицы ошибок
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_confusion_matrix(
            cm, 
            CLASS_NAMES_RU,
            os.path.join(OUTPUT_DIR, 'confusion_matrix.png'),
            show_plot=not args.save_only
        )
        
        # Логирование финальных метрик в TensorBoard
        if writer:
            writer.add_scalar('Final/test_accuracy', report['accuracy'], 0)
            writer.add_scalar('Final/test_precision', report['macro avg']['precision'], 0)
            writer.add_scalar('Final/test_recall', report['macro avg']['recall'], 0)
            writer.add_scalar('Final/test_f1', report['macro avg']['f1-score'], 0)
            
            # Логирование по классам
            for class_name in CLASS_NAMES_RU:
                writer.add_scalar(f'Class/{class_name}_precision', report[class_name]['precision'], 0)
                writer.add_scalar(f'Class/{class_name}_recall', report[class_name]['recall'], 0)
                writer.add_scalar(f'Class/{class_name}_f1', report[class_name]['f1-score'], 0)

    # Экспорт
    if args.export:
        export_to_onnx(model, args.model, ONNX_DIR)

    # Закрыть TensorBoard writer
    if writer:
        writer.close()
        print(f"\n✅ TensorBoard логи сохранены в {args.tb_dir}")

    print("\n" + "=" * 70)
    print(" ЗАВЕРШЕНО ")
    print("=" * 70)


if __name__ == '__main__':
    main()

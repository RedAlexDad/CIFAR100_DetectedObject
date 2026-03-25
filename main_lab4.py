#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа №4
Transfer Learning с предобученными моделями (ResNet20 / MobileNetV2)

Использование:
    # Базовое обучение с замороженной моделью
    python main_lab4.py --model resnet20 --train

    # Обучение с fine-tuning (разморозка после 10 эпох)
    python main_lab4.py --model resnet20 --train --unfreeze-after 10 --epochs 30

    # Сравнение заморозки и fine-tuning
    python main_lab4.py --model mobilenetv2 --train --compare

    # Только оценка модели
    python main_lab4.py --model resnet20 --evaluate --checkpoint checkpoints/lab4_resnet20.pth

    # Обучение с TensorBoard
    python main_lab4.py --model resnet20 --train --tb-logs --tb-dir runs/lab4_resnet20
"""

import argparse
import sys
import os
from datetime import datetime
import json

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
    CHECKPOINT_DIR, ONNX_DIR, OUTPUT_DIR
)
from configs.training_config import get_config, print_config
from models import (
    get_transfer_model,
    print_model_summary,
    count_trainable_params,
    count_total_params,
)
from scripts import (
    load_cifar100_data, get_data_dir,
    train_model_transfer_learning,
    plot_learning_history,
    evaluate_model, save_confusion_matrix,
)
from scripts.data_utils import visualize_data
from scripts.augmentation import get_train_transform, get_test_transform, AugmentedDataset
from torch.utils.data import DataLoader


def create_augmented_dataloaders(train_X, train_y, test_X, test_y, batch_size=128, aug_config=None):
    """
    Создание DataLoader с аугментацией
    
    Args:
        train_X, train_y: Тренировочные данные
        test_X, test_y: Тестовые данные
        batch_size: Размер батча
        aug_config: Конфигурация аугментации
    
    Returns:
        train_loader, test_loader
    """
    # Трансформации
    if aug_config:
        train_transform = get_train_transform(aug_config)
    else:
        train_transform = get_test_transform()
    
    test_transform = get_test_transform()
    
    # Создание датасетов
    tensor_train_x = torch.Tensor(train_X)
    tensor_train_y = torch.Tensor(train_y).long()
    train_dataset = AugmentedDataset(tensor_train_x, tensor_train_y, transform=train_transform)
    
    tensor_test_x = torch.Tensor(test_X)
    tensor_test_y = torch.Tensor(test_y).long()
    test_dataset = AugmentedDataset(tensor_test_x, tensor_test_y, transform=test_transform)
    
    # Создание DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Обучение за одну эпоху"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    
    return train_loss, train_acc


def validate_epoch(model, test_loader, criterion, device):
    """Валидация за одну эпоху"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss_avg = val_loss / len(test_loader)
    val_acc = 100.0 * correct / total
    
    return val_loss_avg, val_acc


def train_model_simple(model, train_loader, test_loader, config, device, writer=None):
    """
    Упрощённое обучение с DataLoader
    
    Args:
        model: Модель
        train_loader, test_loader: DataLoader
        config: Конфигурация
        device: Устройство
        writer: TensorBoard writer
    
    Returns:
        model, history, training_time, best_accuracy
    """
    # Поддержка обоих форматов конфигурации
    lr = config.get('lr', config.get('learning_rate', 0.001))
    batch_size = config.get('batch_size', 64)
    epochs = config.get('epochs', 30)
    momentum = config.get('momentum', 0.9)
    weight_decay = config.get('weight_decay', 1e-4)
    label_smoothing = config.get('label_smoothing', 0.0)
    use_scheduler = config.get('use_scheduler', True)
    
    print(f"\n{'='*70}")
    print(" ОБУЧЕНИЕ МОДЕЛИ")
    print(f"{'='*70}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Device: {device}")
    print()
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_test_acc = 0.0
    best_model_state = None
    
    start_time = datetime.now()

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(val_loss)
        history['test_acc'].append(val_acc)

        if scheduler:
            scheduler.step()

        # TensorBoard
        if writer:
            writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            writer.add_scalar('Loss/val_epoch', val_loss, epoch)
            writer.add_scalar('Accuracy/train_epoch', train_acc, epoch)
            writer.add_scalar('Accuracy/val_epoch', val_acc, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

            if val_acc > best_test_acc:
                writer.add_scalar('Best_Accuracy', val_acc, epoch)

        # Сохранение лучшей модели
        if val_acc > best_test_acc:
            best_test_acc = val_acc
            best_model_state = model.state_dict().copy()

            if writer:
                writer.add_scalar('Best_Accuracy', best_test_acc, epoch)

        # Вывод прогресса
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Эпоха [{epoch+1}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {val_loss:.4f}, Test Acc: {val_acc:.2f}%")

    training_time = (datetime.now() - start_time).total_seconds()

    if best_model_state:
        model.load_state_dict(best_model_state)

    print(f"\nОбучение завершено за {training_time:.2f} сек")
    print(f"Лучшая точность на тесте: {best_test_acc:.2f}%")

    return model, history, training_time, best_test_acc


def main():
    parser = argparse.ArgumentParser(description='ЛР4: Transfer Learning для CIFAR-100')
    parser.add_argument('--model', type=str, default='resnet20',
                        choices=['resnet20', 'mobilenetv2'],
                        help='Модель для обучения')
    parser.add_argument('--train', action='store_true', help='Обучить модель')
    parser.add_argument('--evaluate', action='store_true', help='Оценить модель')
    parser.add_argument('--checkpoint', type=str, help='Путь к чекпоинту для оценки')
    parser.add_argument('--visualize', action='store_true', help='Визуализировать данные')
    parser.add_argument('--save-only', action='store_true',
                        help='Сохранять графики без показа окон')
    parser.add_argument('--tb-logs', action='store_true',
                        help='Включить логирование в TensorBoard')
    parser.add_argument('--tb-dir', type=str, default='runs/lab4',
                        help='Папка для логов TensorBoard')
    parser.add_argument('--compare', action='store_true',
                        help='Сравнить frozen и fine-tuning')
    
    # Гиперпараметры
    parser.add_argument('--epochs', type=int, default=30,
                        help='Количество эпох (по умолчанию 30)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (по умолчанию 0.001)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Размер батча (по умолчанию 64)')
    parser.add_argument('--unfreeze-after', type=int, default=None,
                        help='Эпоха для разморозки весов (для fine-tuning)')
    parser.add_argument('--unfreeze-layers', type=int, default=2,
                        help='Количество слоёв для разморозки')
    parser.add_argument('--fine-tuning-lr', type=float, default=None,
                        help='Learning rate для fine-tuning (по умолчанию lr/10)')
    
    # Пресет конфигурации
    parser.add_argument('--preset', default='base', choices=['base', 'fast', 'accurate'],
                        help='Пресет конфигурации')
    
    args = parser.parse_args()
    
    if not any([args.train, args.evaluate]):
        parser.print_help()
        return
    
    print("=" * 70)
    print(" Лабораторная работа №4: Transfer Learning ")
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
                      os.path.join(OUTPUT_DIR, 'lab4_data_visualization.png'))
    
    # Загрузка конфигурации
    config = get_config(args.preset)
    hp = config['hyperparams']
    aug = config['augmentation']
    
    # Переопределение гиперпараметров из CLI
    hp['epochs'] = args.epochs
    hp['learning_rate'] = args.lr
    hp['batch_size'] = args.batch_size
    if args.fine_tuning_lr:
        hp['fine_tuning_lr'] = args.fine_tuning_lr
    
    # Создание модели
    print(f"Загрузка модели: {args.model}")
    model = get_transfer_model(
        model_name=args.model,
        num_classes=len(CLASSES),
        pretrained=True,
        freeze=True  # Замораживаем веса
    )
    model = model.to(device)
    
    # Вывод информации о модели
    print_model_summary(model, args.model.upper())
    
    # Создание DataLoader с аугментацией
    print("Создание DataLoader с аугментацией...")
    train_loader, test_loader = create_augmented_dataloaders(
        train_X, train_y, test_X, test_y,
        batch_size=hp['batch_size'],
        aug_config=aug
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    print()
    
    # TensorBoard Writer
    writer = None
    if args.tb_logs:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"{args.model}_{args.preset}_{timestamp}"
        exp_dir = os.path.join(args.tb_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        writer = SummaryWriter(log_dir=exp_dir)
        print(f"✅ TensorBoard writer создан: {exp_dir}")
        
        # Логирование архитектуры модели
        writer.add_text('Model_Architecture', str(model), 0)
        writer.add_text('Hyperparameters',
                       '\n'.join([f"{k}: {v}" for k, v in hp.items()]), 0)
        print()
    
    # Обучение
    if args.train:
        # Уникальная папка для сохранения
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = os.path.join(args.tb_dir, f"{args.model}_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        
        if args.compare:
            # ================================================================
            # СРАВНЕНИЕ: Frozen vs Fine-tuning
            # ================================================================
            print("\n" + "="*70)
            print(" ЭКСПЕРИМЕНТ: СРАВНЕНИЕ FROZEN И FINE-TUNING")
            print("="*70)
            
            # Этап 1: Обучение с замороженной моделью
            print("\n[1/2] Обучение с ЗАМОРОЖЕННОЙ моделью...")
            model_frozen = get_transfer_model(args.model, len(CLASSES), pretrained=True, freeze=True)
            model_frozen = model_frozen.to(device)
            
            _, history_frozen, time_frozen, acc_frozen, _ = train_model_simple(
                model_frozen, train_loader, test_loader, hp, device, writer
            )
            
            # Этап 2: Обучение с fine-tuning
            print("\n[2/2] Обучение с FINE-TUNING (разморозка)...")
            model_ft = get_transfer_model(args.model, len(CLASSES), pretrained=True, freeze=False)
            model_ft = model_ft.to(device)
            
            # Уменьшаем LR для fine-tuning
            hp_ft = hp.copy()
            hp_ft['lr'] = hp.get('fine_tuning_lr', hp['lr'] / 10)
            
            _, history_ft, time_ft, acc_ft, _ = train_model_simple(
                model_ft, train_loader, test_loader, hp_ft, device, writer
            )
            
            # Сравнение результатов
            print("\n" + "="*70)
            print(" РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
            print("="*70)
            print(f"  Frozen модель:")
            print(f"    Точность: {acc_frozen:.2f}%")
            print(f"    Время: {time_frozen:.1f} сек")
            print(f"\n  Fine-tuning:")
            print(f"    Точность: {acc_ft:.2f}%")
            print(f"    Время: {time_ft:.1f} сек")
            print(f"\n  Разница: {acc_ft - acc_frozen:+.2f}%")
            print("="*70)
            
            # Сохранение результатов сравнения
            comparison_results = {
                'frozen': {
                    'accuracy': acc_frozen,
                    'time': time_frozen,
                    'history': history_frozen
                },
                'fine_tuning': {
                    'accuracy': acc_ft,
                    'time': time_ft,
                    'history': history_ft
                },
                'difference': acc_ft - acc_frozen
            }
            
            with open(f'{exp_dir}/comparison_results.json', 'w') as f:
                json.dump(comparison_results, f, indent=2)
            
            # Сохранение лучшей модели
            best_model = model_ft if acc_ft > acc_frozen else model_frozen
            best_acc = max(acc_ft, acc_frozen)
            
        elif args.unfreeze_after:
            # ================================================================
            # ДВУХЭТАПНОЕ ОБУЧЕНИЕ: Frozen → Fine-tuning
            # ================================================================
            print("\n" + "="*70)
            print(" ДВУХЭТАПНОЕ ОБУЧЕНИЕ: Заморозка → Fine-tuning")
            print("="*70)
            
            _, history, time_sec, best_acc, phase_acc = train_model_transfer_learning(
                model, train_X, train_y, test_X, test_y, hp, device, writer,
                unfreeze_after=args.unfreeze_after,
                unfreeze_layers=args.unfreeze_layers
            )
            
            # Сохранение информации о фазах
            hp['phase_accuracies'] = phase_acc
        
        else:
            # ================================================================
            # ОБЫЧНОЕ ОБУЧЕНИЕ (только замороженная модель)
            # ================================================================
            _, history, time_sec, best_acc = train_model_simple(
                model, train_loader, test_loader, hp, device, writer
            )
        
        # Сохранение чекпоинта
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'lab4_{args.model}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"\nМодель сохранена в {checkpoint_path}")
        
        # Сохранение в папку эксперимента
        torch.save(model.state_dict(), f'{exp_dir}/model.pth')
        
        # Сохранение истории
        with open(f'{exp_dir}/history.json', 'w') as f:
            json.dump({
                'model': args.model,
                'preset': args.preset,
                'hyperparameters': hp,
                'best_accuracy': best_acc,
                'training_time': time_sec,
                'history': history
            }, f, indent=2, default=str)
        
        # Сохранение графиков
        plot_learning_history(
            history,
            os.path.join(exp_dir, 'learning_history.png'),
            show_plot=not args.save_only
        )
        
        # Сохранение метрик в TensorBoard
        if writer:
            hparams = {
                'model': args.model,
                'learning_rate': hp['learning_rate'],
                'momentum': hp['momentum'],
                'weight_decay': hp['weight_decay'],
                'batch_size': hp['batch_size'],
                'epochs': hp['epochs'],
                'preset': args.preset
            }
            metrics = {
                'Final/test_accuracy': best_acc
            }
            writer.add_hparams(hparams, metrics)
            writer.close()
            print(f"\n✅ TensorBoard логи сохранены в {exp_dir}")
        
        print(f"\n✅ Эксперимент сохранён: {exp_dir}/")
    
    # Оценка
    if args.evaluate:
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
        
        report, cm = evaluate_model(model, test_X, test_y, CLASS_NAMES_RU, device)
        
        # Сохранение матрицы ошибок
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_confusion_matrix(
            cm,
            CLASS_NAMES_RU,
            os.path.join(OUTPUT_DIR, 'lab4_confusion_matrix.png'),
            show_plot=not args.save_only
        )
        
        print(f"\nClassification Report:\n{report}")
    
    print("\n" + "=" * 70)
    print(" ЗАВЕРШЕНО ")
    print("=" * 70)


if __name__ == '__main__':
    main()

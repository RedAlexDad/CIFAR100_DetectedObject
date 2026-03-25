#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Эксперимент: Сравнение оптимизаторов (SGD, Adam, RMSProp, Adagrad)
Задание №5: Сравните разные варианты оптимизатора

Использование:
    python run_exp_optimizers.py --all
    
    # Конкретный оптимизатор
    python run_exp_optimizers.py --optimizer adam
    
    # С кастомными параметрами
    python run_exp_optimizers.py --epochs 100 --lr 0.001
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime


# Конфигурация оптимизаторов
OPTIMIZERS_CONFIG = {
    'sgd': {
        'name': 'SGD',
        'description': 'Стохастический градиентный спуск с momentum',
        'lr': 0.005,  # 5e-3
        'momentum': 0.9,
        'weight_decay': 1e-5,
    },
    'adam': {
        'name': 'Adam',
        'description': 'Adaptive Moment Estimation',
        'lr': 0.005,
        'betas': (0.9, 0.999),
        'weight_decay': 1e-5,
    },
    'rmsprop': {
        'name': 'RMSProp',
        'description': 'Root Mean Square Propagation',
        'lr': 0.005,
        'alpha': 0.99,
        'weight_decay': 1e-5,
    },
    'adagrad': {
        'name': 'Adagrad',
        'description': 'Adaptive Gradient',
        'lr': 0.005,
        'lr_decay': 0.0,
        'weight_decay': 1e-5,
    }
}


def run_training(optimizer: str, lr: float, epochs: int, 
                 tb_dir: str, model: str = 'resnet20',
                 unfreeze_after: int = None, unfreeze_layers: int = 2,
                 dry_run: bool = False):
    """
    Запуск обучения с конкретным оптимизатором
    
    Args:
        optimizer: Название оптимизатора
        lr: Learning rate
        epochs: Количество эпох
        tb_dir: Директория для TensorBoard
        model: Название модели
        unfreeze_after: Эпоха для разморозки (None = не размораживать)
        unfreeze_layers: Количество слоёв для разморозки
        dry_run: Только показать команду без запуска
    """
    cmd = [
        sys.executable, 'main_lab4.py',
        '--model', model,
        '--train',
        '--lr', str(lr),
        '--epochs', str(epochs),
        '--batch-size', '128',
        '--tb-dir', tb_dir,
        '--preset', 'base'
    ]
    
    # Добавляем параметры для fine-tuning если указаны
    if unfreeze_after:
        cmd.extend(['--unfreeze-after', str(unfreeze_after)])
        cmd.extend(['--unfreeze-layers', str(unfreeze_layers)])
    
    cmd_str = ' '.join(cmd)
    print(f"\n{'='*70}")
    print(f" Оптимизатор: {optimizer.upper()}")
    print(f"{'='*70}")
    print(f"  Learning rate: {lr}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: 128")
    print(f"  TensorBoard: {tb_dir}")
    print(f"\nКоманда: {cmd_str}")
    
    if dry_run:
        print("[DRY RUN] Команда не выполнена")
        return True
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка при обучении: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Сравнение оптимизаторов: SGD, Adam, RMSProp, Adagrad'
    )
    parser.add_argument('--optimizer', type=str, default=None,
                        choices=['sgd', 'adam', 'rmsprop', 'adagrad'],
                        help='Конкретный оптимизатор (None = все)')
    parser.add_argument('--all', action='store_true',
                        help='Запустить все оптимизаторы')
    parser.add_argument('--model', type=str, default='resnet20',
                        choices=['resnet20', 'mobilenetv2'],
                        help='Модель для обучения')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Количество эпох (по умолчанию: 60)')
    parser.add_argument('--tb-dir', type=str, default='runs/lab4',
                        help='Базовая директория для TensorBoard')
    parser.add_argument('--dry-run', action='store_true',
                        help='Показать команды без запуска')
    parser.add_argument('--unfreeze-after', type=int, default=None,
                        help='Эпоха для разморозки (для fine-tuning)')
    parser.add_argument('--unfreeze-layers', type=int, default=5,
                        help='Количество слоёв для разморозки')
    
    args = parser.parse_args()
    
    print("="*70)
    print(" ЭКСПЕРИМЕНТ: СРАВНЕНИЕ ОПТИМИЗАТОРОВ")
    print(" Задание №5: Сравните разные варианты оптимизатора")
    print("="*70)
    print()
    
    # Определение оптимизаторов для запуска
    if args.optimizer:
        optimizers_to_run = [args.optimizer]
    elif args.all:
        optimizers_to_run = list(OPTIMIZERS_CONFIG.keys())
    else:
        # По умолчанию запускаем все
        optimizers_to_run = list(OPTIMIZERS_CONFIG.keys())
    
    print(f"📊 Оптимизаторы для сравнения: {len(optimizers_to_run)}")
    for opt in optimizers_to_run:
        config = OPTIMIZERS_CONFIG[opt]
        print(f"  - {config['name']}: lr = {config['lr']}")
    print()
    
    # Запуск обучения
    results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for opt_name in optimizers_to_run:
        config = OPTIMIZERS_CONFIG[opt_name]
        lr = config['lr']
        
        # Формирование имени директории
        tb_dir = os.path.join(
            args.tb_dir,
            f"exp5_optimizer_{opt_name}_lr{lr:.0e}_{args.model}_{timestamp}"
        )
        
        # Запуск обучения
        success = run_training(
            optimizer=opt_name,
            lr=lr,
            epochs=args.epochs,
            tb_dir=tb_dir,
            model=args.model,
            unfreeze_after=args.unfreeze_after,
            unfreeze_layers=args.unfreeze_layers,
            dry_run=args.dry_run
        )
        
        results[opt_name] = '✅' if success else '❌'
        
        # Пауза между запусками
        if not args.dry_run and opt_name != optimizers_to_run[-1]:
            print("\n⏳ Пауза 60 секунд перед следующим запуском...")
            import time
            time.sleep(60)
    
    # Итоговый отчёт
    print("\n" + "="*70)
    print(" РЕЗУЛЬТАТЫ СРАВНЕНИЯ ОПТИМИЗАТОРОВ")
    print("="*70)
    print(f"\n{'Оптимизатор':<15} {'Learning Rate':<15} {'Статус':<10}")
    print("-"*45)
    
    for opt_name in optimizers_to_run:
        config = OPTIMIZERS_CONFIG[opt_name]
        status = results.get(opt_name, '⏸️ Не запускался')
        print(f"{config['name']:<15} {config['lr']:<15.0e} {status:<10}")
    
    print("\n" + "="*70)
    print(" ЗАВЕРШЕНО ")
    print("="*70)
    
    # Инструкция для просмотра результатов
    print(f"\n📊 Для просмотра результатов в TensorBoard:")
    print(f"   tensorboard --logdir {args.tb_dir}/")
    print()
    
    # Рекомендации
    print("📝 Рекомендации по сравнению:")
    print("  1. Откройте TensorBoard и сравните графики Accuracy/loss")
    print("  2. Обратите внимание на скорость сходимости")
    print("  3. Сравните финальную точность на тесте")
    print("  4. SGD обычно даёт лучшую обобщающую способность")
    print("  5. Adam быстрее сходится, но может переобучаться")
    print()


if __name__ == '__main__':
    main()

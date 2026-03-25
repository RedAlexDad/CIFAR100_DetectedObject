#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Запуск эксперимента 4: Влияние количества размороженных слоёв (keep_last)
learning_rate = 1.5e-4

Использование:
    python run_exp4_keep_last.py
    
    # С конкретным вариантом
    python run_exp4_keep_last.py --keep-last 2
    
    # С выбором модели
    python run_exp4_keep_last.py --model mobilenetv2 --keep-last 5
    
    # С кастомными параметрами
    python run_exp4_keep_last.py --epochs 100 --lr 0.0002
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime


# Конфигурация эксперимента
EXPERIMENT_CONFIG = {
    'name': 'exp4_keep_last',
    'description': 'Обучение с заморозкой весов и разным keep_last (lr=1.5e-4)',
    'learning_rate': 0.00015,  # 1.5e-4
    'momentum': 0.9,
    'weight_decay': 0.0001,
    'batch_size': 64,
    'epochs': 50,
    'unfreeze_after': 10,
    'variants': [
        {'keep_last': 2, 'name': 'keep_last_2'},
        {'keep_last': 5, 'name': 'keep_last_5'},
        {'keep_last': 8, 'name': 'keep_last_8'},
    ]
}


def run_training(model: str, keep_last: int, lr: float, epochs: int, 
                 unfreeze_after: int, tb_dir: str, dry_run: bool = False):
    """
    Запуск обучения для одного варианта
    
    Args:
        model: Название модели (resnet20 или mobilenetv2)
        keep_last: Количество размороженных слоёв
        lr: Learning rate
        epochs: Количество эпох
        unfreeze_after: Эпоха для разморозки
        tb_dir: Директория для TensorBoard
        dry_run: Только показать команду без запуска
    """
    cmd = [
        sys.executable, 'main_lab4.py',
        '--model', model,
        '--train',
        '--unfreeze-after', str(unfreeze_after),
        '--unfreeze-layers', str(keep_last),
        '--lr', str(lr),
        '--epochs', str(epochs),
        '--batch-size', str(64),
        '--tb-dir', tb_dir,
        '--preset', 'base'
    ]
    
    cmd_str = ' '.join(cmd)
    print(f"\n{'='*70}")
    print(f" Запуск: {tb_dir}")
    print(f"{'='*70}")
    print(f"Команда: {cmd_str}")
    
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
        description='Запуск эксперимента 4: keep_last с lr=1.5e-4'
    )
    parser.add_argument('--model', type=str, default='resnet20',
                        choices=['resnet20', 'mobilenetv2'],
                        help='Модель для обучения')
    parser.add_argument('--keep-last', type=int, default=None,
                        choices=[2, 5, 8],
                        help='Количество размороженных слоёв (None = все варианты)')
    parser.add_argument('--lr', type=float, default=EXPERIMENT_CONFIG['learning_rate'],
                        help=f'Learning rate (по умолчанию: {EXPERIMENT_CONFIG["learning_rate"]})')
    parser.add_argument('--epochs', type=int, default=EXPERIMENT_CONFIG['epochs'],
                        help=f'Количество эпох (по умолчанию: {EXPERIMENT_CONFIG["epochs"]})')
    parser.add_argument('--unfreeze-after', type=int, 
                        default=EXPERIMENT_CONFIG['unfreeze_after'],
                        help=f'Эпоха для разморозки (по умолчанию: {EXPERIMENT_CONFIG["unfreeze_after"]})')
    parser.add_argument('--tb-dir', type=str, default='runs/lab4',
                        help='Базовая директория для TensorBoard')
    parser.add_argument('--dry-run', action='store_true',
                        help='Показать команды без запуска')
    parser.add_argument('--all', action='store_true',
                        help='Запустить все варианты (2, 5, 8)')
    
    args = parser.parse_args()
    
    print("="*70)
    print(f" ЭКСПЕРИМЕНТ 4: Keep Last Layers ")
    print(f" learning_rate = {args.lr} (1.5e-4)")
    print(f"{'='*70}")
    print(f"\n📋 Конфигурация:")
    print(f"  Модель: {args.model}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Unfreeze after: {args.unfreeze_after} эпох")
    print(f"  Batch size: 64")
    print()
    
    # Определение вариантов для запуска
    if args.keep_last is not None:
        # Один конкретный вариант
        variants_to_run = [
            {'keep_last': args.keep_last, 'name': f'keep_last_{args.keep_last}'}
        ]
    elif args.all:
        # Все варианты
        variants_to_run = EXPERIMENT_CONFIG['variants']
    else:
        # По умолчанию запускаем все варианты
        variants_to_run = EXPERIMENT_CONFIG['variants']
    
    print(f"📊 Варианты для запуска: {len(variants_to_run)}")
    for v in variants_to_run:
        print(f"  - {v['name']}: keep_last = {v['keep_last']}")
    print()
    
    # Запуск обучения
    results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for variant in variants_to_run:
        keep_last = variant['keep_last']
        variant_name = variant['name']
        
        # Формирование имени директории
        tb_dir = os.path.join(
            args.tb_dir,
            f"exp4_{variant_name}_lr{args.lr:.0e}_{args.model}_{timestamp}"
        )
        
        # Запуск обучения
        success = run_training(
            model=args.model,
            keep_last=keep_last,
            lr=args.lr,
            epochs=args.epochs,
            unfreeze_after=args.unfreeze_after,
            tb_dir=tb_dir,
            dry_run=args.dry_run
        )
        
        results[variant_name] = '✅' if success else '❌'
        
        # Пауза между запусками (чтобы GPU остыл)
        if not args.dry_run and variant != variants_to_run[-1]:
            print("\n⏳ Пауза 30 секунд перед следующим запуском...")
            import time
            time.sleep(30)
    
    # Итоговый отчёт
    print("\n" + "="*70)
    print(" РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА 4")
    print("="*70)
    print(f"\n{'Вариант':<25} {'keep_last':<12} {'Статус':<10}")
    print("-"*50)
    
    for variant in EXPERIMENT_CONFIG['variants']:
        variant_name = variant['name']
        keep_last = variant['keep_last']
        status = results.get(variant_name, '⏸️ Не запускался')
        print(f"{variant_name:<25} {keep_last:<12} {status:<10}")
    
    print("\n" + "="*70)
    print(" ЗАВЕРШЕНО ")
    print("="*70)
    
    # Инструкция для просмотра результатов
    print(f"\n📊 Для просмотра результатов в TensorBoard:")
    print(f"   tensorboard --logdir {args.tb_dir}/")
    print()
    
    # Сравнение результатов
    if not args.dry_run:
        print(f"📈 Для сравнения результатов используйте:")
        print(f"   python compare_labs.py --lab4-dir {args.tb_dir}/")
        print()


if __name__ == '__main__':
    main()

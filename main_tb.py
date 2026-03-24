#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Главный скрипт для обучения модели CNN на CIFAR-100
БЕЗ TensorFlow - только PyTorch + TensorBoard

Использование:
    python main.py --model medium --train
    python main.py --model base --train --tb-logs
"""

import argparse
import sys
import os
from datetime import datetime

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# matplotlib до импорта pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Остальные импорты
from configs.config import (
    CLASSES, CLASS_NAMES, CLASS_NAMES_RU,
    MODEL_CONFIG, TRAIN_CONFIG, DEVICE,
    CHECKPOINT_DIR, ONNX_DIR, OUTPUT_DIR
)
from models import Cifar100_CNN_Base, Cifar100_CNN_Medium
from scripts.data_utils import load_cifar100_data, get_data_dir
from scripts.eval_utils import evaluate_model, save_confusion_matrix


def get_model(model_name, device):
    """Получение модели по имени"""
    if model_name == 'base':
        model = Cifar100_CNN_Base(**MODEL_CONFIG['base'])
    elif model_name == 'medium':
        model = Cifar100_CNN_Medium(**MODEL_CONFIG['medium'])
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")
    
    model = model.to(device)
    return model


def train_model_with_tb(model, train_X, train_y, test_X, test_y, config, device, writer=None):
    """Обучение с TensorBoard"""
    
    print(f"\nНАЧАЛО ОБУЧЕНИЯ")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Device: {device}")
    if writer:
        print(f"  TensorBoard: ✅ Включено")
    print()
    
    # DataLoader
    tensor_train_x = torch.Tensor(train_X)
    tensor_train_y = torch.Tensor(train_y).long()
    train_dataset = TensorDataset(tensor_train_x, tensor_train_y)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    tensor_test_x = torch.Tensor(test_X)
    tensor_test_y = torch.Tensor(test_y).long()
    test_dataset = TensorDataset(tensor_test_x, tensor_test_y)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=1e-4)
    
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_test_acc = 0.0
    best_model_state = None
    start_time = time.time()
    
    steps_per_epoch = len(train_loader)
    
    import time as time_module
    start_time = time_module.time()
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
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
            
            # TensorBoard: Batch
            if writer:
                global_step = epoch * steps_per_epoch + batch_idx
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Validation
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
        
        # History
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(val_loss_avg)
        history['test_acc'].append(val_acc)
        
        if scheduler:
            scheduler.step()
        
        # TensorBoard: Epoch
        if writer:
            writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            writer.add_scalar('Loss/val_epoch', val_loss_avg, epoch)
            writer.add_scalar('Accuracy/train_epoch', train_acc, epoch)
            writer.add_scalar('Accuracy/val_epoch', val_acc, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
            
            if val_acc > best_test_acc:
                best_test_acc = val_acc
                writer.add_scalar('Best_Accuracy', best_test_acc, epoch)
        
        # Progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Эпоха [{epoch+1}/{config['epochs']}] | "
                  f"Train: {train_loss:.4f} ({train_acc:.2f}%) | "
                  f"Val: {val_loss_avg:.4f} ({val_acc:.2f}%)")
        
        # Best model
        if val_acc > best_test_acc:
            best_test_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    training_time = time_module.time() - start_time
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\nОбучение завершено за {training_time:.2f} сек")
    print(f"Лучшая точность: {best_test_acc:.2f}%")
    
    return model, history, training_time, best_test_acc


def plot_history(history, output_path, show=False):
    """Графики обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history['train_acc'], label='Train')
    ax1.plot(history['test_acc'], label='Test')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_loss'], label='Train')
    ax2.plot(history['test_loss'], label='Test')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='medium', choices=['base', 'medium'])
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--tb-logs', action='store_true', help='TensorBoard логирование')
    parser.add_argument('--tb-dir', default='runs/exp1')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--save-only', action='store_true')
    
    args = parser.parse_args()
    
    if not args.train and not args.evaluate:
        parser.print_help()
        return
    
    print("=" * 70)
    print(" ЛАБОРАТОРНАЯ РАБОТА №2/№3 ")
    print(f" Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    if args.tb_logs:
        print(f"\n📊 TensorBoard: {args.tb_dir}")
        print(f"   tensorboard --logdir={args.tb_dir}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nУстройство: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Data
    data_dir = get_data_dir()
    train_X, train_y, test_X, test_y = load_cifar100_data(data_dir, CLASSES)
    
    # Model
    model = get_model(args.model, device)
    print(f"\nМодель: {args.model}")
    
    # TensorBoard
    writer = None
    if args.tb_logs:
        os.makedirs(args.tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.tb_dir)
        writer.add_text('Model', str(model), 0)
    
    # Train
    if args.train:
        config = TRAIN_CONFIG[args.model].copy()
        if args.lr: config['lr'] = args.lr
        if args.batch_size: config['batch_size'] = args.batch_size
        if args.epochs: config['epochs'] = args.epochs
        
        model, history, time_sec, acc = train_model_with_tb(
            model, train_X, train_y, test_X, test_y, config, device, writer
        )
        
        # Save checkpoint
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(model.state_dict(), f'{CHECKPOINT_DIR}/cnn_{args.model}.pth')
        
        # Plot
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plot_history(history, f'{OUTPUT_DIR}/learning_history.png', show=not args.save_only)
        
        # Final metrics to TB
        if writer:
            writer.add_scalar('Final/test_accuracy', acc, 0)
            writer.add_scalar('Final/best_accuracy', history['test_acc'][-1], 0)
    
    # Evaluate
    if args.evaluate:
        report, cm = evaluate_model(model, test_X, test_y, CLASS_NAMES_RU, device)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_confusion_matrix(cm, CLASS_NAMES_RU, f'{OUTPUT_DIR}/confusion_matrix.png', show=not args.save_only)
        
        if writer:
            writer.add_scalar('Final/test_accuracy', report['accuracy'], 0)
    
    if writer:
        writer.close()
        print(f"\n✅ TensorBoard: {args.tb_dir}")
    
    print("\n" + "=" * 70)
    print(" ЗАВЕРШЕНО ")
    print("=" * 70)


if __name__ == '__main__':
    import time
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучение CNN с TensorBoard логированием
БЕЗ TensorFlow в импортах - только PyTorch

Использование:
    python train_simple.py --model base --train --tb-logs
    tensorboard --logdir runs/exp1  # в отдельном терминале
"""

import argparse
import sys
import os
from datetime import datetime
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# TensorBoard - импортируем только если нужно
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("⚠️  TensorBoard не установлен: pip install tensorboard")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import CLASSES, CLASS_NAMES_RU, TRAIN_CONFIG, MODEL_CONFIG
from configs.training_config import get_config, print_config, DEFAULT_HYPERPARAMS, DEFAULT_AUGMENTATION
from models import Cifar100_CNN_Base, Cifar100_CNN_Medium
from scripts.data_utils import load_cifar100_data, get_data_dir
from scripts.eval_utils import evaluate_model, save_confusion_matrix
from scripts.augmentation import get_train_transform, get_test_transform, AugmentedDataset
from logger import TensorBoardLogger


def train(model, train_loader, test_loader, hp, device, log_file=None, writer=None):
    """Обучение с аугментацией и расширенными гиперпараметрами"""
    
    criterion = nn.CrossEntropyLoss(label_smoothing=hp['label_smoothing'])
    optimizer = optim.SGD(
        model.parameters(), 
        lr=hp['learning_rate'], 
        momentum=hp['momentum'], 
        weight_decay=hp['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=hp['step_size'], 
        gamma=hp['gamma']
    )
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0.0
    steps = 0
    
    import time
    start = time.time()
    
    for epoch in range(hp['epochs']):
        # Train
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (labels == outputs.argmax(1)).float().mean().item() * 100
            
            # TensorBoard: batch
            if writer:
                writer.add_scalar('Loss/train_batch', loss.item(), steps)
                steps += 1
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Test
        model.eval()
        test_loss, test_acc = 0.0, 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                loss = criterion(model(inputs), labels)
                test_loss += loss.item()
                test_acc += (labels == model(inputs).argmax(1)).float().mean().item() * 100
        
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        # Log CSV
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"{epoch},{train_loss},{train_acc},{test_loss},{test_acc}\n")
        
        # TensorBoard: epoch
        if writer:
            writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            writer.add_scalar('Loss/test_epoch', test_loss, epoch)
            writer.add_scalar('Accuracy/train_epoch', train_acc, epoch)
            writer.add_scalar('Accuracy/test_epoch', test_acc, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('Best_Accuracy', best_acc, epoch)
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{hp['epochs']} | "
                  f"Train: {train_loss:.4f} ({train_acc:.2f}%) | "
                  f"Test: {test_loss:.4f} ({test_acc:.2f}%) | Best: {best_acc:.2f}%")
    
    print(f"\nОбучение за {time.time()-start:.1f} сек | Best: {best_acc:.2f}%")
    
    return model, history, best_acc


def plot_history(history, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_acc'], label='Train')
    ax1.plot(history['test_acc'], label='Test')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_loss'], label='Train')
    ax2.plot(history['test_loss'], label='Test')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='medium', choices=['base', 'medium'])
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--preset', default='base', choices=['base', 'fast', 'accurate'],
                        help='Пресет конфигурации')
    parser.add_argument('--tb-dir', default='tensorboard', help='Папка для TensorBoard логов')
    
    # Гиперпараметры (переопределяют пресет)
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=None, help='Momentum')
    parser.add_argument('--weight-decay', type=float, default=None, help='Weight decay')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Epochs')
    parser.add_argument('--label-smoothing', type=float, default=None, help='Label smoothing')
    parser.add_argument('--step-size', type=int, default=None, help='Step size for scheduler')
    parser.add_argument('--gamma', type=float, default=None, help='Gamma for scheduler')
    
    # Аугментация
    parser.add_argument('--no-aug', action='store_true', help='Отключить аугментацию')
    parser.add_argument('--brightness', type=float, nargs=2, default=None, help='Brightness range')
    parser.add_argument('--contrast', type=float, nargs=2, default=None, help='Contrast range')
    parser.add_argument('--saturation', type=float, nargs=2, default=None, help='Saturation range')
    parser.add_argument('--degrees', type=float, nargs=2, default=None, help='Rotation degrees range')
    
    args = parser.parse_args()
    
    if not args.train and not args.evaluate:
        parser.print_help()
        return
    
    # Загрузка конфигурации
    config = get_config(args.preset)
    hp = config['hyperparams']
    aug = config['augmentation']
    
    # Переопределение гиперпараметров из CLI
    if args.lr: hp['learning_rate'] = args.lr
    if args.momentum: hp['momentum'] = args.momentum
    if args.weight_decay: hp['weight_decay'] = args.weight_decay
    if args.batch_size: hp['batch_size'] = args.batch_size
    if args.epochs: hp['epochs'] = args.epochs
    if args.label_smoothing: hp['label_smoothing'] = args.label_smoothing
    if args.step_size: hp['step_size'] = args.step_size
    if args.gamma: hp['gamma'] = args.gamma
    
    # Переопределение аугментации из CLI
    if args.brightness: aug['brightness'] = tuple(args.brightness)
    if args.contrast: aug['contrast'] = tuple(args.contrast)
    if args.saturation: aug['saturation'] = tuple(args.saturation)
    if args.degrees: aug['degrees'] = tuple(args.degrees)
    
    if args.no_aug:
        aug = DEFAULT_AUGMENTATION.copy()
        aug['brightness'] = None
        aug['contrast'] = None
        aug['saturation'] = None
        aug['degrees'] = None
        aug['translate'] = None
        aug['scale'] = None
        aug['shear'] = None
    
    # Вывод конфигурации
    print_config(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data
    data_dir = get_data_dir()
    train_X, train_y, test_X, test_y = load_cifar100_data(data_dir, CLASSES)
    
    # Transforms
    train_transform = get_train_transform(aug) if not args.no_aug else get_test_transform()
    test_transform = get_test_transform()
    
    # Create datasets with augmentation
    tensor_train_x = torch.Tensor(train_X)
    tensor_train_y = torch.Tensor(train_y).long()
    train_dataset = AugmentedDataset(tensor_train_x, tensor_train_y, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=hp['batch_size'], shuffle=True)
    
    tensor_test_x = torch.Tensor(test_X)
    tensor_test_y = torch.Tensor(test_y).long()
    test_dataset = AugmentedDataset(tensor_test_x, tensor_test_y, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=hp['batch_size'], shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Model
    if args.model == 'base':
        model = Cifar100_CNN_Base(**MODEL_CONFIG['base'])
    else:
        model = Cifar100_CNN_Medium(**MODEL_CONFIG['medium'])
    model.to(device)
    
    # Config
    config = TRAIN_CONFIG[args.model].copy()
    if args.lr: config['lr'] = args.lr
    if args.batch_size: config['batch_size'] = args.batch_size
    if args.epochs: config['epochs'] = args.epochs
    
    # Timestamp и уникальная папка эксперимента
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{args.model}_{timestamp}"
    
    # Создаем папку эксперимента
    exp_dir = os.path.join(args.tb_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # TensorBoard Logger
    logger = TensorBoardLogger()
    writer = None
    
    if HAS_TB:
        writer = logger.initialize_log_dir_with_dir(exp_dir)
        
        # Log model architecture
        logger.log_text('Model_Architecture', str(model), 0)
        logger.log_text('Config', 
                       f"lr={config['lr']}, batch_size={config['batch_size']}, epochs={config['epochs']}", 
                       0)
        
        # Log model graph
        logger.log_model_graph(model, input_size=(32, 32, 3), device=device)
    
    # Файлы в папке эксперимента
    log_file = f'{exp_dir}/metrics.csv'
    
    # Header
    with open(log_file, 'w') as f:
        f.write('epoch,train_loss,train_acc,test_loss,test_acc\n')
    
    print(f"📁 Experiment dir: {exp_dir}")
    print()
    
    # Train
    if args.train:
        model, history, best_acc = train(model, train_loader, test_loader, hp, device, log_file, writer)
        
        # Save в папку эксперимента
        torch.save(model.state_dict(), f'{exp_dir}/model.pth')
        plot_history(history, f'{exp_dir}/history.png')
        
        # Save metrics JSON
        with open(f'{exp_dir}/metrics.json', 'w') as f:
            json.dump({
                'model': args.model,
                'preset': args.preset,
                'hyperparams': hp,
                'augmentation': aug if not args.no_aug else None,
                'best_accuracy': best_acc,
                'history': history
            }, f, indent=2, default=str)
        
        # Final TB metrics
        if writer:
            hparams = {
                'learning_rate': hp['learning_rate'],
                'momentum': hp['momentum'],
                'weight_decay': hp['weight_decay'],
                'batch_size': hp['batch_size'],
                'epochs': hp['epochs'],
                'label_smoothing': hp['label_smoothing'],
                'model': args.model
            }
            metrics = {
                'Final/best_accuracy': best_acc,
                'Final/test_accuracy': history['test_acc'][-1]
            }
            logger.log_hyperparameters(hparams, metrics)
            logger.close()
        
        print(f"\n✅ Saved: {exp_dir}/")
        print(f"   tensorboard --logdir {exp_dir}")
    
    # Evaluate
    if args.evaluate:
        report, cm = evaluate_model(model, test_X, test_y, CLASS_NAMES_RU, device)
        os.makedirs(exp_dir, exist_ok=True)
        save_confusion_matrix(cm, CLASS_NAMES_RU, f'{exp_dir}/confusion_matrix.png', show=False)
        print(f"\nClassification Report:\n{report}")
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()

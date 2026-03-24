#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Запуск обучения с YAML конфигурацией
Использование:
    python train_from_yaml.py configs/config.yaml
    python train_from_yaml.py configs/config.yaml --variant dropout_0.2_0.3
"""

import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json

from configs.yaml_config import load_config, get_model_from_config, get_variants, print_config_summary
from configs.config import CLASSES
from scripts.data_utils import load_cifar100_data, get_data_dir
from scripts.augmentation import get_train_transform, get_test_transform, AugmentedDataset
from logger import TensorBoardLogger

# Классы CIFAR-100
cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'l_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank',
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip',
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


def create_model_from_arch(architecture):
    """Создание модели с Normalize слоем"""
    
    class ModelWithNorm(nn.Module):
        def __init__(self, arch):
            super().__init__()
            self.norm = nn.Sequential()
            # Normalize слой
            self.register_buffer('mean', torch.tensor([0.5074, 0.4867, 0.4411]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.2011, 0.1987, 0.2025]).view(1, 3, 1, 1))
            
            # Архитектура из конфига
            self.arch = get_model_from_config(arch)
        
        def forward(self, x):
            # Поддержка NHWC и NCHW
            if x.dim() == 4 and x.shape[1] > 10:
                x = x.permute(0, 3, 1, 2)
            x = x / 255.0
            x = (x - self.mean) / self.std
            return self.arch(x)
    
    return ModelWithNorm(architecture)


def train_variant(variant, config_name, device, tb_dir):
    """Обучение одного варианта"""
    
    print(f"\n{'='*70}")
    print(f" ОБУЧЕНИЕ ВАРИАНТА: {variant.get('name', 'default')}")
    print(f"{'='*70}")
    
    # Данные
    data_dir = get_data_dir()
    train_X, train_y, test_X, test_y = load_cifar100_data(data_dir, CLASSES)
    
    # Аугментация
    aug = variant.get('augmentation', {})
    train_transform = get_train_transform(aug)
    test_transform = get_test_transform()
    
    tensor_train_x = torch.Tensor(train_X)
    tensor_train_y = torch.Tensor(train_y).long()
    train_dataset = AugmentedDataset(tensor_train_x, tensor_train_y, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    tensor_test_x = torch.Tensor(test_X)
    tensor_test_y = torch.Tensor(test_y).long()
    test_dataset = AugmentedDataset(tensor_test_x, tensor_test_y, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Модель
    architecture = variant.get('model_architecture', [])
    model = create_model_from_arch(architecture)
    model.to(device)
    
    print(f"\nАрхитектура:")
    print(model.arch)
    
    # Гиперпараметры
    hp = variant.get('hyperparameters', {})
    lr = hp.get('learning_rate', 0.005)
    momentum = hp.get('momentum', 0.9)
    weight_decay = hp.get('weight_decay', 1e-5)
    epochs = hp.get('epochs', 500)
    label_smoothing = hp.get('label_smoothing', 0.1)
    
    scheduler_cfg = hp.get('scheduler', {})
    step_size = scheduler_cfg.get('step_size', 240)
    gamma = scheduler_cfg.get('gamma', 0.5)
    
    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{config_name}_{variant.get('name', 'default')}_{timestamp}"
    exp_dir = os.path.join(tb_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    logger = TensorBoardLogger()
    writer = logger.initialize_log_dir_with_dir(exp_dir)
    logger.log_text('Architecture', str(architecture), 0)
    logger.log_text('Hyperparameters', json.dumps(hp, indent=2), 0)
    
    # Обучение
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    log_file = f'{exp_dir}/metrics.csv'
    
    with open(log_file, 'w') as f:
        f.write('epoch,train_loss,train_acc,test_loss,test_acc\n')
    
    import time
    start = time.time()
    
    for epoch in range(epochs):
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
        
        with open(log_file, 'a') as f:
            f.write(f"{epoch},{train_loss},{train_acc},{test_loss},{test_acc}\n")
        
        if writer:
            writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            writer.add_scalar('Loss/test_epoch', test_loss, epoch)
            writer.add_scalar('Accuracy/train_epoch', train_acc, epoch)
            writer.add_scalar('Accuracy/test_epoch', test_acc, epoch)
            writer.add_scalar('Best_Accuracy', best_acc, epoch)
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} ({train_acc:.2f}%) | Test: {test_loss:.4f} ({test_acc:.2f}%) | Best: {best_acc:.2f}%")
    
    training_time = time.time() - start
    print(f"\nОбучение за {training_time:.1f} сек | Best: {best_acc:.2f}%")
    
    # Сохранение
    torch.save(model.state_dict(), f'{exp_dir}/model.pth')
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
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
    plt.savefig(f'{exp_dir}/history.png')
    plt.close()
    
    with open(f'{exp_dir}/metrics.json', 'w') as f:
        json.dump({
            'variant': variant.get('name', 'default'),
            'config': config_name,
            'hyperparameters': hp,
            'best_accuracy': best_acc,
            'training_time': training_time,
            'history': history
        }, f, indent=2, default=str)
    
    if writer:
        logger.log_hyperparameters({'model': variant.get('name', 'default'), **hp}, {'Final/best_accuracy': best_acc})
        logger.close()
    
    print(f"\n✅ Saved: {exp_dir}/")
    print(f"   tensorboard --logdir {exp_dir}")
    
    return best_acc


def main():
    parser = argparse.ArgumentParser(description='Обучение из YAML конфигурации')
    parser.add_argument('config', help='Путь к YAML конфигурации')
    parser.add_argument('--variant', type=str, default=None, help='Название варианта для обучения')
    parser.add_argument('--tb-dir', default='tensorboard', help='Папка для TensorBoard')
    parser.add_argument('--all', action='store_true', help='Обучить все варианты')
    parser.add_argument('--epochs', type=int, default=None, help='Количество эпох (переопределяет YAML)')
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = load_config(args.config)
    print_config_summary(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Получение вариантов
    variants = get_variants(config)
    config_name = config.get('name', 'config')
    
    # Переопределение epochs если указано
    if args.epochs:
        for v in variants:
            if 'hyperparameters' in v:
                v['hyperparameters']['epochs'] = args.epochs
            else:
                v['hyperparameters'] = {'epochs': args.epochs}
        print(f"⚙️  Переопределено epochs: {args.epochs}\n")
    
    if args.variant:
        # Обучение конкретного варианта
        variant = next((v for v in variants if v.get('name') == args.variant), None)
        if not variant:
            print(f"❌ Вариант '{args.variant}' не найден!")
            print(f"Доступные варианты: {[v.get('name') for v in variants]}")
            return
        train_variant(variant, config_name, device, args.tb_dir)
    elif args.all:
        # Обучение всех вариантов
        print(f"\n🔄 Обучение всех {len(variants)} вариантов...")
        results = []
        for variant in variants:
            best_acc = train_variant(variant, config_name, device, args.tb_dir)
            results.append({'name': variant.get('name', 'default'), 'accuracy': best_acc})
        
        print("\n" + "="*70)
        print(" РЕЗУЛЬТАТЫ ВСЕХ ВАРИАНТОВ ")
        print("="*70)
        for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
            print(f"  {r['name']:<30} {r['accuracy']:.2f}%")
        print("="*70)
    else:
        # Обучение базовой конфигурации
        base_variant = {
            'name': config.get('name', 'default'),
            'model_architecture': config.get('model_architecture', []),
            'hyperparameters': config.get('hyperparameters', {}),
            'augmentation': config.get('augmentation', {})
        }
        train_variant(base_variant, config_name, device, args.tb_dir)


if __name__ == '__main__':
    main()

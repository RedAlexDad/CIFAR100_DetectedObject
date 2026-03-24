#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Запуск всех экспериментов с перебором параметров
Сохранение результатов в TensorBoard
"""

import argparse
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torch.utils.data import TensorDataset
import json
import yaml
import pickle
import numpy as np

from configs.yaml_config import get_model_from_config
from configs.config import CLASSES
from scripts.data_utils import load_cifar100_data, get_data_dir
from scripts.augmentation import get_train_transform, get_test_transform, AugmentedDataset
from scripts.run_independent_experiments import (
    load_config,
    generate_dropout_combinations,
    generate_weight_decay_combinations,
    generate_augmentation_combinations
)
from logger import TensorBoardLogger


# ============================================================================
# CifarDataset - как в рабочем коде
# ============================================================================
class CifarDataset(Dataset):
    def __init__(self, X, y, transform=None, p=0.0):
        assert X.size(0) == y.size(0)
        super(Dataset, self).__init__()
        self.X = X
        self.y = y
        self.transform = transform
        self.prob = p

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, index):
        x = self.X[index]
        if self.transform and np.random.random() < self.prob:
            x = self.transform(x.permute(2, 0, 1) / 255.).permute(1, 2, 0) * 255.
        y = self.y[index]
        return x, y


# ============================================================================
# Cifar100_CNN - как в рабочем коде
# ============================================================================
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        # mean/std для формата NHWC (batch, height, width, channels)
        self.register_buffer('mean', torch.tensor(mean).view(1, 1, 1, 3))
        self.register_buffer('std', torch.tensor(std).view(1, 1, 1, 3))

    def forward(self, input):
        # input в формате NHWC (batch, height, width, channels)
        x = input / 255.0
        x = x - self.mean
        x = x / self.std
        return x.permute(0, 3, 1, 2)  # NHWC -> NCHW


class Cifar100_CNN(nn.Module):
    def __init__(self, hidden_size=32, classes=100, dropout_1=0.2, dropout_2=0.3):
        super(Cifar100_CNN, self).__init__()
        self.seq = nn.Sequential(
            Normalize([0.5074, 0.4867, 0.4411], [0.2011, 0.1987, 0.2025]),
            nn.Conv2d(3, hidden_size, 3, stride=4),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_1),
            nn.Conv2d(hidden_size, hidden_size * 2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(4),
            nn.Dropout2d(p=dropout_2),
            nn.Flatten(),
            nn.Linear(hidden_size * 8, classes),
        )

    def forward(self, input):
        return self.seq(input)


def create_model_from_arch(architecture):
    """Создание модели с Normalize слоем"""
    
    class ModelWithNorm(nn.Module):
        def __init__(self, arch):
            super().__init__()
            self.register_buffer('mean', torch.tensor([0.5074, 0.4867, 0.4411]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.2011, 0.1987, 0.2025]).view(1, 3, 1, 1))
            self.arch = get_model_from_config(arch)
        
        def forward(self, x):
            if x.dim() == 4 and x.shape[1] > 10:
                x = x.permute(0, 3, 1, 2)
            x = x / 255.0
            x = (x - self.mean) / self.std
            return self.arch(x)
    
    return ModelWithNorm(architecture)


def train_single_variant(variant, exp_name, device, tb_base_dir, epochs_override=None):
    """Обучение одного варианта"""
    
    variant_name = variant.get('name', 'unknown')
    print(f"\n{'='*70}")
    print(f" ОБУЧЕНИЕ: {exp_name} / {variant_name}")
    print(f"{'='*70}")
    
    # Данные - как в рабочем коде
    with open('cifar-100-python/train', 'rb') as f:
        data_train = pickle.load(f, encoding='latin1')
    with open('cifar-100-python/test', 'rb') as f:
        data_test = pickle.load(f, encoding='latin1')
    
    train_X = data_train['data'].reshape(-1, 3, 32, 32)
    train_X = np.transpose(train_X, [0, 2, 3, 1])
    train_y = np.array(data_train['fine_labels'])
    mask = np.isin(train_y, CLASSES)
    train_X = train_X[mask].copy()
    train_y = train_y[mask].copy()
    train_y = np.unique(train_y, return_inverse=True)[1]
    del data_train
    
    test_X = data_test['data'].reshape(-1, 3, 32, 32)
    test_X = np.transpose(test_X, [0, 2, 3, 1])
    test_y = np.array(data_test['fine_labels'])
    mask = np.isin(test_y, CLASSES)
    test_X = test_X[mask].copy()
    test_y = test_y[mask].copy()
    test_y = np.unique(test_y, return_inverse=True)[1]
    del data_test
    
    print(f"Train: {len(train_X)}, Test: {len(test_y)}")
    
    # Аугментация - как в рабочем коде
    aug = variant.get('augmentation', {})
    if aug:
        hue_val = aug.get('hue', 0)
        if hue_val is None:
            hue_val = 0
        transform = T.Compose([
            T.ColorJitter(
                brightness=aug.get('brightness', [0.9, 1.1]),
                contrast=aug.get('contrast', [0.9, 1.1]),
                saturation=aug.get('saturation', [0.8, 1.2]),
                hue=hue_val
            ),
            T.RandomAffine(
                degrees=aug.get('rotation', [-15, 15]),
                translate=aug.get('translate', (0.1, 0.1)),
                scale=aug.get('scale', (0.8, 1.2)),
                shear=aug.get('shear', [-5, 5])
            ),
        ])
    else:
        transform = None
    
    # DataLoader - как в рабочем коде
    batch_size = 128
    dataloader = {}
    for (X, y), part in zip([(train_X, train_y), (test_X, test_y)], ['train', 'test']):
        tensor_x = torch.Tensor(X)
        tensor_y = F.one_hot(torch.Tensor(y).to(torch.int64), num_classes=len(CLASSES)) / 1.
        dataset = CifarDataset(tensor_x, tensor_y, transform=transform, p=0.5 if part=='train' else 0.0)
        dataloader[part] = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Модель - как в рабочем коде
    architecture = variant.get('model_architecture', [])
    # Находим dropout значения из архитектуры
    dropout_rates = []
    for layer in architecture:
        if layer['type'] == 'Dropout2d':
            dropout_rates.append(layer['args'][0])
    
    # Используем архитектуру из рабочего кода
    HIDDEN_SIZE = 64
    model = Cifar100_CNN(hidden_size=HIDDEN_SIZE, classes=len(CLASSES), 
                         dropout_1=dropout_rates[0] if len(dropout_rates)>0 else 0.2,
                         dropout_2=dropout_rates[1] if len(dropout_rates)>1 else 0.3)
    model.to(device)
    
    # Гиперпараметры - как в рабочем коде
    hp = variant.get('hyperparameters', {})
    lr = hp.get('learning_rate', 0.005)
    momentum = hp.get('momentum', 0.9)
    weight_decay = hp.get('weight_decay', 1e-5)
    # Преобразуем weight_decay в float (из YAML может прийти строка)
    if isinstance(weight_decay, str):
        weight_decay = float(weight_decay)
    epochs = epochs_override if epochs_override else hp.get('epochs', 500)
    label_smoothing = hp.get('label_smoothing', 0.1)
    
    scheduler_cfg = hp.get('scheduler', {})
    step_size = scheduler_cfg.get('step_size', 240)
    gamma = scheduler_cfg.get('gamma', 0.5)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(tb_base_dir, f"{exp_name}_{variant_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    logger = TensorBoardLogger()
    writer = logger.initialize_log_dir_with_dir(exp_dir)
    logger.log_text('Variant_Name', variant_name, 0)
    logger.log_text('Hyperparameters', json.dumps(hp, indent=2), 0)
    
    # Обучение - как в рабочем коде
    best_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [], 
        'test_loss': [], 'test_acc': [],
        'func_loss': []  # Функция потерь
    }
    log_file = f'{exp_dir}/metrics.csv'
    
    with open(log_file, 'w') as f:
        f.write('epoch,train_loss,train_acc,test_loss,test_acc,func_loss,lr\n')
    
    import time
    start = time.time()
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_acc = 0.0, 0.0
        batch_losses = []  # Для статистики по батчам
        for batch_idx, (inputs, labels) in enumerate(dataloader['train']):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += (labels.argmax(dim=-1) == outputs.argmax(dim=-1)).float().mean().item() * 100
            batch_losses.append(loss.item())
            
            # TensorBoard: каждый батч
            if writer and epoch < 5:  # Только первые 5 эпох для батчей
                global_step = epoch * len(dataloader['train']) + batch_idx
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                writer.add_scalar('Accuracy/train_batch', 
                    (labels.argmax(dim=-1) == outputs.argmax(dim=-1)).float().mean().item() * 100, 
                    global_step)
        
        train_loss /= len(dataloader['train'])
        train_acc /= len(dataloader['train'])
        
        # Статистика по батчам
        train_loss_std = np.std(batch_losses)
        train_loss_min = np.min(batch_losses)
        train_loss_max = np.max(batch_losses)
        
        # Test
        model.eval()
        test_loss, test_acc = 0.0, 0.0
        test_batch_losses = []
        with torch.no_grad():
            for inputs, labels in dataloader['test']:
                inputs, labels = inputs.to(device), labels.to(device)
                loss = criterion(model(inputs), labels)
                test_loss += loss.item()
                test_acc += (labels.argmax(dim=-1) == model(inputs).argmax(dim=-1)).float().mean().item() * 100
                test_batch_losses.append(loss.item())
        
        test_loss /= len(dataloader['test'])
        test_acc /= len(dataloader['test'])
        
        # Функция потерь (func_loss) - это test_loss
        func_loss = test_loss
        test_loss_std = np.std(test_batch_losses)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['func_loss'].append(func_loss)
        
        if test_acc > best_acc:
            best_acc = test_acc
        
        with open(log_file, 'a') as f:
            f.write(f"{epoch},{train_loss},{train_acc},{test_loss},{test_acc},{func_loss},{current_lr}\n")
        
        # TensorBoard: максимум информации
        if writer:
            # Основные метрики
            writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            writer.add_scalar('Loss/test_epoch', test_loss, epoch)
            writer.add_scalar('Loss/func_loss', func_loss, epoch)
            writer.add_scalar('Accuracy/train_epoch', train_acc, epoch)
            writer.add_scalar('Accuracy/test_epoch', test_acc, epoch)
            writer.add_scalar('Best_Accuracy', best_acc, epoch)
            
            # Learning Rate
            writer.add_scalar('Optimizer/learning_rate', current_lr, epoch)
            writer.add_scalar('Optimizer/momentum', momentum, epoch)
            writer.add_scalar('Optimizer/weight_decay', weight_decay, epoch)
            
            # Статистика по батчам
            writer.add_scalar('Loss/train_std', train_loss_std, epoch)
            writer.add_scalar('Loss/train_min', train_loss_min, epoch)
            writer.add_scalar('Loss/train_max', train_loss_max, epoch)
            writer.add_scalar('Loss/test_std', test_loss_std, epoch)
            
            # Разницы
            writer.add_scalar('Metrics/train_test_loss_diff', train_loss - test_loss, epoch)
            writer.add_scalar('Metrics/train_test_acc_diff', train_acc - test_acc, epoch)
            writer.add_scalar('Metrics/overfitting_ratio', train_acc / max(test_acc, 0.01), epoch)
            
            # Гиперпараметры
            writer.add_scalar('HParams/label_smoothing', label_smoothing, epoch)
            writer.add_scalar('HParams/step_size', step_size, epoch)
            writer.add_scalar('HParams/gamma', gamma, epoch)
            
            # Dropout rates из названия варианта
            if 'd1_' in variant_name and 'd2_' in variant_name:
                try:
                    d1 = float(variant_name.split('d1_')[1].split('_')[0])
                    d2 = float(variant_name.split('d2_')[1].split('_')[0])
                    writer.add_scalar('HParams/dropout_1', d1, epoch)
                    writer.add_scalar('HParams/dropout_2', d2, epoch)
                except:
                    pass
        
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
            'variant': variant_name,
            'experiment': exp_name,
            'hyperparameters': hp,
            'best_accuracy': best_acc,
            'training_time': training_time,
            'history': history
        }, f, indent=2, default=str)
    
    if writer:
        # Конвертируем hyperparameters в JSON-сериализуемый формат
        hparams = {
            'model': variant_name,
            'learning_rate': float(hp.get('learning_rate', 0.005)),
            'momentum': float(hp.get('momentum', 0.9)),
            'weight_decay': float(hp.get('weight_decay', 1e-5)),
            'batch_size': int(hp.get('batch_size', 128)),
            'epochs': int(epochs),
            'label_smoothing': float(hp.get('label_smoothing', 0.1))
        }
        logger.log_hyperparameters(hparams, {'Final/best_accuracy': float(best_acc)})
        logger.close()
    
    print(f"✅ Saved: {exp_dir}/")
    
    return best_acc


def run_experiment_1(config, device, tb_dir, max_combinations=None, epochs_override=None):
    """Эксперимент 1: Перебор Dropout"""
    print("\n" + "="*70)
    print(" ЭКСПЕРИМЕНТ 1: ПЕРЕБОР DROPOUT")
    print("="*70)
    
    combinations = generate_dropout_combinations(config)
    
    if max_combinations and len(combinations) > max_combinations:
        import random
        combinations = random.sample(combinations, max_combinations)
        print(f"Случайная выборка: {max_combinations} из {len(combinations)}")
    
    print(f"Всего комбинаций: {len(combinations)}")
    
    results = []
    for i, variant in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}]")
        acc = train_single_variant(variant, "exp1_dropout", device, tb_dir, epochs_override)
        results.append({'name': variant['name'], 'accuracy': acc})
    
    # Сортировка результатов
    print("\n" + "="*70)
    print(" РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА 1 (Dropout)")
    print("="*70)
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True)[:10]:
        print(f"  {r['name']:<40} {r['accuracy']:.2f}%")
    
    return results


def run_experiment_2(config, device, tb_dir, max_combinations=None, epochs_override=None):
    """Эксперимент 2: Перебор Weight Decay"""
    print("\n" + "="*70)
    print(" ЭКСПЕРИМЕНТ 2: ПЕРЕБОР WEIGHT DECAY")
    print("="*70)
    
    combinations = generate_weight_decay_combinations(config)
    
    if max_combinations and len(combinations) > max_combinations:
        import random
        combinations = random.sample(combinations, max_combinations)
    
    print(f"Всего комбинаций: {len(combinations)}")
    
    results = []
    for i, variant in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}]")
        acc = train_single_variant(variant, "exp2_weight_decay", device, tb_dir, epochs_override)
        results.append({'name': variant['name'], 'accuracy': acc})
    
    print("\n" + "="*70)
    print(" РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА 2 (Weight Decay)")
    print("="*70)
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"  {r['name']:<40} {r['accuracy']:.2f}%")
    
    return results


def run_experiment_3(config, device, tb_dir, max_combinations=None, epochs_override=None):
    """Эксперимент 3: Перебор Аугментации"""
    print("\n" + "="*70)
    print(" ЭКСПЕРИМЕНТ 3: ПЕРЕБОР АУГМЕНТАЦИИ")
    print("="*70)
    
    combinations = generate_augmentation_combinations(config)
    
    if max_combinations and len(combinations) > max_combinations:
        import random
        combinations = random.sample(combinations, max_combinations)
        print(f"Случайная выборка: {max_combinations} из {len(combinations)}")
    
    print(f"Всего комбинаций: {len(combinations)}")
    
    results = []
    for i, variant in enumerate(combinations):
        print(f"\n[{i+1}/{len(combinations)}]")
        acc = train_single_variant(variant, "exp3_augmentation", device, tb_dir, epochs_override)
        results.append({'name': variant['name'], 'accuracy': acc, 'level': variant.get('level', 0)})
    
    print("\n" + "="*70)
    print(" РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА 3 (Augmentation)")
    print("="*70)
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"  Уровень {r['level']}: {r['name']:<30} {r['accuracy']:.2f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Запуск экспериментов с перебором')
    parser.add_argument('--config', default='configs/independent_experiments.yaml', help='Путь к конфигурации')
    parser.add_argument('--exp', type=int, choices=[1, 2, 3, 0], default=0, help='Номер эксперимента (0=все)')
    parser.add_argument('--tb-dir', default='tensorboard', help='Папка для TensorBoard')
    parser.add_argument('--max', type=int, default=None, help='Максимум комбинаций на эксперимент')
    parser.add_argument('--epochs', type=int, default=None, help='Количество эпох (переопределить)')
    parser.add_argument('--dry-run', action='store_true', help='Только показать что будет')
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = load_config(args.config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"TensorBoard dir: {args.tb_dir}\n")
    
    if args.dry_run:
        from scripts.run_independent_experiments import print_experiment_summary
        for i in [1, 2, 3]:
            if args.exp == 0 or args.exp == i:
                print_experiment_summary(config, i)
        return
    
    # Запуск экспериментов
    if args.exp == 0 or args.exp == 1:
        run_experiment_1(config, device, args.tb_dir, args.max, args.epochs)
    
    if args.exp == 0 or args.exp == 2:
        run_experiment_2(config, device, args.tb_dir, args.max, args.epochs)
    
    if args.exp == 0 or args.exp == 3:
        run_experiment_3(config, device, args.tb_dir, args.max, args.epochs)
    
    print("\n" + "="*70)
    print(" ВСЕ ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ")
    print("="*70)
    print(f"\n📊 TensorBoard: tensorboard --logdir {args.tb_dir}/")


if __name__ == '__main__':
    main()

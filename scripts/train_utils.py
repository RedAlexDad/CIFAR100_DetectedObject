#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Функции для обучения нейронных сетей с TensorBoard
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
import matplotlib.pyplot as plt


def train_model(model, train_X, train_y, test_X, test_y, config, device):
    """
    Обучение модели (базовая версия без TensorBoard)
    """
    print(f"\nНАЧАЛО ОБУЧЕНИЯ")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Device: {device}")
    print()
    
    # Подготовка данных
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
    
    for epoch in range(config['epochs']):
        # Тренировка
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
        
        # Валидация
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
        
        # Сохранение истории
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(val_loss_avg)
        history['test_acc'].append(val_acc)
        
        if scheduler:
            scheduler.step()
        
        # Вывод прогресса
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Эпоха [{epoch+1}/{config['epochs']}] | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {val_loss_avg:.4f}, Test Acc: {val_acc:.2f}%")
        
        # Сохранение лучшей модели
        if val_acc > best_test_acc:
            best_test_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    training_time = time.time() - start_time
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\nОбучение завершено за {training_time:.2f} сек")
    print(f"Лучшая точность на тесте: {best_test_acc:.2f}%")
    
    return model, history, training_time, best_test_acc


def train_model_with_tensorboard(model, train_X, train_y, test_X, test_y, config, device, writer=None):
    """
    Обучение модели с логированием в TensorBoard
    
    Args:
        model: Нейронная сеть
        train_X, train_y: Тренировочные данные
        test_X, test_y: Тестовые данные
        config: Конфигурация обучения
        device: Устройство
        writer: TensorBoard SummaryWriter (опционально)
    
    Returns:
        model, history, training_time, best_accuracy
    """
    print(f"\nНАЧАЛО ОБУЧЕНИЯ")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Device: {device}")
    if writer:
        print(f"  TensorBoard: ✅ Включено")
    print()
    
    # Подготовка данных
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
    
    for epoch in range(config['epochs']):
        # Тренировка
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
            
            # 🔥 TENSORBOARD: Batch metrics
            if writer:
                global_step = epoch * steps_per_epoch + batch_idx
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                accuracy = (labels == predicted).float().mean().item() * 100
                writer.add_scalar('Accuracy/train_batch', accuracy, global_step)
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Валидация
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
        
        # Сохранение истории
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(val_loss_avg)
        history['test_acc'].append(val_acc)
        
        if scheduler:
            scheduler.step()
        
        # 🔥 TENSORBOARD: Epoch metrics
        if writer:
            writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            writer.add_scalar('Loss/val_epoch', val_loss_avg, epoch)
            writer.add_scalar('Accuracy/train_epoch', train_acc, epoch)
            writer.add_scalar('Accuracy/val_epoch', val_acc, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Сохранение лучшей модели
        if val_acc > best_test_acc:
            best_test_acc = val_acc
            best_model_state = model.state_dict().copy()
            
            # 🔥 TENSORBOARD: Best accuracy
            if writer:
                writer.add_scalar('Best_Accuracy', best_test_acc, epoch)
        
        # Вывод прогресса
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Эпоха [{epoch+1}/{config['epochs']}] | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {val_loss_avg:.4f}, Test Acc: {val_acc:.2f}%")
    
    training_time = time.time() - start_time
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\nОбучение завершено за {training_time:.2f} сек")
    print(f"Лучшая точность на тесте: {best_test_acc:.2f}%")
    
    return model, history, training_time, best_test_acc


def plot_learning_history(history, output_path=None, show_plot=True):
    """
    Визуализация истории обучения

    Args:
        history: Словарь с историей обучения
        output_path: Путь для сохранения графика
        show_plot: Показывать график (False = только сохранение)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Точность
    ax1.plot(history['train_acc'], label='Train Acc', linewidth=2)
    ax1.plot(history['test_acc'], label='Test Acc', linewidth=2)
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Точность (%)')
    ax1.set_title('Точность обучения')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Потери
    ax2.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax2.plot(history['test_loss'], label='Test Loss', linewidth=2)
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Потери')
    ax2.set_title('Функция потерь')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Графики сохранены в {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

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


# ============================================================================
# TRANSFER LEARNING - ЗАМОРОЗКА/РАЗМОРОЗКА ВЕСОВ
# ============================================================================

def freeze_model_weights(model, exclude_layers=None):
    """
    Заморозить веса модели, кроме указанных слоёв
    
    Args:
        model: PyTorch модель
        exclude_layers: Список имён слоёв для исключения из заморозки
    
    Returns:
        Количество замороженных параметров
    """
    frozen_count = 0
    exclude_layers = exclude_layers or []
    
    for name, param in model.named_parameters():
        excluded = any(excl in name for excl in exclude_layers)
        if not excluded:
            param.requires_grad = False
            frozen_count += param.numel()
    
    return frozen_count


def unfreeze_model_weights(model):
    """
    Разморозить все веса модели
    
    Args:
        model: PyTorch модель
    
    Returns:
        Количество размороженных параметров
    """
    unfrozen_count = 0
    
    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad = True
            unfrozen_count += param.numel()
    
    return unfrozen_count


def unfreeze_later_layers(model, num_layers=2):
    """
    Разморозить последние num_layers слоёв backbone
    
    Args:
        model: Модель с атрибутами layer1, layer2, layer3, layer4 (ResNet)
               или features (MobileNet)
        num_layers: Количество слоёв для разморозки (с конца)
    
    Returns:
        Количество размороженных параметров
    """
    unfrozen_count = 0
    
    # Для ResNet
    if hasattr(model, 'layer4'):
        layers_map = {
            1: ['layer4'],
            2: ['layer4', 'layer3'],
            3: ['layer4', 'layer3', 'layer2'],
            4: ['layer4', 'layer3', 'layer2', 'layer1'],
        }
        layers_to_unfreeze = layers_map.get(min(num_layers, 4), ['layer4'])
        
        for layer_name in layers_to_unfreeze:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                for param in layer.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        unfrozen_count += param.numel()
    
    # Для MobileNet
    elif hasattr(model, 'features'):
        features = list(model.features)
        layers_to_unfreeze = features[-num_layers:]
        
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen_count += param.numel()
    
    return unfrozen_count


def count_trainable_params(model):
    """Подсчёт обучаемых параметров"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    """Подсчёт общего количества параметров"""
    return sum(p.numel() for p in model.parameters())


def print_transfer_learning_summary(model, model_name, phase='frozen'):
    """
    Вывод информации о модели при transfer learning
    
    Args:
        model: PyTorch модель
        model_name: Название модели
        phase: 'frozen' или 'unfrozen'
    """
    total_params = count_total_params(model)
    trainable_params = count_trainable_params(model)
    frozen_params = total_params - trainable_params
    
    print(f"\n{'='*70}")
    print(f" TRANSFER LEARNING: {model_name} ({phase})")
    print(f"{'='*70}")
    print(f"  Всего параметров:    {total_params:,}")
    print(f"  Обучаемых:           {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"  Заморожено:          {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
    print(f"{'='*70}\n")


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


# ============================================================================
# TRANSFER LEARNING - ДВУХЭТАПНОЕ ОБУЧЕНИЕ
# ============================================================================

def train_model_transfer_learning(
    model,
    train_X, train_y,
    test_X, test_y,
    config,
    device,
    writer=None,
    unfreeze_after=None,
    unfreeze_layers=2
):
    """
    Обучение с Transfer Learning (два этапа: замороженная модель + fine-tuning)
    
    Args:
        model: Модель для transfer learning
        train_X, train_y: Тренировочные данные
        test_X, test_y: Тестовые данные
        config: Конфигурация обучения
        device: Устройство
        writer: TensorBoard SummaryWriter
        unfreeze_after: Номер эпохи для разморозки (None = не размораживать)
        unfreeze_layers: Количество слоёв для разморозки
    
    Returns:
        model, history, training_time, best_accuracy, phase_accuracies
    """
    print(f"\n{'='*70}")
    print(" TRANSFER LEARNING - ОБУЧЕНИЕ")
    print(f"{'='*70}")
    print(f"  Learning rate: {config['lr']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['epochs']}")
    if unfreeze_after:
        print(f"  Разморозка после эпохи: {unfreeze_after}")
        print(f"  Количество слоёв для разморозки: {unfreeze_layers}")
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
    
    criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.0))
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['lr'],
        momentum=config.get('momentum', 0.9),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    
    scheduler = None
    if config.get('use_scheduler', False):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_test_acc = 0.0
    best_model_state = None
    phase_accuracies = {'frozen': 0.0, 'fine_tuned': 0.0}
    
    start_time = time.time()
    is_frozen = True
    
    for epoch in range(config['epochs']):
        # Проверка на разморозку
        if unfreeze_after and epoch == unfreeze_after and is_frozen:
            print(f"\n{'='*70}")
            print(f" ЭПОХА {epoch}: РАЗМОРОЗКА ВЕСОВ (Fine-tuning)")
            print(f"{'='*70}")
            
            # Сохраняем точность замороженной модели
            phase_accuracies['frozen'] = history['test_acc'][-1] if history['test_acc'] else 0.0
            
            # Размораживаем слои
            unfreeze_later_layers(model, unfreeze_layers)
            
            # Создаём новый оптимизатор с меньшим learning rate для fine-tuning
            fine_tuning_lr = config.get('fine_tuning_lr', config['lr'] / 10)
            optimizer = optim.SGD(
                model.parameters(),
                lr=fine_tuning_lr,
                momentum=config.get('momentum', 0.9),
                weight_decay=config.get('weight_decay', 1e-4)
            )
            
            is_frozen = False
            print(f"  Fine-tuning learning rate: {fine_tuning_lr}")
            print_transfer_learning_summary(model, 'model', phase='unfrozen')
        
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
            
            # TensorBoard: batch metrics
            if writer and epoch < 5:
                global_step = epoch * len(train_loader) + train_loader.index((inputs, labels)) if (inputs, labels) in train_loader else epoch
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
        
        # TensorBoard: epoch metrics
        if writer:
            writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            writer.add_scalar('Loss/val_epoch', val_loss_avg, epoch)
            writer.add_scalar('Accuracy/train_epoch', train_acc, epoch)
            writer.add_scalar('Accuracy/val_epoch', val_acc, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Лучшая точность
            if val_acc > best_test_acc:
                writer.add_scalar('Best_Accuracy', val_acc, epoch)
        
        # Сохранение лучшей модели
        if val_acc > best_test_acc:
            best_test_acc = val_acc
            best_model_state = model.state_dict().copy()
            
            if writer:
                writer.add_scalar('Best_Accuracy', best_test_acc, epoch)
        
        # Вывод прогресса
        if (epoch + 1) % 10 == 0 or epoch == 0 or (unfreeze_after and epoch == unfreeze_after):
            phase_marker = "[FROZEN]" if is_frozen else "[FINE-TUNING]"
            print(f"Эпоха [{epoch+1}/{config['epochs']}] {phase_marker} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Test Loss: {val_loss_avg:.4f}, Test Acc: {val_acc:.2f}%")
    
    training_time = time.time() - start_time
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Сохраняем точность fine-tuned модели
    if not is_frozen:
        phase_accuracies['fine_tuned'] = best_test_acc
    else:
        phase_accuracies['frozen'] = best_test_acc
    
    print(f"\n{'='*70}")
    print(" TRANSFER LEARNING - РЕЗУЛЬТАТЫ")
    print(f"{'='*70}")
    print(f"  Обучение завершено за {training_time:.2f} сек")
    print(f"  Лучшая точность на тесте: {best_test_acc:.2f}%")
    if unfreeze_after:
        print(f"\n  Точность замороженной модели: {phase_accuracies['frozen']:.2f}%")
        print(f"  Точность после fine-tuning: {phase_accuracies['fine_tuned']:.2f}%")
        if phase_accuracies['fine_tuned'] > phase_accuracies['frozen']:
            improvement = phase_accuracies['fine_tuned'] - phase_accuracies['frozen']
            print(f"  Улучшение после разморозки: +{improvement:.2f}%")
    print(f"{'='*70}\n")
    
    return model, history, training_time, best_test_acc, phase_accuracies

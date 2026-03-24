#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logger класс для удобной работы с TensorBoard
"""

import os
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """Класс для логирования в TensorBoard"""
    
    def __init__(self):
        self.writer = None
        self.log_dir = None
        self.experiment_name = None
        self.unique_experiment_name = None
    
    def initialize_log_dir(self, experiment_name):
        """
        Создает уникальную директорию для эксперимента и инициализирует SummaryWriter.
        
        Args:
            experiment_name: Название эксперимента
        
        Returns:
            SummaryWriter: Инициализированный writer
        """
        self.experiment_name = experiment_name
        self.log_dir = f'runs/{experiment_name}'
        
        # Проверка существования директории и генерация нового имени
        i = 0
        while os.path.exists(self.log_dir):
            self.log_dir = f'runs/{experiment_name}_{i}'
            i += 1
        
        os.makedirs(self.log_dir)
        print(f"📊 Директория для эксперимента '{experiment_name}': {self.log_dir}")
        
        # Сохраняем уникальное имя
        self.unique_experiment_name = os.path.basename(self.log_dir)
        
        # Создаем writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"   tensorboard --logdir {self.log_dir}")
        
        return self.writer
    
    def initialize_log_dir_with_dir(self, exp_dir):
        """
        Инициализирует SummaryWriter с готовой директорией.
        
        Args:
            exp_dir: Готовая директория для логов
        
        Returns:
            SummaryWriter: Инициализированный writer
        """
        self.log_dir = exp_dir
        self.experiment_name = os.path.basename(exp_dir)
        self.unique_experiment_name = self.experiment_name
        
        # Создаем writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"📊 TensorBoard: {self.log_dir}")
        print(f"   tensorboard --logdir {self.log_dir}")
        
        return self.writer
    
    def log_model_graph(self, model, input_size=(32, 32, 3), device='cpu'):
        """
        Визуализирует граф модели в TensorBoard.
        
        Args:
            model: PyTorch модель
            input_size: Размер входного тензора (H, W, C)
            device: Устройство для dummy input
        """
        if not self.writer:
            print("⚠️  Writer не инициализирован!")
            return
        
        # Создаем dummy input на CPU для совместимости
        dummy_input = torch.randn(1, *input_size).to(device)
        
        try:
            # Добавляем граф
            self.writer.add_graph(model, dummy_input)
            print(f"✅ Граф модели добавлен: {self.log_dir}")
        except Exception as e:
            print(f"⚠️  Не удалось добавить граф: {e}")
    
    def log_scalars(self, tag_scalar_dict, global_step):
        """
        Логирует несколько скаляров одновременно.
        
        Args:
            tag_scalar_dict: dict {'tag': value}
            global_step: Шаг обучения
        """
        if not self.writer:
            return
        
        for tag, scalar in tag_scalar_dict.items():
            self.writer.add_scalar(tag, scalar, global_step)
    
    def log_metrics(self, phase, metrics, epoch):
        """
        Логирует метрики для фазы (train/test).
        
        Args:
            phase: 'train' или 'test'
            metrics: dict {'loss': value, 'accuracy': value}
            epoch: Номер эпохи
        """
        if not self.writer:
            return
        
        self.writer.add_scalar(f'Loss/{phase}_epoch', metrics.get('loss', 0), epoch)
        self.writer.add_scalar(f'Accuracy/{phase}_epoch', metrics.get('accuracy', 0), epoch)
    
    def log_hyperparameters(self, hparams, metrics):
        """
        Логирует гиперпараметры и финальные метрики.
        
        Args:
            hparams: dict гиперпараметров
            metrics: dict финальных метрик
        """
        if not self.writer:
            return
        
        self.writer.add_hparams(hparams, metrics)
        print(f"✅ Гиперпараметры logged: {list(hparams.keys())}")
    
    def log_text(self, tag, text, global_step=0):
        """
        Логирует текст (архитектура модели, конфиг).
        
        Args:
            tag: Тег для текста
            text: Текст для логирования
            global_step: Шаг
        """
        if not self.writer:
            return
        
        self.writer.add_text(tag, text, global_step)
    
    def log_confusion_matrix(self, class_names, true_labels, pred_labels, global_step=0):
        """
        Логирует confusion matrix.
        
        Args:
            class_names: Список названий классов
            true_labels: Истинные метки
            pred_labels: Предсказанные метки
            global_step: Шаг
        """
        if not self.writer:
            return
        
        from sklearn.metrics import confusion_matrix
        import numpy as np
        
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Нормализация
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Создаем figure
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar(im)
        
        tick_marks = np.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        
        # Добавляем значения
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f'{cm_normalized[i, j]:.1f}%', 
                       ha='center', va='center', color='red' if cm_normalized[i, j] > 50 else 'black')
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        self.writer.add_figure('Confusion_Matrix', fig, global_step)
        plt.close()
    
    def close(self):
        """Закрывает writer"""
        if self.writer:
            self.writer.close()
            print(f"✅ TensorBoard writer закрыт: {self.log_dir}")


# Импортируем matplotlib только если нужно
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    pass

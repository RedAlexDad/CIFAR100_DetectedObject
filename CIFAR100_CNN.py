import os
import uuid
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

import onnx
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

class CIFAR100Trainer:
    def __init__(self, classes, batch_size=128, lr_rate=1e-4, patience=2, factor=0.9, dropout_rate=0.5, experiment_name='cifar100_cnn', device=None):
        self.classes = classes
        self.batch_size = batch_size
        self.lr_rate = lr_rate
        self.dropout_rate = dropout_rate
        self.criterion = nn.CrossEntropyLoss()
        self.patience = patience
        self.factor = factor
        self.device = self.get_device(device)
        self.history = {"epoch": [], "loss": []}  # История обучения
        # Загрузка названий классов
        self.class_names = self.load_class_names()
        # Инициализация логгера
        self.writer = self.initialize_log_dir(experiment_name)

        # Чтение тренировочной выборки (обучающих данных)
        with open('cifar-100-python/train', 'rb') as f:
            data_train = pickle.load(f, encoding='latin1')

        # Чтение тестовой выборки (тестовых данных)
        with open('cifar-100-python/test', 'rb') as f:
            data_test = pickle.load(f, encoding='latin1')

        # Фильтрация данных и создание датасетов
        self.train_dataset = self.prepare_data(data_train, 'train')
        self.test_dataset = self.prepare_data(data_test, 'test')

        # Загрузка данных в батчи
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def initialize_log_dir(self, experiment_name):
        """Создает уникальную директорию для эксперимента и инициализирует SummaryWriter."""
        self.experiment_name = experiment_name  # Сохраняем название эксперимента
        self.log_dir = f'logs/{experiment_name}'  # Базовый путь для логов
        # Проверка существования директории и генерация нового имени, если необходимо
        i = 0
        while os.path.exists(self.log_dir):
            self.log_dir = f'logs/{experiment_name}_{i}'
            i += 1
        os.makedirs(self.log_dir)   # Создаем директорию, если она не существует
        # Вывод информации о названии эксперимента
        print(f"Директория для эксперимента '{experiment_name}' инициализирована по пути: {self.log_dir}")
        # Сохраняем уникальное имя для использования в save_model
        self.unique_experiment_name = os.path.basename(self.log_dir)  

        return SummaryWriter(log_dir=self.log_dir) # Возвращаем инициализированный SummaryWriter

    def log_model_graph(self, model):
        """Визуализирует граф модели в TensorBoard."""
        dummy_input = torch.randn(1, 3, 32, 32).to('cpu')  # Размер входа для CIFAR100 (N, C, H, W)
        
        self.writer.add_graph(model, dummy_input)  # Добавляем граф модели
        print(f"Граф модели добавлен в лог: {self.writer.log_dir}")

    def log_hparams_and_metrics(self, accuracy):
        hparams = {
            'batch_size': self.batch_size,
            'learning_rate': self.lr_rate,
            'dropout_rate': self.dropout_rate,
            'epochs': self.epochs
        }
        # Получение текущего времени в читаемом формате
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{timestamp}"  # Читаемое имя
    
        # Логирование гиперпараметров и метрик
        self.writer.add_hparams(hparams, { 
            'accuracy': accuracy,
        }, run_name=run_name)
        
        self.writer.close()
        
    def close_writer(self):
        """Закрывает SummaryWriter."""
        self.writer.close()
        print("SummaryWriter закрыт.")

    def load_class_names(self):
        """Загружает названия классов из мета-файла."""
        with open('cifar-100-python/meta', 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
        return meta['fine_label_names']

    def prepare_data(self, data, part):
        """Фильтрация данных и преобразование в TensorDataset."""
        X = data['data'].reshape(-1, 3, 32, 32)  # Данные в формате NCHW
        y = np.array(data['fine_labels'])

        mask = np.isin(y, self.classes)
        X = X[mask].copy()
        y = y[mask].copy()
        y = np.unique(y, return_inverse=1)[1]

        tensor_x = torch.Tensor(X)
        tensor_y = F.one_hot(torch.Tensor(y).to(torch.int64), num_classes=len(self.classes)) / 1.
        dataset = TensorDataset(tensor_x, tensor_y)
        return dataset
        
    def train(self, model, epochs=10, max_early_stopping_counter=100):
        self.epochs = epochs
        early_stopping_counter = 0
        best_loss = float('inf')
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.patience, factor=self.factor)

        self.history = {"epoch": [], "loss": []}  # Инициализация истории

        for epoch in range(epochs):
            model.train()  # Установка модели в режим обучения
            running_loss = 0.0

            progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit="batch")
            for batch_idx, (data, target) in enumerate(progress_bar):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                # Вызов метода forward() модели
                output = model.forward(data) 
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # Получение текущего значения learning rate
                current_learning_rate = optimizer.param_groups[0]['lr']
                progress_bar.set_postfix(loss=f"{loss:.6f}", lr=f"{current_learning_rate:.6f}")
                
            average_loss = running_loss / len(self.train_loader)
            self.history["epoch"].append(epoch + 1)
            self.history["loss"].append(average_loss)  # Сохранение средней потери
    
            # Логирование потерь в TensorBoard
            self.writer.add_scalar('Loss/train', average_loss, epoch)
            self.writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
            print(f'Epoch: {epoch+1}/{epochs}, Loss: {running_loss/len(self.train_loader):.6f}')
            
            # Обновляем планировщик в каждой эпохе
            self.scheduler.step(average_loss)

            # Early Stopping
            if average_loss < best_loss:
                best_loss = average_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= max_early_stopping_counter:  # Параметр patience для Early Stopping
                print("Early stopping activated.")
                break

    def evaluate(self, model):
        model = model.to(self.device)
        model.eval()  # Установка модели в режим оценки
        
        # Обрабатываем обе выборки: тренировочную и тестовую
        dataloaders = {'train': self.train_loader, 'test': self.test_loader}
        
        for part in ['train', 'test']:
            all_predicted = []
            all_target = []
            with torch.no_grad():
                for data, target in dataloaders[part]:
                    data, target = data.to(self.device), target.to(self.device)

                    output = model(data)
                    
                    # Получаем предсказания
                    _, predicted = torch.max(output.data, 1)
                    
                    # Преобразуем target обратно в одномерный тензор
                    target_indices = torch.argmax(target, dim=1)
                    
                    all_predicted.extend(predicted.cpu().numpy())
                    all_target.extend(target_indices.cpu().numpy())
            
            # Преобразуем списки в массивы NumPy
            all_predicted = np.array(all_predicted)
            all_target = np.array(all_target)
            
            # Выводим наименования классов для выбранных индексов
            selected_class_names = [self.class_names[i] for i in self.classes]
            
            # Выводим отчет о классификации
            print(f"Classification report for {part} dataset:")
            report = classification_report(all_target, all_predicted, target_names=selected_class_names, zero_division=0, digits=4)
            print(report)
            print('-' * 50)
            
        # Вычисляем и логируем точность
        accuracy = np.sum(all_predicted == all_target) / len(all_target)
        self.writer.add_scalar(f'Accuracy/{part}', accuracy, len(self.history["epoch"]) - 1)
        
        self.log_hparams_and_metrics(accuracy)

    def save_model(self, model, path='models/'):
        # Формируем название модели на основе уникального имени эксперимента
        model_name = f"{self.unique_experiment_name}_model.onnx"  # Используем уникальное имя эксперимента
        
        # Создаем полный путь для сохранения модели
        filepath = os.path.join(path, model_name)  # Корректное соединение пути и имени файла
        
        # Убедимся, что директория для сохранения существует
        os.makedirs(path, exist_ok=True)  # Создаем директорию, если она не существует

        # Входной тензор для модели (с учетом правильного порядка: (N, C, H, W))
        dummy_input = torch.randn(1, 3, 32, 32, requires_grad=True).to(self.device)

        # Экспорт модели
        torch.onnx.export(
            model,               # модель
            dummy_input,         # входной тензор
            filepath,            # куда сохранить
            export_params=True,  # сохраняет веса обученных параметров внутри файла модели
            opset_version=9,    # версия ONNX
            do_constant_folding=True,  # следует ли выполнять укорачивание констант для оптимизации
            input_names=['input'],    # имя входного слоя
            output_names=['output'],   # имя выходного слоя
            dynamic_axes={'input': {0: 'batch_size'},    # динамичные оси
                          'output': {0: 'batch_size'}}
        )
        
        # Вывод информации о сохраненной модели
        print(f"Model saved as: {model_name}")

    @staticmethod
    def get_device(select=None):
        """
        Определение устройства для вычислений (CPU или GPU).
        Args:
            select (str, optional): Выбор устройства ('cpu', 'cuda'). По умолчанию None.
            torch.device: Устройство для вычислений.
        """
        if select is None or select == 'cuda':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device('cpu')

    def plot_training_history(self, window_size = 5):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
        # Преобразуем список эпох для оси X
        epochs = self.history["epoch"]
        loss = self.history["loss"]
            
        # Вычисляем скользящее среднее
        smoothed_loss = np.convolve(loss, np.ones(window_size)/window_size, mode='valid')
    
        # Первый график: Полная история с скользящим средним
        axs[0].plot(epochs, loss, marker='o', linestyle='-', color='b', markersize=5, label='Loss')
        axs[0].plot(epochs[window_size-1:], smoothed_loss, color='orange', label='Smoothed Loss (Moving Average)')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Average Loss')
        axs[0].set_title('Loss Function (Full History)')
        axs[0].grid(True)
        axs[0].legend()
    
        # Второй график: Половина истории с скользящим средним
        mid_index = len(epochs) // 2
        axs[1].plot(epochs[mid_index:], loss[mid_index:], marker='o', linestyle='-', color='b', markersize=5, label='Loss')
        axs[1].plot(epochs[mid_index + window_size - 1:], smoothed_loss[mid_index:], color='orange', label='Smoothed Loss (Moving Average)')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Average Loss')
        axs[1].set_title('Loss Function (Second Half of Training)')
        axs[1].grid(True)
        axs[1].legend()
    
        plt.tight_layout()
        plt.show()
                
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, input):
        device = input.device  # Получаем устройство данных
        self.mean = self.mean.to(device)  # Переносим тензоры на это устройство
        self.std = self.std.to(device)

        x = input / 255.0
        x = x - self.mean[None, :, None, None]  # Убедитесь, что mean/std правильно транслируются
        x = x / self.std[None, :, None, None]
        return x

class CIFAR100ModelCNN(nn.Module):
    def __init__(self, hidden_layers=[64, 128, 256], dropout_rate=0.5, num_classes=100):
        super(CIFAR100ModelCNN, self).__init__()
        
        # Устанавливаем параметры нормализации (они могут меняться в зависимости от ваших данных)
        self.normalize = Normalize([0.5074, 0.4867, 0.4411], [0.2011, 0.1987, 0.2025])
        
        layers = []
        input_channels = 3
        for layer_dim in hidden_layers:
            layers.append(nn.Conv2d(input_channels, layer_dim, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            input_channels = layer_dim

        self.conv_layers = nn.Sequential(*layers)
        self.classifier_conv = nn.Conv2d(hidden_layers[-1], num_classes, kernel_size=1)

    def forward(self, x):
        x = self.normalize(x)  # Применяем нормализацию
        x = self.conv_layers(x)
        x = self.classifier_conv(x)
        x = F.adaptive_avg_pool2d(x, 1)  # Global average pooling
        x = torch.flatten(x, 1)
        return x
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
import torchvision.models as models

class CIFAR100Trainer:
    def __init__(self, classes, batch_size=128, lr_rate=1e-4, criterion=nn.CrossEntropyLoss()):
        self.classes = classes
        self.batch_size = batch_size
        self.lr_rate = lr_rate
        self.device = self.check_device()
        self.criterion = criterion
        self.history = {"epoch": [], "loss": []}  # История обучения
        # Загрузка названий классов
        self.class_names = self.load_class_names()
        
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
        
    def load_class_names(self):
        """Загружает названия классов из мета-файла."""
        with open('cifar-100-python/meta', 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
        return meta['fine_label_names']

    def prepare_data(self, data, part):
        """Фильтрация данных и преобразование в TensorDataset."""
        X = data['data'].reshape(-1, 3, 32, 32)  # Данные в формате NCHW
        X = np.transpose(X, [0, 2, 3, 1]) # NCHW -> NHWC
        y = np.array(data['fine_labels'])

        mask = np.isin(y, self.classes)
        X = X[mask].copy()
        y = y[mask].copy()
        y = np.unique(y, return_inverse=1)[1]

        tensor_x = torch.Tensor(X)
        tensor_y = F.one_hot(torch.Tensor(y).to(torch.int64), num_classes=len(self.classes)) / 1.
        dataset = TensorDataset(tensor_x, tensor_y)
        return dataset
        
    def train(self, model, epochs=10):
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr_rate)
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
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                progress_bar.set_postfix(loss=f"{loss:.6f}")
                
            average_loss = running_loss / len(self.train_loader)
            self.history["epoch"].append(epoch + 1)
            self.history["loss"].append(average_loss)  # Сохранение средней потери

            print(f'Epoch: {epoch+1}/{epochs}, Loss: {running_loss/len(self.train_loader):.6f}')

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

    def save_model(self, model, path='models/'):
        # Генерируем уникальный идентификатор
        unique_id = str(uuid.uuid4())  # Генерация уникального ID
        model_name = f"cifar100_{unique_id}.onnx"  # Формируем название модели
        
        # Создаем полный путь для сохранения модели
        filepath = os.path.join(path, model_name)  # Корректное соединение пути и имени файла
        
        # Убедимся, что директория для сохранения существует
        os.makedirs(path, exist_ok=True)  # Создаем директорию, если она не существует

        # Входной тензор для модели (с учетом правильного порядка: (N, H, W, C))
        dummy_input = torch.randn(1, 32, 32, 3, requires_grad=True).to(self.device)  # (N, H, W, C)
    
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
        print(f"Model UUID: {unique_id}")

    def check_device(self):
        """Проверяет доступность CUDA и возвращает устройство."""
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        print(f'Используемое устройство: {device}')
        return device

    def plot_training_history(self):
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
        # Преобразуем список эпох для оси X
        epochs = self.history["epoch"]
    
        # Первый график: Полная история
        axs[0].plot(epochs, self.history["loss"], marker='o', linestyle='-', color='b', markersize=5, label='Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Average Loss')
        axs[0].set_title('Loss Function (Full History)')
        axs[0].grid(True)
        axs[0].legend()
    
        # Второй график: Половина истории
        mid_index = len(epochs) // 2
        axs[1].plot(epochs[mid_index:], self.history["loss"][mid_index:], marker='o', linestyle='-', color='b', markersize=5, label='Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Average Loss')
        axs[1].set_title('Loss Function (Second Half of Training)')
        axs[1].grid(True)
        axs[1].legend()
    
        plt.tight_layout()
        plt.show()

    def load_class_names(self):
        """Загружает названия классов из мета-файла."""
        with open('cifar-100-python/meta', 'rb') as f:
            meta = pickle.load(f, encoding='latin1')
        return meta['fine_label_names']
    
    def display_images_with_predictions(self, model, image_dir='./images'):
        """Отображает изображения и предсказания модели с процентным соотношением для заданных классов."""
        # Определение преобразований для входных изображений
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Пример нормализации, может потребоваться адаптация
        ])
    
        # Создаем отображение названий классов для выбранных индексов
        selected_class_names = [self.class_names[i] for i in self.classes]
    
        # Получаем список изображений
        images = [img for img in os.listdir(image_dir) if img.endswith(('jpg', 'png', 'jpeg'))]
        predictions = []
        probabilities = []
    
        for img_name in images:
            img_path = os.path.join(image_dir, img_name)
            img = Image.open(img_path).convert('RGB')  # Открываем изображение
            img_tensor = transform(img).unsqueeze(0).to(self.device)  # Применяем преобразования и добавляем размерность батча
            
            # Получаем предсказание модели
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)  # Вычисляем вероятности
                predicted = torch.argmax(probs).item()  # Предсказанный класс
                predictions.append(predicted)  # Сохраняем предсказание
                probabilities.append(probs.squeeze().cpu().numpy())  # Сохраняем вероятности
    
        # Визуализация
        num_images = len(images)
        cols = 3
        rows = (num_images + cols - 1) // cols  # Вычисляем количество строк
        
        fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axs = axs.flatten()  # Упрощаем доступ к осям
    
        for i, img_name in enumerate(images):
            img_path = os.path.join(image_dir, img_name)
            img = Image.open(img_path)
            axs[i].imshow(img)
            
            # Выводим предсказания и вероятности для выбранных классов
            title_text = f'Predicted: {self.class_names[predictions[i]]} (ID: {predictions[i]})\n'
            for cls in self.classes:
                # Получаем индекс и название класса
                if cls < len(probabilities[i]):
                    title_text += f'{selected_class_names[self.classes.index(cls)]}: {probabilities[i][cls] * 100:.2f}%, '
    
            axs[i].set_title(title_text[:-2])  # Убираем последний запятую
            axs[i].axis('off')  # Скрываем оси
    
        # Удаляем пустые оси, если их больше чем изображений
        for j in range(i + 1, len(axs)):
            axs[j].axis('off')
    
        plt.tight_layout()
        plt.show()
        
        
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, input):
        x = input / 255.0
        x = x - self.mean
        x = x / self.std
        return torch.flatten(x, start_dim=1) # nhwc -> nm
    

class CIFAR100Model(nn.Module):
    def __init__(self, hidden_layers=[64, 128, 64], dropout_prob=0.5, num_classes=100):
        super(CIFAR100Model, self).__init__()
        self.norm = Normalize([0.5074, 0.4867, 0.4411], [0.2011, 0.1987, 0.2025])
        
        # Определяем входной размер (например, CIFAR100 имеет изображения размером 32x32x3)
        input_size = 32 * 32 * 3

        # Динамически создаем слои на основе переданного списка hidden_layers
        layers = []
        in_features = input_size
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())  # Активационная функция
            layers.append(nn.Dropout(p=dropout_prob))  # Dropout
            in_features = hidden_size

        # Добавляем последний слой для классификации
        layers.append(nn.Linear(in_features, num_classes))

        # Создаем последовательность слоев
        self.seq = nn.Sequential(*layers)

    def forward(self, input):
        x = self.norm(input)
        return self.seq(x)
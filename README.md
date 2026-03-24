# Проект: Сверточная нейронная сеть для классификации CIFAR-100

## 📁 Структура проекта

```
CIFAR100_DetectedObject/       # Основная папка проекта
├── main.py                    # Главный скрипт для обучения
├── requirements.txt           # Зависимости Python
├── README.md                  # Этот файл
├── configs/                   # Конфигурационные файлы
│   ├── __init__.py
│   ├── config.py              # Основная конфигурация
│   ├── config.yaml            # YAML конфигурация
│   ├── grid_search_config.yaml # Конфигурация для поиска по сетке
│   └── training_config.py     # Конфигурация обучения
│
├── models/                    # Модели нейронных сетей
│   ├── __init__.py
│   └── cnn_models.py          # Определения CNN моделей
│
├── scripts/                   # Вспомогательные скрипты
│   ├── __init__.py
│   ├── data_utils.py          # Загрузка данных, визуализация
│   ├── train_utils.py         # Обучение моделей
│   ├── eval_utils.py          # Оценка, метрики, ONNX
│   ├── grid_search.py         # Поиск по сетке
│   └── augmentation.py        # Аугментация данных
│
├── checkpoints/               # Сохранённые веса моделей
│   ├── cnn_base.pth
│   ├── cnn_medium.pth
│   └── cnn_deep_model.pth
│
├── onnx_models/               # ONNX модели
│   └── cnn_lr2_optimized.onnx
│
├── outputs/                   # Результаты (графики, матрицы)
│   ├── confusion_matrix.png
│   ├── learning_history.png
│   └── training_log.txt
│
├── images/                    # Тестовые изображения
│   ├── bicycle_*.jpg
│   ├── flatfish_*.jpg
│   └── train_*.jpg
│
├── tensorboard/               # Логи TensorBoard
│   └── экспериментальные логи
│
├── archive/                   # Архивы экспериментов
└── cifar-100-python/          # Датасет CIFAR-100
```

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Обучение модели

```bash
# Обучить базовую модель (2 conv слоя) с параметрами по умолчанию
python main.py --model base --train

# Обучить среднюю модель (3 conv слоя) с параметрами по умолчанию
python main.py --model medium --train

# Обучить с кастомными параметрами
python main.py --model medium --train --lr 0.001 --batch-size 64 --epochs 200

# Обучить БЕЗ показа графиков (только сохранение в файлы)
python main.py --model medium --train --save-only

# Только оценка модели
python main.py --model medium --evaluate

# Оценка с конкретным чекпоинтом
python main.py --model base --evaluate --checkpoint checkpoints/cnn_base.pth

# Обучение + оценка + визуализация
python main.py --model medium --train --evaluate --visualize

# Экспорт в ONNX
python main.py --model medium --export

# Обучение с логированием в TensorBoard
python main.py --model medium --train --tb-logs

# Обучение на сервере (без GUI)
python main.py --model medium --train --save-only
```

## 📊 Архитектура моделей

Проект включает несколько архитектур CNN:

- **Base CNN**: 2 сверточных слоя (базовая модель)
- **Medium CNN**: 3 сверточных слоя (улучшенная модель)
- **Deep CNN**: 6 сверточных слоев (глубокая модель)
- **Optimized CNN**: 6 сверточных слоев с BatchNorm и Dropout (оптимизированная модель)

## 📈 TensorBoard интеграция

Проект поддерживает логирование в TensorBoard:

```bash
# Запуск TensorBoard для просмотра логов
tensorboard --logdir runs/exp1

# В Jupyter Notebook:
%load_ext tensorboard
%tensorboard --logdir runs/exp1
```

## 📋 Классы CIFAR-100

Проект настроен для работы с тремя классами CIFAR-100:
- Класс 8: 'bicycle' → 'Велосипед'
- Класс 32: 'flatfish' → 'Камбала' 
- Класс 90: 'train' → 'Поезд'

## ⚙️ Конфигурация

Откройте `configs/config.py` для изменения:
- Гиперпараметров обучения (lr, batch_size, epochs)
- Путей к данным
- Названий классов
- Устройства (cuda/cpu)
- Архитектуры моделей

## 🧪 Эксперименты

Проект поддерживает проведение различных экспериментов:
- Поиск по сетке гиперпараметров
- Аугментация данных
- Независимые эксперименты
- Сравнение архитектур

## 📦 Экспорт модели

Модели могут быть экспортированы в формат ONNX для использования в других фреймворках:

```bash
python main.py --model medium --export
```

## 📊 Метрики и визуализация

Проект предоставляет:
- Матрицу ошибок
- Графики обучения (точность и потери)
- Классификационные отчеты
- Визуализацию данных

## 🎯 Особенности проекта

- Поддержка CUDA для ускоренного обучения
- Автоматическая загрузка датасета CIFAR-100
- Нормализация изображений
- Поддержка различных архитектур CNN
- Интеграция с TensorBoard для мониторинга обучения
- Экспорт моделей в ONNX формат
- Поддержка расписаний скорости обучения (CosineAnnealingLR)

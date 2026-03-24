# Проект: Сверточная нейронная сеть для классификации CIFAR-100

## 📁 Структура проекта

```
CIFAR100_DetectedObject/       # Основная папка проекта
├── main.py                    # Главный скрипт для обучения
├── main_tb.py                 # Альтернативный скрипт с TensorBoard
├── train_simple.py            # Упрощенное обучение с расширенными гиперпараметрами
├── train_from_yaml.py         # Обучение из YAML конфигурации
├── run_all_experiments.py     # Запуск всех экспериментов с перебором параметров
├── requirements.txt           # Зависимости Python
├── README.md                  # Этот файл
├── configs/                   # Конфигурационные файлы
│   ├── __init__.py
│   ├── config.py              # Основная конфигурация
│   ├── config.yaml            # YAML конфигурация
│   ├── grid_search_config.yaml # Конфигурация для поиска по сетке
│   ├── independent_experiments.yaml # Конфигурация независимых экспериментов
│   ├── training_config.py     # Конфигурация обучения
│   └── yaml_config.py         # Загрузчик YAML конфигураций
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
│   ├── run_independent_experiments.py # Запуск независимых экспериментов
│   └── augmentation.py        # Аугментация данных
│
├── checkpoints/               # Сохранённые веса моделей
│   ├── cnn_base.pth
│   ├── cnn_medium.pth
│   ├── cnn_base_model.pth
│   ├── cnn_medium_model.pth
│   ├── cnn_deep_model.pth
│   └── cnn_optimized_model.pth
│
├── onnx_models/               # ONNX модели
│   └── cnn_lr2_optimized.onnx
│
├── outputs/                   # Результаты (графики, матрицы)
│   ├── confusion_matrix.png
│   ├── learning_history.png
│   ├── lr2_confusion_matrix.png
│   ├── lr2_data_visualization.png
│   ├── lr2_learning_history.png
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
│
├── web/                       # Веб-интерфейс для классификации
│   ├── index.html             # HTML интерфейс
│   ├── logic.js               # JavaScript логика
│   ├── styles.css             # Стили
│   └── package.json           # Зависимости веб-приложения
│
├── logger.py                  # Класс для логирования в TensorBoard
├── csv_to_tb.py               # Конвертация CSV в TensorBoard
├── TENSORBOARD_USAGE.md       # Руководство по использованию TensorBoard
├── ЭКСПЕРИМЕНТЫ_ИНСТРУКЦИЯ.md # Инструкция по проведению экспериментов
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

# Обучить глубокую модель (6 conv слоев) с параметрами по умолчанию
python main.py --model deep --train

# Обучить оптимизированную модель (6 conv слоев с BatchNorm и Dropout) с параметрами по умолчанию
python main.py --model optimized --train

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

## 🧪 Эксперименты

Проект поддерживает проведение различных экспериментов:

### Запуск всех экспериментов
```bash
# Запуск всех экспериментов с перебором параметров
python run_all_experiments.py

# Запуск конкретного эксперимента (1-3)
python run_all_experiments.py --exp 1

# Запуск с ограничением количества комбинаций
python run_all_experiments.py --max 10

# Просмотр конфигурации без запуска
python run_all_experiments.py --dry-run
```

### Обучение из YAML конфигурации
```bash
# Обучение из YAML конфигурации
python train_from_yaml.py configs/config.yaml

# Обучение конкретного варианта
python train_from_yaml.py configs/config.yaml --variant dropout_0.2_0.3

# Обучение всех вариантов
python train_from_yaml.py configs/config.yaml --all
```

### Упрощенное обучение с расширенными гиперпараметрами
```bash
# Обучение с пресетом конфигурации
python train_simple.py --model medium --train --preset accurate

# Обучение с переопределенными гиперпараметрами
python train_simple.py --model medium --train --lr 0.001 --momentum 0.9 --epochs 100

# Обучение с аугментацией
python train_simple.py --model medium --train --preset accurate --brightness 0.8 1.2 --contrast 0.8 1.2
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

## 🌐 Веб-интерфейс

Проект включает веб-интерфейс для классификации изображений:

1. Откройте `web/index.html` в браузере
2. Загрузите ONNX модель
3. Загрузите изображение для классификации
4. Выберите классы для предсказания
5. Получите результаты классификации

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
- Подробные метрики по каждому классу

## 🎯 Особенности проекта

- Поддержка CUDA для ускоренного обучения
- Автоматическая загрузка датасета CIFAR-100
- Нормализация изображений
- Поддержка различных архитектур CNN
- Интеграция с TensorBoard для мониторинга обучения
- Экспорт моделей в ONNX формат
- Поддержка расписаний скорости обучения (CosineAnnealingLR)
- Аугментация данных
- Поиск по сетке гиперпараметров
- Поддержка YAML конфигураций
- Веб-интерфейс для классификации
- Поддержка Label Smoothing
- Batch Normalization и Dropout для регуляризации
- Поддержка различных оптимизаторов и schedulers

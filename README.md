# Проект: Сверточная нейронная сеть для классификации CIFAR-100

## 📁 Структура проекта

```
CIFAR100_DetectedObject/       # Основная папка проекта
├── main.py                    # Главный скрипт для обучения
├── main_tb.py                 # Альтернативный скрипт с TensorBoard
├── train_simple.py            # Упрощенное обучение с расширенными гиперпараметрами
├── train_from_yaml.py         # Обучение из YAML конфигурации
├── run_all_experiments.py     # Запуск всех экспериментов с перебором параметров
├── main_lab4.py               # ЛР4: Transfer Learning (ResNet20/MobileNetV2)
├── compare_labs.py            # Сравнение результатов всех ЛР
├── requirements.txt           # Зависимости Python
├── README.md                  # Этот файл
├── configs/                   # Конфигурационные файлы
│   ├── __init__.py
│   ├── config.py              # Основная конфигурация
│   ├── config.yaml            # YAML конфигурация
│   ├── grid_search_config.yaml # Конфигурация для поиска по сетке
│   ├── independent_experiments.yaml # Конфигурация независимых экспериментов
│   ├── training_config.py     # Конфигурация обучения
│   ├── yaml_config.py         # Загрузчик YAML конфигураций
│   └── lab4_config.yaml       # Конфигурация для ЛР4
│
├── models/                    # Модели нейронных сетей
│   ├── __init__.py
│   ├── cnn_models.py          # Определения CNN моделей
│   └── transfer_models.py     # Модели для Transfer Learning (ЛР4)
│
├── scripts/                   # Вспомогательные скрипты
│   ├── __init__.py
│   ├── data_utils.py          # Загрузка данных, визуализация
│   ├── train_utils.py         # Обучение моделей (вкл. transfer learning)
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
│   ├── cnn_optimized_model.pth
│   └── lab4_*.pth             # Модели ЛР4
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
│   ├── lab4_*.png             # Результаты ЛР4
│   └── training_log.txt
│
├── images/                    # Тестовые изображения
│   ├── bicycle_*.jpg
│   ├── flatfish_*.jpg
│   └── train_*.jpg
│
├── tensorboard/               # Логи TensorBoard
│   ├── exp1_dropout/          # Эксперименты с dropout
│   ├── exp2_weight_decay/     # Эксперименты с weight decay
│   ├── exp3_augmentation/     # Эксперименты с аугментацией
│   └── lab4/                  # ЛР4: Transfer Learning
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

---

## 🎓 ЛАБОРАТОРНАЯ РАБОТА №4: TRANSFER LEARNING

### 📋 Задание

Использовать предобученную модель (ResNet20 или MobileNetV2), заморозить веса и дообучить на своих классах CIFAR-100.

**Варианты моделей:**
- **Чётный вариант**: ResNet20
- **Нечетный вариант**: MobileNetV2 x0.5

### 🚀 Быстрый старт

```bash
# Базовое обучение с замороженной моделью (30 эпох)
python main_lab4.py --model resnet20 --train

# Обучение с fine-tuning (разморозка после 10 эпохи)
python main_lab4.py --model resnet20 --train --unfreeze-after 10 --epochs 30

# Сравнение frozen и fine-tuning
python main_lab4.py --model mobilenetv2 --train --compare

# Обучение с TensorBoard
python main_lab4.py --model resnet20 --train --tb-logs --tb-dir runs/lab4_resnet20

# Только оценка модели
python main_lab4.py --model resnet20 --evaluate --checkpoint checkpoints/lab4_resnet20.pth
```

### ⚙️ Опции командной строки

| Опция | Описание | Пример |
|-------|----------|--------|
| `--model` | Модель: `resnet20` или `mobilenetv2` | `--model resnet20` |
| `--train` | Обучить модель | `--train` |
| `--evaluate` | Оценить модель | `--evaluate` |
| `--checkpoint` | Путь к чекпоинту | `--checkpoint checkpoints/lab4_resnet20.pth` |
| `--epochs` | Количество эпох | `--epochs 30` |
| `--lr` | Learning rate | `--lr 0.001` |
| `--batch-size` | Размер батча | `--batch-size 64` |
| `--unfreeze-after` | Эпоха для разморозки | `--unfreeze-after 10` |
| `--unfreeze-layers` | Слоёв для разморозки | `--unfreeze-layers 2` |
| `--fine-tuning-lr` | LR для fine-tuning | `--fine-tuning-lr 0.0001` |
| `--compare` | Сравнить frozen vs fine-tuning | `--compare` |
| `--preset` | Пресет: `base`, `fast`, `accurate` | `--preset accurate` |

### 📊 Этапы обучения

**Этап 1: Замороженная модель (Frozen)**
- Все веса backbone заморожены
- Обучается только классификатор
- Высокая скорость обучения (lr=0.001)

**Этап 2: Fine-tuning (после разморозки)**
- Размораживаются последние N слоёв backbone
- Обучается вся модель
- Низкая скорость обучения (lr=0.0001)

### 📈 Сравнение с предыдущими ЛР

После обучения всех моделей используйте скрипт сравнения:

```bash
# Сравнение результатов всех ЛР
python compare_labs.py

# С ручным вводом результатов ЛР1
python compare_labs.py --lab1-accuracy 85.5 --lab1-k 5

# Сохранение отчёта в Markdown
python compare_labs.py --output outputs/lab_comparison.md
```

### 📝 Контрольные вопросы

1. **Что такое перенос обучения?** — Использование знаний, полученных на одной задаче, для решения другой связанной задачи
2. **Что такое заморозка весов?** — Блокировка обновления параметров модели во время обучения
3. **Что такое fine-tuning?** — Дообучение предобученной модели на новых данных
4. **Оптимизаторы:** Adagrad, RMSProp, Adam — методы адаптивной настройки скорости обучения

---

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

# TensorBoard в train_cnn - Инструкция

## ⚠️ Проблема с TensorFlow

При импорте TensorBoard может появляться предупреждение от TensorFlow:
```
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
```

Это **НЕ ошибка**, просто informational message. TensorFlow используется TensorBoard для некоторых функций.

## ✅ Решение 1: Игнорировать предупреждение

Предупреждение не влияет на работу. Просто продолжайте:

```bash
python main.py --model medium --train --tb-logs
```

## ✅ Решение 2: Использовать отдельный процесс для просмотра

**Обучение с логированием:**
```bash
cd train_cnn
python main.py --model medium --train --tb-logs --tb-dir runs/exp1
```

**Просмотр в отдельном терминале:**
```bash
tensorboard --logdir=runs/exp1
```

Откройте в браузере: `http://localhost:6006`

## ✅ Решение 3: Jupyter Lab (правильный способ)

**Ячейка 1: Обучение**
```python
import warnings
warnings.filterwarnings('ignore')

from torch.utils.tensorboard import SummaryWriter

# Обучение
writer = SummaryWriter(log_dir='runs/exp1')
# ... код обучения ...
writer.close()
```

**Ячейка 2: Просмотр (отдельная ячейка!)**
```python
%load_ext tensorboard
%tensorboard --logdir runs/exp1
```

## 📋 Использование в train_cnn

### Базовое обучение (без TensorBoard)
```bash
python main.py --model base --train
```

### С TensorBoard логированием
```bash
python main.py --model medium --train --tb-logs --tb-dir runs/lr3_exp
```

### С кастомными параметрами
```bash
python main.py --model medium --train --tb-logs --lr 0.001 --epochs 100
```

## 📊 Что логируется в TensorBoard

| Метрика | Тег |
|---------|-----|
| Loss по батчам | `Loss/train_batch` |
| Accuracy по батчам | `Accuracy/train_batch` |
| Loss по эпохам | `Loss/train_epoch`, `Loss/val_epoch` |
| Accuracy по эпохам | `Accuracy/train_epoch`, `Accuracy/val_epoch` |
| Learning Rate | `Learning_rate` |
| Лучшая точность | `Best_Accuracy` |
| Финальные метрики | `Final/test_accuracy`, `Final/test_precision`, etc. |
| Метрики по классам | `Class/Велосипед_precision`, etc. |
| Архитектура модели | `Model_Architecture` (текст) |

## 🚀 Пример полного цикла

```bash
# 1. Обучение с логированием
cd train_cnn
python main.py --model medium --train --tb-logs --tb-dir runs/lr3_final

# 2. В отдельном терминале - просмотр
tensorboard --logdir=runs/lr3_final

# 3. Открыть браузер
# http://localhost:6006
```

## 💡 Советы

1. **Именуйте эксперименты**: `runs/exp1_lr0.01_bs128`
2. **Не логируйте каждый батч** если много данных - замедляет обучение
3. **Закрывайте writer** после обучения
4. **Используйте separate process** для просмотра - не в том же терминале где обучение

## ❓ Частые проблемы

### "ModuleNotFoundError: No module named 'tensorboard'"
```bash
pip install tensorboard
```

### "Port 6006 already in use"
```bash
tensorboard --logdir=runs/exp1 --port 6007
```

### Kernel crashes in Jupyter
- Запускайте `%tensorboard` в **отдельной ячейке**
- Не импортируйте `tensorflow` в коде
- Используйте `warnings.filterwarnings('ignore')`

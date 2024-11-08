import os
import base64  
import numpy as np  
import onnxruntime  
from torchvision import transforms

from io import BytesIO  
from PIL import Image  

from django.shortcuts import render  
from django.core.files.storage import FileSystemStorage  
from django.conf import settings
 
# Загружаем модель в соответствии с вариантом:
# Четный номер в списке группы - cifar100_mobile;
# Нечетный номер в списке группы - cifar100_resnet.
# 
# Необходимо поставить ограничение на 3 класса по варианту
# 
# Класс 1 - номер в группе + номер группы
# Класс 2 - номер в группе + номер группы + 30
# Класс 3 - номер в группе + номер группы + 60 **Где номер группы - 1, 2, 3, 4 и т.д.
# Мой вариант: 
# Класс 1: 4 + 21 = 25
# Класс 2: 4 + 21 + 30 = 55
# Класс 3: 4 + 21 + 60 = 85
# Итого: 25, 55, 115
# Классификации: couch, otter, tank - диван, выдра, аквариум

onnx_model = os.path.join(settings.BASE_DIR.parent, 'cifar100/models/cifar100_CNN_RESNET20.onnx')

# imageClassList = {0: 'apple', 1: 'aquarium_fish', 2: 'baby', 3: 'bear', 4: 'beaver', 5: 'bed', 6: 'bee', 7: 'beetle', 8: 'bicycle', 9: 'bottle', 
# 10: 'bowl', 11: 'boy', 12: 'bridge', 13: 'bus', 14: 'butterfly', 15: 'camel', 16: 'can', 17: 'castle', 18: 'caterpillar', 19: 'cattle', 
# 20: 'chair', 21: 'chimpanzee', 22: 'clock', 23: 'cloud', 24: 'cockroach', 25: 'couch', 26: 'cra', 27: 'crocodile', 28: 'cup', 29: 'dinosaur', 
# 30: 'dolphin', 31: 'elephant', 32: 'flatfish', 33: 'forest', 34: 'fox', 35: 'girl', 36: 'hamster', 37: 'house', 38: 'kangaroo', 39: 'keyboard', 
# 40: 'lamp', 41: 'lawn_mower', 42: 'leopard', 43: 'lion', 44: 'lizard', 45: 'lobster', 46: 'man', 47: 'maple_tree', 48: 'motorcycle', 49: 'mountain', 
# 50: 'mouse', 51: 'mushroom', 52: 'oak_tree', 53: 'orange', 54: 'orchid', 55: 'otter', 56: 'palm_tree', 57: 'pear', 58: 'pickup_truck', 59: 'pine_tree', 
# 60: 'plain', 61: 'plate', 62: 'poppy', 63: 'porcupine', 64: 'possum', 65: 'rabbit', 66: 'raccoon', 67: 'ray', 68: 'road', 69: 'rocket', 
# 70: 'rose', 71: 'sea', 72: 'seal', 73: 'shark', 74: 'shrew', 75: 'skunk', 76: 'skyscraper', 77: 'snail', 78: 'snake', 79: 'spider', 
# 80: 'squirrel', 81: 'streetcar', 82: 'sunflower', 83: 'sweet_pepper', 84: 'table', 85: 'tank', 86: 'telephone', 87: 'television', 88: 'tiger', 89: 'tractor', 
# 90: 'train', 91: 'trout', 92: 'tulip', 93: 'turtle', 94: 'wardrobe', 95: 'whale', 96: 'willow_tree', 97: 'wolf', 98: 'woman', 99: 'worm'}  
# Сюда указать классы  

imageClassList = {25: 'Диван', 55: 'Выдра', 85: 'Танк'}

# Главная страница
def scoreImagePage(request):  
    return render(request, 'scorepage.html')  

# Обработка загрузки и предсказания изображения
def predictImage(request):
    if request.method == 'POST' and 'filePath' in request.FILES:
        fileObj = request.FILES['filePath']

        # Открытие изображения в памяти, без сохранения на диск
        img = Image.open(fileObj).convert("RGB")
        img_uri = to_data_uri(img)

        # Вызов функции предсказания
        scorePrediction = predictImageData(img)

        # Если предсказание не найдено в известных классах, устанавливаем "not_class"
        if not scorePrediction:
            scorePrediction = "Не распознанен класс"

        context = {'scorePrediction': scorePrediction, 'img_uri': img_uri}
        return render(request, 'scorepage.html', context)

    return render(request, 'scorepage.html')

# Функция для предсказания изображения
def predictImageData(img):
    # Предобработка изображения
    preprocess = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    # Загрузка модели и выполнение предсказания
    sess = onnxruntime.InferenceSession(onnx_model)
    outputOFModel = np.argmax(sess.run(None, {'input': to_numpy(input_batch)}))

    # Проверяем предсказанный класс
    try:
        return imageClassList[outputOFModel]
    except KeyError:
        return None

# Преобразование PyTorch Tensor в numpy
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Преобразование изображения в Data URI для веб-отображения
def to_data_uri(pil_img):
    data = BytesIO()
    pil_img.save(data, "jpeg")  # Используем формат JPEG
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,' + data64.decode('utf-8')
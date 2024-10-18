import torch

# Загружаем модель в соответствии с вариантом:
# Четный номер в списке группы - cifar100_mobile;
# Нечетный номер в списке группы - cifar100_resnet.
# 
# Необходимо поставить ограничение на 3 класса по варианту
# 
# Класс 1 - номер в группе + номер группы
# Класс 2 - номер в группе + номер группы + 30
# Класс 3 - номер в группе + номер группы + 60 **Где номер группы - 1, 2, 3, 4 и т.д.
# 

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models",
        # "cifar100_mobilenetv2_x0_5",
        "cifar100_resnet20",
        pretrained=True,
    )

    model.to(device)

    x = torch.randn(1, 3, 32, 32, requires_grad=True).to(device)

    torch.onnx.export(
        model,                                              # модель
        x,                                                  # входной тензор (или кортеж нескольких тензоров)
        "cifar100/models/cifar100_CNN_RESNET20.onnx",       # куда сохранить (либо путь к файлу либо fileObject)
        export_params=True,                                 # сохраняет веса обученных параметров внутри файла модели
        opset_version=9,                                    # версия ONNX
        do_constant_folding=True,                           # следует ли выполнять укорачивание констант для оптимизации
        input_names=['input'],                              # имя входного слоя
        output_names=['output'],                            # имя выходного слоя
        dynamic_axes={'input':{0: 'batch_size'},            # динамичные оси, в данном случае только размер пакета
        'output': {0: 'batch_size'}
        },
    )

import os
import torch
import tarfile
import urllib.request
import torch.nn as nn

from CIFAR100 import CIFAR100Model, CIFAR100Trainer

def download_and_extract_cifar100(
    filename="cifar-100-python.tar.gz", 
    url="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", 
    extract_dir="cifar-100-python"
    ):
    """
    Функция для скачивания и распаковки архива CIFAR-100.
    
    Args:
        filename (str): Имя архива, который нужно скачать.
        url (str): URL для скачивания архива.
        extract_dir (str): Папка, куда нужно распаковать архив.
    """
    
    # Проверяем наличие файла
    if not os.path.exists(filename):
        # Файла нет, скачиваем
        print(f"Скачивание {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Файл '{filename}' был успешно скачан.")
    else:
        print(f"Файл '{filename}' уже существует.")

    # Проверяем, был ли архив распакован
    if not os.path.exists(extract_dir):
        print(f"Распаковка {filename}...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()
        print(f"Архив распакован в папку '{extract_dir}'.")
    else:
        print("Архив уже распакован.")

if "__main__" == __name__:
    # Проверка наличия CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        print("CUDA is available!")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        print(f"Driver version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available. Running on CPU.")

    download_and_extract_cifar100()

    CLASSES = [94, 21, 51]

    batch_size=2**5
    epochs=10
    criterion=nn.NLLLoss()
    lr_rate = 1e-5
    hidden_layers=[2**4, 2**6, 2**4]

    trainer = CIFAR100Trainer(CLASSES, batch_size, lr_rate)
        
    cifar100_001 = CIFAR100Model(
        hidden_layers=hidden_layers, 
        dropout_prob=0.5, 
        num_classes=CLASSES.__len__()
    )

    print(cifar100_001)

    trainer.train(cifar100_001, epochs)

    trainer.plot_training_history()

    trainer.evaluate(cifar100_001)
        
    trainer.save_model(cifar100_001)
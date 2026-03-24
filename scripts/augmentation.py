#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аугментация данных для CIFAR-100
"""

import torch
from torchvision import transforms as T


def get_train_transform(aug_config):
    """
    Создание трансформаций для тренировки с аугментацией.
    
    Args:
        aug_config: dict с параметрами аугментации
    
    Returns:
        Transform composition
    """
    transforms = []
    
    # Random Horizontal Flip (всегда включено для CIFAR)
    transforms.append(T.RandomHorizontalFlip())
    
    # Random Crop (всегда включено)
    transforms.append(T.RandomCrop(32, padding=4))
    
    # Color Jitter
    brightness = aug_config.get('brightness')
    contrast = aug_config.get('contrast')
    saturation = aug_config.get('saturation')
    hue = aug_config.get('hue')
    
    if any([brightness, contrast, saturation, hue is not None]):
        transforms.append(T.ColorJitter(
            brightness=brightness if brightness else 0,
            contrast=contrast if contrast else 0,
            saturation=saturation if saturation else 0,
            hue=hue if hue is not None else 0
        ))
    
    # Random Rotation
    if aug_config.get('degrees'):
        transforms.append(T.RandomRotation(
            degrees=aug_config['degrees']
        ))
    
    # Random Affine (translate, scale, shear)
    if any([
        aug_config.get('translate'),
        aug_config.get('scale'),
        aug_config.get('shear')
    ]):
        transforms.append(T.RandomAffine(
            degrees=0,  # 0, т.к. уже есть RandomRotation
            translate=aug_config.get('translate'),
            scale=aug_config.get('scale'),
            shear=aug_config.get('shear')
        ))
    
    # To Tensor
    transforms.append(T.ToTensor())
    
    # Normalize (будет добавлено в модели)
    # transforms.append(T.Normalize([0.5074, 0.4867, 0.4411], [0.2011, 0.1987, 0.2025]))
    
    return T.Compose(transforms)


def get_test_transform():
    """
    Создание трансформаций для тестирования (без аугментации).
    
    Returns:
        Transform composition
    """
    return T.Compose([
        T.ToTensor()
    ])


class AugmentedDataset(torch.utils.data.Dataset):
    """
    Dataset с аугментацией на лету.
    """
    def __init__(self, tensor_x, tensor_y, transform=None):
        self.tensor_x = tensor_x
        self.tensor_y = tensor_y
        self.transform = transform
    
    def __len__(self):
        return len(self.tensor_x)
    
    def __getitem__(self, idx):
        x = self.tensor_x[idx]
        y = self.tensor_y[idx]
        
        # Convert to PIL Image for transforms
        from PIL import Image
        img = Image.fromarray((x * 255).byte().numpy())
        
        if self.transform:
            img = self.transform(img)
        else:
            import torch
            img = torch.from_numpy(x).permute(2, 0, 1).float() / 255.0
        
        return img, y

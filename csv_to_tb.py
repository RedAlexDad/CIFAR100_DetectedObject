#!/usr/bin/env python3
"""
Конвертер CSV логов в TensorBoard формат
Использование: python csv_to_tb.py logs/exp1
"""

import sys
import os
import csv
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def convert_csv_to_tensorboard(csv_file, log_dir):
    """Конвертация CSV в TensorBoard"""
    
    writer = SummaryWriter(log_dir=log_dir)
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(row['epoch'])
            train_loss = float(row['train_loss'])
            train_acc = float(row['train_acc'])
            test_loss = float(row['test_loss'])
            test_acc = float(row['test_acc'])
            
            writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            writer.add_scalar('Loss/test_epoch', test_loss, epoch)
            writer.add_scalar('Accuracy/train_epoch', train_acc, epoch)
            writer.add_scalar('Accuracy/test_epoch', test_acc, epoch)
    
    writer.close()
    print(f"✅ Конвертировано в {log_dir}")
    print(f"🚀 tensorboard --logdir {log_dir}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Использование: python csv_to_tb.py <log_dir>")
        print("Пример: python csv_to_tb.py logs/exp1")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    
    # Найти CSV файл
    csv_files = [f for f in os.listdir(log_dir) if f.startswith('metrics_') and f.endswith('.csv')]
    
    if not csv_files:
        print(f"❌ Не найдено CSV файлов в {log_dir}")
        sys.exit(1)
    
    # Конвертировать каждый
    for csv_file in csv_files:
        csv_path = os.path.join(log_dir, csv_file)
        tb_subdir = os.path.join(log_dir, 'tb_' + csv_file.replace('.csv', '').replace('metrics_', ''))
        
        print(f"\nКонвертация: {csv_file}")
        convert_csv_to_tensorboard(csv_path, tb_subdir)

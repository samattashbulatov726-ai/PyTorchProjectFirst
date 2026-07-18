import os
from PIL import Image

# Путь к твоему датасету
dataset_path = 'dog_dataset'

print("Проверяем датасет на битые файлы...")
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify() # Проверяем, не поврежден ли файл
            except (IOError, SyntaxError, Image.UnidentifiedImageError):
                print(f"Удаляем битый файл: {file_path}")
                os.remove(file_path)
print("Проверка завершена!")
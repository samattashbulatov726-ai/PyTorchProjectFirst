import os
from concurrent.futures import ThreadPoolExecutor
import requests

# Настройки
breeds = ['labrador', 'poodle', 'bulldog', 'husky', 'chihuahua']
images_per_breed = 50
output_dir = 'dog_dataset'


def download_image(url, folder, filename):
    try:
        # Скачиваем напрямую в файл одной командой
        img_data = requests.get(url, timeout=5).content
        with open(os.path.join(folder, filename), 'wb') as f:
            f.write(img_data)
    except Exception:
        pass  # Игнорируем битые ссылки, для датасета это норма


# Сбор всех URL в один список задач
download_tasks = []
for breed in breeds:
    breed_dir = os.path.join(output_dir, breed)
    os.makedirs(breed_dir, exist_ok=True)

    # Получаем ссылки
    res = requests.get(f'https://dog.ceo/api/breed/{breed}/images/random/{images_per_breed}').json()
    if res.get('status') == 'success':
        for i, url in enumerate(res['message']):
            download_tasks.append((url, breed_dir, f'{breed}_{i}.jpg'))

# Скачиваем всё параллельно в 10 потоков (это в 10 раз быстрее!)
print(f"Начинаю скачивание {len(download_tasks)} изображений...")
with ThreadPoolExecutor(max_workers=10) as executor:
    executor.map(lambda p: download_image(*p), download_tasks)

print("Готово!")

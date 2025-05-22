import os
import shutil
import numpy as np
from PIL import Image
from src import config

def organize_images_by_patient():
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)
    for fname in os.listdir(config.IMAGE_FOLDER):
        if fname.lower().endswith(('.png','.bmp')):
            us_number = fname.split('.')[0]
            patient_folder = os.path.join(config.OUTPUT_FOLDER, us_number)
            os.makedirs(patient_folder, exist_ok=True)
            src = os.path.join(config.IMAGE_FOLDER, fname)
            dst = os.path.join(patient_folder, fname)
            shutil.copy(src, dst)

def load_image_for_patient(us_number, size = config.IMAGE_SIZE):
    folder = os.path.join(config.OUTPUT_FOLDER, str(us_number))
    images = []
    failed_images = []

    if os.path.exists(folder):
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            if fname.lower().endswith(('png','bmp')):
                try:
                    img = Image.open(path).convert('RGB')
                    img = img.resize(size)
                    arr = np.array(img, dtype=np.float32) / 255.0
                    images.append(arr)
                except Exception as e:
                    failed_images.append((fname, str(e)))
    if not images:
        images = [np.zeros((size[0], size[1], 3), dtype=np.float32)]
    return images, failed_images

def load_all_images(us_numbers, size = config.IMAGE_SIZE):
    img_list = []
    repeated_us_numbers = []
    all_failed = []
    for us_number in us_numbers:
        imgs, failed = load_image_for_patient(us_number, size)
        img_list.extend(imgs)
        repeated_us_numbers.extend([us_number] * len(imgs))
        if failed:
            all_failed.extend([(us_number, f, err) for f, err in failed])


    return np.stack(img_list), repeated_us_numbers

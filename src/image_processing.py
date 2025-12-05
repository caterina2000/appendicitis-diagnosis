import os
import re
import shutil
import numpy as np
from PIL import Image, ImageOps
from src import config

# === 1. Images organization only for patients with clinical data ===
def organize_images_by_patient(clinical_ids):
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)
    count , skipped = 0, 0

    if not os.path.isdir(config.IMAGE_FOLDER):
        print(f" [ERROR] image folder not found: {config.IMAGE_FOLDER}")
        return

    all_files = os.listdir(config.IMAGE_FOLDER)
    print(f" Found {len(all_files)} file in {config.IMAGE_FOLDER}")

    for fname in all_files:
        if not fname.lower().endswith(('.bmp', '.png')):
            continue

        match = re.match(r'^(\d+)', os.path.splitext(fname)[0])
        if not match:
            skipped += 1
            continue

        us_number = match.group(1).lstrip('0')
        if us_number not in clinical_ids:
            skipped += 1
            continue

        patient_folder = os.path.join(config.OUTPUT_FOLDER, us_number)
        os.makedirs(patient_folder, exist_ok=True)

        src = os.path.join(config.IMAGE_FOLDER, fname)
        dst = os.path.join(patient_folder, fname)
        shutil.copy(src, dst)
        count+=1

    print(f" {count} copied images.")
    print(f" {skipped} skipped images (no matching clinical data)")

# === 2. Uploading all images associated with the single patient ===
def load_image_for_patient(us_number, size = config.IMAGE_SIZE):
    folder = os.path.join(config.OUTPUT_FOLDER, str(us_number))
    images = []

    if not os.path.isdir(folder):
        return images

    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.bmp', '.png')):
            continue
        path = os.path.join(folder, fname)
        try:
            img = Image.open(path).convert('RGB')
            img = ImageOps.contain(img,size)
            img = ImageOps.pad(img,size, color=0)
            arr = np.array(img, dtype=np.float32) / 255.0
            images.append(arr)
        except Exception as e:
            print(f"\n Error loading {fname}: {e}")
    return images

# === 3. Uploading images for all patients ===
def load_all_images(us_numbers, size=config.IMAGE_SIZE):
    img_list ,repeated_ids, skipped = [], [], []

    for uid in us_numbers:
        imgs = load_image_for_patient(uid, size)
        if not imgs :
            skipped.append(uid)
            continue
        img_list.extend(imgs)
        repeated_ids.extend([uid] * len(imgs))

    if not img_list:
        raise ValueError(" No images loaded")

    print(f"\n Images loaded: {len(img_list)} from {len(set(repeated_ids))} patients")
    if skipped:
        print(f" {len(skipped)} Patients had no images")
    return np.stack(img_list), repeated_ids

# === 4. Association images <--> patients with clinical data
def load_all_images_for_valid_patients(X):
    #print("\n Verify valid patients")
    image_ids = set(os.listdir(config.OUTPUT_FOLDER))
    clinical_ids = set(map(str, X.index))
    valid = sorted(image_ids & clinical_ids)

    #print(f"\n Valid multimodal patients found: {len(valid)}")
    only_images = image_ids - clinical_ids
    only_clinical = clinical_ids - image_ids
    print(f" Patients with images but without clinical data: {len(only_images)}")
    print(f" Patients with clinical data but without images: {len(only_clinical)}")
    return valid


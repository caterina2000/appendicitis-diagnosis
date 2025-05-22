import os
import numpy as np
from sklearn.model_selection import train_test_split

from src import config
from src.clinical_data_preprocessing import preprocess_clinical_data
from src.image_processing import organize_images_by_patient, load_all_images
from src.model import multimodal_model
from src.train import train_model, load_best_model
from src.evaluate import evaluate_model

# =======================
# 1. Preprocessing dati
# =======================
print("🔍 Preprocessing dei dati clinici...")
X_clinical, y = preprocess_clinical_data()

print("🧼 Organizzazione immagini per paziente...")
organize_images_by_patient()

print("🖼️ Caricamento immagini associate ai pazienti...")
X_img, repeated_ids = load_all_images(X_clinical.index.tolist())

# Replica i dati clinici in base alle immagini
X_clinical_repeated = X_clinical.loc[repeated_ids].reset_index(drop=True)
y_repeated = y.loc[repeated_ids].reset_index(drop=True)

# =======================
# 2. Train/Test Split
# =======================
print("🔀 Split del dataset in train e test...")
X_img_train, X_img_test, X_train, X_test, y_train, y_test = train_test_split(
    X_img.astype(np.float32),
    X_clinical_repeated.astype(np.float32),
    y_repeated,
    test_size=config.TEST_SIZE,
    stratify=y_repeated,
    random_state=config.RANDOM_STATE
)

# =======================
# 3. Costruzione modello
# =======================
print("🧠 Costruzione del modello multimodale...")
model, base_model = multimodal_model(
    input_shape_img=config.IMAGE_SIZE + (3,),
    input_shape_clinical=(X_train.shape[1],),
    learning_rate=config.LEARNING_RATE_INITIAL
)

# =======================
# 4. Addestramento iniziale
# =======================
model, history_initial = train_model(
    model, base_model,
    X_img_train, X_train, y_train,
    X_img_test, X_test, y_test,
    fine_tune=False
)

# =======================
# 5. Fine-tuning
# =======================
model, history_ft = train_model(
    model, base_model,
    X_img_train, X_train, y_train,
    X_img_test, X_test, y_test,
    fine_tune=True
)

# =======================
# 6. Caricamento modello migliore e valutazione
# =======================
best_model = load_best_model()
evaluate_model(best_model, X_img_test, X_test, y_test, save_plots=True)

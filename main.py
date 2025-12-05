import numpy as np
from sklearn.model_selection import train_test_split
from src import config
from sklearn.utils.class_weight import compute_class_weight
from src.clinical_data_preprocessing import preprocess_clinical_data
from src.image_processing import (
    organize_images_by_patient,
    load_all_images,
    load_all_images_for_valid_patients
)
from src.model import multimodal_model
from src.train import train_model, load_best_model
from src.evaluate import evaluate_model
from src.visualization import plot_training_history

# === STEP 1. Load and preprocess clinical data ===
print("\n ...Preprocessing clinical data...")
X, y = preprocess_clinical_data()

# Use string indices for matching image folders
X.index = X.index.astype(str)
y.index = y.index.astype(str)

# === STEP 2. Organize ultrasound images for valid patients ===
print("\n ...Organizing images per patient...")
organize_images_by_patient(clinical_ids = set(X.index))

# === STEP 3. Identify patients with both clinical data and images ===
print("\n ...Finding valid patients (data + images)...")
valid_patient_ids = load_all_images_for_valid_patients(X)
print(" valid patients found:", len(valid_patient_ids))

# === STEP 4. Load images and align with clinical records ===
print("\n ...Loading images and repeating clinical data...")
#print("Image folders:", os.listdir(config.OUTPUT_FOLDER))
X_img_raw, repeated_us_numbers = load_all_images(valid_patient_ids)

# Filter clinical data by only those repeated US numbers that exist in X
X_img_filtered = []
filter_us_numbers = []

for img, uid in zip(X_img_raw, repeated_us_numbers):
    if uid in X.index:
        X_img_filtered.append(img)
        filter_us_numbers.append(uid)

X_img = np.stack(X_img_filtered)
X_clin = X.loc[filter_us_numbers].reset_index(drop=True)
y_target = y.loc[filter_us_numbers].reset_index(drop=True)

print(f" Total images loaded: {X_img.shape[0]}")
print(f" Clinical data shape: {X_clin.shape}")
print(f" Target array shape: {y_target.shape}")

# === STEP 5. Split into training and test sets ===
print("\n ...Split del dataset in train e test... ")
X_img_train, X_img_test, X_clin_train, X_clin_test, y_train, y_test = train_test_split(
    X_img.astype(np.float32),
    X_clin.astype(np.float32),
    y_target,
    test_size=config.TEST_SIZE,
    stratify=y_target,
    random_state=config.RANDOM_STATE
)
print(" Train/test shapes: ")
print(" X_img_train:", X_img_train.shape)
print(" X_img_test :", X_img_test.shape)
print(" X_train    :", X_clin_train.shape)
print(" X_test     :", X_clin_test.shape)
print(" y_train    :", y_train.shape)
print(" y_test     :", y_test.shape)

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight = dict(zip(np.unique(y_train), class_weights_array))
print(" Class weights:", class_weight)

# === STEP 6. Build the multimodal model ===
print("\n ...Build multimodal model...")
model, base_cnn = multimodal_model(
    input_shape_img=config.IMAGE_SIZE + (3,),
    input_shape_clinical=(X_clin_train.shape[1],),
    learning_rate=config.LEARNING_RATE_INITIAL
)

# === STEP 7.  initial training (frozen ResNet) ===
model, history_initial = train_model(
    model, base_cnn,
    X_img_train, X_clin_train, y_train,
    X_img_test, X_clin_test, y_test,
    fine_tune = False
)

# === STEP 8. Fine-tuning (unfreeze last layers) ===
model, history_ft = train_model(
    model, base_cnn,
    X_img_train, X_clin_train, y_train,
    X_img_test, X_clin_test, y_test,
    fine_tune = True
)

# === STEP 9. Evaluate the best saved model ===
print("\n ...Evaluating best saved model on test set...")
best_model = load_best_model()
evaluate_model(
    best_model,
    X_img_test, X_clin_test, y_test,
    save_plots=True
)

# === STEP 10. Plot training history ===
print("\n ...Plotting training and fine-tuning performance...")
plot_training_history(
    history_initial,
    history_ft,
    title_suffix='(Model Performance)',
    save_plots=True
)

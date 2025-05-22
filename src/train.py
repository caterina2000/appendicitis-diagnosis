from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src import config
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

def get_callbacks():
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=config.PATIENCE,
        restore_best_weights=True,
        verbose = 1
    )

    checkpoint = ModelCheckpoint(
        filepath = config.MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    return [early_stop, checkpoint]

def train_model(model, base_model, X_img_train, X_train, y_train, X_img_test, X_test, y_test, fine_tune=False):

    if fine_tune:
        print("\n🔓 Avvio fine-tuning: sblocco ultimi layer della ResNet50...\n")
        for layer in base_model.layers[-20:]:
            layer.trainable = True

        model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE_FINE_TUNE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        epochs = config.EPOCHS_FINE_TUNE
    else:
        print("\n🚀 Addestramento iniziale con ResNet congelata...\n")
        model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE_INITIAL),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        epochs = config.EPOCHS_INITIAL


    history = model.fit(
        [X_img_train, X_train], y_train,
        epochs=epochs,
        batch_size=config.BATCH_SIZE,
        validation_data=([X_img_test, X_test], y_test),
        class_weight=config.CLASS_WEIGHT,
        callbacks=get_callbacks(),
        verbose=1
    )

    return model, history

def load_best_model():
    print(f"\n✅ Caricamento del miglior modello da: {config.MODEL_PATH}")
    return load_model(config.MODEL_PATH)
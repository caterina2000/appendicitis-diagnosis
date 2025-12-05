from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src import config
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

def get_callbacks(fine_tune=False):
    patience = 2 if fine_tune else config.PATIENCE
    return [
        EarlyStopping(
            monitor='val_loss',  #oppure provo val_loss
            patience=patience,
            mode = 'min',
            restore_best_weights=True,
            verbose = 1
        ),
        ModelCheckpoint(
            filepath = config.MODEL_PATH,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
    ]

def train_model(model, base_model, X_img_train, X_train, y_train, X_img_test, X_test, y_test, class_weight=None,fine_tune=False):
    if fine_tune:
        print(f" Fine-tuning: Unlocking the last layers of ResNet50")
        for layer in base_model.layers:
            layer.trainable = 'conv5_block3' in layer.name
        lr = config.LEARNING_RATE_FINE_TUNE
        epochs = config.EPOCHS_FINE_TUNE
    else:
        print(f" Initial training: ResNet50 locked")
        for layer in base_model.layers:
            layer.trainable = False
        lr = config.LEARNING_RATE_INITIAL
        epochs = config.EPOCHS_INITIAL

    callbacks = get_callbacks(fine_tune)

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(name = 'auc')]
    )

    history = model.fit(
        [X_img_train, X_train],
        y_train,
        epochs=epochs,
        batch_size=config.BATCH_SIZE,
        validation_data=([X_img_test, X_test], y_test),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    return model, history

def load_best_model():
    print(f"\n Loading better model from: {config.MODEL_PATH}")
    return load_model(config.MODEL_PATH)
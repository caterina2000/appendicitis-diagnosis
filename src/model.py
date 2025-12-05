from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Concatenate,
    GlobalAveragePooling2D, Dropout, BatchNormalization, LeakyReLU
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from src import config


def multimodal_model(input_shape_img, input_shape_clinical,learning_rate=config.LEARNING_RATE_INITIAL):
    # === Image input + CNN pre-trained ===
    image_input = Input(shape=input_shape_img, name = 'image_input')
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=image_input)
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D(name = 'gap_cnn_output')(base_model.output)
    #to stabilize feature extraction
    x = Dense(128, name='dense_img_1')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(0.5)(x)

    # === Clinical data input + MLP ===
    clinical_input = Input(shape=input_shape_clinical, name = 'clinical_input')
    y = Dense(128, name='dense_clinical_1')(clinical_input)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    y = Dropout(0.5)(y)

    y = Dense(64, name = 'dense_clinical_2')(y)
    y = BatchNormalization()(y)
    y = LeakyReLU()(y)
    y = Dropout(0.5)(y)

    y = Dense(32, name = 'dense_clinical_3')(y)
    y = LeakyReLU()(y)
    y = Dropout(0.3)(y)

    # === Fusion multimodal ===
    combined = Concatenate(name = 'fusion')([x, y])

    z = Dense(128, name = 'dense_combined_1')(combined)
    z = BatchNormalization()(z)
    z = LeakyReLU()(z)
    z = Dropout(0.5)(z)

    z = Dense(64, name = 'dense_combined_2')(z)
    z = LeakyReLU()(z)
    z = Dropout(0.3)(z)

    z = Dense(32, name = 'dense_combined_3')(z)
    z = BatchNormalization()(z)
    z = LeakyReLU()(z)

    output = Dense(1, activation='sigmoid', name = 'output')(z)

    # === Final model ===
    model = Model(inputs=[image_input, clinical_input], outputs=output, name = 'multimodal_model')
    model.compile(optimizer=Adam(learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', AUC(name = 'auc')])

    return model, base_model

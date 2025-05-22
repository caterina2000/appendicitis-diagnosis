from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from src import config

def multimodal_model(input_shape_img, input_shape_clinical,learning_rate=config.LEARNING_RATE_INITIAL):
    # Input immagini
    image_input = Input(shape=input_shape_img)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=image_input)
    #congelamento dei layer della ResNet50
    for layer in base_model.layers:
        layer.trainable = False
    #estrazione delle feature
    x = GlobalAveragePooling2D()(base_model.output)

    # Input dati clinici
    clinical_input = Input(shape=input_shape_clinical)
    y = Dense(64, activation='relu')(clinical_input)
    y = Dropout(0.5)(y)
    y = Dense(32, activation='relu')(y)
    y = Dropout(0.5)(y)

    # Fusione delle feature
    combined = Concatenate()([x, y])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.5)(z)
    z = Dense(32, activation='relu')(z)
    z = Dropout(0.5)(z)
    output = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[image_input, clinical_input], outputs=output)
    #compilazione del modello
    model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    return model, base_model

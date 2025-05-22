import os

#Percorsi dei file
DATA_PATH = 'data/raw/app_data.xlsx'
IMAGE_FOLDER = 'data/raw/US_Pictures'
OUTPUT_FOLDER = 'data/processed/US_Pictures_new'

#Parametri delle immagini
IMAGE_SIZE = (224,224)

#Parametri di addestramento
BATCH_SIZE = 32
EPOCHS_INITIAL = 20
EPOCHS_FINE_TUNE = 10
LEARNING_RATE_INITIAL = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5
PATIENCE = 5
CLASS_WEIGHT = {0: 1, 1: 2}

#Parametri di test
TEST_SIZE = 0.2
RANDOM_STATE = 42

#Percorso per salvare il modello
MODEL_PATH = 'outputs/models/best_model.h5'
LOG_DIR = 'outputs/logs'
import matplotlib.pyplot as plt
import os
from src import config

def plot_training_history(history, title_suffix='', phase='initial', save_plots=False):
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # === Plot Loss e Accuracy ===
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss{title_suffix}')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy{title_suffix}')
    plt.legend()
    if save_plots:
        plt.savefig(f"{config.LOG_DIR}/loss_e_accuracy.png", bbox_inches='tight')
    plt.show()



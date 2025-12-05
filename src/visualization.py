import matplotlib.pyplot as plt
import os
import numpy as np
from src import config

def plot_training_history(history_initial, history_ft=None, title_suffix='', save_plots=False):
    os.makedirs(config.LOG_DIR, exist_ok=True)

    def save(fig, name):
        if save_plots:
            suffix = title_suffix.replace(" ", "_").lower()
            filename = f"{config.LOG_DIR}/{name}_{suffix}.png"
            plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)

    # === Plot Loss ===
    fig, axes = plt.subplots(1,2 if history_ft else 1, figsize=(14,5), sharey=True)
    fig.suptitle(f"Loss {title_suffix}")

    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    axes[0].plot(history_initial.history['loss'], 'r',label='Train Loss')
    axes[0].plot(history_initial.history['val_loss'], 'b', label='Val Loss')
    axes[0].set_title("Initial Training")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    if history_ft:
        axes[1].plot(history_ft.history['loss'], 'r', label='Train Loss')
        axes[1].plot(history_ft.history['val_loss'], 'b', label='Val Loss')
        axes[1].set_title("Fine-Tuning")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Loss")
        axes[1].legend()

    save(fig, 'loss_history')

    # === Plot Accuracy ===
    fig, axes = plt.subplots(1,2 if history_ft else 1, figsize=(14,5), sharey=True)
    fig.suptitle(f"Accuracy{title_suffix}")

    axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    axes[0].plot(history_initial.history['accuracy'],'g', label='Train Accuracy')
    axes[0].plot(history_initial.history['val_accuracy'], 'm', label='Val Accuracy')
    axes[0].set_title("Initial Training")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    if history_ft:
        axes[1].plot(history_ft.history['accuracy'], 'g', label='Train Accuracy')
        axes[1].plot(history_ft.history['val_accuracy'], 'm', label='Val Accuracy')
        axes[1].set_title("Fine-Tuning")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

    save(fig, 'accuracy_history')

    # === AUC ===
    if 'auc' in history_initial.history:
        fig, axes = plt.subplots(1,2 if history_ft else 1, figsize=(14,5), sharey=True)
        fig.suptitle(f"AUC{title_suffix}")

        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
        axes[0].plot(history_initial.history['auc'],'k', label='Train AUC')
        axes[0].plot(history_initial.history['val_auc'],'c', label='Val AUC')
        axes[0].set_title("Initial Training")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("AUC")
        axes[0].legend()

        if history_ft:
            axes[1].plot(history_ft.history['auc'], 'k',label='Train AUC')
            axes[1].plot(history_ft.history['val_auc'], 'c',label='Val AUC')
            axes[1].set_title("Fine-Tuning")
            axes[1].set_xlabel("Epochs")
            axes[1].set_ylabel("AUC")
            axes[1].legend()

        save(fig, 'auc_history')
    else:
        print(" AUC non presente nei dati di training, salto il grafico.")





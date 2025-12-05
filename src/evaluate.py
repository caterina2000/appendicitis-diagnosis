import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve, auc,
    classification_report,
    confusion_matrix,
)
from src import config

def evaluate_model(model, X_img_test, X_test, y_test, save_plots=False):
    print(f" Evaluating model on test set")
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # === 1. Keras evaluating ===
    loss, accuracy, keras_auc = model.evaluate([X_img_test, X_test], y_test, verbose=1)
    print(f" - Test Loss: {loss:.4f}")
    print(f" - Test Accuracy: {accuracy:.4f}")
    print(f" - Keras AUC: {keras_auc:.4f}")

    # === 2. Predictions and probabilities ===
    y_prob = model.predict([X_img_test, X_test], batch_size=config.BATCH_SIZE).ravel()
    sklearn_auc = roc_auc_score(y_test, y_prob)
    print(f" - Sklearn AUC  : {sklearn_auc:.4f}")

    # === 3. Find optimal threshold using precision-recall curve
    precision, recall, threshold = precision_recall_curve(y_test, y_prob)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_score)
    best_thresh = threshold[best_idx]
    print(f" Best Threshold by F1: {best_thresh:.2f}")

    y_pred = (y_prob >= best_thresh).astype(int)


    # === 4. Confusion Matrix ===
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Appendicitis", "Appendicitis"],
                yticklabels=["No Appendicitis", "Appendicitis"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix @ thresh={best_thresh:.2f}")
    if save_plots:
        plt.savefig(f"{config.LOG_DIR}/confusion_matrix.png", bbox_inches='tight')
    plt.show()

    # === 5. Classification Report ===
    report = classification_report(y_test, y_pred, target_names=['No Appendicitis', 'Appendicitis'], digits=4)
    print("\n Classification Report:")
    print(report)
    if save_plots:
        with open(f"{config.LOG_DIR}/classification_report.txt", "w") as f:
            f.write(report)

    # === 6. ROC Curve ===
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {sklearn_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    if save_plots:
        plt.savefig(f"{config.LOG_DIR}/roc_curve.png", bbox_inches='tight')
    plt.show()

    # === 7. Precision-Recall Curve ===
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    if save_plots:
        plt.savefig(f"{config.LOG_DIR}/precision_recall_curve.png", bbox_inches='tight')
    plt.show()
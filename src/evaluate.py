import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from src import config

def evaluate_model(model, X_img_test, X_test, y_test, save_plots=False):
    """
    Valuta il modello sul test set, stampa le metriche e salva i grafici.
    """
    print("📈 Valutazione sul test set...")
    loss, accuracy = model.evaluate([X_img_test, X_test], y_test, verbose=1)
    print(f"\n✅ Test Loss: {loss:.4f}")
    print(f"✅ Test Accuracy: {accuracy:.4f}")

    predictions = model.predict([X_img_test, X_test], batch_size=config.BATCH_SIZE)
    predictions_binary = (predictions > 0.5).astype('int32').flatten()

    roc_score = roc_auc_score(y_test, predictions)
    print(f"🔍 ROC AUC Score: {roc_score:.4f}")

    # === Confusion Matrix ===
    cm = confusion_matrix(y_test, predictions_binary)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Appendicitis", "Appendicitis"],
                yticklabels=["No Appendicitis", "Appendicitis"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    if save_plots:
        plt.savefig(f"{config.LOG_DIR}/confusion_matrix.png", bbox_inches='tight')
    plt.show()

    # === Classification Report ===
    print("\n📝 Classification Report:")
    print(classification_report(y_test, predictions_binary, target_names=["No Appendicitis", "Appendicitis"]))

    # === ROC Curve ===
    fpr, tpr, _ = roc_curve(y_test, predictions)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_score:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    if save_plots:
        plt.savefig(f"{config.LOG_DIR}/roc_curve.png", bbox_inches='tight')
    plt.show()

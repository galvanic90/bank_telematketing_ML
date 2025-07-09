import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


def plot_confusion_matrix(y_true, y_pred, labels=[0, 1], title='Confusion Matrix', save_path="fig_1.png"):
    """
    Imprime una matriz de confusión bonita con seaborn.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.set(font_scale=1.2)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No', 'Sí'], yticklabels=['No', 'Sí'])
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs('figures', exist_ok=True)
        full_path = os.path.join('figures', save_path)
        plt.savefig(full_path)
        print(full_path)
    plt.show()


def plot_roc_curve(y_true, y_proba, title='ROC Curve', save_path="fig_2.png"):
    """
    Plotea curva ROC y muestra AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Tasa Falsos Positivos')
    plt.ylabel('Tasa Verdaderos Positivos')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    if save_path:
        os.makedirs('figures', exist_ok=True)
        full_path = os.path.join('figures', save_path)
        plt.savefig(full_path)
        print(full_path)
    plt.show()

    return auc


def show_classification_report(y_true, y_pred):
    """
    Imprime el reporte de clasificación completo.
    """
    print("\n=== Reporte de Clasificación ===")
    print(classification_report(y_true, y_pred, target_names=['No', 'Sí']))

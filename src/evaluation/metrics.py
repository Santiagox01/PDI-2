"""
Modulo de computo formal de metricas de clasificacion multiclase (Precision, Recall, F1, Accuracy).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                   class_names: list = None) -> dict:
    """Calcula todas las metricas requeridas sobre el conjunto de evaluacion."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Reporte detallado por clase
    report_dict = classification_report(y_true, y_pred, target_names=class_names,
                                        output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report_dict).transpose()

    # Matrices de confusion
    cm_abs = confusion_matrix(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm_abs.astype("float") / cm_abs.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "classification_report_df": df_report,
        "confusion_matrix_abs": cm_abs,
        "confusion_matrix_norm": cm_norm
    }

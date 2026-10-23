"""
Modulo de visualizacion academica de matrices de confusion absolutas y normalizadas.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrices(cm_abs: np.ndarray, cm_norm: np.ndarray, class_names: list,
                            output_dir: str = "outputs/figures"):
    """Genera y guarda las figuras de matriz de confusion absoluta y normalizada."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Matriz de Confusion Absoluta
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(cm_abs, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                annot_kws={"size": 11, "weight": "bold"})
    ax.set_title("Matriz de Confusión Absoluta (Conjunto de Prueba Reservado)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Denominación Predicha", fontsize=11, fontweight="bold")
    ax.set_ylabel("Denominación Real (Ground Truth)", fontsize=11, fontweight="bold")
    plt.xticks(rotation=25)
    plt.yticks(rotation=0)
    plt.tight_layout()
    path_abs = os.path.join(output_dir, "confusion_matrix_absolute.png")
    fig.savefig(path_abs, dpi=300)
    plt.close()

    # 2. Matriz de Confusion Normalizada
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=True, vmin=0, vmax=1,
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                annot_kws={"size": 11, "weight": "bold"})
    ax.set_title("Matriz de Confusión Normalizada por Recall (Conjunto de Prueba)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Denominación Predicha", fontsize=11, fontweight="bold")
    ax.set_ylabel("Denominación Real (Ground Truth)", fontsize=11, fontweight="bold")
    plt.xticks(rotation=25)
    plt.yticks(rotation=0)
    plt.tight_layout()
    path_norm = os.path.join(output_dir, "confusion_matrix_normalized.png")
    fig.savefig(path_norm, dpi=300)
    plt.close()

    print(f"[INFO] Matrices de confusion guardadas en {path_abs} y {path_norm}")

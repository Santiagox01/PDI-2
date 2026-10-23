"""
Modulo de experimentacion sistematica y comparacion exhaustiva de clasificadores.
Ejecuta los experimentos E1 a E5 con SVM Lineal, SVM RBF, k-NN, Random Forest y MLP,
evalua el impacto de PCA, selecciona el modelo optimo y lo serializa a final_model.joblib.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import matplotlib
matplotlib.use("Agg")

import time
import json
import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.models.train import build_pipeline, optimize_classifier


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_pca_variance(X_train: np.ndarray, feature_name: str = "HOG",
                          save_path: str = "outputs/figures/pca_variance_explained.png"):
    """
    Ajusta PCA estrictamente sobre el conjunto de entrenamiento
    y grafica la curva de varianza acumulada explicada.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    max_comps = min(X_train.shape[0] - 1, X_train.shape[1], 234)
    pca = PCA(n_components=max_comps, random_state=42)
    pca.fit(X_train_scaled)

    cum_variance = np.cumsum(pca.explained_variance_ratio_)

    # Componentes para 85%, 90% y 95%
    n_85 = int(np.argmax(cum_variance >= 0.85) + 1) if any(cum_variance >= 0.85) else max_comps
    n_90 = int(np.argmax(cum_variance >= 0.90) + 1) if any(cum_variance >= 0.90) else max_comps
    n_95 = int(np.argmax(cum_variance >= 0.95) + 1) if any(cum_variance >= 0.95) else max_comps

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(range(1, max_comps + 1), cum_variance, color="#1f77b4", linewidth=2.5, label="Varianza acumulada")
    ax.axhline(0.95, color="crimson", linestyle="--", linewidth=1.2, label=f"95% Varianza ({n_95} componentes)")
    ax.axhline(0.90, color="darkorange", linestyle=":", linewidth=1.2, label=f"90% Varianza ({n_90} componentes)")
    ax.axvline(n_95, color="crimson", linestyle="--", alpha=0.6)

    ax.set_title(f"Análisis de Componentes Principales (PCA) sobre Características {feature_name}",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Número de Componentes Principales", fontsize=11, fontweight="bold")
    ax.set_ylabel("Proporción de Varianza Explicada Acumulada", fontsize=11, fontweight="bold")
    ax.set_ylim([0, 1.02])
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(f"[PCA] Curva de varianza explicada guardada en {save_path}. (95% varianza = {n_95} componentes de {X_train.shape[1]})")
    return n_95


def run_model_comparison(config: dict):
    features_dir = config["paths"]["features"]
    models_dir = config["paths"]["models"]
    cv_folds = config["training"]["cv_folds"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/experiment_results", exist_ok=True)

    experiments = [
        {"id": "E1", "name": "HOG", "file": "features_hog.npz"},
        {"id": "E2", "name": "Forma", "file": "features_shape.npz"},
        {"id": "E3", "name": "Color", "file": "features_color.npz"},
        {"id": "E4", "name": "Textura", "file": "features_texture.npz"},
        {"id": "E5", "name": "Combinada (HOG+Forma+Color+Textura)", "file": "features_combined.npz"}
    ]

    classifiers = ["svm_linear", "svm_rbf", "knn", "random_forest", "mlp"]

    results = []
    trained_pipelines = {}

    print("[INFO] Iniciando ejecucion de experimentos E1 a E5...")

    # Analisis de PCA previo sobre E1 (HOG)
    hog_data = np.load(os.path.join(features_dir, "features_hog.npz"), allow_pickle=True)
    train_mask_hog = (hog_data["splits"] == "train")
    evaluate_pca_variance(hog_data["X"][train_mask_hog], feature_name="HOG")

    for exp in experiments:
        exp_id = exp["id"]
        exp_name = exp["name"]
        npz_path = os.path.join(features_dir, exp["file"])

        data = np.load(npz_path, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        splits = data["splits"]

        # Separar estrictamente train y val
        is_train = (splits == "train")
        is_val = (splits == "val")

        X_train, y_train = X[is_train], y[is_train]
        X_val, y_val = X[is_val], y[is_val]

        print(f"\n==========================================")
        print(f"=== {exp_id}: {exp_name} (Dim: {X.shape[1]}) ===")
        print(f"==========================================")

        # Evaluamos cada clasificador
        for clf_name in classifiers:
            print(f"[Optimizando] {exp_id} - {clf_name}...")
            t0 = time.time()

            pipeline, param_grid = build_pipeline(clf_name, use_pca=False, random_state=42)
            best_pipe, best_params, cv_score = optimize_classifier(
                pipeline, param_grid, X_train, y_train, cv_folds=cv_folds, scoring="f1_macro"
            )
            fit_time = time.time() - t0

            # Evaluacion en conjunto de validacion
            y_val_pred = best_pipe.predict(X_val)
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1_macro = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
            val_f1_weighted = f1_score(y_val, y_val_pred, average="weighted", zero_division=0)

            record_key = f"{exp_id}_{clf_name}"
            trained_pipelines[record_key] = {
                "pipeline": best_pipe,
                "exp_id": exp_id,
                "exp_name": exp_name,
                "clf_name": clf_name,
                "best_params": best_params,
                "feature_file": exp["file"],
                "val_f1_macro": val_f1_macro,
                "val_acc": val_acc
            }

            results.append({
                "experiment_id": exp_id,
                "features_family": exp_name,
                "dimension": X.shape[1],
                "classifier": clf_name,
                "best_params": str(best_params),
                "cv_f1_macro": round(cv_score, 4),
                "val_accuracy": round(val_acc, 4),
                "val_f1_macro": round(val_f1_macro, 4),
                "val_f1_weighted": round(val_f1_weighted, 4),
                "fit_time_seconds": round(fit_time, 2)
            })
            print(f"  -> CV F1-macro: {cv_score:.4f} | Val F1-macro: {val_f1_macro:.4f} | Val Acc: {val_acc:.4f}")

        # En E1 (HOG) y E5 (Combinada), evaluar adicionalmente SVM RBF con PCA (95% varianza)
        if exp_id in ["E1", "E5"]:
            pca_clf_name = "svm_rbf_pca95"
            print(f"[Optimizando] {exp_id} - {pca_clf_name} (con PCA 95%)...")
            t0 = time.time()
            pipe_pca, param_grid_pca = build_pipeline("svm_rbf", use_pca=True, pca_components=0.95, random_state=42)
            best_pipe_pca, best_params_pca, cv_score_pca = optimize_classifier(
                pipe_pca, param_grid_pca, X_train, y_train, cv_folds=cv_folds, scoring="f1_macro"
            )
            fit_time_pca = time.time() - t0

            y_val_pred_pca = best_pipe_pca.predict(X_val)
            val_acc_pca = accuracy_score(y_val, y_val_pred_pca)
            val_f1_macro_pca = f1_score(y_val, y_val_pred_pca, average="macro", zero_division=0)
            val_f1_weighted_pca = f1_score(y_val, y_val_pred_pca, average="weighted", zero_division=0)

            record_key_pca = f"{exp_id}_{pca_clf_name}"
            trained_pipelines[record_key_pca] = {
                "pipeline": best_pipe_pca,
                "exp_id": exp_id,
                "exp_name": f"{exp_name} + PCA(95%)",
                "clf_name": pca_clf_name,
                "best_params": best_params_pca,
                "feature_file": exp["file"],
                "val_f1_macro": val_f1_macro_pca,
                "val_acc": val_acc_pca
            }

            results.append({
                "experiment_id": exp_id,
                "features_family": f"{exp_name} + PCA(95%)",
                "dimension": X.shape[1],
                "classifier": pca_clf_name,
                "best_params": str(best_params_pca),
                "cv_f1_macro": round(cv_score_pca, 4),
                "val_accuracy": round(val_acc_pca, 4),
                "val_f1_macro": round(val_f1_macro_pca, 4),
                "val_f1_weighted": round(val_f1_weighted_pca, 4),
                "fit_time_seconds": round(fit_time_pca, 2)
            })
            print(f"  -> PCA CV F1-macro: {cv_score_pca:.4f} | Val F1-macro: {val_f1_macro_pca:.4f} | Val Acc: {val_acc_pca:.4f}")

    df_results = pd.DataFrame(results)
    df_results.sort_values(by="val_f1_macro", ascending=False, inplace=True)
    df_results.to_csv("outputs/tables/model_comparison_results.csv", index=False, encoding="utf-8")
    print(f"\n[INFO] Tabla de comparacion guardada en outputs/tables/model_comparison_results.csv")

    # Grafica de comparacion de modelos
    fig, ax = plt.subplots(figsize=(13, 6.5))
    chart_data = df_results[~df_results["classifier"].str.contains("pca")].copy()
    sns.barplot(data=chart_data, x="experiment_id", y="val_f1_macro", hue="classifier", ax=ax, palette="deep")
    ax.set_title("Comparativa de Desempeño en Validación (F1-Score Macro) por Experimento y Clasificador",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Experimento (Conjunto de Características)", fontsize=11, fontweight="bold")
    ax.set_ylabel("F1-Score Macro en Validación", fontsize=11, fontweight="bold")
    ax.set_ylim([0, 1.05])
    ax.legend(title="Clasificador", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    fig.savefig("outputs/figures/model_comparison_chart.png", dpi=300)
    plt.close()
    print("[INFO] Figura model_comparison_chart.png guardada.")

    # Criterio estricto de seleccion: Ganador por F1-Macro en Validacion
    best_row = df_results.iloc[0]
    best_key = f"{best_row['experiment_id']}_{best_row['classifier']}"
    best_info = trained_pipelines[best_key]

    print(f"\n=======================================================")
    print(f"=== MODELO GANADOR SELECCIONADO ===")
    print(f"Experimento: {best_row['experiment_id']} ({best_row['features_family']})")
    print(f"Clasificador: {best_row['classifier']}")
    print(f"Parametros: {best_row['best_params']}")
    print(f"Val F1-Macro: {best_row['val_f1_macro']:.4f} | Val Acc: {best_row['val_accuracy']:.4f}")
    print(f"=======================================================")

    # Reentrenar el Pipeline ganador sobre la union de Train + Val para maximizar soporte estadistico
    best_data = np.load(os.path.join(features_dir, best_info["feature_file"]), allow_pickle=True)
    X_all = best_data["X"]
    y_all = best_data["y"]
    splits_all = best_data["splits"]

    train_val_mask = (splits_all == "train") | (splits_all == "val")
    X_train_val = X_all[train_val_mask]
    y_train_val = y_all[train_val_mask]

    final_pipeline = best_info["pipeline"]
    final_pipeline.fit(X_train_val, y_train_val)

    # Serializar modelo definitivo y metadatos
    final_model_path = os.path.join(models_dir, "final_model.joblib")
    model_artifact = {
        "pipeline": final_pipeline,
        "experiment_id": best_row["experiment_id"],
        "features_family": best_row["features_family"],
        "classifier": best_row["classifier"],
        "best_params": best_info["best_params"],
        "feature_file": best_info["feature_file"],
        "classes": config["classes"],
        "val_f1_macro": best_row["val_f1_macro"],
        "val_accuracy": best_row["val_accuracy"]
    }
    joblib.dump(model_artifact, final_model_path)
    print(f"[INFO] Modelo final serializado exitosamente en: {final_model_path}")

    # Guardar metadatos en JSON
    json_meta = {
        "experiment_id": best_row["experiment_id"],
        "features_family": best_row["features_family"],
        "classifier": best_row["classifier"],
        "best_params": str(best_info["best_params"]),
        "val_f1_macro": float(best_row["val_f1_macro"]),
        "val_accuracy": float(best_row["val_accuracy"]),
        "feature_file": best_info["feature_file"],
        "saved_path": final_model_path
    }
    return df_results, best_info


if __name__ == "__main__":
    cfg = load_config("config/config.yaml")
    run_model_comparison(cfg)

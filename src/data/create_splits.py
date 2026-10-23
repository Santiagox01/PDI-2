"""
Generacion de particiones estratificadas y agrupadas por imagen original (70% Train, 15% Val, 15% Test)
para la prevencion estricta de fuga de informacion (Data Leakage).
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

from collections import Counter
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_grouped_stratified_splits(config: dict) -> tuple:
    manifest_path = config["paths"]["manifest"]
    splits_dir = config["paths"]["splits"]
    seed = config.get("seed", 42)
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    df = pd.read_csv(manifest_path)
    print(f"[INFO] Creando particiones para {len(df)} monedas agrupadas por original_image_id...")

    unique_images = df["original_image_id"].unique()
    img_to_coins = df.groupby("original_image_id")["class_id"].apply(list).to_dict()
    total_per_class = df["class_id"].value_counts().to_dict()
    num_classes = len(config["classes"])

    target_train = {c: total_per_class.get(c, 0) * 0.70 for c in range(num_classes)}
    target_val = {c: total_per_class.get(c, 0) * 0.15 for c in range(num_classes)}
    target_test = {c: total_per_class.get(c, 0) * 0.15 for c in range(num_classes)}

    rng = np.random.RandomState(seed)
    best_score = float("inf")
    best_assignment = None

    # Optimizacion combinatoria para balance estratificado por grupos disjuntos
    for _ in range(12000):
        shuffled = rng.permutation(unique_images)
        train_imgs = set(shuffled[:105])
        val_imgs = set(shuffled[105:128])
        test_imgs = set(shuffled[128:])

        counts_train = Counter()
        counts_val = Counter()
        counts_test = Counter()

        for img in train_imgs:
            counts_train.update(img_to_coins[img])
        for img in val_imgs:
            counts_val.update(img_to_coins[img])
        for img in test_imgs:
            counts_test.update(img_to_coins[img])

        if len(counts_val) < num_classes or len(counts_test) < num_classes:
            continue
        if any(counts_val[c] < 2 for c in range(num_classes)) or any(counts_test[c] < 2 for c in range(num_classes)):
            continue

        score = 0.0
        for c in range(num_classes):
            score += ((counts_train[c] - target_train[c]) ** 2) / (target_train[c] + 1e-5)
            score += ((counts_val[c] - target_val[c]) ** 2) / (target_val[c] + 1e-5)
            score += ((counts_test[c] - target_test[c]) ** 2) / (target_test[c] + 1e-5)

        if score < best_score:
            best_score = score
            best_assignment = (train_imgs, val_imgs, test_imgs)

    if best_assignment is None:
        raise RuntimeError("No se encontro una asignacion valida que satisfaga las restricciones de particion.")

    train_imgs, val_imgs, test_imgs = best_assignment

    # Asignar particion en el dataframe
    def assign_split(row):
        img_id = row["original_image_id"]
        if img_id in train_imgs:
            return "train"
        elif img_id in val_imgs:
            return "val"
        elif img_id in test_imgs:
            return "test"
        return "unknown"

    df["split"] = df.apply(assign_split, axis=1)

    # Validar que los conjuntos sean estrictamente disjuntos por imagen de origen
    train_ids = set(df[df["split"] == "train"]["original_image_id"])
    val_ids = set(df[df["split"] == "val"]["original_image_id"])
    test_ids = set(df[df["split"] == "test"]["original_image_id"])

    assert len(train_ids.intersection(val_ids)) == 0, "Fuga de informacion detectada entre train y val!"
    assert len(train_ids.intersection(test_ids)) == 0, "Fuga de informacion detectada entre train y test!"
    assert len(val_ids.intersection(test_ids)) == 0, "Fuga de informacion detectada entre val y test!"
    print("[VALIDACION] Verificado: Los conjuntos de imagenes son estrictamente disjuntos (0% de fuga).")

    # Guardar particiones separadas y actualizar manifiesto
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    df.to_csv(manifest_path, index=False, encoding="utf-8")
    df_train.to_csv(os.path.join(splits_dir, "train.csv"), index=False, encoding="utf-8")
    df_val.to_csv(os.path.join(splits_dir, "validation.csv"), index=False, encoding="utf-8")
    df_test.to_csv(os.path.join(splits_dir, "test.csv"), index=False, encoding="utf-8")

    print(f"[INFO] Train split: {len(df_train)} monedas ({len(train_ids)} fotos)")
    print(f"[INFO] Val split: {len(df_val)} monedas ({len(val_ids)} fotos)")
    print(f"[INFO] Test split: {len(df_test)} monedas ({len(test_ids)} fotos)")

    # Resumen y tabla comparativa
    summary_records = []
    classes_cfg = config["classes"]
    for c in range(num_classes):
        c_name = classes_cfg[c]["name"]
        n_tr = (df_train["class_id"] == c).sum()
        n_va = (df_val["class_id"] == c).sum()
        n_te = (df_test["class_id"] == c).sum()
        n_tot = n_tr + n_va + n_te
        summary_records.append({
            "class_id": c,
            "class_name": c_name,
            "train_count": n_tr,
            "val_count": n_va,
            "test_count": n_te,
            "total_count": n_tot,
            "train_pct": (n_tr / n_tot) * 100 if n_tot > 0 else 0,
            "val_pct": (n_va / n_tot) * 100 if n_tot > 0 else 0,
            "test_pct": (n_te / n_tot) * 100 if n_tot > 0 else 0
        })

    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv("outputs/tables/splits_summary.csv", index=False, encoding="utf-8")
    print("\n=== RESUMEN DE PARTICIONES ESTRATIFICADAS ===")
    print(df_summary[["class_id", "class_name", "train_count", "val_count", "test_count", "total_count"]].to_string(index=False))

    # Grafico comparativo
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(num_classes)
    width = 0.25

    ax.bar(x - width, df_summary["train_count"], width, label="Entrenamiento (70%)", color="#2b5c8f")
    ax.bar(x, df_summary["val_count"], width, label="Validación (15%)", color="#e27c38")
    ax.bar(x + width, df_summary["test_count"], width, label="Prueba (15%)", color="#3ca55c")

    ax.set_title("Distribución de Clases por Partición (Agrupamiento por Escena sin Fuga)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Denominación de Euro", fontsize=11, fontweight="bold")
    ax.set_ylabel("Número de Instancias", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df_summary["class_name"], rotation=15)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig("outputs/figures/splits_distribution.png", dpi=300)
    plt.close()
    print("[INFO] Figura splits_distribution.png guardada.")

    return df_train, df_val, df_test


if __name__ == "__main__":
    cfg = load_config("config/config.yaml")
    create_grouped_stratified_splits(cfg)

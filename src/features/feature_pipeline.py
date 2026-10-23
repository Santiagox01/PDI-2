"""
Pipeline unificado y modular de extraccion de caracteristicas de vision por computadora.
Ejecuta HOG, Forma, Color y Textura de forma independiente y combinada,
registrando dimensionalidad, tiempos de computo y guardando matrices reproducibles (.npz y .csv).
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.features.hog_features import extract_hog_dataset, generate_hog_visualization_figure
from src.features.shape_features import extract_shape_dataset, SHAPE_FEATURE_NAMES
from src.features.color_features import extract_color_dataset
from src.features.texture_features import extract_texture_dataset


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_feature_extraction_pipeline(config: dict) -> dict:
    manifest_path = config["paths"]["manifest"]
    features_dir = config["paths"]["features"]
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)

    df_manifest = pd.read_csv(manifest_path)
    n_samples = len(df_manifest)
    print(f"[INFO] Iniciando extraccion de caracteristicas para {n_samples} monedas...")

    roi_paths = df_manifest["roi_path"].tolist()
    mask_paths = df_manifest["mask_path"].tolist()
    labels = df_manifest["class_id"].to_numpy(dtype=np.int64)
    coin_ids = df_manifest["coin_id"].to_numpy()
    splits = df_manifest["split"].to_numpy()

    # 1. HOG Obligatorio
    print("\n--- Extrayendo Familia 1: HOG (Histogram of Oriented Gradients) ---")
    t0 = time.time()
    hog_cfg = config["features"]["hog"]
    X_hog = extract_hog_dataset(
        roi_paths,
        orientations=hog_cfg["orientations"],
        pixels_per_cell=tuple(hog_cfg["pixels_per_cell"]),
        cells_per_block=tuple(hog_cfg["cells_per_block"])
    )
    t_hog = (time.time() - t0)
    ms_hog = (t_hog / n_samples) * 1000.0
    print(f"[HOG] Dimension del vector: {X_hog.shape[1]} | Tiempo total: {t_hog:.2f}s ({ms_hog:.2f} ms/moneda)")

    # 2. Caracteristicas de Forma Obligatorias
    print("\n--- Extrayendo Familia 2: Forma y Momentos de Hu ---")
    t0 = time.time()
    X_shape = extract_shape_dataset(mask_paths)
    t_shape = (time.time() - t0)
    ms_shape = (t_shape / n_samples) * 1000.0
    print(f"[Forma] Dimension del vector: {X_shape.shape[1]} | Tiempo total: {t_shape:.2f}s ({ms_shape:.2f} ms/moneda)")

    # 3. Caracteristicas de Color (HSV, CIELAB, RGB, Histogramas)
    print("\n--- Extrayendo Familia 3: Color en Mascara (HSV, CIELAB, RGB) ---")
    t0 = time.time()
    color_bins = config["features"]["color"]["hist_bins_per_channel"]
    X_color = extract_color_dataset(roi_paths, mask_paths, hist_bins=color_bins)
    t_color = (time.time() - t0)
    ms_color = (t_color / n_samples) * 1000.0
    print(f"[Color] Dimension del vector: {X_color.shape[1]} | Tiempo total: {t_color:.2f}s ({ms_color:.2f} ms/moneda)")

    # 4. Caracteristicas de Textura (LBP uniforme y GLCM)
    print("\n--- Extrayendo Familia 4: Textura (LBP uniforme + GLCM) ---")
    t0 = time.time()
    X_texture = extract_texture_dataset(roi_paths, mask_paths)
    t_texture = (time.time() - t0)
    ms_texture = (t_texture / n_samples) * 1000.0
    print(f"[Textura] Dimension del vector: {X_texture.shape[1]} | Tiempo total: {t_texture:.2f}s ({ms_texture:.2f} ms/moneda)")

    # 5. Vector Combinado (E5)
    print("\n--- Construyendo Familia 5: Representacion Combinada (Multimodal) ---")
    X_combined = np.hstack([X_hog, X_shape, X_color, X_texture])
    ms_combined = ms_hog + ms_shape + ms_color + ms_texture
    print(f"[Combinado] Dimension total: {X_combined.shape[1]} ({X_hog.shape[1]} HOG + {X_shape.shape[1]} Forma + {X_color.shape[1]} Color + {X_texture.shape[1]} Textura)")

    # Guardar matrices serializadas en formato NPZ reproducible
    np.savez_compressed(os.path.join(features_dir, "features_hog.npz"),
                        X=X_hog, y=labels, coin_ids=coin_ids, splits=splits)
    np.savez_compressed(os.path.join(features_dir, "features_shape.npz"),
                        X=X_shape, y=labels, coin_ids=coin_ids, splits=splits)
    np.savez_compressed(os.path.join(features_dir, "features_color.npz"),
                        X=X_color, y=labels, coin_ids=coin_ids, splits=splits)
    np.savez_compressed(os.path.join(features_dir, "features_texture.npz"),
                        X=X_texture, y=labels, coin_ids=coin_ids, splits=splits)
    np.savez_compressed(os.path.join(features_dir, "features_combined.npz"),
                        X=X_combined, y=labels, coin_ids=coin_ids, splits=splits)
    print(f"[INFO] Matrices .npz guardadas exitosamente en {features_dir}")

    # Tabla resumen de familias de caracteristicas
    families_summary = [
        {
            "Familia": "HOG (Histogram of Oriented Gradients)",
            "Experimento": "E1",
            "Dimension": X_hog.shape[1],
            "Tiempo_ms_por_moneda": round(ms_hog, 2),
            "Propiedades_Fisicas": "Bordes de relieve, mapa de orientacion de gradientes del cuño",
            "Invarianzas": "Invarianza a iluminacion monotonica (bloques L2-Hys)"
        },
        {
            "Familia": "Geometría y Forma (Contorno + Hu)",
            "Experimento": "E2",
            "Dimension": X_shape.shape[1],
            "Tiempo_ms_por_moneda": round(ms_shape, 2),
            "Propiedades_Fisicas": "Circularidad, convexidad, radio aparente, 7 momentos invariantes de Hu",
            "Invarianzas": "Invarianza estricta a traslacion, escala y rotacion 2D"
        },
        {
            "Familia": "Colorimetría (HSV + CIELAB + RGB)",
            "Experimento": "E3",
            "Dimension": X_color.shape[1],
            "Tiempo_ms_por_moneda": round(ms_color, 2),
            "Propiedades_Fisicas": "Aleaciones metalicas: Cobre (1c,2c,5c), Oro nordico (10c,20c,50c), Bimetalicas (1e,2e)",
            "Invarianzas": "Invarianza a sombras (espacio perceptual CIELAB canal a*, b*)"
        },
        {
            "Familia": "Textura (LBP Uniforme + GLCM)",
            "Experimento": "E4",
            "Dimension": X_texture.shape[1],
            "Tiempo_ms_por_moneda": round(ms_texture, 2),
            "Propiedades_Fisicas": "Micro-textura superficial, contraste y homogeneidad del grabado",
            "Invarianzas": "Invarianza a rotacion (LBP circular uniforme y promedio GLCM)"
        },
        {
            "Familia": "Combinada Multimodal",
            "Experimento": "E5",
            "Dimension": X_combined.shape[1],
            "Tiempo_ms_por_moneda": round(ms_combined, 2),
            "Propiedades_Fisicas": "Fusion completa de estructura, geometria, aleacion y textura",
            "Invarianzas": "Complementariedad robusta multivariada"
        }
    ]
    df_fam = pd.DataFrame(families_summary)
    df_fam.to_csv("outputs/tables/feature_families_summary.csv", index=False, encoding="utf-8")
    print("\n=== RESUMEN DE FAMILIAS DE CARACTERISTICAS ===")
    print(df_fam[["Familia", "Experimento", "Dimension", "Tiempo_ms_por_moneda"]].to_string(index=False))

    # Generar visualizaciones academicas
    # 1. HOG Visualizacion
    sample_coins = df_manifest.drop_duplicates(subset=["class_id"]).sort_values("class_id").to_dict("records")
    generate_hog_visualization_figure(sample_coins)

    # 2. Scatter plot de color (CIELAB a* vs b* y RGB ratios)
    generate_color_scatter_figure(X_color, df_manifest)

    # 3. Scatter plot de Forma (Circularidad vs Solidez)
    generate_shape_scatter_figure(X_shape, df_manifest)

    return {
        "X_hog": X_hog,
        "X_shape": X_shape,
        "X_color": X_color,
        "X_texture": X_texture,
        "X_combined": X_combined,
        "y": labels,
        "splits": splits
    }


def generate_color_scatter_figure(X_color: np.ndarray, df_manifest: pd.DataFrame):
    """Muestra la capacidad discriminante del espacio CIELAB (canales a* y b*) para separar aleaciones."""
    # En X_color: lab_stats comienza en indice 9 (media l, std l, med l, media a, std a, med a, media b...)
    # media_a esta en indice 9 + 3 = 12; media_b esta en indice 9 + 6 = 15
    mean_a = X_color[:, 12]
    mean_b = X_color[:, 15]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    palette = sns.color_palette("tab10", 8)
    class_names = df_manifest["class_name"].unique()

    for cls_id in range(8):
        mask = (df_manifest["class_id"] == cls_id)
        c_name = df_manifest[mask]["class_name"].iloc[0]
        ax.scatter(mean_a[mask], mean_b[mask], label=c_name, s=70, alpha=0.85,
                   color=palette[cls_id], edgecolors="black", linewidth=0.7)

    ax.set_title("Espacio Cromático CIELAB ($a^*$ vs $b^*$): Discriminación de Aleaciones",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Canal $a^*$ (Verde $\\leftarrow$ Rojo)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Canal $b^*$ (Azul $\\leftarrow$ Amarillo)", fontsize=11, fontweight="bold")
    ax.legend(title="Denominación", loc="best", framealpha=0.9)
    plt.tight_layout()
    fig.savefig("outputs/figures/color_features_scatter.png", dpi=300)
    plt.close()
    print("[INFO] Figura color_features_scatter.png guardada.")


def generate_shape_scatter_figure(X_shape: np.ndarray, df_manifest: pd.DataFrame):
    """Muestra la distribucion geometrica de circularidad vs solidez."""
    # circularity es indice 2, solidity es indice 4
    circ = X_shape[:, 2]
    solidity = X_shape[:, 4]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    palette = sns.color_palette("tab10", 8)

    for cls_id in range(8):
        mask = (df_manifest["class_id"] == cls_id)
        c_name = df_manifest[mask]["class_name"].iloc[0]
        ax.scatter(circ[mask], solidity[mask], label=c_name, s=65, alpha=0.8,
                   color=palette[cls_id], edgecolors="black", linewidth=0.6)

    ax.set_title("Descriptores de Forma: Circularidad vs Solidez por Denominación",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Circularidad ($4\\pi A / P^2$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Solidez ($A / A_{envolvente}$)", fontsize=11, fontweight="bold")
    ax.legend(title="Denominación", loc="lower left", framealpha=0.9)
    plt.tight_layout()
    fig.savefig("outputs/figures/shape_features_scatter.png", dpi=300)
    plt.close()
    print("[INFO] Figura shape_features_scatter.png guardada.")


if __name__ == "__main__":
    cfg = load_config("config/config.yaml")
    run_feature_extraction_pipeline(cfg)

"""
Extraccion, normalizacion a 128x128 y generacion de mascaras binarias para todas las monedas.
Compara alternativas de segmentacion y produce figuras analiticas para el informe.
"""

import os
import sys
# Asegurar que la raiz del proyecto este en sys.path
sys.path.insert(0, os.path.abspath("."))

import yaml
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.preprocessing.image_loader import load_image, resize_maintaining_aspect_ratio
from src.preprocessing.filtering import apply_clahe_color, apply_clahe_gray, apply_gaussian_filter
from src.preprocessing.segmentation import (
    segment_otsu, segment_adaptive, segment_color_hsv,
    segment_coins_watershed, segment_single_coin_roi
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_all_rois(config: dict) -> pd.DataFrame:
    manifest_path = config["paths"]["manifest"]
    rois_dir = config["paths"]["rois"]
    masks_dir = config["paths"]["masks"]
    target_size = tuple(config["preprocessing"]["target_roi_size"])  # (128, 128)
    padding_ratio = config["preprocessing"]["padding_ratio"]

    os.makedirs(rois_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    df_manifest = pd.read_csv(manifest_path)
    print(f"[INFO] Extrayendo y normalizando ROIs para {len(df_manifest)} monedas...")

    roi_paths = []
    mask_paths = []

    # Cache de imagenes originales para no recargarlas innecesariamente
    current_img_id = None
    current_img = None

    for _, row in df_manifest.iterrows():
        img_id = row["original_image_id"]
        if img_id != current_img_id:
            current_img = load_image(row["image_path"])
            current_img_id = img_id

        h_full, w_full = current_img.shape[:2]
        xmin = row["bbox_xmin"]
        ymin = row["bbox_ymin"]
        xmax = row["bbox_xmax"]
        ymax = row["bbox_ymax"]

        bw = xmax - xmin
        bh = ymax - ymin

        # Aplicar margen de seguridad
        pad_x = int(bw * padding_ratio)
        pad_y = int(bh * padding_ratio)

        crop_xmin = max(0, xmin - pad_x)
        crop_ymin = max(0, ymin - pad_y)
        crop_xmax = min(w_full, xmax + pad_x)
        crop_ymax = min(h_full, ymax + pad_y)

        coin_crop = current_img[crop_ymin:crop_ymax, crop_xmin:crop_xmax].copy()

        # Normalizar dimensiones conservando proporcion
        roi_norm, _, _ = resize_maintaining_aspect_ratio(coin_crop, target_size=target_size)

        # Generar mascara binaria precisa
        mask_norm = segment_single_coin_roi(roi_norm)

        coin_id = row["coin_id"]
        roi_filename = f"{coin_id}.png"
        mask_filename = f"{coin_id}_mask.png"

        roi_save_path = os.path.join(rois_dir, roi_filename)
        mask_save_path = os.path.join(masks_dir, mask_filename)

        cv2.imwrite(roi_save_path, roi_norm)
        cv2.imwrite(mask_save_path, mask_norm)

        roi_paths.append(roi_save_path)
        mask_paths.append(mask_save_path)

    df_manifest["roi_path"] = roi_paths
    df_manifest["mask_path"] = mask_paths
    df_manifest.to_csv(manifest_path, index=False, encoding="utf-8")
    print(f"[INFO] ROIs guardadas en: {rois_dir} ({len(roi_paths)} archivos)")
    print(f"[INFO] Mascaras guardadas en: {masks_dir} ({len(mask_paths)} archivos)")

    # 1. Comparacion sistematica de metodos de segmentacion sobre una escena representativa
    generate_segmentation_comparison_figure(config)

    # 2. Galeria de etapas de procesamiento por ROI
    generate_roi_stages_figure(df_manifest)

    return df_manifest


def generate_segmentation_comparison_figure(config: dict):
    """Compara metodos clasicos de segmentacion sobre una fotografia completa con multiples monedas."""
    sample_img_path = os.path.join(config["paths"]["raw_images"], "001.jpg")
    if not os.path.exists(sample_img_path):
        return

    bgr = load_image(sample_img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

    # Metodo A: Otsu
    otsu_mask = segment_otsu(blurred, invert=True)

    # Metodo B: Adaptativo Gaussiano
    adaptive_mask = segment_adaptive(blurred, block_size=31, C=5, invert=True)

    # Metodo C: Espacio de Color HSV
    hsv_mask = segment_color_hsv(bgr)

    # Metodo D: Watershed con Transformada de Distancia
    ws_mask, markers, dist = segment_coins_watershed(bgr)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title("(a) Imagen Original", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(otsu_mask, cmap="gray")
    axes[1].set_title("(b) Umbralización Otsu", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(adaptive_mask, cmap="gray")
    axes[2].set_title("(c) Adaptativo Gaussiano", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(hsv_mask, cmap="gray")
    axes[3].set_title("(d) Espacio de Color HSV", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    axes[4].imshow(ws_mask, cmap="gray")
    axes[4].set_title("(e) Distancia + Watershed", fontsize=11, fontweight="bold")
    axes[4].axis("off")

    plt.suptitle("Comparativa Experimental de Métodos de Segmentación en Escena con Múltiples Monedas",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig("outputs/figures/segmentation_comparison.png", dpi=300)
    plt.close()
    print("[INFO] Figura segmentation_comparison.png guardada.")


def generate_roi_stages_figure(df_manifest: pd.DataFrame):
    """Muestra para 4 denominaciones representativas las etapas: Original, CLAHE, Mascara Binaria y Contorno."""
    sample_coins = df_manifest.drop_duplicates(subset=["class_id"]).head(4)
    fig, axes = plt.subplots(len(sample_coins), 4, figsize=(14, 12))

    for idx, (_, row) in enumerate(sample_coins.iterrows()):
        roi_bgr = cv2.imread(row["roi_path"])
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)

        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        clahe_bgr = apply_clahe_color(roi_bgr, clip_limit=2.5)
        clahe_rgb = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB)

        # Superponer contorno verde en la imagen CLAHE
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_vis = clahe_rgb.copy()
        cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)

        axes[idx, 0].imshow(roi_rgb)
        axes[idx, 0].set_ylabel(row["class_name"], fontsize=12, fontweight="bold")
        if idx == 0:
            axes[idx, 0].set_title("1. ROI Normalizada (128x128)", fontsize=11, fontweight="bold")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(clahe_rgb)
        if idx == 0:
            axes[idx, 1].set_title("2. Realce CLAHE (CIELAB)", fontsize=11, fontweight="bold")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(mask, cmap="gray")
        if idx == 0:
            axes[idx, 2].set_title("3. Máscara Binaria Limpia", fontsize=11, fontweight="bold")
        axes[idx, 2].axis("off")

        axes[idx, 3].imshow(contour_vis)
        if idx == 0:
            axes[idx, 3].set_title("4. Detección de Contorno", fontsize=11, fontweight="bold")
        axes[idx, 3].axis("off")

    plt.suptitle("Etapas del Pipeline de Preprocesamiento y Segmentación de Monedas",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig("outputs/figures/roi_segmentation_stages.png", dpi=300)
    plt.close()
    print("[INFO] Figura roi_segmentation_stages.png guardada.")


if __name__ == "__main__":
    cfg = load_config("config/config.yaml")
    extract_all_rois(cfg)

"""
Modulo de extraccion de descriptores HOG (Histogram of Oriented Gradients).
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import cv2
import numpy as np
from skimage.feature import hog
import matplotlib.pyplot as plt


def extract_hog_single(roi_img: np.ndarray, orientations: int = 9,
                       pixels_per_cell: tuple = (8, 8), cells_per_block: tuple = (2, 2),
                       block_norm: str = "L2-Hys", visualize: bool = False):
    """
    Extrae el vector HOG a partir de una ROI normalizada (128x128).
    Si visualize=True, retorna (descriptor, hog_image).
    """
    if len(roi_img.shape) == 3:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_img

    # Aplicar realce suave de contraste para destacar grabados y bordes
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    res = hog(
        enhanced,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm=block_norm,
        visualize=visualize,
        feature_vector=True
    )
    return res


def extract_hog_dataset(roi_paths: list, orientations: int = 9,
                        pixels_per_cell: tuple = (8, 8), cells_per_block: tuple = (2, 2)) -> np.ndarray:
    """Extrae descriptores HOG para un conjunto de rutas de ROIs."""
    features = []
    for p in roi_paths:
        img = cv2.imread(p)
        desc = extract_hog_single(img, orientations=orientations,
                                  pixels_per_cell=pixels_per_cell,
                                  cells_per_block=cells_per_block,
                                  visualize=False)
        features.append(desc)
    return np.array(features, dtype=np.float32)


def generate_hog_visualization_figure(sample_coins_df: list, save_path: str = "outputs/figures/hog_visualization.png"):
    """Genera una figura comparativa de la imagen ROI original y su representacion de gradientes HOG."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = len(sample_coins_df)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.5))

    for i, item in enumerate(sample_coins_df):
        roi_bgr = cv2.imread(item["roi_path"])
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        desc, hog_vis = extract_hog_single(roi_bgr, visualize=True)

        axes[0, i].imshow(roi_rgb)
        axes[0, i].set_title(f"{item['class_name']}\n(ROI 128x128)", fontsize=11, fontweight="bold")
        axes[0, i].axis("off")

        axes[1, i].imshow(hog_vis, cmap="inferno")
        axes[1, i].set_title(f"HOG ({len(desc)} dim)", fontsize=10, fontweight="bold")
        axes[1, i].axis("off")

    plt.suptitle("Representación de Gradientes Orientados (HOG) por Denominación", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Visualizacion HOG guardada en {save_path}")

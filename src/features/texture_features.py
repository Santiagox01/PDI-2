"""
Modulo de extraccion de caracteristicas de textura: LBP uniforme (P=8,R=1 y P=16,R=2)
y matrices de co-ocurrencia de niveles de gris (GLCM: contraste, disimilitud, homogeneidad, energia, correlacion).
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops


def extract_texture_single(roi_img: np.ndarray, mask: np.ndarray = None,
                           glcm_distances: list = None,
                           glcm_angles: list = None) -> np.ndarray:
    """
    Extrae descriptores de textura combinando LBP uniforme multiescala
    y estadisticos de matrices GLCM.
    """
    if glcm_distances is None:
        glcm_distances = [1, 2, 3]
    if glcm_angles is None:
        glcm_angles = [0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]

    if len(roi_img.shape) == 3:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_img

    # Realce adaptativo suave para resaltar el micro-relieve del cuño de la moneda
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    if mask is not None:
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask
        coin_pixels = mask_gray > 127
    else:
        coin_pixels = np.ones(enhanced.shape, dtype=bool)

    # 1. LBP uniforme P=8, R=1 (10 patrones uniformes posibles)
    p1, r1 = 8, 1
    lbp1 = local_binary_pattern(enhanced, p1, r1, method="uniform")
    lbp1_coin = lbp1[coin_pixels]
    n_bins1 = p1 + 2  # 10 bins
    hist_lbp1, _ = np.histogram(lbp1_coin, bins=n_bins1, range=(0, n_bins1), density=True)

    # 2. LBP uniforme P=16, R=2 (18 patrones uniformes posibles)
    p2, r2 = 16, 2
    lbp2 = local_binary_pattern(enhanced, p2, r2, method="uniform")
    lbp2_coin = lbp2[coin_pixels]
    n_bins2 = p2 + 2  # 18 bins
    hist_lbp2, _ = np.histogram(lbp2_coin, bins=n_bins2, range=(0, n_bins2), density=True)

    # 3. GLCM (Gray-Level Co-occurrence Matrix)
    # Cuantizar a 32 niveles de gris para robustez y densidad espectral
    quantized = (enhanced // 8).astype(np.uint8)

    # Enmascarar fondo para que no interfiera en la co-ocurrencia
    quantized[~coin_pixels] = 0

    glcm = graycomatrix(quantized, distances=glcm_distances, angles=glcm_angles,
                        levels=32, symmetric=True, normed=True)

    glcm_props = []
    props_to_calc = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    for prop in props_to_calc:
        val_matrix = graycoprops(glcm, prop)  # shape (len(distances), len(angles))
        # Promediar a traves de los angulos para obtener invarianza a rotacion
        val_mean_per_dist = np.mean(val_matrix, axis=1)
        glcm_props.extend(val_mean_per_dist.tolist())

    features = np.concatenate([hist_lbp1, hist_lbp2, np.array(glcm_props, dtype=np.float32)])
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    return features


def extract_texture_dataset(roi_paths: list, mask_paths: list) -> np.ndarray:
    """Extrae descriptores de textura para un conjunto de imagenes."""
    features = []
    for r_p, m_p in zip(roi_paths, mask_paths):
        roi = cv2.imread(r_p)
        mask = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
        feat = extract_texture_single(roi, mask)
        features.append(feat)
    return np.array(features, dtype=np.float32)

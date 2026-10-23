"""
Modulo de extraccion de caracteristicas cromaticas (HSV, CIELAB, RGB) calculadas
exclusivamente dentro de la mascara de la moneda (excluyendo el fondo).
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import cv2
import numpy as np


def extract_color_single(roi_bgr: np.ndarray, mask: np.ndarray, hist_bins: int = 8) -> np.ndarray:
    """
    Extrae descriptores de color (medias, desviaciones, medianas, ratios e histogramas reducidos)
    evaluados estrictamente sobre los pixeles pertenecientes a la moneda.
    """
    if len(mask.shape) == 3:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask

    # Identificar indices de pixeles de primer plano (moneda)
    coin_indices = np.where(mask_gray > 127)

    # Si la mascara esta vacia por anomalia, usar el 60% central circular
    if len(coin_indices[0]) < 50:
        h, w = roi_bgr.shape[:2]
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(temp_mask, (w // 2, h // 2), int(min(w, h) * 0.4), 255, -1)
        coin_indices = np.where(temp_mask > 0)

    # Espacios de color
    bgr = roi_bgr.astype(np.float32)
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Extraer pixeles bajo la mascara
    r_pix = rgb[coin_indices[0], coin_indices[1], 0]
    g_pix = rgb[coin_indices[0], coin_indices[1], 1]
    b_pix = rgb[coin_indices[0], coin_indices[1], 2]

    h_pix = hsv[coin_indices[0], coin_indices[1], 0]
    s_pix = hsv[coin_indices[0], coin_indices[1], 1]
    v_pix = hsv[coin_indices[0], coin_indices[1], 2]

    l_pix = lab[coin_indices[0], coin_indices[1], 0]
    a_pix = lab[coin_indices[0], coin_indices[1], 1]
    b_lab_pix = lab[coin_indices[0], coin_indices[1], 2]

    # 1. Estadisticos HSV (9 caracteristicas)
    hsv_stats = [
        np.mean(h_pix), np.std(h_pix), np.median(h_pix),
        np.mean(s_pix), np.std(s_pix), np.median(s_pix),
        np.mean(v_pix), np.std(v_pix), np.median(v_pix)
    ]

    # 2. Estadisticos CIELAB (9 caracteristicas)
    lab_stats = [
        np.mean(l_pix), np.std(l_pix), np.median(l_pix),
        np.mean(a_pix), np.std(a_pix), np.median(a_pix),
        np.mean(b_lab_pix), np.std(b_lab_pix), np.median(b_lab_pix)
    ]

    # 3. Estadisticos RGB y Ratios Cromaticos (8 caracteristicas)
    mean_r = np.mean(r_pix)
    mean_g = np.mean(g_pix)
    mean_b = np.mean(b_pix)
    ratio_rg = mean_r / (mean_g + 1e-5)
    ratio_rb = mean_r / (mean_b + 1e-5)
    rgb_stats = [
        mean_r, np.std(r_pix),
        mean_g, np.std(g_pix),
        mean_b, np.std(b_pix),
        ratio_rg, ratio_rb
    ]

    # 4. Histogramas reducidos (8 bins por canal, normalizados a suma 1)
    hist_h, _ = np.histogram(h_pix, bins=hist_bins, range=(0, 180), density=True)
    hist_s, _ = np.histogram(s_pix, bins=hist_bins, range=(0, 256), density=True)
    hist_v, _ = np.histogram(v_pix, bins=hist_bins, range=(0, 256), density=True)

    hist_l, _ = np.histogram(l_pix, bins=hist_bins, range=(0, 256), density=True)
    hist_a, _ = np.histogram(a_pix, bins=hist_bins, range=(0, 256), density=True)
    hist_b, _ = np.histogram(b_lab_pix, bins=hist_bins, range=(0, 256), density=True)

    hists = np.concatenate([hist_h, hist_s, hist_v, hist_l, hist_a, hist_b])

    features = np.concatenate([hsv_stats, lab_stats, rgb_stats, hists])
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
    return features


def extract_color_dataset(roi_paths: list, mask_paths: list, hist_bins: int = 8) -> np.ndarray:
    """Extrae descriptores de color para todas las muestras."""
    features = []
    for r_p, m_p in zip(roi_paths, mask_paths):
        roi = cv2.imread(r_p)
        mask = cv2.imread(m_p, cv2.IMREAD_GRAYSCALE)
        feat = extract_color_single(roi, mask, hist_bins=hist_bins)
        features.append(feat)
    return np.array(features, dtype=np.float32)

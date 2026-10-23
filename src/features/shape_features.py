"""
Modulo de extraccion de caracteristicas de forma y momentos de Hu invariantes.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import cv2
import numpy as np


SHAPE_FEATURE_NAMES = [
    "area",
    "perimeter",
    "circularity",
    "aspect_ratio",
    "solidity",
    "extent",
    "equiv_diameter",
    "convexity",
    "eccentricity",
    "area_ratio",
    "hu_0", "hu_1", "hu_2", "hu_3", "hu_4", "hu_5", "hu_6"
]


def extract_shape_single(mask_img: np.ndarray, original_crop_width: float = None) -> np.ndarray:
    """
    Extrae descriptores geometricos y momentos de Hu invariantes a partir de la mascara binaria.
    Si se proporciona original_crop_width, se incluye como indicador del diametro real observado.
    """
    if len(mask_img.shape) == 3:
        mask_gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask_img

    _, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(len(SHAPE_FEATURE_NAMES), dtype=np.float32)

    # Tomar el contorno de mayor area
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))

    h_img, w_img = binary.shape[:2]
    total_pixels = float(h_img * w_img)

    # 1. Circularidad (4 * pi * Area / Perimeter^2)
    if perimeter > 0:
        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
    else:
        circularity = 0.0

    # 2. Bounding Box y Relacion de Aspecto
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / float(h) if h > 0 else 1.0

    # 3. Envolvente Convexa y Solidez
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    hull_perimeter = float(cv2.arcLength(hull, True))

    solidity = (area / hull_area) if hull_area > 0 else 1.0

    # 4. Extension (Area / Bounding Box Area)
    bbox_area = float(w * h)
    extent = (area / bbox_area) if bbox_area > 0 else 0.0

    # 5. Diametro equivalente (diametro de un circulo con igual area)
    equiv_diameter = np.sqrt(4.0 * area / np.pi) if area > 0 else 0.0

    # 6. Convexidad (Perimetro Convexo / Perimetro)
    convexity = (hull_perimeter / perimeter) if perimeter > 0 else 1.0

    # 7. Excentricidad ajustando elipse (si hay al menos 5 puntos)
    eccentricity = 0.0
    if len(cnt) >= 5:
        try:
            (cx, cy), (ma, ma_major), angle = cv2.fitEllipse(cnt)
            major = max(ma, ma_major)
            minor = min(ma, ma_major)
            if major > 0:
                eccentricity = np.sqrt(1.0 - (minor / major) ** 2)
        except Exception:
            eccentricity = 0.0

    # 8. Razon de area ocupada respecto a la ROI normalizada
    area_ratio = area / total_pixels if total_pixels > 0 else 0.0

    # 9. Momentos de Hu log-transformados
    moments = cv2.moments(binary)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = []
    for h_val in hu:
        # Transformacion logaritmica con preservacion de signo
        sign = np.sign(h_val)
        val = np.abs(h_val)
        log_h = -sign * np.log10(val + 1e-12)
        hu_log.append(log_h)

    features = [
        area,
        perimeter,
        circularity,
        aspect_ratio,
        solidity,
        extent,
        equiv_diameter,
        convexity,
        eccentricity,
        area_ratio
    ] + hu_log

    vec = np.array(features, dtype=np.float32)
    # Reemplazar NaNs o Infs por seguridad
    vec = np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
    return vec


def extract_shape_dataset(mask_paths: list) -> np.ndarray:
    """Extrae descriptores de forma para un conjunto de mascaras binarias."""
    features = []
    for p in mask_paths:
        mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        feat = extract_shape_single(mask)
        features.append(feat)
    return np.array(features, dtype=np.float32)

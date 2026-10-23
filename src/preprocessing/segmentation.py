"""
Modulo de segmentacion clasica: Otsu, Umbral Adaptativo, Espacio de Color,
Transformada de Distancia y Watershed para separacion de monedas en contacto.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import cv2
import numpy as np
from src.preprocessing.filtering import apply_gaussian_filter, apply_clahe_color, apply_clahe_gray
from src.preprocessing.morphology import clean_mask, fill_holes


def segment_otsu(gray_img: np.ndarray, invert: bool = False) -> np.ndarray:
    """Segmentacion bimodal global mediante metodo de Otsu."""
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray_img, 0, 255, flag + cv2.THRESH_OTSU)
    return mask


def segment_adaptive(gray_img: np.ndarray, block_size: int = 31, C: float = 4.0, invert: bool = False) -> np.ndarray:
    """Segmentacion local adaptativa con distribucion Gaussiana."""
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    mask = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, flag, block_size, C)
    return mask


def segment_color_hsv(bgr_img: np.ndarray) -> np.ndarray:
    """
    Segmentacion cromatico-intensidad combinada en espacio HSV
    para discriminar monedas de fondo no metalico.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    # Rango de saturacion y brillo para monedas metalicas
    lower_val = np.array([0, 15, 30])
    upper_val = np.array([180, 255, 250])
    mask = cv2.inRange(hsv, lower_val, upper_val)
    return mask


def segment_coins_watershed(bgr_img: np.ndarray, min_distance_ratio: float = 0.40) -> tuple:
    """
    Pipeline completo de segmentacion y separacion de monedas en contacto
    mediante Transformada de Distancia Euclidiana y Watershed controlado por marcadores.
    
    Retorna: (mascara_binaria_total, marcadores_watershed, lista_de_contornos)
    """
    # 1. Correccion de iluminacion y suavizado
    enhanced = apply_clahe_color(bgr_img, clip_limit=2.0)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blurred = apply_gaussian_filter(gray, ksize=(7, 7), sigma=1.5)

    # 2. Umbralizacion Otsu con analisis de polaridad de fondo
    # En fondos claros las monedas son mas oscuras; en fondos oscuros son mas claras.
    border_pixels = np.concatenate([blurred[0, :], blurred[-1, :], blurred[:, 0], blurred[:, -1]])
    mean_border = np.mean(border_pixels)
    invert = mean_border > 120  # Fondo claro -> invertir

    thresh = segment_otsu(blurred, invert=invert)

    # 3. Limpieza morfologica y cierre de cavidades
    clean = clean_mask(thresh, open_ksize=5, close_ksize=9)
    filled = fill_holes(clean)

    # 4. Fondo seguro mediante dilatacion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sure_bg = cv2.dilate(filled, kernel, iterations=3)

    # 5. Transformada de Distancia Euclidiana para centros seguros de monedas
    dist_transform = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    max_val = dist_transform.max()
    if max_val > 0:
        _, sure_fg = cv2.threshold(dist_transform, min_distance_ratio * max_val, 255, 0)
    else:
        sure_fg = np.zeros_like(filled)

    sure_fg = np.uint8(sure_fg)

    # 6. Region desconocida
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 7. Marcadores conectados y Watershed
    num_markers, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(bgr_img, markers)

    # Reconstruir mascara binaria donde marcadores > 1
    final_mask = np.zeros(gray.shape, dtype=np.uint8)
    final_mask[markers > 1] = 255

    return final_mask, markers, dist_transform


def filter_coin_contours(binary_mask: np.ndarray, min_area: float = 500.0,
                         min_circularity: float = 0.40) -> list:
    """
    Detecta y filtra contornos por consistencia geometrica de moneda
    (area minima y circularidad = 4*pi*Area / Perimetro^2).
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_coins = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
        if circularity < min_circularity:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0

        # En monedas circulares la relacion de aspecto esta proxima a 1.0 (0.6 a 1.6 tolerando perspectiva)
        if not (0.65 <= aspect_ratio <= 1.55):
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)

        # Mascara individual para esta moneda
        coin_mask = np.zeros(binary_mask.shape, dtype=np.uint8)
        cv2.drawContours(coin_mask, [cnt], -1, 255, -1)

        valid_coins.append({
            "contour": cnt,
            "bbox": (x, y, w, h),
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
            "center": (int(cx), int(cy)),
            "radius": int(radius),
            "mask": coin_mask
        })

    # Ordenar de izquierda a derecha y arriba a abajo
    valid_coins.sort(key=lambda c: (c["bbox"][1] // 100, c["bbox"][0]))
    return valid_coins


def segment_single_coin_roi(roi_bgr: np.ndarray) -> np.ndarray:
    """
    Genera la mascara binaria precisa para un recorte individual de moneda
    centrado en la ROI.
    """
    h, w = roi_bgr.shape[:2]
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    
    # Mascara circular de referencia centrada
    center = (w // 2, h // 2)
    radius = min(w, h) // 2 - 2

    # Otsu threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Si las esquinas son blancas, invertimos
    corners = [thresh[0, 0], thresh[0, -1], thresh[-1, 0], thresh[-1, -1]]
    if np.mean(corners) > 127:
        thresh = cv2.bitwise_not(thresh)

    cleaned = clean_mask(thresh, open_ksize=3, close_ksize=7)
    filled = fill_holes(cleaned)

    # Asegurar que cubra principalmente el disco central
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, 255, -1)

    combined = cv2.bitwise_and(filled, circle_mask)
    if cv2.countNonZero(combined) < 0.25 * np.pi * (radius ** 2):
        # Si la segmentacion fallo por iluminacion especular, usar el disco circular solido
        return circle_mask
    return combined

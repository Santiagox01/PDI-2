"""
Pruebas para los modulos de preprocesamiento, filtrado, morfologia y segmentacion.
"""

import os
import cv2
import numpy as np
import pytest

from src.preprocessing.filtering import apply_clahe_color, apply_gaussian_filter
from src.preprocessing.morphology import clean_mask, fill_holes
from src.preprocessing.segmentation import (
    segment_otsu, segment_adaptive, segment_coins_watershed,
    filter_coin_contours, segment_single_coin_roi
)


def test_filtering_and_clahe():
    dummy = np.full((128, 128, 3), 120, dtype=np.uint8)
    # Agregar gradiente
    for i in range(128):
        dummy[i, :, :] = i

    filtered = apply_gaussian_filter(dummy, ksize=(5, 5))
    assert filtered.shape == dummy.shape
    assert filtered.dtype == np.uint8

    enhanced = apply_clahe_color(dummy, clip_limit=2.0)
    assert enhanced.shape == dummy.shape
    assert enhanced.dtype == np.uint8


def test_morphology_and_hole_filling():
    # Crear un anillo binario con un agujero central
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mask, (50, 50), 30, 255, -1)
    cv2.circle(mask, (50, 50), 10, 0, -1)  # Agujero interior

    assert cv2.countNonZero(mask) < (np.pi * 30**2)

    filled = fill_holes(mask)
    # Al rellenar, el centro debe ser blanco
    assert filled[50, 50] == 255


def test_segmentation_single_roi():
    # Probar con un recorte real del dataset
    sample_roi_path = "data/processed/rois/001_coin_00.png"
    assert os.path.exists(sample_roi_path), f"No existe {sample_roi_path}"

    roi = cv2.imread(sample_roi_path)
    mask = segment_single_coin_roi(roi)

    assert mask.shape == (128, 128)
    assert mask.dtype == np.uint8
    non_zero = cv2.countNonZero(mask)
    # Una moneda normalizada debe ocupar entre el 30% y 95% del area de 128x128
    ratio = non_zero / (128 * 128)
    assert 0.30 <= ratio <= 0.95, f"Proporcion de mascara fuera de rango: {ratio:.2f}"


def test_watershed_scene_detection():
    sample_scene_path = "data/raw/images/001.jpg"
    assert os.path.exists(sample_scene_path)
    scene = cv2.imread(sample_scene_path)

    mask, markers, dist = segment_coins_watershed(scene)
    assert mask.shape == scene.shape[:2]
    assert dist.shape == scene.shape[:2]

    coins = filter_coin_contours(mask, min_area=400, min_circularity=0.35)
    # La imagen 001 contiene 3 monedas
    assert len(coins) >= 1, "Debe detectar al menos una moneda en la imagen 001"

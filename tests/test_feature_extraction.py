"""
Pruebas para las familias de extraccion de caracteristicas (HOG, Forma, Color, Textura).
Verifica dimensiones exactas, ausencia de NaNs/Infs y rangos fisicos validos.
"""

import os
import cv2
import numpy as np
import pytest

from src.features.hog_features import extract_hog_single
from src.features.shape_features import extract_shape_single
from src.features.color_features import extract_color_single
from src.features.texture_features import extract_texture_single


@pytest.fixture
def sample_roi_and_mask():
    roi_path = "data/processed/rois/001_coin_00.png"
    mask_path = "data/processed/masks/001_coin_00_mask.png"
    roi = cv2.imread(roi_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return roi, mask


def test_hog_extraction(sample_roi_and_mask):
    roi, _ = sample_roi_and_mask
    feat = extract_hog_single(roi, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    assert feat.shape == (8100,), f"Dimension HOG esperada (8100,), obtenida {feat.shape}"
    assert not np.isnan(feat).any(), "HOG contiene valores NaN"
    assert not np.isinf(feat).any(), "HOG contiene valores Inf"
    assert np.all(feat >= 0.0), "Los valores de HOG deben ser no negativos"


def test_shape_extraction(sample_roi_and_mask):
    _, mask = sample_roi_and_mask
    feat = extract_shape_single(mask)
    assert feat.shape == (17,), f"Dimension de forma esperada (17,), obtenida {feat.shape}"
    assert not np.isnan(feat).any(), "Forma contiene valores NaN"
    assert not np.isinf(feat).any(), "Forma contiene valores Inf"
    circularity = feat[2]
    # Circularidad teorica en [0, 1] o ligeramente superior por discretizacion
    assert 0.0 <= circularity <= 1.25, f"Circularidad fuera de rango: {circularity}"


def test_color_extraction(sample_roi_and_mask):
    roi, mask = sample_roi_and_mask
    feat = extract_color_single(roi, mask, hist_bins=8)
    assert feat.shape == (74,), f"Dimension de color esperada (74,), obtenida {feat.shape}"
    assert not np.isnan(feat).any(), "Color contiene valores NaN"
    assert not np.isinf(feat).any(), "Color contiene valores Inf"


def test_texture_extraction(sample_roi_and_mask):
    roi, mask = sample_roi_and_mask
    feat = extract_texture_single(roi, mask)
    assert feat.shape == (46,), f"Dimension de textura esperada (46,), obtenida {feat.shape}"
    assert not np.isnan(feat).any(), "Textura contiene valores NaN"
    assert not np.isinf(feat).any(), "Textura contiene valores Inf"

"""
Pruebas unitarias y de integracion para la integridad de datos, imagenes y manifiestos.
"""

import os
import cv2
import pandas as pd
import numpy as np
import pytest

from src.preprocessing.image_loader import load_image, resize_maintaining_aspect_ratio


def test_raw_dataset_structure():
    assert os.path.exists("data/raw/images"), "Directorio data/raw/images no existe"
    assert os.path.exists("data/raw/labels"), "Directorio data/raw/labels no existe"
    images = [f for f in os.listdir("data/raw/images") if f.endswith(".jpg")]
    labels = [f for f in os.listdir("data/raw/labels") if f.endswith(".txt")]
    assert len(images) == 150, f"Se esperaban 150 imagenes, se encontraron {len(images)}"
    assert len(labels) == 150, f"Se esperaban 150 etiquetas, se encontraron {len(labels)}"


def test_manifest_integrity():
    manifest_path = "data/interim/dataset_manifest.csv"
    assert os.path.exists(manifest_path), "dataset_manifest.csv no existe"
    df = pd.read_csv(manifest_path)
    assert len(df) == 336, f"Se esperaban 336 monedas en el manifiesto, se encontraron {len(df)}"
    assert df["class_id"].nunique() == 8, "Debe haber exactamente 8 clases de monedas"
    assert not df["image_path"].isnull().any(), "Hay rutas de imagen nulas"
    assert not df["roi_path"].isnull().any(), "Hay rutas de ROI nulas"
    assert not df["mask_path"].isnull().any(), "Hay rutas de mascara nulas"


def test_image_loading_and_resizing():
    sample_path = "data/raw/images/001.jpg"
    img = load_image(sample_path)
    assert img is not None, "Fallo al cargar la imagen con load_image"
    assert img.shape == (960, 720, 3), f"Dimension inesperada: {img.shape}"

    resized, scale, padding = resize_maintaining_aspect_ratio(img, target_size=(128, 128))
    assert resized.shape == (128, 128, 3), f"Dimension normalizada incorrecta: {resized.shape}"
    assert scale > 0, "Factor de escala invalido"
    assert len(padding) == 4, "Formato de padding incorrecto"

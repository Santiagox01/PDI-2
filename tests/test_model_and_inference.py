"""
Pruebas para el modelo serializado, recuperacion, consistencia de inferencia y predictor de escenas.
"""

import os
import cv2
import joblib
import numpy as np
import pytest

from src.models.predict import EuroCoinPredictor


def test_final_model_artifact():
    model_path = "models/final_model.joblib"
    assert os.path.exists(model_path), f"El modelo final {model_path} no existe"

    artifact = joblib.load(model_path)
    assert "pipeline" in artifact
    assert "classes" in artifact
    assert "features_family" in artifact
    assert len(artifact["classes"]) == 8


def test_predictor_single_roi_inference():
    predictor = EuroCoinPredictor("models/final_model.joblib")
    sample_roi_path = "data/processed/rois/001_coin_00.png"
    assert os.path.exists(sample_roi_path)

    roi = cv2.imread(sample_roi_path)
    result = predictor.predict_single_roi(roi)

    assert "class_id" in result
    assert "class_name" in result
    assert "face_value_eur" in result
    assert "confidence" in result
    assert "latency_ms" in result
    assert 0 <= result["class_id"] <= 7
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["latency_ms"] > 0.0


def test_predictor_scene_inference():
    predictor = EuroCoinPredictor("models/final_model.joblib")
    sample_scene_path = "data/raw/images/001.jpg"
    assert os.path.exists(sample_scene_path)

    scene = cv2.imread(sample_scene_path)
    result = predictor.detect_and_classify_scene(scene)

    assert "num_coins" in result
    assert "total_value_eur" in result
    assert "annotated_image" in result
    assert result["num_coins"] >= 1
    assert result["total_value_eur"] > 0.0
    assert result["annotated_image"].shape == scene.shape

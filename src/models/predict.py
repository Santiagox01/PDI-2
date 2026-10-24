"""
Modulo de inferencia para clasificacion de una moneda individual
o deteccion y conteo de multiples monedas en fotografias de escenas completas.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import time
import joblib
import cv2
import numpy as np

from src.preprocessing.image_loader import load_image, resize_maintaining_aspect_ratio
from src.preprocessing.segmentation import segment_single_coin_roi, segment_coins_watershed, filter_coin_contours
from src.features.hog_features import extract_hog_single
from src.features.shape_features import extract_shape_single
from src.features.color_features import extract_color_single
from src.features.texture_features import extract_texture_single


class EuroCoinPredictor:
    def __init__(self, model_path: str = "models/final_model.joblib"):
        if not os.path.exists(model_path):
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            alt_path = os.path.join(repo_root, model_path)
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                raise FileNotFoundError(f"No se encontro el modelo serializado en: {model_path} ni en {alt_path}")
        
        self.artifact = joblib.load(model_path)
        self.pipeline = self.artifact["pipeline"]
        self.experiment_id = self.artifact.get("experiment_id", "E1")
        self.features_family = self.artifact.get("features_family", "HOG")
        self.classes = self.artifact["classes"]
        print(f"[PREDICTOR] Modelo cargado: {self.features_family} | Pipeline: {type(self.pipeline[-1]).__name__}")

    def extract_features_for_roi(self, roi_norm: np.ndarray, mask_norm: np.ndarray) -> np.ndarray:
        """Extrae el vector de caracteristicas en el orden exacto esperado por el modelo."""
        if self.experiment_id == "E1":
            feat = extract_hog_single(roi_norm)
        elif self.experiment_id == "E2":
            feat = extract_shape_single(mask_norm)
        elif self.experiment_id == "E3":
            feat = extract_color_single(roi_norm, mask_norm, hist_bins=8)
        elif self.experiment_id == "E4":
            feat = extract_texture_single(roi_norm, mask_norm)
        elif self.experiment_id == "E5":
            f_hog = extract_hog_single(roi_norm)
            f_shape = extract_shape_single(mask_norm)
            f_color = extract_color_single(roi_norm, mask_norm, hist_bins=8)
            f_texture = extract_texture_single(roi_norm, mask_norm)
            feat = np.hstack([f_hog, f_shape, f_color, f_texture])
        else:
            feat = extract_hog_single(roi_norm)
        return feat.reshape(1, -1)

    def predict_single_roi(self, roi_bgr: np.ndarray) -> dict:
        """
        Clasifica un recorte de moneda.
        Retorna diccionario con clase predicha, confianza, valor facial y visualizaciones intermedias.
        """
        t0 = time.time()
        # Normalizar a 128x128
        roi_norm, _, _ = resize_maintaining_aspect_ratio(roi_bgr, target_size=(128, 128))
        mask_norm = segment_single_coin_roi(roi_norm)

        # Extraer caracteristicas
        X = self.extract_features_for_roi(roi_norm, mask_norm)

        # Inferencia
        pred_class_id = int(self.pipeline.predict(X)[0])

        # Confianza / Probabilidad
        confidence = 0.0
        probabilities = None
        if hasattr(self.pipeline, "predict_proba"):
            try:
                probs = self.pipeline.predict_proba(X)[0]
                confidence = float(np.max(probs))
                probabilities = probs.tolist()
            except Exception:
                pass
        
        if confidence == 0.0 and hasattr(self.pipeline, "decision_function"):
            decision = self.pipeline.decision_function(X)[0]
            # Softmax aproximado de decision_function
            exp_dec = np.exp(decision - np.max(decision))
            probs = exp_dec / np.sum(exp_dec)
            confidence = float(probs[pred_class_id])
            probabilities = probs.tolist()

        latency_ms = (time.time() - t0) * 1000.0
        cls_info = self.classes.get(pred_class_id, {
            "name": f"Clase {pred_class_id}",
            "value": 0.0,
            "color_group": "unknown"
        })

        return {
            "class_id": pred_class_id,
            "class_name": cls_info["name"],
            "face_value_eur": cls_info["value"],
            "color_group": cls_info["color_group"],
            "confidence": round(confidence, 4),
            "probabilities": probabilities,
            "latency_ms": round(latency_ms, 2),
            "roi_norm": roi_norm,
            "mask_norm": mask_norm
        }

    def detect_and_classify_scene(self, scene_bgr: np.ndarray, min_area: float = 500.0,
                                   min_circularity: float = 0.45) -> dict:
        """
        Detecta, segmenta y clasifica todas las monedas presentes en una escena completa.
        Retorna imagen anotada, conteo por denominacion, suma total en EUR y latencia.
        """
        t0 = time.time()
        h, w = scene_bgr.shape[:2]

        # 1. Segmentacion con Watershed y Transformada de Distancia
        binary_mask, markers, dist = segment_coins_watershed(scene_bgr)

        # 2. Deteccion de regiones candidatas
        coin_regions = filter_coin_contours(binary_mask, min_area=min_area, min_circularity=min_circularity)

        annotated_img = scene_bgr.copy()
        detected_coins = []
        counts_per_class = {c: 0 for c in range(len(self.classes))}
        total_eur = 0.0

        for idx, region in enumerate(coin_regions):
            rx, ry, rw, rh = region["bbox"]

            # Padding de seguridad 5%
            pad_x = int(rw * 0.05)
            pad_y = int(rh * 0.05)
            x1 = max(0, rx - pad_x)
            y1 = max(0, ry - pad_y)
            x2 = min(w, rx + rw + pad_x)
            y2 = min(h, ry + rh + pad_y)

            coin_crop = scene_bgr[y1:y2, x1:x2]
            if coin_crop.shape[0] < 10 or coin_crop.shape[1] < 10:
                continue

            pred = self.predict_single_roi(coin_crop)
            cls_id = pred["class_id"]
            c_name = pred["class_name"]
            val_eur = pred["face_value_eur"]
            conf = pred["confidence"]

            counts_per_class[cls_id] += 1
            total_eur += val_eur

            # Dibujar bounding box y etiqueta
            cv2.rectangle(annotated_img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 3)
            label_text = f"{c_name} ({conf*100:.0f}%)"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_img, (rx, max(0, ry - 25)), (rx + tw + 4, ry), (0, 255, 0), -1)
            cv2.putText(annotated_img, label_text, (rx + 2, max(18, ry - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            detected_coins.append({
                "coin_index": idx + 1,
                "bbox": (rx, ry, rw, rh),
                "class_id": cls_id,
                "class_name": c_name,
                "face_value_eur": val_eur,
                "confidence": conf,
                "circularity": round(region["circularity"], 3),
                "area_px": int(region["area"])
            })

        total_latency_ms = (time.time() - t0) * 1000.0

        return {
            "num_coins": len(detected_coins),
            "coins": detected_coins,
            "counts_per_class": counts_per_class,
            "total_value_eur": round(total_eur, 2),
            "latency_ms": round(total_latency_ms, 2),
            "annotated_image": annotated_img,
            "binary_mask": binary_mask,
            "distance_transform": dist
        }

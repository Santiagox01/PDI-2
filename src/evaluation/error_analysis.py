"""
Modulo de evaluacion final sobre el conjunto de prueba reservado y analisis riguroso de errores
(falsos positivos, falsos negativos, galeria visual e hipotesis tecnicas plausibles).
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import matplotlib
matplotlib.use("Agg")

import time
import yaml
import joblib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation.metrics import compute_comprehensive_metrics
from src.evaluation.confusion_matrices import plot_confusion_matrices


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_test_set_and_analyze_errors(config: dict):
    model_path = os.path.join(config["paths"]["models"], "final_model.joblib")
    features_dir = config["paths"]["features"]
    test_csv_path = os.path.join(config["paths"]["splits"], "test.csv")
    manifest_path = config["paths"]["manifest"]

    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/predictions", exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo final no encontrado en: {model_path}")

    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    feature_file = artifact["feature_file"]
    classes_cfg = artifact["classes"]
    num_classes = len(classes_cfg)
    class_names = [classes_cfg[i]["name"] for i in range(num_classes)]

    print(f"[INFO] Evaluando modelo final '{artifact['classifier']}' con caracteristicas '{artifact['features_family']}'...")

    # Cargar datos de caracteristicas correspondientes
    data = np.load(os.path.join(features_dir, feature_file), allow_pickle=True)
    X = data["X"]
    y = data["y"]
    coin_ids = data["coin_ids"]
    splits = data["splits"]

    # Extraer estrictamente el conjunto de prueba reservado (53 monedas)
    is_test = (splits == "test")
    X_test = X[is_test]
    y_test = y[is_test]
    coin_ids_test = coin_ids[is_test]

    print(f"[INFO] Tamaño del conjunto de prueba reservado: {len(y_test)} monedas.")

    # Medir tiempo de inferencia total y por muestra
    t0 = time.time()
    y_pred = pipeline.predict(X_test)
    inference_time_total = time.time() - t0
    inference_ms_per_coin = (inference_time_total / len(y_test)) * 1000.0
    print(f"[TEST] Inferencia completada en {inference_time_total:.4f}s ({inference_ms_per_coin:.2f} ms/moneda)")

    # Probabilidades / Confianza estimada
    confidences = []
    if hasattr(pipeline, "predict_proba"):
        try:
            probs = pipeline.predict_proba(X_test)
            confidences = [float(np.max(p)) for p in probs]
        except Exception:
            pass

    if not confidences and hasattr(pipeline, "decision_function"):
        decisions = pipeline.decision_function(X_test)
        for dec in decisions:
            exp_d = np.exp(dec - np.max(dec))
            p = exp_d / np.sum(exp_d)
            confidences.append(float(np.max(p)))

    # Metricas integrales
    metrics = compute_comprehensive_metrics(y_test, y_pred, class_names=class_names)
    acc = metrics["accuracy"]
    f1_m = metrics["f1_macro"]
    f1_w = metrics["f1_weighted"]
    prec_m = metrics["precision_macro"]
    rec_m = metrics["recall_macro"]

    print("\n=== METRICAS SOBRE CONJUNTO DE PRUEBA RESERVADO ===")
    print(f"Accuracy:          {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1-Score Macro:    {f1_m:.4f}")
    print(f"F1-Score Ponderado:{f1_w:.4f}")
    print(f"Precisión Macro:   {prec_m:.4f}")
    print(f"Recall Macro:      {rec_m:.4f}")
    print(f"Latencia inferencia: {inference_ms_per_coin:.2f} ms/moneda")

    # Guardar resumen de metricas
    df_metrics = pd.DataFrame([{
        "Modelo": artifact["classifier"],
        "Familia_Caracteristicas": artifact["features_family"],
        "Accuracy": round(acc, 4),
        "F1_Macro": round(f1_m, 4),
        "F1_Ponderado": round(f1_w, 4),
        "Precision_Macro": round(prec_m, 4),
        "Recall_Macro": round(rec_m, 4),
        "Latencia_ms_por_moneda": round(inference_ms_per_coin, 2),
        "Total_Muestras_Prueba": len(y_test)
    }])
    df_metrics.to_csv("outputs/tables/test_evaluation_metrics.csv", index=False, encoding="utf-8")

    # Guardar reporte detallado por clase
    df_report = metrics["classification_report_df"]
    df_report.to_csv("outputs/tables/classification_report.csv", index=True, encoding="utf-8")
    print("\n=== REPORTE DE CLASIFICACION POR DENOMINACION ===")
    print(df_report.to_string())

    # Generar matrices de confusion
    plot_confusion_matrices(metrics["confusion_matrix_abs"], metrics["confusion_matrix_norm"],
                            class_names=class_names, output_dir="outputs/figures")

    # Predicciones detalladas
    df_manifest = pd.read_csv(manifest_path).set_index("coin_id")

    records_pred = []
    for idx, (cid, y_t, y_p) in enumerate(zip(coin_ids_test, y_test, y_pred)):
        c_conf = confidences[idx] if confidences else 1.0
        m_row = df_manifest.loc[cid]
        records_pred.append({
            "coin_id": cid,
            "original_image_id": m_row["original_image_id"],
            "true_class_id": y_t,
            "true_class_name": class_names[y_t],
            "pred_class_id": y_p,
            "pred_class_name": class_names[y_p],
            "confidence": round(c_conf, 4),
            "is_correct": bool(y_t == y_p),
            "roi_path": m_row["roi_path"],
            "mask_path": m_row["mask_path"],
            "crop_width_px": m_row["crop_width_px"]
        })

    df_preds = pd.DataFrame(records_pred)
    df_preds.to_csv("outputs/predictions/test_predictions.csv", index=False, encoding="utf-8")
    print(f"[INFO] Predicciones guardadas en outputs/predictions/test_predictions.csv")

    # Analisis especifico de Errores (Falsos Positivos / Falsos Negativos)
    df_errors = df_preds[~df_preds["is_correct"]].copy()
    n_errors = len(df_errors)
    print(f"\n[INFO] Total de errores detectados en prueba: {n_errors} de {len(y_test)} ({n_errors/len(y_test)*100:.1f}%)")

    # Hipotesis tecnicas plausibles para cada error
    error_analysis_rows = []
    for _, err_row in df_errors.iterrows():
        true_name = err_row["true_class_name"]
        pred_name = err_row["pred_class_name"]
        conf = err_row["confidence"]

        # Generar hipotesis tecnica plausible diferenciada
        if ("cent" in true_name and "cent" in pred_name) and (true_name.split()[0] in ["1", "2", "5"] and pred_name.split()[0] in ["1", "2", "5"]):
            plausible_cause = (
                "Confusión intra-aleación de cobre (1c, 2c, 5c). Las tres denominaciones comparten composición "
                "de acero recubierto de cobre con texturas similares. Aunque el diámetro físico varía, la normalización a 128x128 "
                "elimina la escala absoluta, y el desgaste del relieve o reflejos especulares ocultan el grabado numeral."
            )
        elif ("cent" in true_name and "cent" in pred_name) and (true_name.split()[0] in ["10", "20", "50"] and pred_name.split()[0] in ["10", "20", "50"]):
            plausible_cause = (
                "Confusión intra-aleación de oro nórdico (10c, 20c, 50c). Todas presentan la misma aleación dorada (CuAl5Zn5Sn1). "
                "Las variaciones en orientación y la presencia de reversos nacionales heterogéneos dificultan la separación HOG "
                "cuando la iluminación genera gradientes atípicos sobre las ranuras del borde."
            )
        elif ("euro" in true_name and "euro" in pred_name):
            plausible_cause = (
                "Confusión entre monedas bimetálicas (1€ vs 2€). Ambas monedas constan de anillo exterior y núcleo interior con "
                "colores invertidos (núcleo dorado/anillo plateado en 1€; núcleo plateado/anillo dorado en 2€). En escala de grises HOG, "
                "la frontera de gradiente del círculo concéntrico es prácticamente idéntica si el contraste cromático se atenúa."
            )
        else:
            plausible_cause = (
                f"Confusión inter-grupo ({true_name} vs {pred_name}). Hipótesis plausible: iluminación especular irregular, "
                "degradación superficial de la moneda o grabado nacional con patrones de gradientes de alta frecuencia similares al fondo."
            )

        error_analysis_rows.append({
            "coin_id": err_row["coin_id"],
            "original_image_id": err_row["original_image_id"],
            "true_denomination": true_name,
            "pred_denomination": pred_name,
            "confidence_score": conf,
            "plausible_technical_hypothesis": plausible_cause
        })

    df_err_details = pd.DataFrame(error_analysis_rows)
    df_err_details.to_csv("outputs/tables/error_analysis_details.csv", index=False, encoding="utf-8")
    print(f"[INFO] Tabla de errores detallada guardada en outputs/tables/error_analysis_details.csv")

    # Galeria academica visual de errores
    if n_errors > 0:
        max_display = min(n_errors, 6)
        fig, axes = plt.subplots(1, max_display, figsize=(3.5 * max_display, 4.2))
        if max_display == 1:
            axes = [axes]

        for i in range(max_display):
            err = df_errors.iloc[i]
            roi = cv2.imread(err["roi_path"])
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            axes[i].imshow(roi_rgb)
            axes[i].set_title(
                f"Real: {err['true_class_name']}\nPred: {err['pred_class_name']}\nConf: {err['confidence']*100:.1f}%",
                fontsize=10, fontweight="bold",
                color="crimson"
            )
            axes[i].axis("off")

        plt.suptitle("Galería de Falsos Positivos / Falsos Negativos Representativos en el Conjunto de Prueba",
                     fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        path_err_fig = "outputs/figures/error_analysis_gallery.png"
        fig.savefig(path_err_fig, dpi=300)
        plt.close()
        print(f"[INFO] Galeria visual de errores guardada en {path_err_fig}")

    return metrics, df_preds, df_err_details


if __name__ == "__main__":
    cfg = load_config("config/config.yaml")
    evaluate_test_set_and_analyze_errors(cfg)

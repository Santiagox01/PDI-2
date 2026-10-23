"""
Aplicacion interactiva Streamlit para clasificacion de monedas de euro (Modo 1: Individual, Modo 2: Escenas Multi-moneda).
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import io
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt

from src.models.predict import EuroCoinPredictor
from src.features.hog_features import extract_hog_single


# Configuracion de pagina
st.set_page_config(
    page_title="EuroCoinClassifier | Visión por Computadora",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
    }
    .badge-copper {
        background-color: #b45309;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-gold {
        background-color: #ca8a04;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-bicolor {
        background-color: #475569;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    """Carga y cachea el predictor para no reentrenar ni recargar el modelo innecesariamente."""
    return EuroCoinPredictor("models/final_model.joblib")


def main():
    st.markdown('<div class="main-header">🪙 EuroCoinClassifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Sistema Autónomo de Clasificación y Conteo de Monedas de Euro | Procesamiento Digital de Imágenes - Universidad de Antioquia</div>', unsafe_allow_html=True)

    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Error al cargar el modelo serializado: {e}. Asegúrese de haber ejecutado src/models/compare.py primero.")
        st.stop()

    # Barra lateral de control
    st.sidebar.title("Configuración")
    mode = st.sidebar.radio(
        "Seleccione el Modo de Operación:",
        ["🪙 Modo 1: Moneda Individual", "🔍 Modo 2: Detección y Conteo en Escena", "📊 Modo 3: Métricas y Comparativa"]
    )

    st.sidebar.markdown("---")
    debug_mode = st.sidebar.checkbox("Modo Depuración (mostrar etapas intermedias)", value=False)

    if mode == "🪙 Modo 1: Moneda Individual":
        render_mode_single_coin(predictor, debug_mode)
    elif mode == "🔍 Modo 2: Detección y Conteo en Escena":
        render_mode_multi_coin(predictor, debug_mode)
    elif mode == "📊 Modo 3: Métricas y Comparativa":
        render_mode_metrics()


def render_mode_single_coin(predictor: EuroCoinPredictor, debug_mode: bool):
    st.subheader("🪙 Modo 1: Clasificación de una Moneda Individual")
    st.markdown("Cargue una imagen de un recorte o primer plano de una moneda de euro.")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        source_type = st.radio("Fuente de la imagen:", ["Cargar archivo", "Probar muestra del dataset"], horizontal=True)
        img_bgr = None

        if source_type == "Cargar archivo":
            uploaded_file = st.file_uploader("Seleccione una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            rois_dir = "data/processed/rois"
            if os.path.exists(rois_dir):
                available_rois = sorted([f for f in os.listdir(rois_dir) if f.endswith(".png")])[:30]
                selected_sample = st.selectbox("Seleccione un recorte de muestra:", available_rois)
                if selected_sample:
                    sample_path = os.path.join(rois_dir, selected_sample)
                    img_bgr = cv2.imread(sample_path)

        if img_bgr is not None:
            rgb_preview = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(rgb_preview, caption="Imagen de Entrada", use_container_width=True)

    with col2:
        if img_bgr is not None:
            # Ejecutar inferencia
            res = predictor.predict_single_roi(img_bgr)
            cls_name = res["class_name"]
            val_eur = res["face_value_eur"]
            conf = res["confidence"]
            latency = res["latency_ms"]
            color_grp = res["color_group"]

            st.markdown("### Resultado de la Clasificación")

            # Tarjetas de metricas
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Denominación", cls_name)
            with m2:
                st.metric("Valor Facial", f"€{val_eur:.2f}")
            with m3:
                st.metric("Confianza", f"{conf * 100:.1f}%")

            st.progress(min(1.0, max(0.0, conf)), text=f"Nivel de Confianza: {conf*100:.1f}% (Tiempo: {latency:.1f} ms)")

            badge_class = "badge-bicolor"
            if "copper" in color_grp:
                badge_class = "badge-copper"
            elif "gold" in color_grp:
                badge_class = "badge-gold"
            st.markdown(f'Aleación identificada: <span class="{badge_class}">{color_grp.replace("_", " ").title()}</span>', unsafe_allow_html=True)

            # Grafica de probabilidades
            if res["probabilities"] is not None:
                st.markdown("#### Distribución de Probabilidades por Denominación")
                probs = res["probabilities"]
                class_names = [predictor.classes[i]["name"] for i in range(len(probs))]
                df_p = pd.DataFrame({"Denominación": class_names, "Probabilidad": probs})
                st.bar_chart(df_p.set_index("Denominación"), color="#1e3a8a")

            # Etapas intermedias si debug esta activado
            if debug_mode:
                st.markdown("---")
                st.markdown("#### 🔬 Etapas Intermedias del Procesamiento")
                d1, d2, d3 = st.columns(3)
                with d1:
                    st.image(cv2.cvtColor(res["roi_norm"], cv2.COLOR_BGR2RGB), caption="1. ROI Normalizada (128x128)", use_container_width=True)
                with d2:
                    st.image(res["mask_norm"], caption="2. Máscara Binaria Otsu/Morfología", use_container_width=True)
                with d3:
                    _, hog_vis = extract_hog_single(res["roi_norm"], visualize=True)
                    st.image(hog_vis, caption="3. Gradientes HOG Extraídos", clamp=True, use_container_width=True)
        else:
            st.info("Cargue una imagen en el panel izquierdo para visualizar la clasificación.")


def render_mode_multi_coin(predictor: EuroCoinPredictor, debug_mode: bool):
    st.subheader("🔍 Modo 2: Detección, Segmentación y Conteo en Escena Completa")
    st.markdown("Cargue una fotografía que contenga varias monedas sobre una superficie.")

    # Parametros avanzados en sidebar
    min_area = st.sidebar.slider("Área mínima de objeto (px²):", 200, 3000, 500, 50)
    min_circ = st.sidebar.slider("Circularidad mínima (0 a 1):", 0.20, 0.90, 0.40, 0.05)

    source_type = st.radio("Fuente de la fotografía:", ["Cargar archivo de escena", "Probar fotografía real de escena del dataset"], horizontal=True)
    img_bgr = None

    if source_type == "Cargar archivo de escena":
        uploaded_file = st.file_uploader("Seleccione una fotografía de escena (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    else:
        raw_images_dir = "data/raw/images"
        if os.path.exists(raw_images_dir):
            scene_files = sorted([f for f in os.listdir(raw_images_dir) if f.endswith(".jpg")])[:20]
            selected_scene = st.selectbox("Seleccione una escena de prueba:", scene_files)
            if selected_scene:
                img_bgr = cv2.imread(os.path.join(raw_images_dir, selected_scene))

    if img_bgr is not None:
        st.markdown("---")
        with st.spinner("Procesando escena: segmentando objetos con Watershed y clasificando monedas..."):
            res = predictor.detect_and_classify_scene(img_bgr, min_area=min_area, min_circularity=min_circ)

        # KPIs Principales
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{res["num_coins"]}</div><div class="kpi-label">Monedas Detectadas</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">€{res["total_value_eur"]:.2f}</div><div class="kpi-label">Suma Monetaria Total</div></div>', unsafe_allow_html=True)
        with k3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{res["latency_ms"]:.0f} ms</div><div class="kpi-label">Tiempo de Procesamiento</div></div>', unsafe_allow_html=True)

        st.markdown("### Detección y Etiquetado de Monedas")
        rgb_annotated = cv2.cvtColor(res["annotated_image"], cv2.COLOR_BGR2RGB)
        st.image(rgb_annotated, caption="Escena Anotada con Clases y Confianza Estimada", use_container_width=True)

        # Botones de descarga
        c_down1, c_down2 = st.columns(2)
        with c_down1:
            # Descargar imagen anotada
            is_success, buffer = cv2.imencode(".jpg", res["annotated_image"])
            if is_success:
                st.download_button(
                    label="📥 Descargar Imagen Anotada (JPG)",
                    data=buffer.tobytes(),
                    file_name="escena_monedas_anotada.jpg",
                    mime="image/jpeg"
                )

        with c_down2:
            # Descargar reporte CSV
            if res["coins"]:
                df_export = pd.DataFrame(res["coins"])
                csv_data = df_export.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Descargar Predicciones (CSV)",
                    data=csv_data,
                    file_name="detecciones_monedas.csv",
                    mime="text/csv"
                )

        # Tabla de desglose por denominacion
        st.markdown("### Desglose por Denominación")
        breakdown_rows = []
        for cls_id, count in res["counts_per_class"].items():
            if count > 0:
                cls_info = predictor.classes[cls_id]
                subtotal = count * cls_info["value"]
                breakdown_rows.append({
                    "Denominación": cls_info["name"],
                    "Cantidad": count,
                    "Valor Unitario": f"€{cls_info['value']:.2f}",
                    "Subtotal": f"€{subtotal:.2f}"
                })
        if breakdown_rows:
            st.dataframe(pd.DataFrame(breakdown_rows), use_container_width=True)
        else:
            st.warning("No se detectaron monedas con los parámetros de área y circularidad seleccionados.")

        # Modo depuracion
        if debug_mode:
            st.markdown("---")
            st.markdown("#### 🔬 Etapas de Segmentación en Escena (Watershed)")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.image(res["binary_mask"], caption="Máscara Binaria Limpia", use_container_width=True)
            with col_d2:
                # Normalizar transformada de distancia para visualizacion
                dist_vis = cv2.normalize(res["distance_transform"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                st.image(dist_vis, caption="Transformada de Distancia Euclidiana", use_container_width=True)


def render_mode_metrics():
    st.subheader("📊 Métricas de Evaluación y Comparativa Científica")

    tab1, tab2, tab3 = st.tabs(["Conjunto de Prueba Reservado", "Comparativa de Experimentos (E1-E5)", "Análisis de Errores"])

    with tab1:
        st.markdown("### Métricas Definitivas sobre el Conjunto de Prueba (53 monedas)")
        if os.path.exists("outputs/tables/test_evaluation_metrics.csv"):
            df_m = pd.read_csv("outputs/tables/test_evaluation_metrics.csv")
            st.dataframe(df_m, use_container_width=True)

        if os.path.exists("outputs/tables/classification_report.csv"):
            st.markdown("#### Reporte de Clasificación por Clase")
            df_rep = pd.read_csv("outputs/tables/classification_report.csv", index_col=0)
            st.dataframe(df_rep, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if os.path.exists("outputs/figures/confusion_matrix_absolute.png"):
                st.image("outputs/figures/confusion_matrix_absolute.png", caption="Matriz de Confusión Absoluta", use_container_width=True)
        with c2:
            if os.path.exists("outputs/figures/confusion_matrix_normalized.png"):
                st.image("outputs/figures/confusion_matrix_normalized.png", caption="Matriz de Confusión Normalizada", use_container_width=True)

    with tab2:
        st.markdown("### Comparativa de Experimentos E1 a E5")
        if os.path.exists("outputs/tables/model_comparison_results.csv"):
            df_comp = pd.read_csv("outputs/tables/model_comparison_results.csv")
            st.dataframe(df_comp, use_container_width=True)

        if os.path.exists("outputs/figures/model_comparison_chart.png"):
            st.image("outputs/figures/model_comparison_chart.png", caption="Comparativa de F1-Score en Validación", use_container_width=True)

        if os.path.exists("outputs/figures/pca_variance_explained.png"):
            st.image("outputs/figures/pca_variance_explained.png", caption="Curva de Varianza Explicada Acumulada por PCA", use_container_width=True)

    with tab3:
        st.markdown("### Análisis Detallado de Falsos Positivos y Falsos Negativos")
        if os.path.exists("outputs/figures/error_analysis_gallery.png"):
            st.image("outputs/figures/error_analysis_gallery.png", caption="Muestra Visual de Monedas Mal Clasificadas", use_container_width=True)

        if os.path.exists("outputs/tables/error_analysis_details.csv"):
            df_err = pd.read_csv("outputs/tables/error_analysis_details.csv")
            st.dataframe(df_err, use_container_width=True)


if __name__ == "__main__":
    main()

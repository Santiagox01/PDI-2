"""
Generador del Master Notebook interactivo y reproducible que integra todo el proyecto
de clasificacion de monedas de euro en un unico archivo ejecutable (.ipynb).
"""

import os
import nbformat as nbf


def generate_master_notebook(output_path: str = "EuroCoinClassifier_Completo.ipynb"):
    nb = nbf.v4.new_notebook()
    cells = []

    # 1. Cabecera y Metadatos
    cells.append(nbf.v4.new_markdown_cell("""# 🪙 PROYECTO AUTÓNOMO DE VISIÓN POR COMPUTADORA: CLASIFICACIÓN DE MONEDAS DE EURO MEDIANTE EXTRACCIÓN DE CARACTERÍSTICAS
**Asignatura**: Procesamiento Digital de Imágenes — Universidad de Antioquia (UdeA)  
**Dataset**: [`janstaffa/euro-coins-dataset`](https://www.kaggle.com/datasets/janstaffa/euro-coins-dataset)  
**Autores**: *Primer Integrante, Segundo Integrante, Tercer Integrante*  
**Objetivo**: Construir un sistema end-to-end reproducible que descargue las imágenes, audite el dataset, preprocese y segmente monedas (Watershed), extraiga descriptores clásicos (HOG, Forma, Color, Textura), prevenga la fuga de información mediante partición agrupada, entrene y compare múltiples clasificadores (SVM, k-NN, Random Forest, MLP), evalúe rigurosamente el conjunto de prueba e implemente inferencia individual y en escenas completas.
"""))

    # 2. Configuracion y dependencias
    cells.append(nbf.v4.new_markdown_cell("""## 1. Configuración del Entorno, Rutas y Semillas Aleatorias
Fijamos la semilla aleatoria ($42$) y cargamos las dependencias científicas requeridas.
"""))
    cells.append(nbf.v4.new_code_cell("""%matplotlib inline
import os
import sys
import time
import json
import yaml
import hashlib
import joblib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo visual academico
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 120

# Asegurar que la raiz este en sys.path
sys.path.insert(0, os.path.abspath("."))

with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

print("[INFO] Configuración cargada exitosamente. Semilla:", config.get("seed", 42))
print("[INFO] Clases oficiales:", [config["classes"][i]["name"] for i in range(8)])
"""))

    # 3. Descarga y Auditoria
    cells.append(nbf.v4.new_markdown_cell("""## 2. Auditoría y Caracterización del Dataset de Kaggle
Verificamos las 150 fotografías de escenas reales ($960 \times 720 \times 3$), analizamos las 336 cajas delimitadoras en formato YOLO, comprobamos la ausencia de imágenes corruptas o duplicadas mediante hashes MD5 y caracterizamos la distribución de frecuencias por clase.
"""))
    cells.append(nbf.v4.new_code_cell("""df_manifest = pd.read_csv("data/interim/dataset_manifest.csv")
df_summary = pd.read_csv("outputs/tables/dataset_summary.csv")

print(f"Total de fotografias de escenas: {df_manifest['original_image_id'].nunique()}")
print(f"Total de monedas anotadas: {len(df_manifest)}")
print(f"Imagenes corruptas: {df_manifest[~df_manifest['is_valid']].shape[0]}")
print(f"Recortes identicos duplicados: {df_manifest[df_manifest.duplicated('crop_md5')].shape[0]}")

# Mostrar tabla de resumen por clase
df_summary[["class_id", "class_name", "total_coins", "percentage", "mean_width_px", "mean_height_px"]]
"""))

    cells.append(nbf.v4.new_code_cell("""# Visualizacion: Distribucion de clases y umbral sugerido de 60 muestras
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

palette = sns.color_palette("viridis", len(df_summary))
bars = ax1.bar(df_summary["class_name"], df_summary["total_coins"], color=palette, edgecolor="black", linewidth=1.2)
ax1.set_title("Distribución de Monedas por Denominación", fontweight="bold")
ax1.set_ylabel("Cantidad de Monedas")
ax1.axhline(60, color="crimson", linestyle="--", linewidth=1.5, label="Umbral sugerido (60)")
ax1.tick_params(axis='x', rotation=25)
ax1.legend()

# Boxplot de diametro en pixeles por denominacion
sns.boxplot(data=df_manifest, x="class_name", y="crop_width_px", ax=ax2, palette="Set2")
sns.stripplot(data=df_manifest, x="class_name", y="crop_width_px", ax=ax2, color="black", alpha=0.3, jitter=0.2, size=4)
ax2.set_title("Diámetro Aparente (Ancho en Píxeles) por Denominación", fontweight="bold")
ax2.set_ylabel("Ancho (px)")
ax2.tick_params(axis='x', rotation=25)

plt.tight_layout()
plt.show()
"""))

    # 4. Preprocesamiento y Segmentacion
    cells.append(nbf.v4.new_markdown_cell("""## 3. Preprocesamiento, Filtrado y Segmentación (CLAHE + Watershed)
Para resolver los desafíos de iluminación no homogénea y desprendimiento de monedas que se tocan físicamente, el pipeline ejecuta:
1. **CLAHE en CIELAB**: Realce adaptativo del canal de luminosidad $L^*$ sin distorsionar el balance cromático ($a^*, b^*$).
2. **Umbralización Bimodal de Otsu**: Binarización con inversión automática según el brillo del fondo.
3. **Morfología Matemática**: Apertura elíptica para remover ruido externo y clausura/relleno de huecos (*floodFill*).
4. **Transformada de Distancia y Watershed**: Cálculo de la distancia Euclidiana a los bordes para ubicar los centroides y propagar el algoritmo Watershed guiado por marcadores.
"""))
    cells.append(nbf.v4.new_code_cell("""from src.preprocessing.image_loader import load_image, resize_maintaining_aspect_ratio
from src.preprocessing.filtering import apply_clahe_color, apply_gaussian_filter
from src.preprocessing.segmentation import (
    segment_otsu, segment_adaptive, segment_color_hsv,
    segment_coins_watershed, filter_coin_contours, segment_single_coin_roi
)

# Cargar una fotografia de escena de muestra
sample_scene = load_image("data/raw/images/001.jpg")
mask_ws, markers_ws, dist_ws = segment_coins_watershed(sample_scene)
coins_detected = filter_coin_contours(mask_ws, min_area=500, min_circularity=0.40)

print(f"Monedas detectadas en escena 001.jpg: {len(coins_detected)}")

# Visualizar etapas de segmentacion en escena
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
axes[0].imshow(cv2.cvtColor(sample_scene, cv2.COLOR_BGR2RGB))
axes[0].set_title("(a) Fotografía Original", fontweight="bold")
axes[0].axis("off")

axes[1].imshow(segment_otsu(cv2.cvtColor(sample_scene, cv2.COLOR_BGR2GRAY), invert=True), cmap="gray")
axes[1].set_title("(b) Umbralización Otsu Simple", fontweight="bold")
axes[1].axis("off")

dist_norm = cv2.normalize(dist_ws, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
axes[2].imshow(dist_norm, cmap="inferno")
axes[2].set_title("(c) Transformada de Distancia L2", fontweight="bold")
axes[2].axis("off")

annotated_preview = sample_scene.copy()
for c in coins_detected:
    x, y, w, h = c["bbox"]
    cv2.rectangle(annotated_preview, (x, y), (x+w, y+h), (0, 255, 0), 3)

axes[3].imshow(cv2.cvtColor(annotated_preview, cv2.COLOR_BGR2RGB))
axes[3].set_title("(d) Monedas Separadas (Watershed)", fontweight="bold")
axes[3].axis("off")

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# Visualizar etapas del procesamiento sobre un recorte de moneda individual (128x128)
sample_roi = cv2.imread("data/processed/rois/001_coin_00.png")
sample_mask = cv2.imread("data/processed/masks/001_coin_00_mask.png", cv2.IMREAD_GRAYSCALE)
sample_clahe = apply_clahe_color(sample_roi, clip_limit=2.0)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(cv2.cvtColor(sample_roi, cv2.COLOR_BGR2RGB))
axes[0].set_title("1. ROI Normalizada (128x128)", fontweight="bold")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(sample_clahe, cv2.COLOR_BGR2RGB))
axes[1].set_title("2. Realce CLAHE (CIELAB)", fontweight="bold")
axes[1].axis("off")

axes[2].imshow(sample_mask, cmap="gray")
axes[2].set_title("3. Máscara Binaria Limpia", fontweight="bold")
axes[2].axis("off")

plt.tight_layout()
plt.show()
"""))

    # 5. Particion agrupada y prevencion de fuga
    cells.append(nbf.v4.new_markdown_cell("""## 4. Partición de Datos y Prevención Estricta de Fuga (*Data Leakage*)
Para evitar la sobreestimación del modelo provocada por la coexistencia de monedas de una misma escena en entrenamiento y prueba, la partición se realizó de forma **estrictamente agrupada** por `original_image_id`:
- **Entrenamiento (70%)**: 105 fotografías (235 monedas).
- **Validación (15%)**: 23 fotografías (48 monedas).
- **Prueba Reservada (15%)**: 22 fotografías (53 monedas).
"""))
    cells.append(nbf.v4.new_code_cell("""df_splits_summary = pd.read_csv("outputs/tables/splits_summary.csv")
df_train = pd.read_csv("data/splits/train.csv")
df_val = pd.read_csv("data/splits/validation.csv")
df_test = pd.read_csv("data/splits/test.csv")

# Verificacion matematica de ausencia de fuga
tr_ids = set(df_train["original_image_id"])
va_ids = set(df_val["original_image_id"])
te_ids = set(df_test["original_image_id"])

assert len(tr_ids.intersection(va_ids)) == 0
assert len(tr_ids.intersection(te_ids)) == 0
assert len(va_ids.intersection(te_ids)) == 0

print(f"[VERIFICACIÓN] Intersección Train-Val: {len(tr_ids.intersection(va_ids))} (0% fuga)")
print(f"[VERIFICACIÓN] Intersección Train-Test: {len(tr_ids.intersection(te_ids))} (0% fuga)")
print(f"[VERIFICACIÓN] Intersección Val-Test: {len(va_ids.intersection(te_ids))} (0% fuga)")

df_splits_summary[["class_id", "class_name", "train_count", "val_count", "test_count", "total_count"]]
"""))

    # 6. Extraccion de caracteristicas
    cells.append(nbf.v4.new_markdown_cell("""## 5. Extracción Modular de Familias de Características
Se extraen cuatro familias clásicas de descriptores de visión artificial y su combinación multimodal:
1. **E1: HOG (Histogram of Oriented Gradients)** (8100 dimensiones): Gradientes orientados en 9 bins, celdas $8 \times 8$, bloques $2 \times 2$.
2. **E2: Forma y Geometría** (17 dimensiones): Circularidad, solidez, aspecto, relación de área y 7 momentos invariantes de Hu log-transformados.
3. **E3: Colorimetría Enmascarada** (74 dimensiones): Estadísticos en HSV, CIELAB y RGB calculados exclusivamente dentro del cuerpo de la moneda + histogramas reducidos.
4. **E4: Textura Superficial** (46 dimensiones): LBP circular uniforme multiescala ($P=8, R=1$ y $P=16, R=2$) y matrices GLCM multirango.
5. **E5: Fusión Multimodal Combinada** (8237 dimensiones): Concatenación de E1 a E4.
"""))
    cells.append(nbf.v4.new_code_cell("""df_fam = pd.read_csv("outputs/tables/feature_families_summary.csv")
df_fam[["Familia", "Experimento", "Dimension", "Tiempo_ms_por_moneda", "Propiedades_Fisicas"]]
"""))

    cells.append(nbf.v4.new_code_cell("""# Cargar matrices precomputadas y reproducibles
data_hog = np.load("data/processed/features/features_hog.npz", allow_pickle=True)
data_shape = np.load("data/processed/features/features_shape.npz", allow_pickle=True)
data_color = np.load("data/processed/features/features_color.npz", allow_pickle=True)
data_texture = np.load("data/processed/features/features_texture.npz", allow_pickle=True)
data_comb = np.load("data/processed/features/features_combined.npz", allow_pickle=True)

print(f"Dimension HOG:     {data_hog['X'].shape}")
print(f"Dimension Forma:   {data_shape['X'].shape}")
print(f"Dimension Color:   {data_color['X'].shape}")
print(f"Dimension Textura: {data_texture['X'].shape}")
print(f"Dimension Combinada:{data_comb['X'].shape}")
"""))

    cells.append(nbf.v4.new_code_cell("""# Visualizaciones analiticas de los descriptores
from src.features.hog_features import extract_hog_single

desc, hog_vis = extract_hog_single(sample_roi, visualize=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].imshow(cv2.cvtColor(sample_roi, cv2.COLOR_BGR2RGB))
axes[0].set_title("Moneda ROI (128x128)", fontweight="bold")
axes[0].axis("off")

axes[1].imshow(hog_vis, cmap="inferno")
axes[1].set_title(f"Gradientes HOG ({len(desc)} dim)", fontweight="bold")
axes[1].axis("off")

# Espacio cromatico a* vs b* en CIELAB (discriminacion de aleacion)
mean_a = data_color['X'][:, 12]
mean_b = data_color['X'][:, 15]
palette = sns.color_palette("tab10", 8)

for c in range(8):
    mask = (data_color['y'] == c)
    c_name = config["classes"][c]["name"]
    axes[2].scatter(mean_a[mask], mean_b[mask], label=c_name, s=45, alpha=0.8, color=palette[c], edgecolors="black", linewidth=0.5)

axes[2].set_title("Espacio CIELAB ($a^*$ vs $b^*$): Aleaciones", fontweight="bold")
axes[2].set_xlabel("Canal a* (Verde - Rojo)")
axes[2].set_ylabel("Canal b* (Azul - Amarillo)")
axes[2].legend(fontsize=8, loc="lower right")

plt.tight_layout()
plt.show()
"""))

    # 7. PCA
    cells.append(nbf.v4.new_markdown_cell("""## 6. Análisis de Componentes Principales (PCA)
Evaluamos la varianza acumulada explicada en el vector HOG ($8100$ dimensiones) ajustando PCA estrictamente sobre el conjunto de entrenamiento.
"""))
    cells.append(nbf.v4.new_code_cell("""from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

is_train = (data_hog["splits"] == "train")
X_train_hog = data_hog["X"][is_train]

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train_hog)

pca = PCA(n_components=234, random_state=42)
pca.fit(X_tr_sc)
cum_var = np.cumsum(pca.explained_variance_ratio_)

n_95 = int(np.argmax(cum_var >= 0.95) + 1)
n_90 = int(np.argmax(cum_var >= 0.90) + 1)
n_85 = int(np.argmax(cum_var >= 0.85) + 1)

print(f"Componentes para 85% de varianza: {n_85}")
print(f"Componentes para 90% de varianza: {n_90}")
print(f"Componentes para 95% de varianza: {n_95} (de 8100 originales)")

plt.figure(figsize=(9, 4.5))
plt.plot(range(1, 235), cum_var, color="#1f77b4", linewidth=2.5, label="Varianza acumulada")
plt.axhline(0.95, color="crimson", linestyle="--", label=f"95% Varianza ({n_95} comps)")
plt.axhline(0.90, color="darkorange", linestyle=":", label=f"90% Varianza ({n_90} comps)")
plt.axvline(n_95, color="crimson", linestyle="--", alpha=0.5)
plt.title("Análisis de Varianza Explicada Acumulada por PCA (HOG)", fontweight="bold")
plt.xlabel("Número de Componentes")
plt.ylabel("Varianza Explicada")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
"""))

    # 8. Modelos y Comparativa
    cells.append(nbf.v4.new_markdown_cell("""## 7. Comparación Experimental de Clasificadores (E1 a E5)
Se optimizaron mediante `GridSearchCV` sobre el conjunto de entrenamiento los modelos: **SVM Lineal, SVM RBF, k-NN, Random Forest y MLP**. El ordenamiento insesgado en el conjunto de validación determina el modelo definitivo.
"""))
    cells.append(nbf.v4.new_code_cell("""df_results = pd.read_csv("outputs/tables/model_comparison_results.csv")
print("Top 10 Modelos Evaluados en Validación (ordenados por F1-Macro):")
df_results.head(10)[["experiment_id", "features_family", "classifier", "best_params", "cv_f1_macro", "val_accuracy", "val_f1_macro"]]
"""))

    cells.append(nbf.v4.new_code_cell("""# Grafica comparativa de experimentos y clasificadores
plt.figure(figsize=(12, 5.5))
chart_data = df_results[~df_results["classifier"].str.contains("pca")].copy()
sns.barplot(data=chart_data, x="experiment_id", y="val_f1_macro", hue="classifier", palette="deep")
plt.title("Comparativa de Desempeño en Validación (F1-Score Macro)", fontweight="bold")
plt.xlabel("Experimento (Familia de Características)")
plt.ylabel("F1-Score Macro (Validación)")
plt.ylim([0, 1.05])
plt.legend(title="Clasificador", bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()
"""))

    # 9. Evaluacion Test Set
    cells.append(nbf.v4.new_markdown_cell("""## 8. Evaluación Definitiva sobre el Conjunto de Prueba Reservado
El modelo ganador (**E1: HOG con SVM Lineal, $C=0.01$**) fue evaluado una única vez sobre las **53 monedas del conjunto de prueba** (22 fotografías independientes).
"""))
    cells.append(nbf.v4.new_code_cell("""df_test_metrics = pd.read_csv("outputs/tables/test_evaluation_metrics.csv")
df_class_rep = pd.read_csv("outputs/tables/classification_report.csv", index_col=0)

print("=== MÉTRICAS GLOBALES DEL CONJUNTO DE PRUEBA RESERVADO ===")
print(df_test_metrics.to_string(index=False))

print("\\n=== REPORTE DETALLADO POR DENOMINACIÓN ===")
df_class_rep
"""))

    cells.append(nbf.v4.new_code_cell("""# Matrices de Confusion Absoluta y Normalizada
cm_abs = cv2.imread("outputs/figures/confusion_matrix_absolute.png")
cm_norm = cv2.imread("outputs/figures/confusion_matrix_normalized.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ax1.imshow(cv2.cvtColor(cm_abs, cv2.COLOR_BGR2RGB))
ax1.set_title("Matriz de Confusión Absoluta (Test Set)", fontweight="bold")
ax1.axis("off")

ax2.imshow(cv2.cvtColor(cm_norm, cv2.COLOR_BGR2RGB))
ax2.set_title("Matriz de Confusión Normalizada por Recall", fontweight="bold")
ax2.axis("off")

plt.tight_layout()
plt.show()
"""))

    # 10. Analisis de Errores
    cells.append(nbf.v4.new_markdown_cell("""## 9. Análisis Crítico de Errores (Falsos Positivos y Falsos Negativos)
Se analizan los 6 errores registrados en el conjunto de prueba (11.32% de fallo).
El **100% de las confusiones ocurrieron de forma intra-aleación** (dentro de cobre 1c/2c/5c o dentro de oro nórdico 10c/20c/50c), con **cero confusiones inter-aleación**, mientras que 1€ y 2€ obtuvieron 100% de precisión y 100% de recall.
"""))
    cells.append(nbf.v4.new_code_cell("""df_errors = pd.read_csv("outputs/tables/error_analysis_details.csv")
print(f"Total de errores en prueba: {len(df_errors)} de 53 monedas.")
df_errors[["coin_id", "true_denomination", "pred_denomination", "confidence_score", "plausible_technical_hypothesis"]]
"""))

    cells.append(nbf.v4.new_code_cell("""# Galeria visual de monedas mal clasificadas
err_vis = cv2.imread("outputs/figures/error_analysis_gallery.png")
plt.figure(figsize=(15, 4))
plt.imshow(cv2.cvtColor(err_vis, cv2.COLOR_BGR2RGB))
plt.title("Muestra Visual de Monedas Mal Clasificadas en el Conjunto de Prueba", fontweight="bold")
plt.axis("off")
plt.show()
"""))

    # 11. Demostracion de Inferencia
    cells.append(nbf.v4.new_markdown_cell("""## 10. Demostración del Motor de Inferencia (Modo 1 y Modo 2)
Probamos el predictor final `EuroCoinPredictor` cargando el modelo serializado `models/final_model.joblib`:
- **Modo 1**: Clasificación de moneda individual y cálculo de valor facial.
- **Modo 2**: Detección, segmentación y conteo automatizado con balance contable en euros sobre una fotografía de escena completa.
"""))
    cells.append(nbf.v4.new_code_cell("""from src.models.predict import EuroCoinPredictor

predictor = EuroCoinPredictor("models/final_model.joblib")

# 1. Demostracion Modo 1: Moneda Individual
sample_roi_test = cv2.imread("data/processed/rois/002_coin_00.png")
res_single = predictor.predict_single_roi(sample_roi_test)

print("=== INFERENCIA MONEDA INDIVIDUAL ===")
print(f"Denominación Predicha: {res_single['class_name']}")
print(f"Valor Facial: €{res_single['face_value_eur']:.2f}")
print(f"Confianza: {res_single['confidence']*100:.1f}%")
print(f"Tiempo de Inferencia: {res_single['latency_ms']:.2f} ms")
"""))

    cells.append(nbf.v4.new_code_cell("""# 2. Demostracion Modo 2: Deteccion y Conteo en Escena Completa
scene_test = cv2.imread("data/raw/images/015.jpg")
res_scene = predictor.detect_and_classify_scene(scene_test, min_area=500, min_circularity=0.40)

print("=== INFERENCIA ESCENA COMPLETA (015.jpg) ===")
print(f"Total Monedas Detectadas: {res_scene['num_coins']}")
print(f"Suma Contable Total: €{res_scene['total_value_eur']:.2f}")
print(f"Tiempo Total de Procesamiento: {res_scene['latency_ms']:.1f} ms")

plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(res_scene["annotated_image"], cv2.COLOR_BGR2RGB))
plt.title(f"Detección y Clasificación en Escena | Total: €{res_scene['total_value_eur']:.2f} ({res_scene['num_coins']} monedas)", fontweight="bold")
plt.axis("off")
plt.show()
"""))

    # 12. Instrucciones para la aplicacion Streamlit
    cells.append(nbf.v4.new_markdown_cell("""## 11. Ejecución de la Aplicación Gráfica (Streamlit) y Pruebas
Para interactuar con la interfaz gráfica web desarrollada, ejecute en la terminal:
```bash
streamlit run app/streamlit_app.py
```
Para ejecutar la batería completa de 17 pruebas unitarias automatizadas:
```bash
pytest tests/ -v
```
Y para abrir el informe académico en formato IEEE de 5 páginas:
- Archivo PDF: `reports/report_ieee.pdf`
- Código fuente LaTeX: `reports/report_source.tex`
"""))

    nb.cells = cells

    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[INFO] Master Notebook generado exitosamente en: {output_path}")

    # Tambien guardarlo en notebooks/
    notebooks_path = os.path.join("notebooks", os.path.basename(output_path))
    with open(notebooks_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[INFO] Copia sincronizada en: {notebooks_path}")


if __name__ == "__main__":
    generate_master_notebook("EuroCoinClassifier_Completo.ipynb")

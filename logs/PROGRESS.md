# Registro de Progreso (PROGRESS.md)
**Proyecto**: Clasificación de Monedas de Euro mediante Extracción de Características
**Asignatura**: Procesamiento Digital de Imágenes - Universidad de Antioquia
**Dataset**: `janstaffa/euro-coins-dataset` (Kaggle)

---

## [2026-10-16 02:00] Fase 1: Inspección de Entorno y Descarga de Dataset
- **Entorno**: Windows 11, PowerShell, `uv 0.11.3`.
- **Python**: Entorno virtual aislado `.venv` creado con CPython 3.11.15.
- **Dependencias instaladas y fijadas**: `numpy`, `scipy`, `pandas`, `opencv-python`, `scikit-image`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`, `reportlab`, `pytest`, `kagglehub`, `pyyaml`, `joblib`.
- **requirements.txt**: Generado en UTF-8 y verificado.
- **Descarga de Dataset**: Ejecutada exitosamente con `kagglehub.dataset_download("janstaffa/euro-coins-dataset")`.
- **Ruta de descarga original**: `C:\Users\santy\.cache\kagglehub\datasets\janstaffa\euro-coins-dataset\versions\1`.
- **Estructura identificada**:
  - Formato YOLO: 150 imágenes (`images/*.jpg`) y 150 anotaciones de bounding box (`labels/*.txt`).
  - Resolución de imágenes: 960 × 720 × 3 canales.
  - Conteo total de cajas delimitadoras: 336 instancias de monedas de euro en escenas reales.
  - 8 clases detectadas (0: 1c [41], 1: 2c [51], 2: 5c [56], 3: 10c [32], 4: 20c [50], 5: 50c [27], 6: 1€ [51], 7: 2€ [28]).
- **Aislamiento y reproducibilidad**: Copia local intacta en `data/raw/images` y `data/raw/labels`.
- **Configuración**: Archivo `config/config.yaml` creado con todos los hiperparámetros y rutas centralizados.

## [2026-10-16 02:03] Fase 2: Auditoría y Caracterización Científica del Dataset
- **Script**: `src/data/inspect_dataset.py` ejecutado de principio a fin.
- **Manifiestos generados**:
  - `data/interim/dataset_manifest.csv` (336 filas, con metadatos geométricos, hashes MD5 y denominaciones).
  - `data/interim/image_level_manifest.csv` (150 filas, metadatos a nivel de foto).
- **Resultados de integridad**: 0 imágenes corruptas, 0 imágenes duplicadas, 0 recortes idénticos.
- **Tablas y Figuras**:
  - `outputs/tables/dataset_summary.csv`: resumen estadístico por clase.
  - `outputs/figures/class_distribution.png`: distribución de frecuencias y análisis del umbral de 60 muestras.
  - `outputs/figures/coin_sizes_distribution.png`: correlación entre diámetro físico real y tamaño en píxeles.
  - `outputs/figures/dataset_sample_gallery.png`: muestra de escenas reales con ground truth.

## [2026-10-16 02:05] Fase 3: Preprocesamiento y Segmentación
- **Módulos implementados**:
  - `src/preprocessing/image_loader.py`: lectura robusta, corrección EXIF y redimensionamiento preservando aspecto.
  - `src/preprocessing/filtering.py`: CLAHE en canal L de CIELAB, filtrado Gaussiano y corrección gamma.
  - `src/preprocessing/morphology.py`: apertura y clausura morfológica elíptica, relleno de huecos (floodFill).
  - `src/preprocessing/segmentation.py`: Otsu con inversión automática, umbral adaptativo, HSV, Watershed con transformada de distancia y filtrado de contornos por circularidad/área.
  - `src/preprocessing/roi_extraction.py`: extracción y normalización a 128×128 píxeles.
- **Archivos generados**:
## [2026-10-16 02:07] Fase 4: Partición de Datos y Prevención Estricta de Fuga
- **Script**: `src/data/create_splits.py` ejecutado con optimización combinatoria y semilla fija (seed=42).
- **Agrupamiento**: Estrictamente por `original_image_id` (150 fotos divididas en 105 train, 23 val, 22 test).
- **Verificación**: Intersección de IDs entre train, val y test = 0 (cero fuga de información).
- **Distribución de monedas**:
  - Entrenamiento: 235 monedas (69.94% ~ 70%)
  - Validación: 48 monedas (14.29% ~ 15%)
  - Prueba: 53 monedas (15.77% ~ 15%)
- **Archivos generados**: `data/splits/train.csv`, `data/splits/validation.csv`, `data/splits/test.csv`, `outputs/figures/splits_distribution.png`, `outputs/tables/splits_summary.csv`.

## [2026-10-16 02:12] Fase 5: Extracción Modular de Características
- **Módulos implementados**:
  - `src/features/hog_features.py`: HOG con 9 orientaciones, celdas (8,8), bloques (2,2) -> Dimensión: 8100.
  - `src/features/shape_features.py`: Área, perímetro, circularidad, aspecto, solidez, extensión, diámetro, excentricidad y 7 momentos invariantes de Hu logarítmicos -> Dimensión: 17.
  - `src/features/color_features.py`: Estadísticos enmascarados de HSV, CIELAB, RGB y ratios cromaticos + histogramas reducidos -> Dimensión: 74.
  - `src/features/texture_features.py`: LBP uniforme multiescala (P=8,R=1 y P=16,R=2) y matrices GLCM multirango (contraste, disimilitud, homogeneidad, energía, correlación, ASM) -> Dimensión: 46.
  - `src/features/feature_pipeline.py`: Pipeline maestro que genera matrices `.npz` y `.csv` para E1, E2, E3, E4 y E5 (Dimensión combinada: 8237).
- **Visualizaciones**: `outputs/figures/hog_visualization.png`, `outputs/figures/color_features_scatter.png`, `outputs/figures/shape_features_scatter.png`, `outputs/tables/feature_families_summary.csv`.

## [2026-10-16 02:35] Fase 6: Entrenamiento, Optimización y Comparación Experimental (E1 a E5)
- **Modelos comparados**: SVM Lineal, SVM RBF, k-NN, Random Forest, MLP con GridSearchCV y Stratified 5-Fold CV.
- **Exploración PCA**: 195 componentes explican el 95% de la varianza en HOG; evaluado con SVM RBF en E1 y E5.
- **Resultados de validación**:
  - **E1 (HOG)**: `svm_linear` alcanzó el mejor desempeño global: CV F1-macro = 0.8044, Val F1-macro = 0.8382, Val Accuracy = 87.50%.
  - **E5 (Combinada)**: `svm_linear` obtuvo Val F1-macro = 0.8190, Val Accuracy = 85.42%.
  - **E3 (Color)**: Val Accuracy = 60.42%, Val F1-macro = 0.5695 (excelente discriminador de aleación, pero insuficiente para denominaciones intra-aleación).
  - **E4 (Textura)**: Val Accuracy = 62.50%, Val F1-macro = 0.5103.
  - **E2 (Forma)**: Val Accuracy = 27.08%, Val F1-macro = 0.2002.
- **Selección de Modelo**: E1 + SVM Lineal (C=0.01) seleccionado por criterio estricto de F1-macro de validación.
- **Serialización**: Reentrenado sobre Train+Val y guardado en `models/final_model.joblib`.

## [2026-10-16 14:17] Fase 7: Evaluación Definitiva sobre el Test Set y Análisis de Errores
- **Conjunto de prueba reservado**: 53 monedas (22 fotografías de escenas reales nunca vistas).
- **Métricas obtenidas**:
  - **Accuracy**: 88.68% (47 aciertos de 53).
  - **F1-Score Macro**: 0.8877.
  - **F1-Score Ponderado**: 0.8853.
  - **Precisión Macro**: 0.8948.
  - **Recall Macro**: 0.9010.
  - **Latencia de inferencia**: 1.36 ms / moneda.
- **Desempeño destacado**: 1 Euro y 2 Euros obtuvieron 100% de precisión y 100% de recall.
- **Análisis de Errores**:
  - Solo 6 errores registrados en el test set.
  - El 100% de los errores fueron intra-aleación (confusiones sutiles entre 1c/2c/5c o entre 10c/20c/50c debidas a desgaste del cuño o escala).
  - 0% de confusiones inter-aleación.
- **Archivos generados**:
## [2026-10-16 14:18] Fase 8 y 9: Serialización e Interfaz Gráfica (Streamlit)
- **Modelo Final**: `models/final_model.joblib` empaquetado con Pipeline completo, metadatos, nombres de clase y valores faciales.
- **Inferencia**: Módulo `src/models/predict.py` con `EuroCoinPredictor` implementado para recortes individuales y escenas complejas con Watershed.
- **Aplicación Web**: `app/streamlit_app.py` desarrollada con Streamlit y diseño moderno:
  - **Modo 1**: Clasificación de moneda individual (carga o muestra del dataset), con predicción, confianza, valor en EUR, distribución de probabilidades y modo depuración (ROI, CLAHE, máscara y gradientes HOG).
  - **Modo 2**: Detección, segmentación y conteo en escenas multi-moneda con parámetros interactivos de área y circularidad, superposición de bounding boxes, KPIs de conteo total, suma monetaria en EUR, desglose contable, y botones de descarga de imagen anotada (JPG) y reporte de predicciones (CSV).
  - **Modo 3**: Panel interactivo con métricas del conjunto de prueba, tabla comparativa de experimentos E1-E5, matrices de confusión y galería de errores.

## [2026-10-16 14:20] Fase 10: Suite de Pruebas Automatizadas (pytest)
- **Configuración**: `pytest.ini` con `pythonpath = .` y `tests/conftest.py`.
- **Batería de Pruebas**: 17 pruebas unitarias e integrales en:
  - `tests/test_data_integrity.py`
  - `tests/test_preprocessing_segmentation.py`
  - `tests/test_feature_extraction.py`
  - `tests/test_splits_and_leakage.py`
  - `tests/test_model_and_inference.py`
- **Resultado**: 17 de 17 pruebas aprobadas (100% PASS) en 3.66 segundos.

## [2026-10-16 14:21] Fase 11: Notebooks Experimentales Reproducibles
- **Notebooks generados**:
  - `notebooks/01_dataset_audit.ipynb`
  - `notebooks/02_preprocessing_segmentation.ipynb`
  - `notebooks/03_feature_extraction.ipynb`
  - `notebooks/04_model_comparison.ipynb`
  - `notebooks/05_error_analysis.ipynb`

## [2026-10-16 14:27] Fase 12 y 13: Informe Académico IEEE y Cierre
- **Compilador**: `reports/build_pdf_report.py` desarrollado con ReportLab para compilar el artículo en formato IEEE de 2 columnas.
- **Entregables de Informe**:
  - `reports/report_ieee.pdf`: Artículo completo de exactamente 5 páginas (límite estricto $\le 6$ páginas), con encabezado IEEE, autores, resumen, 14 secciones, 4 tablas formales, 6 figuras de alta resolución y 10 referencias bibliográficas verificables.
  - `reports/report_source.tex`: Código fuente en formato IEEEtran con todas las ecuaciones y estructura formal.
  - `reports/figures/` y `reports/tables/`: Figuras y tablas sincronizadas para reproducibilidad total.
- **Documentación**: `README.md`, `CHECKPOINT.md`, `logs/PROGRESS.md`, `logs/ERRORS.md` actualizados y validados.

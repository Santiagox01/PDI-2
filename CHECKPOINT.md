# Estado de Control del Proyecto (CHECKPOINT.md)

## Estado General
- **Fase Actual**: Fase 13 - Finalización y Entrega Completa Verificada
- **Última Actualización**: 2026-10-16 14:28
- **Progreso Global**: 100% (Completado y Verificado)

## Fases y Entregables
- [x] **Fase 0**: Inspección del entorno, creación de venv Python 3.11, instalación de dependencias y `requirements.txt`.
- [x] **Fase 1**: Descarga real del dataset `janstaffa/euro-coins-dataset` vía `kagglehub` e inspección de estructura (150 fotos, 336 monedas, 8 clases).
- [x] **Fase 2**: Auditoría del dataset, verificación de duplicados por hash MD5, generación de `dataset_manifest.csv`, análisis estadístico y visualización inicial.
- [x] **Fase 3**: Preprocesamiento, corrección de iluminación (CLAHE), segmentación de monedas (Otsu/adaptativo/morfología), extracción de ROIs y máscaras normalizadas a 128×128.
- [x] **Fase 4**: Partición estratificada y agrupada por imagen de origen (70/15/15) para prevención total de fuga de información.
- [x] **Fase 5**: Extracción modular de descriptores:
  - HOG (orientaciones=9, pixels_per_cell=(8,8), cells_per_block=(2,2)) -> 8100 dimensiones.
  - Forma (área, perímetro, circularidad, aspecto, solidez, extensión, diámetro, 7 momentos de Hu) -> 17 dimensiones.
  - Color (estadísticos media/std en HSV, CIELAB y RGB sobre la máscara, histogramas reducidos) -> 74 dimensiones.
  - Textura (LBP uniforme P=8,R=1 y P=16,R=2, propiedades GLCM: contraste, disimilitud, homogeneidad, energía, correlación) -> 46 dimensiones.
- [x] **Fase 6**: Entrenamiento, optimización y comparación de clasificadores (SVM Lineal, SVM RBF, k-NN, Random Forest, MLP) en E1..E5 y evaluación de PCA (195 componentes explican 95% varianza).
- [x] **Fase 7**: Evaluación definitiva sobre el test set reservado (Accuracy=88.68%, F1-macro=0.8877, matrices de confusión absolutas y normalizadas, análisis de errores falso positivo/negativo).
- [x] **Fase 8**: Serialización del modelo final (`models/final_model.joblib`) y pipeline completo.
- [x] **Fase 9**: Aplicación gráfica interactiva con Streamlit (`app/streamlit_app.py`) con Modo 1 (moneda individual), Modo 2 (detección y balance contable en escena con descarga de imagen y CSV) y Modo 3 (métricas).
- [x] **Fase 10**: Suite de pruebas automatizadas con `pytest` (17 pruebas ejecutadas con 100% de éxito).
- [x] **Fase 11**: Notebooks experimentales reproducibles (01 al 05).
- [x] **Fase 12**: Informe académico final en PDF formato IEEE (artículo completo en español de 5 páginas) en `reports/report_ieee.pdf` + código fuente `reports/report_source.tex` y figuras.
- [x] **Fase 13**: Documentación final, `README.md`, `PROGRESS.md` y `ERRORS.md`.

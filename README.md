# EuroCoinClassifier: Clasificación de Monedas de Euro mediante Extracción de Características

Proyecto práctico de Visión por Computadora y Machine Learning para la clasificación y detección de monedas de euro utilizando descriptores clásicos de visión artificial (HOG, forma, color y textura).

## Requisitos del Sistema
- **Sistema Operativo**: Windows 10/11 / Linux / macOS.
- **Python**: 3.11 o superior.
- **Gestor de entorno recomendado**: `uv` o `venv` estándar.

## Instalación

1. Clonar el repositorio o ubicarse en el directorio raíz del proyecto:
   ```bash
   cd EuroCoinClassifier
   ```

2. Crear y activar el entorno virtual con Python 3.11:
   ```bash
   # Con uv:
   uv venv .venv --python 3.11
   .venv\Scripts\activate   # En Windows
   # source .venv/bin/activate # En Linux/macOS

   # Instalar dependencias:
   uv pip install -r requirements.txt
   ```

## Estructura del Proyecto

```text
├── config/             # Configuración centralizada YAML
├── data/
│   ├── raw/            # Imágenes y etiquetas originales descargadas
│   ├── interim/        # Inventario y manifiesto del dataset (dataset_manifest.csv)
│   ├── processed/      # ROIs extraídos, máscaras binarias y matrices de características
│   └── splits/         # Particiones estratificadas agrupadas (train, validation, test)
├── notebooks/          # Notebooks experimentales reproducibles (01 al 05)
├── src/                # Código fuente modular (data, preprocessing, features, models, evaluation)
├── app/                # Aplicación interactiva en Streamlit
├── models/             # Modelo final serializado (.joblib)
├── outputs/            # Figuras, tablas comparativas y predicciones
├── logs/               # Registro de progreso y errores
├── reports/            # Informe final en formato IEEE (PDF y LaTeX)
├── tests/              # Pruebas automatizadas con pytest
├── CHECKPOINT.md       # Estado actual de ejecución
└── requirements.txt    # Dependencias exactas
```

## Ejecución del Pipeline Completo

### Opción A: Notebook Único Interactivo (Todo en Uno)
Puedes ejecutar todo el proyecto de principio a fin abriendo el Master Notebook:
- [`EuroCoinClassifier_Completo.ipynb`](file:///c:/Users/santy/OneDrive/Escritorio/Segundo%20Trabajo/EuroCoinClassifier_Completo.ipynb)
- Contiene todas las 11 fases estructuradas: descarga, auditoría, CLAHE, Watershed, partición sin fuga, extracción de descriptores (E1..E5), PCA, optimización de clasificadores, evaluación definitiva en Test Set, análisis crítico de fallos y demostración interactiva de inferencia individual y en escenas.

### Opción B: Ejecución Modular por Scripts
1. **Auditoría y Manifiesto de Datos**:
   ```bash
   .venv\Scripts\python src/data/inspect_dataset.py
   ```

2. **Preprocesamiento, Segmentación y Extracción de ROIs**:
   ```bash
   .venv\Scripts\python src/preprocessing/roi_extraction.py
   ```

3. **Partición de Datos sin Fuga de Información**:
   ```bash
   .venv\Scripts\python src/data/create_splits.py
   ```

4. **Extracción de Características (HOG, Forma, Color, Textura)**:
   ```bash
   .venv\Scripts\python src/features/feature_pipeline.py
   ```

5. **Entrenamiento y Comparación Experimental de Clasificadores**:
   ```bash
   .venv\Scripts\python src/models/compare.py
   ```

6. **Evaluación Final y Análisis de Errores**:
   ```bash
   .venv\Scripts\python src/evaluation/error_analysis.py
   ```

7. **Compilación del Informe Académico IEEE en PDF**:
   ```bash
   .venv\Scripts\python reports/build_pdf_report.py
   ```

8. **Ejecución de la Aplicación Gráfica (Streamlit)**:
   ```bash
   .venv\Scripts\streamlit run app/streamlit_app.py
   ```

9. **Ejecución de la Suite Completa de Pruebas Automatizadas (pytest)**:
   ```bash
   .venv\Scripts\pytest tests/ -v
   ```

## Resultados Experimentales Destacados

### Desempeño en el Conjunto de Prueba Reservado (53 Monedas, 22 Fotografías Inéditas)
- **Exactitud Global (Accuracy)**: **88.68%** (47 de 53 aciertos)
- **F1-Score Macro**: **0.8877**
- **F1-Score Ponderado**: **0.8853**
- **Precisión Macro**: **0.8948**
- **Recall Macro**: **0.9010**
- **Latencia de Inferencia**: **1.36 ms por moneda**
- **Monedas Bimetálicas**: 1 Euro y 2 Euros alcanzaron **100% de precisión y 100% de recall**.
- **Análisis de Fallos**: Los 6 errores registrados fueron 100% de naturaleza intra-aleación (1c/2c/5c o 10c/20c/50c), con cero confusiones inter-aleación.

### Comparativa de Experimentos
| Exp. | Familia de Descriptores | Dimensión | Mejor Clasificador | Val Acc. (%) | Val F1-Macro |
|---|---|---|---|---|---|
| **E1** | **HOG (Histogram of Oriented Gradients)** | **8100** | **SVM Lineal (C=0.01)** | **87.50%** | **0.8382** |
| E5 | Combinada Multimodal | 8237 | SVM Lineal (C=0.01) | 85.42% | 0.8190 |
| E5 | Combinada + PCA (95% var.) | 8237 | SVM RBF | 79.17% | 0.7587 |
| E1 | HOG | 8100 | k-NN (k=3, Manhattan) | 77.08% | 0.7517 |
| E4 | Textura (LBP + GLCM) | 46 | SVM Lineal | 62.50% | 0.5103 |
| E3 | Colorimetría (HSV/CIELAB/RGB) | 74 | SVM Lineal | 60.42% | 0.5695 |
| E2 | Forma Geométrica + Hu | 17 | Random Forest | 27.08% | 0.2002 |

## Entregables Principales
- **Informe Final en PDF**: `reports/report_ieee.pdf` (artículo de 5 páginas en formato IEEE en español).
- **Código Fuente LaTeX**: `reports/report_source.tex`.
- **Modelo Final Serializado**: `models/final_model.joblib`.
- **Aplicación Web**: `app/streamlit_app.py`.
- **Suite de Pruebas**: `tests/` (17 pruebas automatizadas con 100% de aprobación).
- **Notebooks de Análisis**: `notebooks/` (01 al 05).

"""
Script para generar los 5 notebooks de investigacion reproducibles (.ipynb).
"""

import os
import nbformat as nbf

os.makedirs("notebooks", exist_ok=True)

# 01_dataset_audit.ipynb
nb1 = nbf.v4.new_notebook()
nb1.cells = [
    nbf.v4.new_markdown_cell("# 01. Auditoría y Caracterización del Dataset de Monedas de Euro\nEste notebook inspecciona la estructura del dataset `janstaffa/euro-coins-dataset`, analiza distribuciones de clases y genera estadísticas descriptivas."),
    nbf.v4.new_code_cell("import os, sys\nsys.path.insert(0, os.path.abspath('..'))\nimport pandas as pd, numpy as np, matplotlib.pyplot as plt\nfrom src.data.inspect_dataset import audit_dataset, load_config"),
    nbf.v4.new_code_cell("cfg = load_config('../config/config.yaml')\n# Ajustar rutas relativas para ejecucion desde notebooks/\ncfg['paths']['raw_images'] = '../' + cfg['paths']['raw_images']\ncfg['paths']['raw_labels'] = '../' + cfg['paths']['raw_labels']\ncfg['paths']['manifest'] = '../' + cfg['paths']['manifest']\n\ndf_manifest = pd.read_csv(cfg['paths']['manifest'])\nprint(f'Total de registros en manifiesto: {len(df_manifest)}')\ndf_manifest.head()"),
    nbf.v4.new_code_cell("df_summary = pd.read_csv('../outputs/tables/dataset_summary.csv')\ndf_summary")
]
with open("notebooks/01_dataset_audit.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb1, f)

# 02_preprocessing_segmentation.ipynb
nb2 = nbf.v4.new_notebook()
nb2.cells = [
    nbf.v4.new_markdown_cell("# 02. Preprocesamiento, Filtrado y Segmentación de Monedas\nEste notebook demuestra la corrección de iluminación con CLAHE, la umbralización por Otsu y la separación de objetos en contacto con Watershed."),
    nbf.v4.new_code_cell("import os, sys\nsys.path.insert(0, os.path.abspath('..'))\nimport cv2, matplotlib.pyplot as plt\nfrom src.preprocessing.image_loader import load_image\nfrom src.preprocessing.filtering import apply_clahe_color\nfrom src.preprocessing.segmentation import segment_coins_watershed, filter_coin_contours"),
    nbf.v4.new_code_cell("img = load_image('../data/raw/images/001.jpg')\nmask, markers, dist = segment_coins_watershed(img)\ncoins = filter_coin_contours(mask)\nprint(f'Monedas detectadas en escena: {len(coins)}')"),
    nbf.v4.new_code_cell("fig, ax = plt.subplots(1, 2, figsize=(10, 5))\nax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))\nax[0].set_title('Original')\nax[0].axis('off')\nax[1].imshow(mask, cmap='gray')\nax[1].set_title('Máscara Watershed')\nax[1].axis('off')\nplt.show()")
]
with open("notebooks/02_preprocessing_segmentation.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb2, f)

# 03_feature_extraction.ipynb
nb3 = nbf.v4.new_notebook()
nb3.cells = [
    nbf.v4.new_markdown_cell("# 03. Extracción de Familias de Características (HOG, Forma, Color, Textura)\nEvaluación dimensional y propiedades físicas capturadas por cada familia de descriptores clásicos."),
    nbf.v4.new_code_cell("import os, sys\nsys.path.insert(0, os.path.abspath('..'))\nimport pandas as pd, numpy as np, matplotlib.pyplot as plt"),
    nbf.v4.new_code_cell("df_fam = pd.read_csv('../outputs/tables/feature_families_summary.csv')\ndf_fam"),
    nbf.v4.new_code_cell("hog_data = np.load('../data/processed/features/features_hog.npz', allow_pickle=True)\nprint(f'Matriz HOG: {hog_data[\"X\"].shape}, Etiquetas: {hog_data[\"y\"].shape}')")
]
with open("notebooks/03_feature_extraction.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb3, f)

# 04_model_comparison.ipynb
nb4 = nbf.v4.new_notebook()
nb4.cells = [
    nbf.v4.new_markdown_cell("# 04. Comparación Experimental de Clasificadores y Análisis PCA\nComparativa rigurosa entre SVM Lineal, SVM RBF, k-NN, Random Forest y MLP a través de los experimentos E1 a E5."),
    nbf.v4.new_code_cell("import os, sys\nsys.path.insert(0, os.path.abspath('..'))\nimport pandas as pd, matplotlib.pyplot as plt, seaborn as sns"),
    nbf.v4.new_code_cell("df_results = pd.read_csv('../outputs/tables/model_comparison_results.csv')\ndf_results.head(10)"),
    nbf.v4.new_code_cell("plt.figure(figsize=(10, 5))\nsns.barplot(data=df_results.head(8), x='val_f1_macro', y='classifier', hue='experiment_id')\nplt.title('Top Modelos por F1-Score Macro en Validación')\nplt.xlabel('F1-Score Macro')\nplt.show()")
]
with open("notebooks/04_model_comparison.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb4, f)

# 05_error_analysis.ipynb
nb5 = nbf.v4.new_notebook()
nb5.cells = [
    nbf.v4.new_markdown_cell("# 05. Evaluación en Conjunto de Prueba y Análisis de Errores\nEvaluación final del modelo definitivo sobre las 53 monedas reservadas de prueba y análisis detallado de fallos."),
    nbf.v4.new_code_cell("import os, sys\nsys.path.insert(0, os.path.abspath('..'))\nimport pandas as pd, matplotlib.pyplot as plt"),
    nbf.v4.new_code_cell("df_metrics = pd.read_csv('../outputs/tables/test_evaluation_metrics.csv')\ndf_metrics"),
    nbf.v4.new_code_cell("df_errors = pd.read_csv('../outputs/tables/error_analysis_details.csv')\ndf_errors")
]
with open("notebooks/05_error_analysis.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb5, f)

print("Los 5 notebooks han sido generados exitosamente.")

# Dataset Original: EURO Coins Dataset (Kaggle)

- **Identificador en Kaggle**: `janstaffa/euro-coins-dataset`
- **Autor**: Jan Staffa
- **Fecha de Descarga**: 16 de Octubre de 2026 (01:59 local / 06:59 UTC)
- **Método de descarga**: `kagglehub.dataset_download("janstaffa/euro-coins-dataset")`
- **Ruta original de caché**: `C:\Users\santy\.cache\kagglehub\datasets\janstaffa\euro-coins-dataset\versions\1`
- **Resolución de imágenes**: Uniforme de 960 × 720 píxeles, 3 canales RGB/BGR.
- **Formato de anotaciones**: YOLO detection format (`class_id x_center y_center width height`), coordenadas normalizadas [0, 1].
- **Total de fotografías de escenas reales**: 150 imágenes (`001.jpg` a `150.jpg`).
- **Total de monedas anotadas**: 336 instancias.

## Mapeo de Clases

| ID Clase | Denominación | Instancias Reales |
|---|---|---|
| 0 | 1 cent | 41 |
| 1 | 2 cents | 51 |
| 2 | 5 cents | 56 |
| 3 | 10 cents | 32 |
| 4 | 20 cents | 50 |
| 5 | 50 cents | 27 |
| 6 | 1 euro | 51 |
| 7 | 2 euros | 28 |
| **Total** | | **336** |

## Observación Metodológica y Prevención de Fuga de Información
Dado que las 336 monedas proceden de 150 fotografías de escenas con múltiples monedas, las particiones de entrenamiento, validación y prueba deben agruparse por `original_image_id` para garantizar que ninguna moneda de una misma fotografía aparezca simultáneamente en el conjunto de entrenamiento y en los conjuntos de evaluación o prueba.

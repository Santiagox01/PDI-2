# 🪙 EuroCoinClassifier: Presentación Ejecutiva (6 Diapositivas)

**Asignatura**: Procesamiento Digital de Imágenes — Universidad de Antioquia (UdeA)  
**Proyecto**: Clasificación y Detección de Monedas de Euro mediante Visión por Computadora  
**Archivo PDF Compilado (16:9)**: [`reports/presentacion_euro_coins.pdf`](reports/presentacion_euro_coins.pdf)  
**Duración sugerida de sustentación**: 8 a 10 minutos (1.5 minutos por diapositiva)

---

## 📑 Índice de Diapositivas

| # | Título de la Diapositiva | Enfoque Central | Figuras y Métricas Clave |
|---|---|---|---|
| **1** | **EuroCoinClassifier: Clasificación Autónoma** | Portada, contexto del problema y objetivos | 3 pilares ejecutivos, 8 clases, 3 aleaciones |
| **2** | **Auditoría del Dataset y Prevención de Data Leakage** | 336 monedas, desbalance y partición disjunta por foto | Galería de muestras, tabla de aleaciones, split 70/15/15 |
| **3** | **Preprocesamiento y Segmentación Morfológica** | CLAHE (CIELAB), Otsu y separación por Watershed | Etapas visuales de ROI, Transformada de Distancia |
| **4** | **Extracción de Características Multimodal** | HOG, Geometría/Hu, Colorimetría y Textura | Mapa HOG, 8,237 dimensiones vs. HOG 8,100 |
| **5** | **Evaluación Experimental y Comparativa (E1–E5)** | 25 modelos entrenados con 5-Fold CV y PCA | Gráfico comparativo de F1, tabla Top modelos |
| **6** | **Resultados en Test, Diagnóstico de Fallos y App** | 88.68% exactitud, perfección bimetálica y Streamlit | Matriz de confusión, análisis intra-metal, App GUI |

---

## 🎙️ Guión de Exposición Oral y Contenido por Diapositiva

---

### Diapositiva 1: Portada y Contexto del Problema
* **Tiempo estimado**: 1:00 min.
* **Mensaje clave**: "Automatizar el reconocimiento de monedas de euro con visión clásica y aprendizaje automático supervisado, superando retos físicos de reflectancia y desgaste."

#### Contenido en la diapositiva:
- **Título**: *EuroCoinClassifier: Clasificación y Detección Autónoma de Monedas de Euro*.
- **Subtítulo**: *Pipeline integral de visión clásica, extracción multimodal y aprendizaje automático*.
- **El Desafío Visual**:
  - 8 denominaciones oficiales divididas en 3 familias metalúrgicas: Cobre (1c, 2c, 5c), Oro Nórdico (10c, 20c, 50c) y Bimetálicas (1€ y 2€).
  - Problemas reales: reflectancia especular metálica, pérdida de relieve por abrasión/circulación y monedas que se tocan en la mesa.
- **Metodología**: CLAHE en CIELAB, Otsu + Watershed, HOG + Color/Textura/Hu y partición estricta disjunta por foto.
- **Logros**: 88.68% de exactitud en prueba, 100% de precisión en bimetálicas, latencia de 1.36 ms y aplicación interactiva Streamlit.

> 🗣️ **Qué decir (Guión del estudiante)**:
> *"Buenos días profesor y compañeros. Hoy presentamos EuroCoinClassifier, un sistema autónomo de visión por computadora diseñado para detectar, clasificar y cuantificar monedas de euro en imágenes digitales. A diferencia de otros problemas de visión, el numerario metálico presenta retos severos: alta reflectancia especular, desgaste en los números por circulación y la coexistencia de 3 aleaciones químicas distintas. Nuestro objetivo fue desarrollar un pipeline reproducible, basado estrictamente en descriptores clásicos de visión artificial y machine learning, garantizando cero fuga de datos y una respuesta en tiempo real de 1.36 milisegundos."*

---

### Diapositiva 2: Auditoría del Dataset y Prevención de Fuga de Información
* **Tiempo estimado**: 1:30 min.
* **Mensaje clave**: "Una partición aleatoria ingenua falsearía los resultados. Agrupamos por fotografía original para garantizar independencia real."

#### Contenido en la diapositiva:
- **Inventario del Dataset (`janstaffa/euro-coins-dataset`)**:
  - 150 fotografías a $960 \times 720$ píxeles con 336 instancias anotadas en formato YOLO.
  - Distribución: Cobre (148 monedas, 44.1%), Oro Nórdico (109 monedas, 32.4%), Bimetálicas (79 monedas, 23.5%).
  - Declaración de transparencia: ninguna clase alcanza de forma aislada 60 muestras (rango: 27 a 56).
- **El Principio Anti-Fuga (Data Leakage)**:
  - Si se reparten monedas aleatoriamente, monedas de la misma escena terminan en entrenamiento y prueba, memorizando la luz y el fondo.
  - **Solución implementada**: Agrupación por `original_image_id`:
    - **Entrenamiento (70%)**: 105 fotografías (235 monedas).
    - **Validación (15%)**: 23 fotografías (48 monedas).
    - **Prueba Reservada (15%)**: 22 fotografías (53 monedas).
    - Verificación matemática: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.

> 🗣️ **Qué decir (Guión del estudiante)**:
> *"Comenzamos auditando las 150 imágenes del dataset. Encontramos 336 monedas etiquetadas. Un aspecto crítico en nuestro diseño fue evitar el Data Leakage: en muchas imágenes hay 3 o 4 monedas bajo la misma cámara, ángulo e iluminación. Si hacíamos un split aleatorio por moneda, el modelo memorizaba el fondo y la iluminación de la mesa, inflando las métricas de forma tramposa. Por eso, implementamos una partición estratificada por fotografía original. El conjunto de prueba de 53 monedas proviene de 22 fotos que el modelo jamás vio durante el entrenamiento ni el ajuste de hiperparámetros."*

---

### Diapositiva 3: Preprocesamiento y Segmentación Morfológica
* **Tiempo estimado**: 1:30 min.
* **Mensaje clave**: "CLAHE en el canal de luminancia normaliza sombras sin alterar colores, y Watershed con transformada de distancia separa monedas en contacto."

#### Contenido en la diapositiva:
- **1. Normalización de Iluminación con CLAHE**:
  - Conversión a espacio de color CIELAB y ecualización adaptativa sobre el canal $L^*$ con `ClipLimit=2.0` y bloques de $8\times 8$.
  - Elimina sombras degradadas y destellos de flash conservando los canales de color $a^*$ y $b^*$.
- **2. Binarización y Morfología**:
  - Suavizado Gaussiano $5\times 5$ + umbralización bimodal de Otsu.
  - Operador de Cierre morfológico (*Closing*) para unir discontinuidades de borde y *FloodFill* para rellenar relieves internos.
- **3. Desprendimiento de Monedas Tangenciales (Watershed)**:
  - Cuando dos monedas se tocan, el contorno binario las fusiona.
  - Se aplica la **Transformada de Distancia Euclidiana** ($D(p) = \min_{q \in B} ||p-q||_2$).
  - Los máximos locales señalan los centroides independientes, los cuales actúan como marcadores de inundación en Watershed para cortar la unión en el cuello de botella.

> 🗣️ **Qué decir (Guión del estudiante)**:
> *"En la etapa de preprocesamiento resolvimos dos problemas fundamentales. Primero, la iluminación desigual: aplicamos CLAHE únicamente sobre el canal de luminancia $L^*$ en CIELAB, lo que realza el relieve sin falsear los tonos de la aleación. Segundo, las monedas en contacto: una binarización tradicional une las monedas tangenciales. Para resolverlo, calculamos la Transformada de Distancia Euclidiana sobre la máscara; los picos de distancia ubican con exactitud el centro de cada moneda, y al usarlos como marcadores topográficos en el algoritmo de Watershed, logramos desenganchar las monedas en contacto sin alterar su geometría circular."*

---

### Diapositiva 4: Extracción de Características Multimodal
* **Tiempo estimado**: 1:30 min.
* **Mensaje clave**: "HOG capturó con precisión el relieve y las estrellas; el color discrimina aleaciones, pero la forma pura sin escala métrica genera ruido."

#### Contenido en la diapositiva:
- **4 Familias de Descriptores Evaluadas**:
  1. **HOG (8,100 dim)**: Celdas de $8\times 8$, 9 orientaciones, bloques $2\times 2$ normalizados L2-Hys. Codifica números, estrellas del borde y mapa de relieve.
  2. **Forma y Momentos de Hu (17 dim)**: Área, perímetro, circularidad ($4\pi A/P^2$), solidez, convexidad y los 7 momentos invariantes de Hu.
  3. **Colorimetría Multiespacial (74 dim)**: Ratios $R/G$, $R/B$, media/desviación en HSV y CIELAB, e histogramas reducidos enmascarados al interior del contorno.
  4. **Microtextura Superficial (46 dim)**: LBP circular uniforme ($P=8, R=1$, 10 bins) y matrices GLCM de Haralick a 4 orientaciones (contraste, energía, homogeneidad, correlación).
- **Lección de Diseño Dimensional**:
  - El vector combinado sumó **8,237 dimensiones**.
  - Sin embargo, los descriptores de forma pura (E2) obtuvieron bajo desempeño (~20% F1) debido a la **ausencia de un objeto de escala métrica externa**: en una foto, una moneda pequeña cercana se proyecta con el mismo radio en píxeles que una moneda grande lejana.
  - HOG aislado (E1) preservó la mayor densidad de señal discriminante.

> 🗣️ **Qué decir (Guión del estudiante)**:
> *"Extrajimos cuatro familias de características con un total de 8,237 dimensiones. Descubrimos algo muy valioso metodológicamente: aunque el color distingue perfectamente si una moneda es de cobre o de oro nórdico, no puede diferenciar una de 10 cents de una de 20 cents. Por su parte, la forma geométrica pura tuvo un rendimiento bajo debido a que el dataset no posee un objeto de calibración milimétrica (como una regla o moneda de referencia), por lo que el diámetro aparente en píxeles varía según la distancia de la cámara. El descriptor HOG fue la verdadera estrella: al codificar los gradientes de los números tallados y las coronas de estrellas en celdas de $8\times 8$, proporcionó la huella digital más robusta."*

---

### Diapositiva 5: Evaluación Experimental y Comparativa de Modelos
* **Tiempo estimado**: 1:30 min.
* **Mensaje clave**: "Una SVM Lineal sobre HOG superó a redes neuronales densas y Random Forest, y demostramos por qué PCA degrada información crítica."

#### Contenido en la diapositiva:
- **Protocolo de Experimentación**:
  - 5 familias experimentales (E1 a E5) $\times$ 5 clasificadores (SVM Lineal, SVM RBF, $k$-NN, Random Forest, MLP) = **25 modelos optimizados** con `GridSearchCV` y 5-Fold Cross Validation.
- **Resultados en Validación**:
  - 🥇 **E1: SVM Lineal ($C=0.01$) sobre HOG**: **F1-Macro 0.8382** | Exactitud: **87.50%** (Modelo Ganador).
  - 🥈 **E5: SVM Lineal sobre Vector Combinado**: F1-Macro 0.8190 | Exactitud: 85.42%.
  - 🥉 **E5: SVM RBF + PCA(95%)**: F1-Macro 0.7587 | Exactitud: 79.17%.
  - **k-NN ($k=3$, Manhattan)**: F1-Macro 0.7517.
  - **MLP (50 neuronas)**: F1-Macro 0.7128.
  - **Random Forest (200 árboles)**: F1-Macro 0.6998.
- **Conclusión sobre PCA**:
  - 195 componentes explican el 95% de la varianza. No obstante, PCA atenúa los microgradientes locales que diferencian los números del relieve, disminuyendo el F1 de 0.8382 a 0.7587.

> 🗣️ **Qué decir (Guión del estudiante)**:
> *"Entrenamos 25 modelos bajo validación cruzada de 5 particiones. El clasificador óptimo fue la Support Vector Machine Lineal con regularización $C=0.01$ sobre el espacio HOG, alcanzando 87.5% de exactitud en validación. ¿Por qué una SVM Lineal superó a modelos más complejos como Random Forest o Redes Neuronales? Porque en un espacio disperso de 8,100 dimensiones, el problema es prácticamente separable de forma lineal; los métodos de ensamble sufren de dispersión dimensional y el MLP requiere miles de muestras más para converger. También probamos PCA: aunque 195 componentes resumían el 95% de varianza, el F1 cayó porque PCA tiende a descartar las variaciones sutiles de alta frecuencia donde precisamente residen los grabados de las monedas."*

---

### Diapositiva 6: Evaluación en Test, Diagnóstico de Fallos y App
* **Tiempo estimado**: 1:30 min.
* **Mensaje clave**: "88.68% de exactitud en prueba independiente, 100% perfecto en bimetálicas, errores explicados por metalurgia y software completo en Streamlit."

#### Contenido en la diapositiva:
- **Métricas Finales en Test Reservado (53 monedas disjuntas)**:
  - **Exactitud Global (*Accuracy*)**: **88.68%** (47 aciertos de 53).
  - **F1-Score Macro**: **0.8877** | **Precisión Macro**: 0.8948 | **Recall Macro**: 0.9010.
  - **Latencia de Inferencia**: **1.36 ms por moneda** (tiempo real en CPU).
- **El Fenómeno de las Familias Metalúrgicas**:
  - **1€ y 2€ (Bimetálicas)**: **100% Precisión, 100% Recall, 100% F1-Score**. La unión concéntrica cuproníquel-latón genera gradientes circulares inconfundibles.
  - **Análisis de los 6 Errores (11.32%)**:
    - **100% intra-aleación**: 3 errores en cobre (1c $\leftrightarrow$ 2c, 5c $\rightarrow$ 2c) y 3 en oro nórdico (20c $\rightarrow$ 10c y 20c $\rightarrow$ 50c).
    - **0% de confusión inter-aleación**: el sistema jamás confunde una moneda de cobre con una de oro o bimetálica.
- **Demostración de la Aplicación Streamlit**:
  - Modo 1 (Moneda Individual), Modo 2 (Escena con Watershed, conteo contable de dinero en euros y descarga CSV) y Modo 3 (Pestaña analítica de métricas).

> 🗣️ **Qué decir (Guión del estudiante)**:
> *"Finalmente, evaluamos el modelo final una sola vez sobre el conjunto de prueba reservado de 53 monedas. Obtuvimos un 88.68% de exactitud y un F1-score macro de 0.8877. Al estudiar la matriz de confusión, descubrimos un patrón fascinante: las monedas bimetálicas de 1 y 2 euros obtuvieron un 100% perfecto de precisión y recall. Por su parte, los únicos 6 errores ocurrieron estrictamente dentro de la misma aleación: monedas de cobre confundidas con cobre y monedas de oro nórdico con oro nórdico, cuando el grabado está desgastado. Por último, desplegamos todo el sistema en una aplicación web interactiva en Streamlit, que detecta múltiples monedas en una escena, las clasifica y entrega el balance total de dinero acumulado en euros con soporte de exportación a CSV. Muchas gracias por su atención, quedamos atentos a sus preguntas."*

---

## ❓ Preguntas Frecuentes del Jurado / Profesor y Cómo Defenderlas

1. **¿Por qué usaron SVM Lineal en lugar de una Red Neuronal Convolucional (CNN)?**
   - *Respuesta*: El enunciado del taller exige expresamente el uso de descriptores clásicos de visión artificial (HOG, Hu, LBP, GLCM) y modelos tradicionales de machine learning. Además, con un dataset limitado de 336 instancias, una CNN profunda sufriría de sobreajuste severo (*overfitting*), mientras que una SVM lineal con regularización $C=0.01$ ofrece generalización matemática garantizada mediante la maximización del margen.

2. **¿Por qué la combinación de todas las características (E5) dio menor resultado que HOG solo (E1)?**
   - *Respuesta*: Es la clásica "maldición de la dimensionalidad" (*curse of dimensionality*). Al agregar los descriptores de forma (17 dimensiones) que tenían baja capacidad discriminante por falta de escala externa, se introdujo ruido en la distancia del hiperplano de decisión, reduciendo ligeramente el margen de separación.

3. **¿Cómo garantizan que su conteo de monedas en Streamlit no cuente cosas que no son monedas?**
   - *Respuesta*: El algoritmo de segmentación en `src/preprocessing/segmentation.py` implementa filtros geométricos estrictos: solo se aceptan contornos con un área mínima ($>500 \text{ px}^2$), una circularidad mínima ($4\pi A/P^2 > 0.40$) y una solidez ($A/A_{hull} > 0.80$). Cualquier mancha o sombra irregular queda descartada automáticamente.

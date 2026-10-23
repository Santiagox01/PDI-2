"""
Compilador autonomo de informe academico en formato IEEE (PDF) utilizando ReportLab.
Incorpora diseno a dos columnas, figuras de alta resolucion, tablas con resultados reales
y referencias bibliograficas verificables en espanol segun la guia de la UdeA.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Agrega pie de pagina formal con numeracion de pagina 'Página X de Y' y encabezado IEEE."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 8)
        self.setFillColor(colors.HexColor("#4b5563"))

        # Encabezado (a partir de pagina 2)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "Universidad de Antioquia | Procesamiento Digital de Imágenes")
            self.drawRightString(letter[0] - 54, letter[1] - 36, "EuroCoinClassifier: Clasificación de Monedas de Euro")
            self.setStrokeColor(colors.HexColor("#d1d5db"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 40, letter[0] - 54, letter[1] - 40)

        # Pie de pagina
        page_text = f"Página {self._pageNumber} de {page_count}"
        self.drawRightString(letter[0] - 54, 30, page_text)
        self.drawString(54, 30, "Proyecto Autónomo de Visión por Computadora -- Formato IEEE")
        self.setStrokeColor(colors.HexColor("#d1d5db"))
        self.setLineWidth(0.5)
        self.line(54, 40, letter[0] - 54, 40)

        self.restoreState()


def build_pdf_report(output_pdf_path: str = "reports/report_ieee.pdf"):
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Estilos tipograficos tipo IEEE
    title_style = ParagraphStyle(
        "IEEETitle",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=16,
        leading=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#111827"),
        spaceAfter=6
    )

    authors_style = ParagraphStyle(
        "IEEEAuthors",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=9.5,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=8
    )

    abstract_heading = ParagraphStyle(
        "IEEEAbstractHead",
        parent=styles["Normal"],
        fontName="Times-BoldItalic",
        fontSize=8.5,
        leading=11,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#111827")
    )

    abstract_text = ParagraphStyle(
        "IEEEAbstract",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=8.5,
        leading=11,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=5
    )

    sec_style = ParagraphStyle(
        "IEEESection",
        parent=styles["Normal"],
        fontName="Times-Bold",
        fontSize=10.5,
        leading=13.5,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=8,
        spaceAfter=3
    )

    subsec_style = ParagraphStyle(
        "IEEESubSection",
        parent=styles["Normal"],
        fontName="Times-BoldItalic",
        fontSize=9.5,
        leading=12,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#1f2937"),
        spaceBefore=5,
        spaceAfter=2
    )

    body_style = ParagraphStyle(
        "IEEEBody",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=9,
        leading=11.5,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#111827"),
        spaceAfter=4
    )

    caption_style = ParagraphStyle(
        "IEEECaption",
        parent=styles["Normal"],
        fontName="Times-Italic",
        fontSize=8,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4b5563"),
        spaceAfter=5
    )

    ref_style = ParagraphStyle(
        "IEEERef",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=7.5,
        leading=9.5,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#111827"),
        spaceAfter=2
    )

    story = []

    # 1. Titulo y Autores
    story.append(Paragraph("Clasificación y Detección de Monedas de Euro mediante Descriptores Clásicos de Visión por Computadora y Aprendizaje Automático", title_style))
    story.append(Paragraph(
        "<b>Primer Integrante</b>, <b>Segundo Integrante</b>, <b>Tercer Integrante</b><br/>"
        "<i>Facultad de Ingeniería, Departamento de Ingeniería Electrónica y de Telecomunicaciones</i><br/>"
        "<i>Universidad de Antioquia UdeA, Medellín, Colombia</i><br/>"
        "Procesamiento Digital de Imágenes | {integrante1, integrante2, integrante3}@udea.edu.co",
        authors_style
    ))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#9ca3af"), spaceAfter=8))

    # 2. Resumen y Palabras Clave
    resumen_text = (
        "<b><i>Resumen</i>—En este trabajo se presenta el diseño, implementación y validación experimental de un sistema autónomo "
        "para la detección, segmentación y clasificación de monedas de euro en fotografías de escenas reales mediante técnicas clásicas de visión por computadora. "
        "A partir del dataset público <i>janstaffa/euro-coins-dataset</i> (150 fotografías de alta resolución de 960 × 720 píxeles con 336 monedas etiquetadas en formato YOLO), "
        "se implementó un pipeline riguroso compuesto por realce de iluminación CLAHE en espacio CIELAB, umbralización bimodal de Otsu, apertura y cierre morfológico, "
        "y separación de objetos en contacto mediante Transformada de Distancia y Watershed. Se extrajeron de forma modular cuatro familias de características: "
        "Histograma de Gradientes Orientados (HOG, 8100 dimensiones), descriptores geométricos y 7 momentos invariantes de Hu (17 dimensiones), "
        "colorimetría enmascarada HSV/CIELAB/RGB e histogramas reducidos (74 dimensiones), y textura superficial basada en LBP uniforme y matrices GLCM (46 dimensiones). "
        "Para prevenir de forma estricta la fuga de información (<i>data leakage</i>), las particiones de datos (70% entrenamiento, 15% validación y 15% prueba) "
        "se agruparon por fotografía de origen, impidiendo que recortes de una misma escena aparezcan simultáneamente en entrenamiento y prueba. "
        "Se evaluaron sistemáticamente cinco clasificadores (SVM Lineal, SVM RBF, k-NN, Random Forest y MLP) a través de cinco experimentos (E1 a E5) y reducción por PCA. "
        "El modelo seleccionado, SVM con kernel lineal sobre descriptores HOG (C=0.01), alcanzó una exactitud del 88.68% y un F1-score macro de 0.8877 sobre el conjunto de prueba reservado, "
        "con una latencia de inferencia de 1.36 ms por moneda. Finalmente, se integró una aplicación gráfica interactiva en Streamlit que permite la clasificación individual "
        "y el conteo automatizado con balance contable en euros sobre fotografías de escenas completas.</b>"
    )
    story.append(Paragraph(resumen_text, abstract_text))
    story.append(Paragraph("<b><i>Palabras clave</i>—Visión por computadora, Monedas de euro, HOG, Momentos de Hu, LBP, GLCM, Watershed, SVM, Prevención de fuga de información.</b>", abstract_text))
    story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#9ca3af"), spaceAfter=10))

    # 3. Introduccion
    story.append(Paragraph("I. INTRODUCCIÓN", sec_style))
    story.append(Paragraph(
        "El reconocimiento y conteo automatizado de numerario metálico representa una aplicación crítica en máquinas expendedoras, "
        "sistemas de transporte público y terminales bancarias de autoservicio. A diferencia de los billetes, las monedas metálicas introducen dificultades "
        "ópticas sustanciales: reflexión especular desmedida, desgaste físico del relieve por abrasión, suciedad y pátinas de óxido, además de la heterogeneidad de diseños "
        "nacionales conmemorativos adoptados por los distintos bancos centrales de la Eurozona.", body_style))
    story.append(Paragraph(
        "Las ocho denominaciones oficiales se organizan en tres grupos metalúrgicos: (1) Aleación de Cobre (1 cent, 2 cents, 5 cents), compuestas de acero recubierto de cobre; "
        "(2) Oro Nórdico (10 cents, 20 cents, 50 cents), basadas en la aleación CuAl5Zn5Sn1; y (3) Bimetálicas (1 euro y 2 euros), integradas por anillos y núcleos concéntricos alternados. "
        "El objetivo de este proyecto consiste en diseñar, optimizar y validar experimentalmente un sistema autónomo de clasificación multiclase que resuelva la ambigüedad "
        "entre denominaciones utilizando estrictamente descriptores clásicos de visión artificial, modelos de machine learning interpretables y un protocolo que impida la fuga de datos.", body_style))

    # 4. Marco Teorico
    story.append(Paragraph("II. MARCO TEÓRICO", sec_style))
    story.append(Paragraph("A. Corrección de Iluminación: CLAHE en Espacio CIELAB", subsec_style))
    story.append(Paragraph(
        "La técnica de Ecualización Adaptativa de Histograma con Limitación de Contraste (CLAHE) subdivide la imagen en bloques contextuales de 8 × 8 píxeles. "
        "Para mitigar el ruido sobre zonas homogéneas, limita la pendiente acumulada a un umbral de recorte (clip limit = 2.0). "
        "Al transformar previamente el espacio de color de BGR a CIELAB y aplicar CLAHE exclusivamente sobre el canal de luminosidad L*, "
        "se corrigen sombras pronunciadas e iluminación no uniforme sin introducir aberraciones cromáticas en las coordenadas a* y b*.", body_style))

    story.append(Paragraph("B. Segmentación y Desprendimiento con Watershed y Transformada de Distancia", subsec_style))
    story.append(Paragraph(
        "Cuando múltiples monedas se ubican en contacto físico en una misma fotografía, la umbralización bimodal de Otsu produce agrupaciones conexas fusionadas. "
        "Para segregarlas sin distorsionar sus límites reales, se aplica la Transformada de Distancia Euclidiana (L2) sobre la máscara binarizada y limpia. "
        "Los máximos locales correspondientes al 40% del valor máximo de distancia identifican las 'zonas seguras' de primer plano, que sirven como marcadores "
        "invariables en la propagación por inmersión topográfica del algoritmo Watershed [2].", body_style))

    story.append(Paragraph("C. Histograma de Gradientes Orientados (HOG)", subsec_style))
    story.append(Paragraph(
        "El descriptor HOG cuantifica la distribución de orientaciones de los gradientes espaciales de intensidad [1]. En una cuadrícula normalizada de 128 × 128 píxeles, "
        "se calculan gradientes horizontales y verticales mediante máscaras [-1, 0, 1]. Los ángulos de gradiente se dividen en 9 canales de orientación entre 0° y 180°. "
        "Agrupando celdas de 8 × 8 píxeles en bloques de 2 × 2 celdas con normalización L2-Hys y solapamiento del 50%, se obtiene un vector estructurado de 8100 dimensiones "
        "que codifica el micro-relieve, las estrellas, las figuras numerales y el perfil de canto de cada moneda.", body_style))

    story.append(Paragraph("D. Descriptores Geométricos, Colorimétricos y de Textura", subsec_style))
    story.append(Paragraph(
        "Para complementar la estructura de gradientes, se extraen descriptores invariantes: (1) Forma: área, perímetro, circularidad (4πA/P²), solidez, relación de aspecto y "
        "los 7 momentos invariantes de Hu log-transformados [3]; (2) Color: medias, desviaciones estándar, medianas y ratios cromáticos (R/G, R/B) en HSV, CIELAB y RGB calculados "
        "estrictamente dentro de la máscara binaria del objeto (evitando contaminación del fondo); y (3) Textura: Local Binary Patterns (LBP) circulares uniformes en dos escalas "
        "(P=8, R=1 y P=16, R=2) [4] junto con estadísticos de matrices de co-ocurrencia de niveles de gris (GLCM) de Haralick [5] (contraste, homogeneidad, energía, correlación y disimilitud).", body_style))

    # 5. Metodologia y Auditoria
    story.append(Paragraph("III. METODOLOGÍA Y AUDITORÍA DEL DATASET", sec_style))
    story.append(Paragraph(
        "La auditoría formal ejecutada mediante 'src/data/inspect_dataset.py' sobre el dataset 'janstaffa/euro-coins-dataset' reveló la existencia de 150 fotografías "
        "de escenas reales tomadas a 960 × 720 píxeles con un total de 336 monedas anotadas. Todas las imágenes fueron validadas sin corrupción ni duplicados.", body_style))

    # Tabla 1: Distribucion
    t1_data = [
        ["ID", "Denominación", "Aleación", "Monedas", "Proporción (%)", "Diámetro (px)"],
        ["0", "1 cent", "Cobre", "41", "12.20%", "130.0 ± 10.4"],
        ["1", "2 cents", "Cobre", "51", "15.18%", "154.1 ± 11.2"],
        ["2", "5 cents", "Cobre", "56", "16.67%", "158.8 ± 12.1"],
        ["3", "10 cents", "Oro Nórdico", "32", "9.52%", "156.5 ± 11.8"],
        ["4", "20 cents", "Oro Nórdico", "50", "14.88%", "166.7 ± 13.0"],
        ["5", "50 cents", "Oro Nórdico", "27", "8.04%", "173.6 ± 12.5"],
        ["6", "1 euro", "Bimetálica", "51", "15.18%", "176.5 ± 13.4"],
        ["7", "2 euros", "Bimetálica", "28", "8.33%", "184.8 ± 14.1"],
        ["--", "TOTAL", "--", "336", "100.00%", "--"]
    ]
    t1 = Table(t1_data, colWidths=[24, 70, 75, 50, 75, 80])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#f3f4f6")),
        ("FONTNAME", (0, -1), (-1, -1), "Times-Bold")
    ]))
    story.append(t1)
    story.append(Paragraph("Tabla I: Inventario analítico y caracterización geométrica del conjunto de datos.", caption_style))

    story.append(Paragraph(
        "<b>Declaración de Limitación</b>: Conforme a la auditoría, ninguna clase individual alcanza 60 instancias en el dataset crudo original "
        "(oscilando entre 27 y 56 instancias). Esta cifra se declara de forma transparente para cumplir con los estándares de rigor científico.", body_style))

    # Figura: Distribucion de clases
    if os.path.exists("reports/figures/class_distribution.png"):
        story.append(RLImage("reports/figures/class_distribution.png", width=5.6 * inch, height=1.7 * inch))
        story.append(Paragraph("Figura 1: Distribución empírica de frecuencias por denominación y análisis del umbral de 60 muestras.", caption_style))

    # Prevencion de fuga de informacion
    story.append(Paragraph("IV. PARTICIÓN ESTRATIFICADA Y PREVENCIÓN DE FUGA DE INFORMACIÓN", sec_style))
    story.append(Paragraph(
        "Para garantizar que los modelos no aprendan artefactos de la cámara o iluminación particular de una escena, la partición se configuró de forma "
        "<b>estrictamente agrupada</b> por `original_image_id`. Las 150 fotografías completas fueron asignadas de forma disjunta mediante optimización combinatoria: "
        "(1) Entrenamiento: 105 fotografías (235 monedas, 69.94%); (2) Validación: 23 fotografías (48 monedas, 14.29%); y (3) Prueba Reservada: 22 fotografías (53 monedas, 15.77%). "
        "Se confirmó formalmente que ningún objeto de una misma escena coexiste en más de un conjunto (0% de fuga de información). "
        "Asimismo, los escaladores (StandardScaler) y modelos de PCA fueron ajustados exclusivamente con los datos de entrenamiento.", body_style))

    # Preprocesamiento y Segmentacion
    story.append(Paragraph("V. PREPROCESAMIENTO Y EXTRACCIÓN DE ROIs", sec_style))
    story.append(Paragraph(
        "Cada moneda detectada fue recortada con un margen de seguridad del 5% perimetral y normalizada mediante redimensionamiento conservando la relación de aspecto "
        "a un tamaño unificado de 128 × 128 píxeles. En la Figura 2 se contrastan las alternativas de segmentación sobre una escena con múltiples monedas en contacto.", body_style))

    if os.path.exists("reports/figures/segmentation_comparison.png"):
        story.append(RLImage("reports/figures/segmentation_comparison.png", width=5.8 * inch, height=1.2 * inch))
        story.append(Paragraph("Figura 2: Comparativa experimental de segmentación en escena completa: (a) Original, (b) Otsu, (c) Adaptativo, (d) Color HSV, (e) Distancia + Watershed.", caption_style))

    if os.path.exists("reports/figures/roi_segmentation_stages.png"):
        story.append(RLImage("reports/figures/roi_segmentation_stages.png", width=5.5 * inch, height=2.1 * inch))
        story.append(Paragraph("Figura 3: Etapas del pipeline por moneda: (1) ROI 128×128, (2) Realce CLAHE, (3) Máscara binaria limpia, (4) Detección de contorno.", caption_style))

    # Familias de caracteristicas
    story.append(Paragraph("VI. EXTRACCIÓN DE CARACTERÍSTICAS", sec_style))
    story.append(Paragraph(
        "Se implementaron cuatro familias independientes y una representación combinada multimodal (Tabla II). "
        "HOG demandó 23.73 ms por moneda generando 8100 dimensiones; las características de forma requirieron 6.52 ms (17 dimensiones); "
        "color requirió 4.92 ms (74 dimensiones); y textura requirió 10.40 ms (46 dimensiones).", body_style))

    t2_data = [
        ["Experimento", "Familia de Descriptores", "Dimensión", "Latencia (ms/img)", "Propiedad Física Principal"],
        ["E1", "HOG (Histogram of Oriented Gradients)", "8100", "23.73", "Micro-relieve, bordes y motivos del cuño"],
        ["E2", "Forma Geométrico-Morfológica + Hu", "17", "6.52", "Compacidad, circularidad y momentos de Hu"],
        ["E3", "Colorimetría (HSV + CIELAB + RGB)", "74", "4.92", "Discriminación de aleación (Cobre, Oro, Bicolor)"],
        ["E4", "Textura (LBP Uniforme + GLCM)", "46", "10.40", "Abrasión superficial y contraste microscópico"],
        ["E5", "Combinada Multimodal (E1..E4)", "8237", "45.56", "Fusión multivariada complementaria"]
    ]
    t2 = Table(t2_data, colWidths=[45, 150, 45, 65, 175])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (1, 1), (1, -1), "LEFT"),
        ("ALIGN", (4, 1), (4, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db"))
    ]))
    story.append(t2)
    story.append(Paragraph("Tabla II: Resumen dimensional, latencia computacional y propiedades físicas de las familias de características.", caption_style))

    if os.path.exists("reports/figures/hog_visualization.png"):
        story.append(RLImage("reports/figures/hog_visualization.png", width=5.6 * inch, height=1.5 * inch))
        story.append(Paragraph("Figura 4: Visualización de gradientes HOG por denominación mostrando los patrones del relieve facial.", caption_style))

    # Experimentos y comparacion
    story.append(Paragraph("VII. EXPERIMENTACIÓN Y COMPARACIÓN DE CLASIFICADORES", sec_style))
    story.append(Paragraph(
        "Se ejecutó una búsqueda en malla (GridSearchCV) con validación cruzada estratificada (5-folds) sobre el conjunto de entrenamiento para cada par clasificador-descriptor. "
        "La evaluación en el conjunto de validación independiente (48 monedas) permitió ordenar los modelos de forma insesgada (Tabla III).", body_style))

    t3_data = [
        ["Exp.", "Características", "Clasificador", "Parámetros Óptimos", "CV F1-Macro", "Val Acc. (%)", "Val F1-Macro"],
        ["E1", "HOG (8100)", "SVM Lineal", "C=0.01", "0.8044", "87.50%", "0.8382"],
        ["E5", "Combinada (8237)", "SVM Lineal", "C=0.01", "0.8044", "85.42%", "0.8190"],
        ["E5", "Combinada + PCA95", "SVM RBF", "C=10, gamma='scale'", "0.7333", "79.17%", "0.7587"],
        ["E1", "HOG (8100)", "k-NN", "k=3, dist, manhattan", "0.6826", "77.08%", "0.7517"],
        ["E5", "Combinada (8237)", "k-NN", "k=3, dist, manhattan", "0.7095", "77.08%", "0.7403"],
        ["E1", "HOG (8100)", "MLP", "50 hidden, lr=0.001", "0.6798", "75.00%", "0.7128"],
        ["E5", "Combinada (8237)", "Random Forest", "n=200, depth=None", "0.6785", "75.00%", "0.6998"],
        ["E3", "Color (74)", "SVM Lineal", "C=10.0", "0.5901", "60.42%", "0.5695"],
        ["E4", "Textura (46)", "SVM Lineal", "C=1.0", "0.5816", "62.50%", "0.5103"],
        ["E2", "Forma (17)", "Random Forest", "n=200, depth=None", "0.1213", "27.08%", "0.2002"]
    ]
    t3 = Table(t3_data, colWidths=[25, 80, 75, 95, 60, 55, 60])
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#dbeafe")),
        ("FONTNAME", (0, 1), (-1, 1), "Times-Bold")
    ]))
    story.append(t3)
    story.append(Paragraph("Tabla III: Resultados comparativos de los modelos en validación (ordenados por F1-Macro). En azul el modelo seleccionado.", caption_style))

    story.append(Paragraph(
        "<b>Análisis de PCA</b>: Para el vector HOG de 8100 dimensiones, se determinó que 195 componentes explican el 95% de la varianza acumulada (141 para 85% y 166 para 90%). "
        "Sin embargo, la proyección lineal sin reducción retuvo mayor poder discriminativo en SVM Lineal (F1 0.8382 vs 0.7587 con PCA en validación).", body_style))

    # Evaluacion en Test Set
    story.append(Paragraph("VIII. RESULTADOS EN EL CONJUNTO DE PRUEBA RESERVADO", sec_style))
    story.append(Paragraph(
        "El modelo definitivo (E1 con SVM Lineal) fue evaluado una sola vez sobre el conjunto de prueba reservado (53 monedas de 22 fotografías independientes). "
        "El sistema alcanzó una exactitud global del <b>88.68%</b> (47 de 53 monedas correctas), F1-Score Macro de <b>0.8877</b>, "
        "F1-Score Ponderado de <b>0.8853</b> y una latencia de inferencia ultrarrápida de <b>1.36 ms por moneda</b> (Tabla IV).", body_style))

    t4_data = [
        ["Denominación", "Precisión", "Recall", "F1-Score", "Soporte (Monedas Reales)"],
        ["1 cent", "0.833", "0.833", "0.833", "6"],
        ["2 cents", "0.778", "0.875", "0.824", "8"],
        ["5 cents", "1.000", "0.875", "0.933", "8"],
        ["10 cents", "0.714", "1.000", "0.833", "5"],
        ["20 cents", "1.000", "0.625", "0.769", "8"],
        ["50 cents", "0.833", "1.000", "0.909", "5"],
        ["1 euro", "1.000", "1.000", "1.000", "8"],
        ["2 euros", "1.000", "1.000", "1.000", "5"],
        ["PROMEDIO MACRO", "0.895", "0.901", "0.888", "53"],
        ["PROMEDIO PONDERADO", "0.905", "0.887", "0.885", "53"]
    ]
    t4 = Table(t4_data, colWidths=[90, 80, 80, 80, 120])
    t4.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("BACKGROUND", (0, -2), (-1, -1), colors.HexColor("#f3f4f6")),
        ("FONTNAME", (0, -2), (-1, -1), "Times-Bold")
    ]))
    story.append(t4)
    story.append(Paragraph("Tabla IV: Métricas de desempeño por denominación en el conjunto de prueba reservado.", caption_style))

    # Matrices de confusion
    story.append(Paragraph("IX. MATRICES DE CONFUSIÓN", sec_style))
    story.append(Paragraph(
        "En la Figura 5 se presentan las matrices de confusión absoluta y normalizada. Se destaca que las monedas bimetálicas (1€ y 2€) "
        "obtuvieron una separación perfecta (100% de recall y 100% de precisión).", body_style))

    if os.path.exists("reports/figures/confusion_matrix_absolute.png") and os.path.exists("reports/figures/confusion_matrix_normalized.png"):
        c_table_data = [[
            RLImage("reports/figures/confusion_matrix_absolute.png", width=2.8 * inch, height=2.2 * inch),
            RLImage("reports/figures/confusion_matrix_normalized.png", width=2.8 * inch, height=2.2 * inch)
        ]]
        c_table = Table(c_table_data, colWidths=[2.9 * inch, 2.9 * inch])
        c_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story.append(c_table)
        story.append(Paragraph("Figura 5: Matriz de confusión absoluta (izquierda) y normalizada por recall (derecha) sobre el conjunto de prueba.", caption_style))

    # Analisis critico de errores
    story.append(Paragraph("X. ANÁLISIS CRÍTICO DE ERRORES", sec_style))
    story.append(Paragraph(
        "Se registraron únicamente 6 clasificaciones erróneas sobre el conjunto de prueba reservado. "
        "Un hallazgo científico de máxima relevancia es que el <b>100% de los errores fueron estrictamente intra-aleación</b> (cero confusiones entre familias metalúrgicas distintas):", body_style))
    story.append(Paragraph(
        "1. <i>Confusiones en Aleación de Cobre (1c, 2c, 5c)</i>: Tres errores se produjeron dentro de este grupo (ej. 2c como 1c con confianza 26.0%, y 5c como 2c con confianza 50.3%). "
        "<b>Hipótesis técnica plausible</b>: La normalización a 128 × 128 píxeles prescinde de la escala métrica absoluta. Al poseer idéntica textura de cobre y diseño de globo terráqueo en el anverso común, "
        "pequeños desgastes por abrasión o reflejos especulares atenúan los bordes del número facial.", body_style))
    story.append(Paragraph(
        "2. <i>Confusiones en Oro Nórdico (10c, 20c, 50c)</i>: Se registraron tres confusiones en monedas de 20 cents (clasificadas como 10c y 50c). "
        "<b>Hipótesis técnica plausible</b>: Las ranuras 'Spanish flower' de la moneda de 20 cents generan gradientes HOG que, bajo ángulos de iluminación oblicua, "
        "pueden confundirse con las ranuras continuas de 10c o 50c, especialmente en monedas envejecidas.", body_style))

    if os.path.exists("reports/figures/error_analysis_gallery.png"):
        story.append(RLImage("reports/figures/error_analysis_gallery.png", width=5.8 * inch, height=1.3 * inch))
        story.append(Paragraph("Figura 6: Galería de falsos positivos y falsos negativos representativos en el conjunto de prueba con confianza estimada.", caption_style))

    # Aplicacion grafica
    story.append(Paragraph("XI. APLICACIÓN GRÁFICA INTERACTIVA", sec_style))
    story.append(Paragraph(
        "Se implementó una interfaz ergonómica con Streamlit (`app/streamlit_app.py`) articulada en dos modos: "
        "(1) <i>Modo Moneda Individual</i>: recibe un recorte, extrae HOG en tiempo real, predice denominación, calcula valor facial y exhibe la distribución de probabilidades; "
        "(2) <i>Modo Escena Multi-moneda</i>: detecta objetos con Watershed, superpone rectángulos y etiquetas con nivel de confianza, calcula el sumatorio contable total en euros "
        "y habilita la exportación de la escena anotada y el archivo CSV con las predicciones. Dispone además de un modo de depuración para visualizar máscaras y gradientes.", body_style))

    # Discusion y Limitaciones
    story.append(Paragraph("XII. DISCUSIÓN Y LIMITACIONES", sec_style))
    story.append(Paragraph(
        "1. <i>Ausencia de escala métrica externa</i>: Al no disponer de un patrón de calibración métrica en las escenas (como una diana o regla milimetrada), "
        "el tamaño aparente depende de la distancia sensor-moneda, justificando por qué las características de forma puras (E2) tuvieron bajo rendimiento discriminante aisladamente.", body_style))
    story.append(Paragraph(
        "2. <i>Reversos nacionales heterogéneos</i>: El numerario de la Eurozona exhibe un anverso común pero reversos nacionales variables según el país acuñador. "
        "El clasificador HOG lineal demostró gran resiliencia hacia esta variabilidad, aunque un dataset multianual más voluminoso consolidaría aún más las fronteras de decisión.", body_style))

    # Conclusiones
    story.append(Paragraph("XIII. CONCLUSIONES", sec_style))
    story.append(Paragraph(
        "Se construyó y validó un sistema autónomo y reproducible de visión artificial para clasificación de monedas de euro. "
        "La integración de CLAHE en CIELAB, segmentación por Watershed y descriptores HOG con SVM Lineal alcanzó un 88.68% de exactitud y 0.8877 de F1-Score macro, "
        "con 100% de precisión en denominaciones de 1€ y 2€. El estricto protocolo de partición por grupos garantizó cero fuga de información. "
        "Tanto el código fuente, la batería de pruebas unitarias (17 tests superados) y la aplicación gráfica en Streamlit conforman un entregable científico completo y verificable.", body_style))

    # Referencias bibliograficas verificables
    story.append(Paragraph("XIV. REFERENCIAS BIBLIOGRÁFICAS", sec_style))
    refs = [
        "[1] N. Dalal and B. Triggs, ``Histograms of oriented gradients for human detection,'' in <i>IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)</i>, vol. 1, 2005, pp. 886--893.",
        "[2] L. Vincent and P. Soille, ``Watersheds in digital spaces: an efficient algorithm based on immersion simulations,'' <i>IEEE Transactions on Pattern Analysis and Machine Intelligence</i>, vol. 13, no. 6, pp. 583--598, 1991.",
        "[3] M.-K. Hu, ``Visual pattern recognition by moment invariants,'' <i>IRE Transactions on Information Theory</i>, vol. 8, no. 2, pp. 179--187, 1962.",
        "[4] T. Ojala, M. Pietikäinen, and T. Mäenpää, ``Multiresolution gray-scale and rotation invariant texture classification with local binary patterns,'' <i>IEEE Transactions on Pattern Analysis and Machine Intelligence</i>, vol. 24, no. 7, pp. 971--987, 2002.",
        "[5] R. M. Haralick, K. Shanmugam, and I. Dinstein, ``Textural features for image classification,'' <i>IEEE Transactions on Systems, Man, and Cybernetics</i>, vol. SMC-3, no. 6, pp. 610--621, 1973.",
        "[6] N. Otsu, ``A threshold selection method from gray-level histograms,'' <i>IEEE Transactions on Systems, Man, and Cybernetics</i>, vol. 9, no. 1, pp. 62--66, 1979.",
        "[7] C. Cortes and V. Vapnik, ``Support-vector networks,'' <i>Machine Learning</i>, vol. 20, no. 3, pp. 273--297, 1995.",
        "[8] F. Pedregosa et al., ``Scikit-learn: Machine learning in Python,'' <i>Journal of Machine Learning Research</i>, vol. 12, pp. 2825--2830, 2011.",
        "[9] G. Bradski, ``The OpenCV Library,'' <i>Dr. Dobb's Journal of Software Tools</i>, 2000.",
        "[10] J. Staffa, ``EURO coins dataset,'' Kaggle Datasets, DOI: 10.34740/KAGGLE/DSV/2192736, 2021. Disponible en: https://www.kaggle.com/datasets/janstaffa/euro-coins-dataset."
    ]
    for r in refs:
        story.append(Paragraph(r, ref_style))

    # Construir documento PDF con NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[INFO] Documento PDF generado exitosamente en: {output_pdf_path}")


if __name__ == "__main__":
    build_pdf_report("reports/report_ieee.pdf")

"""
Generador de Presentación Ejecutiva en PDF (Exactamente 6 diapositivas panorámicas 16:9).
Proyecto: Clasificación y Detección de Monedas de Euro mediante Visión por Computadora.
Asignatura: Procesamiento Digital de Imágenes - Universidad de Antioquia.
"""

import os
import sys
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from PIL import Image as PILImage

# Dimensiones 16:9 widescreen (960 x 540 puntos)
SLIDE_W = 960
SLIDE_H = 540

# Paleta de colores profesional
C_BG_PAGE = colors.HexColor("#0f172a")        # Slate 900 oscuro para portada
C_BG_LIGHT = colors.HexColor("#f8fafc")       # Slate 50 para diapositivas de contenido
C_NAVY_DARK = colors.HexColor("#0f172a")      # Titulo / texto principal
C_BLUE_BRAND = colors.HexColor("#1e3a8a")     # Azul institucional UdeA / Euro
C_BLUE_ACCENT = colors.HexColor("#0284c7")    # Azul cian
C_GOLD_ACCENT = colors.HexColor("#d97706")    # Dorado moneda euro
C_EMERALD = colors.HexColor("#059669")        # Verde éxito
C_RED_ACCENT = colors.HexColor("#dc2626")     # Rojo errores
C_CARD_BG = colors.HexColor("#ffffff")        # Blanco puro para tarjetas
C_CARD_BG_ALT = colors.HexColor("#f1f5f9")    # Gris claro
C_CARD_BORDER = colors.HexColor("#e2e8f0")    # Borde sutil
C_TEXT_MUTED = colors.HexColor("#64748b")     # Texto secundario


class PresentationBuilder:
    def __init__(self, filename="reports/presentacion_euro_coins.pdf"):
        self.filename = filename
        self.c = canvas.Canvas(filename, pagesize=(SLIDE_W, SLIDE_H))
        self.total_slides = 6

    def draw_slide_header(self, slide_num, category, title, subtitle):
        """Dibuja el encabezado estándar para diapositivas de contenido."""
        c = self.c
        # Fondo general
        c.setFillColor(C_BG_LIGHT)
        c.rect(0, 0, SLIDE_W, SLIDE_H, fill=True, stroke=False)

        # Barra superior decorativa
        c.setFillColor(C_BLUE_BRAND)
        c.rect(0, SLIDE_H - 6, SLIDE_W, 6, fill=True, stroke=False)

        # Categoría / Chip
        c.setFillColor(C_BLUE_ACCENT)
        c.roundRect(40, SLIDE_H - 46, 160, 20, 4, fill=True, stroke=False)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(48, SLIDE_H - 42, category.upper())

        # Título principal
        c.setFillColor(C_NAVY_DARK)
        c.setFont("Helvetica-Bold", 20)
        c.drawString(40, SLIDE_H - 72, title)

        # Subtítulo descriptivo
        c.setFillColor(C_TEXT_MUTED)
        c.setFont("Helvetica", 11)
        c.drawString(40, SLIDE_H - 90, subtitle)

        # Línea divisoria suave
        c.setStrokeColor(C_CARD_BORDER)
        c.setLineWidth(1)
        c.line(40, SLIDE_H - 100, SLIDE_W - 40, SLIDE_H - 100)

    def draw_slide_footer(self, slide_num):
        """Pie de página institucional con numeración."""
        c = self.c
        c.setStrokeColor(C_CARD_BORDER)
        c.setLineWidth(1)
        c.line(40, 36, SLIDE_W - 40, 36)

        c.setFillColor(C_TEXT_MUTED)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(40, 20, "Universidad de Antioquia | Procesamiento Digital de Imágenes")
        c.setFont("Helvetica", 9)
        c.drawString(400, 20, "Proyecto EuroCoinClassifier — Clasificación y Detección de Monedas")

        # Contador de diapositivas
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(C_BLUE_BRAND)
        c.drawRightString(SLIDE_W - 40, 20, f"Diapositiva {slide_num} de {self.total_slides}")

    def draw_card(self, x, y, w, h, title=None, subtitle=None, bg=C_CARD_BG, border=C_CARD_BORDER):
        """Dibuja un contenedor tipo tarjeta moderna con sombra sutil."""
        c = self.c
        # Fondo y borde
        c.setFillColor(bg)
        c.setStrokeColor(border)
        c.setLineWidth(1)
        c.roundRect(x, y, w, h, 8, fill=True, stroke=True)

        # Título interno si aplica
        if title:
            c.setFillColor(C_NAVY_DARK)
            c.setFont("Helvetica-Bold", 13)
            c.drawString(x + 16, y + h - 26, title)
            if subtitle:
                c.setFillColor(C_TEXT_MUTED)
                c.setFont("Helvetica", 9)
                c.drawString(x + 16, y + h - 40, subtitle)
            c.setStrokeColor(border)
            c.line(x + 16, y + h - 46, x + w - 16, y + h - 46)

    def draw_metric_badge(self, x, y, w, h, value, label, subtext="", color_val=C_BLUE_BRAND):
        """Tarjeta pequeña para destacar KPIs o métricas."""
        c = self.c
        c.setFillColor(C_CARD_BG)
        c.setStrokeColor(C_CARD_BORDER)
        c.roundRect(x, y, w, h, 6, fill=True, stroke=True)

        c.setFillColor(color_val)
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(x + w / 2, y + h - 26, value)

        c.setFillColor(C_NAVY_DARK)
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(x + w / 2, y + h - 40, label)

        if subtext:
            c.setFillColor(C_TEXT_MUTED)
            c.setFont("Helvetica", 7.5)
            c.drawCentredString(x + w / 2, y + 10, subtext)

    # -------------------------------------------------------------
    # SLIDE 1: PORTADA Y CONTEXTO
    # -------------------------------------------------------------
    def render_slide_1(self):
        c = self.c
        # Fondo oscuro elegante
        c.setFillColor(colors.HexColor("#090d16"))
        c.rect(0, 0, SLIDE_W, SLIDE_H, fill=True, stroke=False)

        # Acento geométrico lateral
        c.setFillColor(colors.HexColor("#1e293b"))
        c.rect(0, 0, 16, SLIDE_H, fill=True, stroke=False)
        c.setFillColor(C_BLUE_ACCENT)
        c.rect(16, 0, 8, SLIDE_H, fill=True, stroke=False)

        # Chip de Asignatura
        c.setFillColor(C_GOLD_ACCENT)
        c.roundRect(60, SLIDE_H - 85, 340, 24, 6, fill=True, stroke=False)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(72, SLIDE_H - 69, "PROCESAMIENTO DIGITAL DE IMÁGENES | UdeA")

        # Título Principal
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 30)
        c.drawString(60, SLIDE_H - 140, "EuroCoinClassifier")
        c.setFont("Helvetica-Bold", 20)
        c.setFillColor(colors.HexColor("#93c5fd"))
        c.drawString(60, SLIDE_H - 172, "Clasificación y Detección Autónoma de Monedas de Euro")

        # Subtítulo metodológico
        c.setFillColor(colors.HexColor("#cbd5e1"))
        c.setFont("Helvetica", 13)
        c.drawString(60, SLIDE_H - 204, "Pipeline integral de visión clásica, extracción multimodal y aprendizaje automático")

        # Línea divisoria
        c.setStrokeColor(colors.HexColor("#334155"))
        c.setLineWidth(1.5)
        c.line(60, SLIDE_H - 225, SLIDE_W - 60, SLIDE_H - 225)

        # 3 Tarjetas de Resumen Ejecutivo en Portada
        box_w = 265
        box_h = 175
        y_box = 100

        # Tarjeta 1: Desafío
        c.setFillColor(colors.HexColor("#1e293b"))
        c.roundRect(60, y_box, box_w, box_h, 8, fill=True, stroke=False)
        c.setFillColor(C_GOLD_ACCENT)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(80, y_box + box_h - 30, "🎯 El Desafío Visual")
        c.setFillColor(colors.HexColor("#e2e8f0"))
        c.setFont("Helvetica", 10)
        c.drawString(80, y_box + box_h - 55, "• 8 denominaciones oficiales.")
        c.drawString(80, y_box + box_h - 75, "• 3 aleaciones metalúrgicas:")
        c.drawString(95, y_box + box_h - 93, "Cobre, Oro Nórdico y Bimetálicas.")
        c.drawString(80, y_box + box_h - 115, "• Reflejos, desgaste por uso y")
        c.drawString(95, y_box + box_h - 133, "monedas en contacto mutuo.")

        # Tarjeta 2: Enfoque Riguroso
        c.setFillColor(colors.HexColor("#1e293b"))
        c.roundRect(345, y_box, box_w, box_h, 8, fill=True, stroke=False)
        c.setFillColor(C_BLUE_ACCENT)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(365, y_box + box_h - 30, "🔬 Metodología Rigurosa")
        c.setFillColor(colors.HexColor("#e2e8f0"))
        c.setFont("Helvetica", 10)
        c.drawString(365, y_box + box_h - 55, "• CLAHE en espacio CIELAB (L*).")
        c.drawString(365, y_box + box_h - 75, "• Transformada Distancia + Watershed.")
        c.drawString(365, y_box + box_h - 95, "• 4 familias: HOG, Forma, Color,")
        c.drawString(380, y_box + box_h - 113, "y Microtextura (LBP / GLCM).")
        c.drawString(365, y_box + box_h - 135, "• Cero fuga de datos (split por foto).")

        # Tarjeta 3: Logros Clave
        c.setFillColor(colors.HexColor("#1e293b"))
        c.roundRect(630, y_box, box_w, box_h, 8, fill=True, stroke=False)
        c.setFillColor(C_EMERALD)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(650, y_box + box_h - 30, "🏆 Logros Principales")
        c.setFillColor(colors.HexColor("#e2e8f0"))
        c.setFont("Helvetica", 10)
        c.drawString(650, y_box + box_h - 55, "• 88.68% Exactitud en Test.")
        c.drawString(650, y_box + box_h - 75, "• F1-Macro 0.8877 en 53 monedas.")
        c.drawString(650, y_box + box_h - 95, "• 100% Precisión en Bimetálicas.")
        c.drawString(650, y_box + box_h - 115, "• Latencia ultra-baja: 1.36 ms.")
        c.drawString(650, y_box + box_h - 135, "• App Streamlit + Paper IEEE (5 pp).")

        # Pie de portada
        c.setFillColor(colors.HexColor("#94a3b8"))
        c.setFont("Helvetica", 10)
        c.drawString(60, 45, "Equipo de Desarrollo: Estudiantes PDI | Departamento de Ingeniería Electrónica y Telecomunicaciones | UdeA (2026)")
        c.showPage()

    # -------------------------------------------------------------
    # SLIDE 2: DATASET, AUDITORÍA Y PARTICIÓN SIN FUGA
    # -------------------------------------------------------------
    def render_slide_2(self):
        self.draw_slide_header(
            2,
            "Auditoría y Datos",
            "Dataset: Caracterización y Prevención de Fuga de Información",
            "Inspección de 336 instancias y formulación de particiones disjuntas por fotografía original"
        )

        # Columna Izquierda: Tarjeta descriptiva del dataset
        self.draw_card(40, 56, 440, 360, "Inventario y Distribución de Clases", "Kaggle: janstaffa/euro-coins-dataset (150 escenas a 960x720 px)")

        c = self.c
        # Texto descriptivo
        c.setFillColor(C_NAVY_DARK)
        c.setFont("Helvetica", 10)
        c.drawString(56, 350, "• 150 fotografías con 336 cajas delimitadoras (YOLO format).")
        c.drawString(56, 332, "• 8 clases con desbalance natural (desde 27 hasta 56 instancias).")
        c.drawString(56, 314, "• Declaración de rigor: ninguna clase supera las 60 muestras de forma")
        c.drawString(70, 298, "aislada, lo que se documenta abiertamente en el informe técnico.")

        # Tabla de distribución condensada
        y_t = 280
        c.setFillColor(C_CARD_BG_ALT)
        c.rect(56, y_t - 15, 408, 18, fill=True, stroke=False)
        c.setFillColor(C_NAVY_DARK)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(62, y_t - 10, "Denominación")
        c.drawString(160, y_t - 10, "Familia Metal")
        c.drawString(275, y_t - 10, "Instancias")
        c.drawString(365, y_t - 10, "Ancho Prom.")

        classes_info = [
            ("1, 2, 5 cents", "Cobre (Acero/Cu)", "148 (44.1%)", "130 - 158 px"),
            ("10, 20, 50 cents", "Oro Nórdico (CuAlZnSn)", "109 (32.4%)", "156 - 173 px"),
            ("1 y 2 euros", "Bimetálicas (CuNi/Latón)", "79 (23.5%)", "176 - 184 px"),
        ]
        y_r = y_t - 32
        for d, f, inst, px in classes_info:
            c.setFont("Helvetica-Bold" if "Bimet" in f else "Helvetica", 8.5)
            c.drawString(62, y_r, d)
            c.drawString(160, y_r, f)
            c.drawString(275, y_r, inst)
            c.drawString(365, y_r, px)
            y_r -= 18

        # Recuadro crítico de Data Leakage
        c.setFillColor(colors.HexColor("#eff6ff"))
        c.setStrokeColor(C_BLUE_BRAND)
        c.roundRect(56, 75, 408, 90, 6, fill=True, stroke=True)

        c.setFillColor(C_BLUE_BRAND)
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(68, 145, "🛡️ Regla de Oro: Prevención de Fuga de Datos (Data Leakage)")
        c.setFillColor(C_NAVY_DARK)
        c.setFont("Helvetica", 9)
        c.drawString(68, 128, "Una partición aleatoria de monedas mezclaría monedas de la misma foto")
        c.drawString(68, 112, "en Train y Test (mismo fondo, sensor e iluminación = desempeño inflado).")
        c.setFont("Helvetica-Bold", 9)
        c.drawString(68, 94, "Solución: Partición estratificada por foto original (original_image_id):")
        c.setFont("Helvetica", 8.5)
        c.drawString(68, 80, "Train: 105 fotos (235 mon.) | Val: 23 fotos (48 mon.) | Test: 22 fotos (53 mon.)")

        # Columna Derecha: Imagen ilustrativa y galería del dataset
        self.draw_card(500, 56, 420, 360, "Muestra Visual del Conjunto de Datos", "Fotografías de escenas con iluminación variada y monedas en contacto")

        img_path = "outputs/figures/dataset_sample_gallery.png"
        if os.path.exists(img_path):
            c.drawImage(img_path, 515, 175, width=390, height=210, preserveAspectRatio=True)

        # Gráfico o resumen inferior
        self.draw_metric_badge(515, 75, 120, 75, "150", "Fotos Totales", "960x720 RGB")
        self.draw_metric_badge(650, 75, 120, 75, "336", "Monedas Ground Truth", "8 Clases oficiales")
        self.draw_metric_badge(785, 75, 120, 75, "0%", "Fuga de Datos", "Partición disjunta")

        self.draw_slide_footer(2)
        c.showPage()

    # -------------------------------------------------------------
    # SLIDE 3: PREPROCESAMIENTO Y SEGMENTACIÓN ROBUSTA
    # -------------------------------------------------------------
    def render_slide_3(self):
        c = self.c
        self.draw_slide_header(
            3,
            "Preprocesamiento",
            "Pipeline de Preprocesamiento y Segmentación Morfológica",
            "Corrección lumínica por CLAHE y desprendimiento de monedas tangenciales mediante Watershed"
        )

        # Columna Izquierda: Imagen de etapas del procesamiento
        self.draw_card(40, 56, 440, 360, "Etapas de Segmentación de Objetos", "Pipeline visual desde la imagen cruda hasta la máscara normalizada")

        img_path = "outputs/figures/roi_segmentation_stages.png"
        if os.path.exists(img_path):
            c.drawImage(img_path, 50, 80, width=420, height=300, preserveAspectRatio=True)

        # Columna Derecha: 3 bloques metodológicos
        self.draw_card(500, 56, 420, 360, "Módulos Clave del Preprocesamiento", "Fundamentación algorítmica para garantizar robustez")

        # Bloque 1: CLAHE en CIELAB
        c.setFillColor(C_CARD_BG_ALT)
        c.roundRect(515, 275, 390, 80, 6, fill=True, stroke=False)
        c.setFillColor(C_BLUE_BRAND)
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(525, 338, "1. Corrección de Iluminación: CLAHE en CIELAB")
        c.setFillColor(C_NAVY_DARK)
        c.setFont("Helvetica", 9)
        c.drawString(525, 322, "• Se procesa exclusivamente el canal de luminancia L* (L=8x8 bloques).")
        c.drawString(525, 306, "• ClipLimit=2.0 mitiga sombras duras y brillos especulares de flash.")
        c.drawString(525, 290, "• Preserva intacto el balance cromático intrínseco (a* y b*).")

        # Bloque 2: Otsu + Morfología
        c.setFillColor(C_CARD_BG_ALT)
        c.roundRect(515, 180, 390, 80, 6, fill=True, stroke=False)
        c.setFillColor(C_GOLD_ACCENT)
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(525, 243, "2. Binarización Adaptativa y Morfología")
        c.setFillColor(C_NAVY_DARK)
        c.setFont("Helvetica", 9)
        c.drawString(525, 227, "• Umbralización global bimodal de Otsu maximiza varianza inter-clase.")
        c.drawString(525, 211, "• Filtro Gaussiano 5x5 previo para atenuar ruido de alta frecuencia.")
        c.drawString(525, 195, "• Operador de Cierre (Closing) 5x5 y rellenado de huecos por FloodFill.")

        # Bloque 3: Watershed y Transformada de Distancia
        c.setFillColor(C_CARD_BG_ALT)
        c.roundRect(515, 75, 390, 92, 6, fill=True, stroke=False)
        c.setFillColor(C_EMERALD)
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(525, 149, "3. Desprendimiento de Monedas en Contacto (Watershed)")
        c.setFillColor(C_NAVY_DARK)
        c.setFont("Helvetica", 9)
        c.drawString(525, 133, "• Monedas que se tocan generan un contorno fusionado erróneo.")
        c.drawString(525, 117, "• Se calcula la Transformada de Distancia Euclidiana sobre la máscara.")
        c.drawString(525, 101, "• Los máximos locales actúan como marcadores en la inundación")
        c.drawString(525, 85, "  topográfica de Watershed, separando con precisión los bordes.")

        self.draw_slide_footer(3)
        c.showPage()

    # -------------------------------------------------------------
    # SLIDE 4: EXTRACCIÓN DE CARACTERÍSTICAS
    # -------------------------------------------------------------
    def render_slide_4(self):
        self.draw_slide_header(
            4,
            "Ingeniería de Características",
            "Extracción Multimodal: De Contornos HOG a Microtextura y Color",
            "Caracterización discriminante en 4 familias y 8,237 dimensiones totales"
        )

        c = self.c
        # 4 Tarjetas de familias de características
        col_w = 205
        y_card = 210
        h_card = 205

        # Familia 1: HOG
        self.draw_card(40, y_card, col_w, h_card, "1. Gradientes (HOG)", "8,100 dimensiones", bg=colors.HexColor("#f0fdf4"), border=C_EMERALD)
        c.setFillColor(C_EMERALD)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(54, y_card + h_card - 62, "⭐ Familia Ganadora")
        c.setFillColor(C_NAVY_DARK)
        c.setFont("Helvetica", 8.5)
        c.drawString(54, y_card + h_card - 80, "• Celdas de 8x8 px, 9 bins.")
        c.drawString(54, y_card + h_card - 96, "• Bloques 2x2 normalizados.")
        c.drawString(54, y_card + h_card - 112, "• Captura el micro-relieve,")
        c.drawString(54, y_card + h_card - 128, "  dígitos faciales y estrellas.")
        c.drawString(54, y_card + h_card - 144, "• Invariante a iluminación.")
        c.drawString(54, y_card + h_card - 160, "• Alto F1-Score: 0.8877.")

        # Familia 2: Forma y Hu
        self.draw_card(260, y_card, col_w, h_card, "2. Geometría / Hu", "17 dimensiones", border=C_CARD_BORDER)
        c.setFillColor(C_NAVY_DARK)
        c.setFont("Helvetica", 8.5)
        c.drawString(274, y_card + h_card - 62, "• 7 Momentos de Hu.")
        c.drawString(274, y_card + h_card - 78, "• Área, perímetro, circularidad.")
        c.drawString(274, y_card + h_card - 94, "• Solidez y relación de aspecto.")
        c.drawString(274, y_card + h_card - 110, "• Radio aparente y convexidad.")
        c.setFillColor(C_RED_ACCENT)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(274, y_card + h_card - 132, "Limitación declarada:")
        c.setFont("Helvetica", 8)
        c.drawString(274, y_card + h_card - 146, "Sin escala milimétrica externa,")
        c.drawString(274, y_card + h_card - 160, "el diámetro depende de la cámara.")

        # Familia 3: Colorimetría
        self.draw_card(480, y_card, col_w, h_card, "3. Color Multiespacio", "74 dimensiones", border=C_CARD_BORDER)
        c.setFillColor(C_NAVY_DARK)
        c.setFont("Helvetica", 8.5)
        c.drawString(494, y_card + h_card - 62, "• Ratios R/G y R/B en RGB.")
        c.drawString(494, y_card + h_card - 78, "• Media y desviación en HSV.")
        c.drawString(494, y_card + h_card - 94, "• Canales a* y b* en CIELAB.")
        c.drawString(494, y_card + h_card - 110, "• Histogramas reducidos (8 bins).")
        c.setFillColor(C_BLUE_BRAND)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(494, y_card + h_card - 132, "Excelente separador:")
        c.setFont("Helvetica", 8)
        c.drawString(494, y_card + h_card - 146, "Distingue al 100% Cobre de")
        c.drawString(494, y_card + h_card - 160, "Oro Nórdico y Bimetálicas.")

        # Familia 4: Textura
        self.draw_card(700, y_card, col_w, h_card, "4. Textura (LBP/GLCM)", "46 dimensiones", border=C_CARD_BORDER)
        c.setFillColor(C_NAVY_DARK)
        c.setFont("Helvetica", 8.5)
        c.drawString(714, y_card + h_card - 62, "• LBP circular uniforme:")
        c.drawString(724, y_card + h_card - 76, "P=8, R=1 (10 bins).")
        c.drawString(714, y_card + h_card - 92, "• GLCM (Matrices de Haralick):")
        c.drawString(724, y_card + h_card - 106, "Contraste, homogeneidad,")
        c.drawString(724, y_card + h_card - 120, "energía, correlación y ASM.")
        c.drawString(714, y_card + h_card - 136, "• 4 orientaciones (0°, 45°, 90°, 135°).")
        c.drawString(714, y_card + h_card - 152, "• Evalúa la rugosidad superficial.")

        # Fila inferior: Visualización de HOG y Conclusión Dimensional
        self.draw_card(40, 56, 420, 140, "Visualización de Gradientes HOG", "Captura de relieves circulares y bordes numerales")
        img_hog = "outputs/figures/hog_visualization.png"
        if os.path.exists(img_hog):
            c.drawImage(img_hog, 50, 62, width=400, height=95, preserveAspectRatio=True)

        self.draw_card(480, 56, 440, 140, "Conclusión de Ingeniería de Características", "Por qué más dimensiones no siempre es mejor")
        c.setFillColor(C_NAVY_DARK)
        c.setFont("Helvetica", 9)
        c.drawString(496, 146, "• Vector Multimodal Combinado (E5): 8,100 + 17 + 74 + 46 = 8,237 variables.")
        c.drawString(496, 130, "• Aunque el color y textura aportan discriminación, la inclusión de Forma")
        c.drawString(496, 114, "  pura sin escala métrica absoluta introdujo ruido en la frontera de decisión.")
        c.setFillColor(C_BLUE_BRAND)
        c.setFont("Helvetica-Bold", 9.5)
        c.drawString(496, 92, "• HOG aislado (E1) preservó la mayor densidad de señal discriminante")
        c.drawString(496, 76, "  logrando el mejor equilibrio entre generalización y coste de cómputo.")

        self.draw_slide_footer(4)
        c.showPage()

    # -------------------------------------------------------------
    # SLIDE 5: COMPARACIÓN EXPERIMENTAL DE MODELOS
    # -------------------------------------------------------------
    def render_slide_5(self):
        c = self.c
        self.draw_slide_header(
            5,
            "Evaluación Experimental",
            "Comparativa de Modelos (E1 a E5) y Optimización de Hiperparámetros",
            "Entrenamiento de 25 configuraciones mediante 5-Fold Cross Validation estratificado"
        )

        # Columna Izquierda: Gráfica de modelos
        self.draw_card(40, 56, 440, 360, "F1-Score en Validación por Experimento", "Comparación de SVM, k-NN, Random Forest y MLP")
        img_comp = "outputs/figures/model_comparison_chart.png"
        if os.path.exists(img_comp):
            c.drawImage(img_comp, 50, 75, width=420, height=295, preserveAspectRatio=True)

        # Columna Derecha: Tabla resumen de resultados y hallazgos
        self.draw_card(500, 56, 420, 360, "Tabla Comparativa de Rendimiento", "Conjunto de Validación (48 monedas disjuntas)")

        # Tabla de modelos clave
        y_t = 345
        c.setFillColor(C_CARD_BG_ALT)
        c.rect(515, y_t - 15, 390, 18, fill=True, stroke=False)
        c.setFillColor(C_NAVY_DARK)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(520, y_t - 10, "Exp.")
        c.drawString(560, y_t - 10, "Clasificador")
        c.drawString(670, y_t - 10, "Dim.")
        c.drawString(725, y_t - 10, "Val Acc.")
        c.drawString(810, y_t - 10, "Val F1-Macro")

        top_models = [
            ("E1", "SVM Lineal (C=0.01) [GANADOR]", "8100", "87.50%", "0.8382", True),
            ("E5", "SVM Lineal (C=0.01)", "8237", "85.42%", "0.8190", False),
            ("E5", "SVM RBF + PCA(95%)", "8237", "79.17%", "0.7587", False),
            ("E1", "k-NN (k=3, Manhattan)", "8100", "77.08%", "0.7517", False),
            ("E1", "MLP (50 neuronas)", "8100", "75.00%", "0.7128", False),
            ("E5", "Random Forest (200 est.)", "8237", "75.00%", "0.6998", False),
            ("E3", "SVM Lineal (Color)", "74", "60.42%", "0.5695", False),
            ("E4", "SVM Lineal (Textura)", "46", "62.50%", "0.5103", False),
            ("E2", "Random Forest (Forma)", "17", "27.08%", "0.2002", False),
        ]
        y_r = y_t - 32
        for exp, name, dim, acc, f1, is_winner in top_models:
            if is_winner:
                c.setFillColor(colors.HexColor("#dcfce7"))
                c.rect(515, y_r - 4, 390, 16, fill=True, stroke=False)
                c.setFillColor(C_EMERALD)
                c.setFont("Helvetica-Bold", 8.5)
            else:
                c.setFillColor(C_NAVY_DARK)
                c.setFont("Helvetica", 8)
            c.drawString(520, y_r, exp)
            c.drawString(560, y_r, name[:28])
            c.drawString(675, y_r, dim)
            c.drawString(730, y_r, acc)
            c.drawString(820, y_r, f1)
            y_r -= 18

        # Bloque de Análisis Crítico
        c.setFillColor(colors.HexColor("#f8fafc"))
        c.setStrokeColor(C_CARD_BORDER)
        c.roundRect(515, 68, 390, 85, 6, fill=True, stroke=True)

        c.setFillColor(C_BLUE_BRAND)
        c.setFont("Helvetica-Bold", 9.5)
        c.drawString(525, 137, "💡 Hallazgos Científicos:")
        c.setFillColor(C_NAVY_DARK)
        c.setFont("Helvetica", 8.5)
        c.drawString(525, 122, "1. SVM Lineal fue óptimo: En espacios de alta dimensión (8100 HOG),")
        c.drawString(525, 108, "   las clases son linealmente separables sin requerir kernels pesados.")
        c.drawString(525, 94, "2. Efecto de PCA: 195 componentes explicaron 95% de varianza, pero")
        c.drawString(525, 80, "   redujeron el F1 a 0.7587 al filtrar microgradientes de números y estrellas.")

        self.draw_slide_footer(5)
        c.showPage()

    # -------------------------------------------------------------
    # SLIDE 6: RESULTADOS EN TEST, ANÁLISIS DE ERRORES Y APP
    # -------------------------------------------------------------
    def render_slide_6(self):
        self.draw_slide_header(
            6,
            "Resultados y Aplicación",
            "Evaluación en Test Reservado, Análisis Metalúrgico y App Streamlit",
            "Exactitud del 88.68% con perfección en bimetálicas e integración en software interactivo"
        )

        c = self.c
        # 4 Tarjetas de Métricas en Test (Arriba)
        m_w = 210
        m_h = 75
        self.draw_metric_badge(40, 340, m_w, m_h, "88.68%", "Exactitud Global (Test)", "47 de 53 monedas correctas", C_EMERALD)
        self.draw_metric_badge(265, 340, m_w, m_h, "0.8877", "F1-Score Macro", "Sin sesgo de clase", C_BLUE_BRAND)
        self.draw_metric_badge(490, 340, m_w, m_h, "100%", "F1 Bimetálicas (1€ y 2€)", "0 errores en bimetálicas", C_GOLD_ACCENT)
        self.draw_metric_badge(715, 340, m_w, m_h, "1.36 ms", "Latencia por Moneda", "Tiempo real en CPU", C_BLUE_ACCENT)

        # Fila Media/Baja: Izquierda (Matriz de confusión), Derecha (Análisis de errores y App)
        self.draw_card(40, 56, 440, 270, "Matriz de Confusión Normalizada (Test)", "Diagnóstico pormenorizado sobre las 8 clases oficiales")
        img_cm = "outputs/figures/confusion_matrix_normalized.png"
        if os.path.exists(img_cm):
            c.drawImage(img_cm, 50, 62, width=420, height=225, preserveAspectRatio=True)

        self.draw_card(500, 56, 420, 270, "Análisis de Fallos y Despliegue en Streamlit", "Diagnóstico físico y software funcional para el usuario")

        # Diagnóstico de fallos
        c.setFillColor(C_NAVY_DARK)
        c.setFont("Helvetica-Bold", 9.5)
        c.drawString(516, 280, "🔍 Análisis de los 6 Errores (11.32% de fallo):")
        c.setFont("Helvetica", 8.5)
        c.drawString(516, 264, "• El 100% de los errores ocurrió de forma intra-aleación:")
        c.drawString(528, 250, "- Cobre (3 casos): 2c -> 1c, 5c -> 2c, 1c -> 2c.")
        c.drawString(528, 236, "- Oro Nórdico (3 casos): 20c -> 10c (2 veces) y 20c -> 50c (1 vez).")
        c.drawString(516, 222, "• Confusión inter-aleación: 0% (nunca se confunde cobre con oro).")
        c.drawString(516, 208, "• Causa física: Desgaste del relieve numeral y pérdida de referencia")
        c.drawString(528, 194, "  de escala métrica absoluta al normalizar a 128x128 píxeles.")

        # Recuadro de la Aplicación Streamlit
        c.setFillColor(colors.HexColor("#f0fdf4"))
        c.setStrokeColor(C_EMERALD)
        c.roundRect(516, 68, 388, 110, 6, fill=True, stroke=True)

        c.setFillColor(C_EMERALD)
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(528, 160, "🚀 Aplicación Gráfica Interactiva (Streamlit)")
        c.setFillColor(C_NAVY_DARK)
        c.setFont("Helvetica", 8.5)
        c.drawString(528, 142, "• Modo 1: Clasificación de moneda individual y visualización HOG.")
        c.drawString(528, 126, "• Modo 2: Detección multimoneda en escenas con Watershed, conteo")
        c.drawString(540, 112, "  automático de valor monetario (€) y descarga de imagen y CSV.")
        c.drawString(528, 96, "• Modo 3: Pestaña analítica embebida de métricas y experimentos.")
        c.setFillColor(C_BLUE_BRAND)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(528, 80, "Desplegado en GitHub con packages.txt y opencv-python-headless.")

        self.draw_slide_footer(6)
        c.showPage()

    def generate(self):
        print(f"[PRESENTACIÓN] Generando {self.filename}...")
        self.render_slide_1()
        self.render_slide_2()
        self.render_slide_3()
        self.render_slide_4()
        self.render_slide_5()
        self.render_slide_6()
        self.c.save()
        print(f"[PRESENTACIÓN] ¡Presentación guardada exitosamente en {self.filename} ({self.total_slides} diapositivas)!")


if __name__ == "__main__":
    builder = PresentationBuilder()
    builder.generate()

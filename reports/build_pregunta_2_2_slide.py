"""
Generador de Diapositiva Extra para la Pregunta 2.2:
'¿Por qué el Accuracy es una métrica engañosa en clases desbalanceadas? ¿Qué métrica usar y por qué?'
Formato: PDF 16:9 widescreen (960 x 540 pt) listo para proyectar.
"""

import os
from reportlab.pdfgen import canvas
from reportlab.lib import colors

SLIDE_W = 960
SLIDE_H = 540

# Paleta de colores consistente
C_BG_LIGHT = colors.HexColor("#f8fafc")
C_NAVY_DARK = colors.HexColor("#0f172a")
C_BLUE_BRAND = colors.HexColor("#1e3a8a")
C_BLUE_ACCENT = colors.HexColor("#0284c7")
C_GOLD_ACCENT = colors.HexColor("#d97706")
C_EMERALD = colors.HexColor("#059669")
C_RED_ACCENT = colors.HexColor("#dc2626")
C_CARD_BG = colors.HexColor("#ffffff")
C_CARD_BG_ALT = colors.HexColor("#f1f5f9")
C_CARD_BORDER = colors.HexColor("#cbd5e1")
C_TEXT_MUTED = colors.HexColor("#64748b")


def build_pregunta_2_2_slide(filename="reports/diapositiva_pregunta_2_2.pdf"):
    c = canvas.Canvas(filename, pagesize=(SLIDE_W, SLIDE_H))

    # 1. Fondo y Encabezado
    c.setFillColor(C_BG_LIGHT)
    c.rect(0, 0, SLIDE_W, SLIDE_H, fill=True, stroke=False)

    # Barra superior decorativa
    c.setFillColor(C_BLUE_BRAND)
    c.rect(0, SLIDE_H - 6, SLIDE_W, 6, fill=True, stroke=False)

    # Chip de Pregunta
    c.setFillColor(C_GOLD_ACCENT)
    c.roundRect(40, SLIDE_H - 46, 210, 20, 4, fill=True, stroke=False)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(48, SLIDE_H - 42, "PREGUNTA 2.2 | EVALUACIÓN")

    # Título principal
    c.setFillColor(C_NAVY_DARK)
    c.setFont("Helvetica-Bold", 19)
    c.drawString(40, SLIDE_H - 72, "¿Por qué el Accuracy es una Métrica Engañosa en Clases Desbalanceadas?")

    # Subtítulo descriptivo
    c.setFillColor(C_TEXT_MUTED)
    c.setFont("Helvetica", 11)
    c.drawString(40, SLIDE_H - 90, "La trampa de la exactitud global y la justificación matemática del F1-Score Macro")

    # Línea divisoria
    c.setStrokeColor(C_CARD_BORDER)
    c.setLineWidth(1)
    c.line(40, SLIDE_H - 100, SLIDE_W - 40, SLIDE_H - 100)

    # 2. Las 3 Columnas Comparativas
    col_w = 275
    col_h = 285
    y_col = 140

    # ------------------ COLUMNA 1: EL PROBLEMA (ACCURACY) ------------------
    c.setFillColor(C_CARD_BG)
    c.setStrokeColor(C_RED_ACCENT)
    c.setLineWidth(1.5)
    c.roundRect(40, y_col, col_w, col_h, 8, fill=True, stroke=True)

    # Encabezado tarjeta 1
    c.setFillColor(colors.HexColor("#fef2f2"))
    c.roundRect(40, y_col + col_h - 40, col_w, 40, 8, fill=True, stroke=False)
    c.setFillColor(C_RED_ACCENT)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(55, y_col + col_h - 25, "❌ 1. La Paradoja del Accuracy")

    c.setFillColor(C_NAVY_DARK)
    c.setFont("Helvetica", 9)
    c.drawString(55, y_col + col_h - 60, "Fórmula simple de aciertos globales:")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(75, y_col + col_h - 78, "Accuracy = (TP + TN) / Total")

    # Recuadro con el ejemplo de la trampa
    c.setFillColor(C_CARD_BG_ALT)
    c.roundRect(50, y_col + 45, col_w - 20, 115, 6, fill=True, stroke=False)

    c.setFillColor(C_NAVY_DARK)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(60, y_col + 140, "Ejemplo Intuitivo:")
    c.setFont("Helvetica", 8.5)
    c.drawString(60, y_col + 125, "• 95 monedas de 5 cents (95%)")
    c.drawString(60, y_col + 110, "• 5 monedas de 2 euros (5%)")

    c.setFont("Helvetica-Bold", 8.5)
    c.setFillColor(C_RED_ACCENT)
    c.drawString(60, y_col + 90, "Clasificador Trivial (predice siempre 5c):")
    c.setFont("Helvetica-Bold", 9)
    c.drawString(60, y_col + 74, "• Accuracy = 95.0%  (¡Aparenta éxito!)")
    c.drawString(60, y_col + 58, "• Recall en 2€ = 0.0%  (Inútil)")

    c.setFillColor(C_TEXT_MUTED)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(55, y_col + 20, "⚠️ Ciego total ante la clase minoritaria.")

    # ------------------ COLUMNA 2: LA SOLUCIÓN (F1-SCORE) ------------------
    c.setFillColor(C_CARD_BG)
    c.setStrokeColor(C_BLUE_BRAND)
    c.setLineWidth(1.5)
    c.roundRect(342, y_col, col_w, col_h, 8, fill=True, stroke=True)

    # Encabezado tarjeta 2
    c.setFillColor(colors.HexColor("#eff6ff"))
    c.roundRect(342, y_col + col_h - 40, col_w, 40, 8, fill=True, stroke=False)
    c.setFillColor(C_BLUE_BRAND)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(357, y_col + col_h - 25, "⚖️ 2. El Remedio: F1-Score")

    c.setFillColor(C_NAVY_DARK)
    c.setFont("Helvetica", 9)
    c.drawString(357, y_col + col_h - 60, "Media armónica entre dos fuerzas:")

    # Fórmulas de Precisión y Recall
    c.setFillColor(C_CARD_BG_ALT)
    c.roundRect(352, y_col + 80, col_w - 20, 95, 6, fill=True, stroke=False)

    c.setFillColor(C_BLUE_BRAND)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(362, y_col + 155, "Precisión (Certeza):")
    c.setFont("Helvetica", 8.5)
    c.setFillColor(C_NAVY_DARK)
    c.drawString(362, y_col + 140, "TP / (TP + FP)  -> ¿Es certero?")

    c.setFillColor(C_BLUE_BRAND)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(362, y_col + 120, "Recall (Cobertura):")
    c.setFont("Helvetica", 8.5)
    c.setFillColor(C_NAVY_DARK)
    c.drawString(362, y_col + 105, "TP / (TP + FN)  -> ¿Detecta todas?")

    c.setFont("Helvetica-Bold", 9.5)
    c.setFillColor(C_NAVY_DARK)
    c.drawString(357, y_col + 50, "F1 = 2 × (Prec × Rec) / (Prec + Rec)")
    c.setFont("Helvetica", 8.5)
    c.setFillColor(C_TEXT_MUTED)
    c.drawString(357, y_col + 32, "Si Recall = 0, F1 se derrumba a 0.")
    c.drawString(357, y_col + 18, "No permite camuflar omisiones.")

    # ------------------ COLUMNA 3: ¿POR QUÉ MACRO? ------------------
    c.setFillColor(C_CARD_BG)
    c.setStrokeColor(C_EMERALD)
    c.setLineWidth(1.5)
    c.roundRect(645, y_col, col_w, col_h, 8, fill=True, stroke=True)

    # Encabezado tarjeta 3
    c.setFillColor(colors.HexColor("#f0fdf4"))
    c.roundRect(645, y_col + col_h - 40, col_w, 40, 8, fill=True, stroke=False)
    c.setFillColor(C_EMERALD)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(660, y_col + col_h - 25, "🏆 3. ¿Por qué F1-Macro?")

    c.setFillColor(C_NAVY_DARK)
    c.setFont("Helvetica", 9)
    c.drawString(660, y_col + col_h - 60, "Promedio aritmético NO ponderado:")

    c.setFillColor(C_CARD_BG_ALT)
    c.roundRect(655, y_col + 90, col_w - 20, 85, 6, fill=True, stroke=False)

    c.setFillColor(C_EMERALD)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(670, y_col + 150, "F1-Macro = (1 / K) ∑ F1_k")

    c.setFillColor(C_NAVY_DARK)
    c.setFont("Helvetica", 8.5)
    c.drawString(670, y_col + 130, "• K = 8 denominaciones oficiales.")
    c.drawString(670, y_col + 112, "• Cada clase pesa exactamente 12.5%")
    c.drawString(670, y_col + 98, "  en la calificación global.")

    c.setFillColor(C_NAVY_DARK)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(660, y_col + 65, "Equidad absoluta en el dataset:")
    c.setFont("Helvetica", 8.5)
    c.drawString(660, y_col + 50, "50 cents (27 fotos) se evalúa con el")
    c.drawString(660, y_col + 36, "mismo rigor que 5 cents (56 fotos).")
    c.setFillColor(C_EMERALD)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(660, y_col + 18, "Protección total ante el desbalance.")

    # 3. Banner Inferior: Conexión con los Resultados Reales del Proyecto
    c.setFillColor(colors.HexColor("#eff6ff"))
    c.setStrokeColor(C_BLUE_ACCENT)
    c.setLineWidth(1)
    c.roundRect(40, 48, SLIDE_W - 80, 72, 6, fill=True, stroke=True)

    c.setFillColor(C_BLUE_BRAND)
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(55, 100, "🪙 Resultados Empíricos en EuroCoinClassifier (Conjunto de Prueba Reservado, 53 monedas):")

    c.setFillColor(C_NAVY_DARK)
    c.setFont("Helvetica", 9)
    c.drawString(55, 84, "• Accuracy Global: 88.68%  |  F1-Score Macro: 0.8877  |  F1-Score Ponderado: 0.8853")
    c.drawString(55, 68, "• Ambos valores son prácticamente idénticos porque nuestro modelo ganador (SVM Lineal) aprendió a clasificar con alto balance")
    c.drawString(55, 54, "  todas las clases, alcanzando precisión y recall perfectos del 100% (F1 = 1.000) en monedas bimetálicas de 1 y 2 euros.")

    # Pie institucional
    c.setFillColor(C_TEXT_MUTED)
    c.setFont("Helvetica", 8.5)
    c.drawString(40, 20, "Universidad de Antioquia | PDI 2026 | Sustentación Parte 2 — Pregunta 2.2")
    c.drawRightString(SLIDE_W - 40, 20, "EuroCoinClassifier | reports/diapositiva_pregunta_2_2.pdf")

    c.save()
    print(f"[OK] Diapositiva generada en: {filename}")


if __name__ == "__main__":
    build_pregunta_2_2_slide()

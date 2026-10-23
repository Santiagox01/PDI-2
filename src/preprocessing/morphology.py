"""
Operaciones morfologicas matematicas para limpieza, union y refinamiento de mascaras binarias.
"""

import cv2
import numpy as np


def get_structuring_element(shape: str = "ellipse", ksize: int = 5) -> np.ndarray:
    """Crea un elemento estructurante circular/eliptico o rectangular."""
    if shape == "ellipse":
        morph_shape = cv2.MORPH_ELLIPSE
    elif shape == "cross":
        morph_shape = cv2.MORPH_CROSS
    else:
        morph_shape = cv2.MORPH_RECT
    return cv2.getStructuringElement(morph_shape, (ksize, ksize))


def clean_mask(mask: np.ndarray, open_ksize: int = 3, close_ksize: int = 7) -> np.ndarray:
    """
    Aplica apertura morfologica para remover ruido espurio exterior
    seguida de clausura morfologica para cerrar poros e inscripciones internas.
    """
    kernel_open = get_structuring_element("ellipse", open_ksize)
    kernel_close = get_structuring_element("ellipse", close_ksize)

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    return closed


def fill_holes(binary_mask: np.ndarray) -> np.ndarray:
    """
    Rellena cavidades y relieves internos dentro de los cuerpos de las monedas
    mediante propagacion floodFill.
    """
    im_floodfill = binary_mask.copy()
    h, w = binary_mask.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    # Iniciar relleno desde esquina (fondo)
    cv2.floodFill(im_floodfill, flood_mask, (0, 0), 255)

    # Invertir la imagen inundada
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combinar con la mascara original para obtener el objeto relleno
    filled = binary_mask | im_floodfill_inv
    return filled

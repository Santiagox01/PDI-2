"""
Modulo de filtrado espacial, transformaciones de espacio de color y correccion de iluminacion.
"""

import cv2
import numpy as np


def apply_gaussian_filter(image: np.ndarray, ksize: tuple = (5, 5), sigma: float = 1.0) -> np.ndarray:
    """Aplica suavizado Gaussiano para reduccion de ruido de alta frecuencia."""
    return cv2.GaussianBlur(image, ksize, sigmaX=sigma, sigmaY=sigma)


def apply_bilateral_filter(image: np.ndarray, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
    """Aplica filtrado bilateral preservando bordes definidos de las monedas."""
    return cv2.bilateralFilter(image, d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def convert_color_spaces(image_bgr: np.ndarray) -> dict:
    """Convierte una imagen BGR a RGB, Grises, HSV y CIELAB."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    return {
        "bgr": image_bgr,
        "rgb": rgb,
        "gray": gray,
        "hsv": hsv,
        "lab": lab
    }


def apply_clahe_color(image_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    Aplica CLAHE al canal de luminancia L en el espacio CIELAB
    para corregir sombras y contrastes heterogeneos sin alterar el balance cromatico.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def apply_clahe_gray(gray_img: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> np.ndarray:
    """Aplica CLAHE sobre una imagen monocromatica."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray_img)


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Aplica correccion no lineal de gamma para balancear sobre o subexposicion."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

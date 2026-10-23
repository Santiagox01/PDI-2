"""
Modulo de carga, lectura robusta y normalizacion geometrica de imagenes.
"""

import os
import cv2
import numpy as np
from PIL import Image, ExifTags


def load_image(image_path: str) -> np.ndarray:
    """
    Carga de forma robusta una imagen desde disco en formato BGR (OpenCV)
    corrigiendo la orientacion EXIF si corresponde.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontro el archivo de imagen: {image_path}")

    # Corregir orientacion EXIF con Pillow si esta disponible
    try:
        pil_img = Image.open(image_path)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        
        exif = pil_img._getexif()
        if exif is not None and orientation in exif:
            orientation_val = exif[orientation]
            if orientation_val == 3:
                pil_img = pil_img.rotate(180, expand=True)
            elif orientation_val == 6:
                pil_img = pil_img.rotate(270, expand=True)
            elif orientation_val == 8:
                pil_img = pil_img.rotate(90, expand=True)
        
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img
    except Exception:
        # Fallback a cv2.imread directo
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Fallo al decodificar la imagen: {image_path}")
        return img


def resize_maintaining_aspect_ratio(image: np.ndarray, target_size: tuple = (128, 128),
                                     interpolation=cv2.INTER_AREA, pad_color=(0, 0, 0)) -> tuple:
    """
    Redimensiona una imagen conservando estrictamente su relacion de aspecto,
    agregando un padding simetrico si es necesario.
    Retorna (imagen_redimensionada, factor_escala, (pad_top, pad_bottom, pad_left, pad_right)).
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_w = target_w - new_w
    pad_h = target_h - new_h

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if len(image.shape) == 3:
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=pad_color)
    else:
        padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=pad_color[0] if isinstance(pad_color, tuple) else pad_color)

    return padded, scale, (pad_top, pad_bottom, pad_left, pad_right)

"""
Pruebas para verificar la ausencia total de fuga de informacion (Data Leakage)
y el balance de clases en las particiones.
"""

import os
import pandas as pd
import pytest


def test_splits_exist_and_counts():
    train_path = "data/splits/train.csv"
    val_path = "data/splits/validation.csv"
    test_path = "data/splits/test.csv"

    assert os.path.exists(train_path), "Falta train.csv"
    assert os.path.exists(val_path), "Falta validation.csv"
    assert os.path.exists(test_path), "Falta test.csv"

    df_tr = pd.read_csv(train_path)
    df_va = pd.read_csv(val_path)
    df_te = pd.read_csv(test_path)

    total = len(df_tr) + len(df_va) + len(df_te)
    assert total == 336, f"La suma de particiones ({total}) no coincide con 336"

    # Verificar proporciones aproximadas (70 / 15 / 15)
    assert 0.65 <= (len(df_tr) / total) <= 0.75
    assert 0.10 <= (len(df_va) / total) <= 0.20
    assert 0.10 <= (len(df_te) / total) <= 0.20


def test_zero_data_leakage():
    df_tr = pd.read_csv("data/splits/train.csv")
    df_va = pd.read_csv("data/splits/validation.csv")
    df_te = pd.read_csv("data/splits/test.csv")

    tr_imgs = set(df_tr["original_image_id"])
    va_imgs = set(df_va["original_image_id"])
    te_imgs = set(df_te["original_image_id"])

    # Verificacion rigurosa: Interseccion de conjuntos debe ser vacia
    assert len(tr_imgs.intersection(va_imgs)) == 0, "Fuga detectada: Imagenes compartidas entre Train y Val"
    assert len(tr_imgs.intersection(te_imgs)) == 0, "Fuga detectada: Imagenes compartidas entre Train y Test"
    assert len(va_imgs.intersection(te_imgs)) == 0, "Fuga detectada: Imagenes compartidas entre Val y Test"


def test_all_classes_present_in_all_splits():
    df_tr = pd.read_csv("data/splits/train.csv")
    df_va = pd.read_csv("data/splits/validation.csv")
    df_te = pd.read_csv("data/splits/test.csv")

    assert df_tr["class_id"].nunique() == 8, "No todas las 8 clases estan en train"
    assert df_va["class_id"].nunique() == 8, "No todas las 8 clases estan en val"
    assert df_te["class_id"].nunique() == 8, "No todas las 8 clases estan en test"

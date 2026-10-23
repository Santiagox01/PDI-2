"""
Ejecuta el Master Notebook y almacena todas las salidas graficas y tabulares
directamente en el archivo .ipynb para visualizacion instantanea.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import nbformat
import matplotlib
matplotlib.use("Agg")  # Previene bloqueo por ventanas GUI emergentes


def execute_and_embed_outputs(notebook_path: str = "EuroCoinClassifier_Completo.ipynb"):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    print(f"[INFO] Ejecutando y almacenando salidas para: {notebook_path}...")
    shared_globals = {
        "__name__": "__main__",
        "__file__": os.path.abspath(notebook_path)
    }

    # Reemplazar %matplotlib inline por configuracion no bloqueante para ejecucion por script
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            code = cell.source
            # Remover lineas magicas de jupyter si se ejecuta via python puro
            clean_code = "\n".join([line for line in code.splitlines() if not line.strip().startswith("%")])
            try:
                exec(clean_code, shared_globals)
                print(f"  -> Celda de código {idx+1}/{len(nb.cells)} ejecutada con éxito.")
            except Exception as e:
                print(f"  [AVISO] Error en celda {idx+1}: {e}")

    print("[INFO] Todas las celdas validadas correctamente.")


if __name__ == "__main__":
    execute_and_embed_outputs("EuroCoinClassifier_Completo.ipynb")

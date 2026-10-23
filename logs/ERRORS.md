# Registro de Errores y Mitigaciones (ERRORS.md)

| ID | Fecha/Hora | Componente | Error Encontrado | Causa Raíz | Solución Aplicada | Estado |
|---|---|---|---|---|---|---|
| ERR-001 | 2026-10-16 02:00 | PowerShell redirection | Codificación UTF-16LE en `requirements.txt` | Redirección `>` en PowerShell utiliza UTF-16LE por defecto | Generación controlada mediante script Python con encoding explícito UTF-8 | RESUELTO |
| ERR-002 | 2026-10-16 02:04 | Scripts de ejecución directa | `ModuleNotFoundError: No module named 'src'` | En Windows al invocar `python src/dir/script.py`, Python coloca `src/dir` en sys.path | Se incorporó `sys.path.insert(0, os.path.abspath('.'))` en la cabecera de los scripts para ejecución dual (módulo/script) | RESUELTO |
| ERR-003 | 2026-10-16 02:21 | Model Comparison / Matplotlib | `Tcl_AsyncDelete: async handler deleted by wrong thread` | Matplotlib inicializaba Tkinter en el hilo principal y colisionaba con subprocesos de joblib | Configuración de backend headless explícito `matplotlib.use('Agg')` y serialización controlada `n_jobs=1` | RESUELTO |
| ERR-004 | 2026-10-16 14:20 | Pruebas Automatizadas / Pytest | `ModuleNotFoundError: No module named 'src'` durante recolección | Pytest no incluía la raíz de trabajo en `sys.path` por defecto | Creación de `pytest.ini` con `pythonpath = .` y `tests/conftest.py` | RESUELTO |

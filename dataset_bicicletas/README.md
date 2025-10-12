# Dataset Bicicletas — Guía de Uso

Este módulo organiza el flujo de trabajo para limpiar datos, seleccionar atributos (features) y entrenar modelos (baseline con scikit‑learn, econométricos con statsmodels y un modelo simple en PyTorch).

## Requisitos

- Python 3.10+ (probado con 3.11)
- Instalar dependencias (desde la raíz del repo o dentro de `dataset_bicicletas`):
  - `pip install -r dataset_bicicletas/requirements.txt`
  - Nota: `torch` es opcional. Si tu entorno GPU requiere una build específica, instala PyTorch según su guía oficial para tu CUDA/cuDNN.

Recomendación: ejecutar los comandos situándote dentro de la carpeta `dataset_bicicletas` para rutas simples.

## Flujo Rápido

1) Limpiar CSV crudo y guardar el procesado
- `python mains/run_cleaning.py --csv-in data/raw/all_data.csv --csv-out data/processed/dataset_bicicletas_clean.csv`

2) Explorar columnas y seleccionar features
- Interactivo con guardado y comando sugerido:
  - `python mains/run_features.py --csv data/processed/dataset_bicicletas_clean.csv --interactive --save utils/feature_sets/exp1.json --print-cmd --no-clean`
- No interactivo (add/remove + guardado):
  - `python mains/run_features.py --csv data/processed/dataset_bicicletas_clean.csv --add hr sdnn --remove mean_scr --save utils/feature_sets/exp2.json --print-cmd --no-clean`

3) Entrenar modelos con la selección (baseline siempre; MNLogit/Torch opcionales)
- Baseline con CSV procesado y sin volver a limpiar:
  - `python mains/run_training.py --csv data/processed/dataset_bicicletas_clean.csv --no-clean --features-file utils/feature_sets/exp1.json --prefix exp1`
- Agregando MNLogit:
  - `python mains/run_training.py --csv data/processed/dataset_bicicletas_clean.csv --no-clean --features-file utils/feature_sets/exp1.json --mnlogit --prefix exp1`
- Agregando Torch:
  - `python mains/run_training.py --csv data/processed/dataset_bicicletas_clean.csv --no-clean --features-file utils/feature_sets/exp1.json --torch --prefix exp1`

## Carpetas y Rol

- `data/`
  - `raw/`: CSVs crudos de entrada (ej. `all_data.csv`).
  - `processed/`: CSVs limpios (ej. `dataset_bicicletas_clean.csv`).
- `mains/`: Ejecutables CLI para el flujo end‑to‑end.
  - `run_cleaning.py`: limpia un CSV de entrada y guarda el procesado.
  - `run_features.py`: lista columnas, permite armar una selección (interactiva o por flags) y guardar a archivo.
  - `run_training.py`: entrena baseline y, opcionalmente, MNLogit y/o Torch; guarda resultados.
- `src/`: Código modular reutilizable.
  - `data_cleaning/cleaning.py`: funciones de limpieza (timestamps, binarias, categorías, `fillna(0)`).
  - `data_loading/load.py`: lectura de CSV (ruta obligatoria, falla explícitamente si no existe).
  - `features/prepare.py`: `features_labels`, split train/test, `build_preprocessor`, codificación de labels.
  - `models/`
    - `baseline.py`: pipeline sklearn (escala + one‑hot + `LogisticRegression`).
    - `econ.py`: `MNLogit` (statsmodels) sobre features preprocesados (y labels codificados).
    - `embeddings.py`: modelo lineal simple estilo “embedding” con entrenamiento básico en PyTorch.
- `utils/`
  - `features.py`: utilidades para columnas y features (paginado, add/remove con preservación de orden, hash corto, guardar/cargar listas de features a `.json`/`.txt`).
  - `results_io.py`: utilidades para guardar resultados (`.txt`, probabilidades a `.csv`) y modelos (pickle `.pkl`, `state_dict` de PyTorch `.pt`).

## Detalles de los Mains

### `mains/run_cleaning.py`
- Uso: `python mains/run_cleaning.py --csv-in <ruta_in> --csv-out <ruta_out>`
- Lee el CSV de entrada, aplica `limpiar_dataset` y guarda el resultado.

### `mains/run_features.py`
- Uso básico: `python mains/run_features.py --csv <ruta_csv> [--interactive] [--add ...] [--remove ...] [--save <archivo>] [--print-cmd]`
- Imprime columnas paginadas (por defecto 10 por bloque). `--page-size` cambia el tamaño.
- Modo interactivo: comandos `add col1,col2`, `remove col3`, `show`, `done`.
- `--save` guarda la selección en `.json` o `.txt` (preferible `.json`).
- `--print-cmd` imprime un comando sugerido para `run_training` (prioriza `--features-file` si guardaste la selección).

### `mains/run_training.py`
- Uso típico con CSV ya procesado: `--no-clean` para omitir limpieza.
- Flags principales:
  - `--csv`: ruta al CSV (requerido).
  - `--features` o `--features-file`: lista de features (inline o desde archivo).
  - `--label`: columna objetivo (por defecto `action`).
  - `--no-clean`: usa el CSV tal cual (recomendado con `data/processed/...`).
  - `--mnlogit` y `--torch`: activan los modelos opcionales.
  - `--prefix`: prefijo manual para artefactos en `results/`. Si lo omites, se autogenera con timestamp + cantidad de features + hash corto.
- Artefactos que se guardan en `results/`:
  - `{prefix}_config.json`: configuración de la corrida (csv, features, flags).
  - `{prefix}_baseline_report.txt`: `classification_report` del test.
  - `{prefix}_baseline_proba.csv`: probabilidades por clase en el test.
  - `{prefix}_baseline_model.pkl`: pipeline sklearn (pickle con joblib).
  - Si `--mnlogit`: `{prefix}_mnlogit_summary.txt`.
  - Si `--torch`: `{prefix}_torch_model.pt` y `{prefix}_torch_history.csv` (pérdida y accuracy por época).

## Ejemplos Adicionales

- Entrenar solo baseline con selección inline:
  - `python mains/run_training.py --csv data/processed/dataset_bicicletas_clean.csv --no-clean --features mean_scl hr sdnn`
- Listar columnas paginadas sin interacción:
  - `python mains/run_features.py --csv data/processed/dataset_bicicletas_clean.csv --page-size 20`
- Construir selección a partir de la totalidad removiendo algunas y guardar:
  - `python mains/run_features.py --csv data/processed/dataset_bicicletas_clean.csv --remove timestamp participant --save utils/feature_sets/exp_remove_time.json --print-cmd --no-clean`

## Notas y Buenas Prácticas

- Ejecuta los mains desde la carpeta `dataset_bicicletas` para que las rutas relativas funcionen tal cual en los ejemplos.
- Para CSV procesado, usa `--no-clean` en `run_training` para evitar reprocesar.
- Si instalas PyTorch con CUDA específica, puedes omitir `torch` del `requirements.txt` y hacer la instalación manual según tu GPU.
- Las utilidades de features mantienen el orden original de las columnas y generan un hash corto útil para etiquetar experimentos.


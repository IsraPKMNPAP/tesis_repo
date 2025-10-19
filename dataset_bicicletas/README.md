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

### Ejecutar como módulo (recomendado)

Para asegurar la resolución correcta de imports de `src/` y `utils/`, puedes ejecutar los scripts como módulos desde la carpeta `dataset_bicicletas`:

- `python -m mains.run_cleaning --csv-in data/raw/all_data.csv --csv-out data/processed/dataset_bicicletas_clean.csv`
- `python -m mains.run_features --csv data/processed/dataset_bicicletas_clean.csv --interactive --save utils/feature_sets/exp1.json --print-cmd --no-clean`
- `python -m mains.run_training --csv data/processed/dataset_bicicletas_clean.csv --no-clean --features-file utils/feature_sets/exp1.json --prefix exp1`

## Modelos de Video por Ventanas (CNN+LSTM)

Se incluye un flujo para entrenar modelos CNN+LSTM sobre ventanas de video ya convertidas a tensores de PyTorch por ventana.

- Scripts:
  - `mains/run_link_video_tensors.py`: enlaza (linkea) las rutas reales en GPU de los `window_*.pt` al pickle crudo y guarda un pickle procesado con una nueva columna (`gpu_tensor_path`).
  - `mains/run_video_training.py`: entrena el modelo CNN+LSTM leyendo el pickle procesado y utilizando directamente `gpu_tensor_path`.
- Entrada: un pickle con DataFrame (p. ej. `X_proc_final.pkl`) con columnas
  `participant, timestamp, window, paths, paths_list, paths_list_fixed, delta_t, delta_t_round, is_imputed, session_id, action`.
- Por ventana, el archivo `.pt` correspondiente (en el GPU) debe contener un diccionario como:
  `{ "frames": Tensor[T,C,H,W], "label": int, "participant": str, "timestamp": str, "window_id": int }`.

Paso 1 — Link de rutas a GPU

- Ejecuta el enlace desde el pickle crudo al procesado con la columna `gpu_tensor_path`:
  - `python -m mains.run_link_video_tensors --pickle-in "~/projects/tesis_repo/dataset_bicicletas/data/raw/X_proc_final.pkl" --linux-root "/mnt/otra_particion/home/israel_gpu_data/video_tensors" --out-pickle "~/projects/tesis_repo/dataset_bicicletas/data/processed/X_proc_final_linked.pkl"`

- El script escanea `--linux-root` para `window_*.pt` y asocia por `timestamp` si está en los `.pt` (o por orden como respaldo). Añade la columna `gpu_tensor_path` y guarda el pickle en `data/processed`.

Paso 2 — Entrenamiento con el pickle procesado

- Entrena usando la columna `gpu_tensor_path` (sin volver a enlazar rutas):
  - `python -m mains.run_video_training --pickle "~/projects/tesis_repo/dataset_bicicletas/data/processed/X_proc_final_linked.pkl" --arkoudi`

Opciones alternativas de mapeo de rutas (flujo combinado)

- Reemplazo de prefijo (OneDrive → Linux): usar `--onedrive-prefix` y `--linux-root` si el DataFrame trae rutas Windows y quieres reemplazarlas por el root Linux donde están los `.pt`.
- Mapeo robusto por `timestamp` u orden a `window_i.pt`: pasar `--linux-root` y `--map-by-timestamp` para escanear los `window_*.pt` en el GPU y asociarlos por timestamp (si existe en los `.pt`) o por orden cronológico como respaldo.

Uso típico (desde `dataset_bicicletas`)

- Flujo en dos pasos (recomendado y único):
  1. `python -m mains.run_link_video_tensors --pickle-in "~/projects/tesis_repo/dataset_bicicletas/data/raw/X_proc_final.pkl" --linux-root "/mnt/otra_particion/home/israel_gpu_data/video_tensors" --out-pickle "~/projects/tesis_repo/dataset_bicicletas/data/processed/X_proc_final_linked.pkl"`
  2. `python -m mains.run_video_training --pickle "~/projects/tesis_repo/dataset_bicicletas/data/processed/X_proc_final_linked.pkl" --arkoudi`



Parámetros clave

- `--label-col` (por defecto `action`), `--timestamp-col` (por defecto `timestamp`), `--window-id-col` (por defecto `window`).
- Hiperparámetros: `--cnn-emb`, `--lstm-hidden`, `--lstm-layers`, `--bidirectional`, `--batch-size`, `--epochs`, `--lr`, `--weight-decay`.
- Arkoudi: `--arkoudi` activa la cabeza de embeddings de clase (interpretable); `--arkoudi-no-norm` desactiva normalización L2.
- `--num-classes` se infiere del DataFrame si no se pasa.

Artefactos en `results/`

- `{prefix}_cnn_lstm.pt` (pesos), `{prefix}_history.csv` (entrenamiento), `{prefix}_val_report.txt` y `{prefix}_val_proba.csv` (validación).
- `{prefix}_embeddings.csv` (embeddings por ventana con `emb_*`, `label`, `timestamp`, `window_id`, `participant`).
- `{prefix}_mnlogit_embeddings_summary.txt` (MNLogit sobre embeddings; si hay labels disponibles).
- `{prefix}_config.json` (configuración de la corrida).

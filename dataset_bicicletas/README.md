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
- Artefactos que se guardan en `results/` (nuevo esquema de nombres):
  - Formato: `Modelo-Artifact-Hash.ext`.
  - Para baseline (LogReg):
    - `LogReg-config-<hash>.json`, `LogReg-eval_report-<hash>.txt`, `LogReg-eval_proba-<hash>.csv`, `LogReg-model-<hash>.pkl`.
  - Si `--mnlogit`: `MNLogit-summary-<hash>.txt`.
  - Si `--torch`: `TorchEmbed-model-<hash>.pt` y `TorchEmbed-history-<hash>.csv`.
  - Cada corrida agrega una entrada en `results/run_index.txt` con el `hash`, `model`, comando y configuración.

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
  2. `python -m mains.run_video_training --pickle "~/projects/tesis_repo/dataset_bicicletas/data/processed/X_proc_final_linked.pkl" --label-col label_proc --prefer-df-label`



Parámetros clave

- `--label-col` (por defecto `action`), `--timestamp-col` (por defecto `timestamp`), `--window-id-col` (por defecto `window`).
- `--prefer-df-label` (por defecto activo) para usar siempre el label del DataFrame (p. ej. `label_proc`) e ignorar el `label` dentro de los `.pt`.
- `--no-default-class-map` desactiva el mapeo string→int por defecto: `{'accelerate':0,'brake':1,'decelerate':2,'maintain speed':3,'wait':4}`. `--class-map-json` permite pasar un JSON con tu propio mapping.
- `--class-weighted` usa pesos de clase inversos a la frecuencia durante el entrenamiento (CrossEntropy con `weight`).
- Hiperparámetros: `--cnn-emb`, `--lstm-hidden`, `--lstm-layers`, `--bidirectional`, `--batch-size`, `--epochs`, `--lr`, `--weight-decay`.
- Arkoudi: `--arkoudi` activa la cabeza de embeddings de clase (interpretable); `--arkoudi-no-norm` desactiva normalización L2.
- `--num-classes` se infiere del DataFrame si no se pasa.

Artefactos en `results/` (nuevo esquema)

- Para CNN+LSTM (videos):
  - `CNNLSTM-model-<hash>.pt`, `CNNLSTM-history-<hash>.csv`, `CNNLSTM-eval_report-<hash>.txt`, `CNNLSTM-eval_proba-<hash>.csv`.
  - `CNNLSTM-embeddings-<hash>.csv` (embeddings por ventana con `emb_*`, `label`, `timestamp`, `window_id`, `participant`).
  - `CNNLSTM-mnlogit_summary-<hash>.txt` (MNLogit sobre embeddings; si hay labels disponibles).
- `CNNLSTM-config-<hash>.json` (configuración de la corrida) y una entrada en `results/run_index.txt`.

## Backbones Visuales + LSTM (CLIP/ViT)

Entrena modelos que extraen embeddings por frame con CLIP o ViT y modelan la secuencia con LSTM.

- Script: `mains/run_video_backbones.py`
- Entrada: pickle procesado (ver paso de link) con `gpu_tensor_path` y labels (`action` o `action_proc`).

Uso típico:
- CLIP finetune (congelado por defecto), LSTM y Arkoudi, coseno con scheduler coseno:
  - `python -m mains.run_video_backbones --pickle "~/projects/tesis_repo/dataset_bicicletas/data/processed/X_proc_final_linked.pkl" --label-col action_proc --backbone clip --backbone-name ViT-B-16 --arkoudi --scheduler cosine --epochs 30 --lr 1e-4`
- ViT (torchvision vit_b_16), LSTM bidireccional con pesos de clase:
  - `python -m mains.run_video_backbones --pickle "~/projects/.../X_proc_final_linked.pkl" --label-col action_proc --backbone vit --backbone-name vit_b_16 --bidirectional --class-weighted --epochs 30 --lr 1e-4`

Flags principales:
- `--backbone {vit,clip}` y `--backbone-name` (vit_b_16 o ViT-B-16 para CLIP).
- `--backbone-trainable` para permitir fine-tuning; por defecto congelado.
- `--backbone-unfreeze-last N` para descongelar únicamente los últimos N bloques del transformador (recomendado 1–2 al comenzar).
- `--target-size` (por defecto 224) para reescalar frames antes del backbone.
- `--lstm-hidden`, `--lstm-layers`, `--bidirectional`, `--dropout` para regularización en LSTM/salida.
- `--arkoudi` / `--arkoudi-no-norm`.
- Entrenamiento: `--batch-size`, `--epochs`, `--lr` (por defecto 1e-4), `--weight-decay`, `--class-weighted`, `--scheduler {step,cosine,plateau}` y parámetros asociados.
- Debug: `--debug-eval` imprime balance de clases y stats de logits.

Requisitos adicionales:
- Se añadieron `torchvision`, `timm` y `open-clip-torch` al `requirements.txt` para ViT/CLIP. Instala builds acordes a tu CUDA según la guía oficial de PyTorch.

## Verificación de Insumos (DataFrame y Tensors)

Para inspeccionar rápidamente que el pickle y los `.pt` tengan lo esperado, usa el verificador:

- `python -m utils.verify --pickle "~/projects/tesis_repo/dataset_bicicletas/data/processed/X_proc_final_linked.pkl" --path-col gpu_tensor_path --label-col action_proc --sample-pt 5 --random --show-dtypes`

Qué muestra:
- Columnas, shape y dtypes del DataFrame.
- Head de columnas clave (`participant`, `timestamp`, `window`, `gpu_tensor_path`, `action_proc`, etc.).
- Conteos de labels (soporta strings o ints).
- Valida existencia de archivos y “probea” algunos `.pt` mostrando: tipo (dict/Tensor), claves, shape de `frames`, `label`, `timestamp`, `window_id`, `participant`.
Recomendaciones de fine-tuning y LR

- Backbones congelados (solo LSTM/cabeza): `--lr` en 1e-4 a 3e-4 suele funcionar bien.
- Descongelando últimos 1–2 bloques (`--backbone-unfreeze-last 1` o `2`): usa LR más bajo, p. ej. 5e-5 a 1e-5, y `--scheduler cosine` o `plateau`.
- Regularización: `--dropout` 0.1–0.3, `--class-weighted` para desbalance, `--weight-decay` 1e-4–1e-3.
- Estabilidad: considera `--grad-clip 1.0` (ya activado por defecto en backbones) y `--label-smoothing 0.05` (activado en backbones).

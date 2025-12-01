# Dataset Neuma

Armazón mínimo para replicar el pipeline de la tesis sobre el nuevo dataset **neuma**.

- Datos: en GPU bajo `/mnt/otra_particion/home/israel_gpu_data/dataset_neuma/` (subcarpetas `processed/` y luego `raw/`). La carpeta `data/` local queda para muestras o pruebas rápidas.
- Código: `mains/` para scripts ejecutables (`python dataset_neuma/mains/<script>.py --args`), `src/` para módulos (`data_loading/`, `models/`, `features/`), `utils/` para utilitarios compartidos, `configs/` para YAML/JSON de config y `results/` para salidas y logs.
- Script inicial: `dataset_neuma/mains/inspect_processed_csv.py` imprime columnas, forma y dtypes, y convierte a `category` las columnas con <50 valores únicos.

Estructura propuesta:

```
dataset_neuma/
  data/
    processed/        # enlaza o monta datos en GPU si se requiere localmente
    raw/
  mains/
    inspect_processed_csv.py
  src/
    data_loading/
    models/
    features/
  utils/
  configs/
  results/
  notebooks/
  requirements.txt
```

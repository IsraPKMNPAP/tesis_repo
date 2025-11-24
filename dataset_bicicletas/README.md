# Dataset Bicicletas – Guia de Uso (Comandos Primero)

Esta guia lista primero los comandos de ejecucion, con opciones y defaults, para reproducir desde los modelos unimodales hasta el pipeline multimodal. Debajo quedan detalles y notas.

Recomendacion: situarse dentro de `dataset_bicicletas` antes de ejecutar.

## Rutas y Atajos
- Tabular limpio: `data/processed/dataset_bicicletas_clean.csv`
- Pickle video linkeado: `data/processed/X_proc_final_linked.pkl`
- Join multimodal: `data/processed/multimodal_join.pkl`
- Conjunto de features (tabular): `utils/feature_sets/exp1.json`
- Etiqueta recomendada: `action_proc`

## Comandos Clave (Modelos y Pipelines)

- Limpieza tabular basica
  - `python mains/run_cleaning.py --csv-in data/raw/all_data.csv --csv-out data/processed/dataset_bicicletas_clean.csv`

- Seleccion de features (tabular)
  - `python mains/run_features.py --csv data/processed/dataset_bicicletas_clean.csv --base  --add  --remove  --page-size 10 --interactive --save utils/feature_sets/exp1.json --print-cmd --label action --no-clean --mnlogit --torch`

- Baseline (tabular, sklearn) + opcionales (MNLogit/Torch)
  - `python mains/run_training.py --csv data/processed/dataset_bicicletas_clean.csv --features-file utils/feature_sets/exp1.json --label action --mnlogit --torch --no-clean --prefix baseline_exp`
  - Flags y defaults: `--features` (lista) | `--features-file` | `--label action` | `--mnlogit` | `--torch` | `--no-clean` | `--prefix None`

- Linkeo de tensores de video a pickle (GPU)
  - `python mains/run_link_video_tensors.py --pickle-in data/raw/X_proc_final.pkl --linux-root /mnt/otra_particion/home/israel_gpu_data/video_tensors --timestamp-col timestamp --out-pickle data/processed/X_proc_final_linked.pkl --out-column gpu_tensor_path`

- Verificacion de pickle de video
  - `python utils/verify.py --pickle data/processed/X_proc_final_linked.pkl --path-col gpu_tensor_path --label-col action --timestamp-col timestamp --window-id-col window --head 5 --show-dtypes --sample-pt 3 --random`

- Alineacion tabular → ancla de video (relleno huecos + sesiones + paths)
  - `python src/data_cleaning/align_to_video_anchor.py --csv-in data/processed/dataset_bicicletas_clean.csv --pkl-ref data/processed/X_proc_final_linked.pkl --csv-out data/processed/dataset_bicicletas_clean_aligned.csv --timestamp-col timestamp --participant-col participant --session-id-col session_id --imputed-col is_imputed --expected-step-seconds 5 --raw-csv data/raw/all_data.csv --paths-col paths --max-gap 120`

- Revision post-alineacion
  - `python mains/run_review_alignment.py --csv-aligned data/processed/dataset_bicicletas_clean_aligned.csv --pkl-ref data/processed/X_proc_final_linked.pkl --timestamp-col timestamp --participant-col participant --session-id-col session_id --imputed-col is_imputed --expected-step-seconds 5`

- Join multimodal por timestamp (para VAE)
  - `python mains/run_join_modalities.py --csv-in data/processed/dataset_bicicletas_clean_aligned.csv --pkl-in data/processed/X_proc_final_linked.pkl --timestamp-col timestamp --out data/processed/multimodal_join.pkl --how one-to-one --suffixes _csv _vid`

- Gestión de audios crudos (renombrado + validación)
  - Renombrar archivos `Copia de final_audio_PXX.wav` a `raw_audio_PXX.wav`:  
    `python utils/rename_audio_files.py --audio-root /mnt/otra_particion/home/israel_gpu_data/audio_data_raw/audio_participantes_validos --dry-run`
  - Validar cobertura temporal de audios vs. `X_vid_aud.pkl`:  
    `python src/data_cleaning/validate_audio_segments.py --pickle /mnt/otra_particion/home/israel_gpu_data/audio_data_raw/X_vid_aud.pkl --audio-root /mnt/otra_particion/home/israel_gpu_data/audio_data_raw/audio_participantes_validos --out-csv data/processed/audio_segments_validation.csv --out-json data/processed/audio_segments_validation.json --fail-on-error`
  - Parchear desincronías (pad de audios y session_id desde video):  
    `python src/data_cleaning/patch_audio_mismatches.py --audio-pickle /mnt/otra_particion/home/israel_gpu_data/audio_data_raw/X_vid_aud.pkl --audio-root /mnt/otra_particion/home/israel_gpu_data/audio_data_raw/audio_participantes_validos --video-pickle data/processed/X_proc_final_linked.pkl --participants-pad P14 P27 --out-pickle data/processed/X_vid_aud_patched.pkl`

- Entrenamiento unimodal audio (CNN/logit baseline)
  - `python mains/run_audio_training.py --pickle /mnt/otra_particion/home/israel_gpu_data/audio_data_raw/X_vid_aud.pkl --audio-root /mnt/otra_particion/home/israel_gpu_data/audio_data_raw/audio_participantes_validos --results-prefix AudioCNN_v1 --class-weighted --epochs 30 --batch-size 16`
  - TCN: `python mains/run_audio_training.py --pickle /mnt/otra_particion/home/israel_gpu_data/audio_data_raw/X_vid_aud.pkl --audio-root /mnt/otra_particion/home/israel_gpu_data/audio_data_raw/audio_participantes_validos --arch tcn --results-prefix AudioTCN_v1 --tcn-channels 64 128 256 --class-weighted --epochs 30 --batch-size 16`
  - wav2vec (freeze por defecto): `python mains/run_audio_training.py --pickle /mnt/otra_particion/home/israel_gpu_data/audio_data_raw/X_vid_aud.pkl --audio-root /mnt/otra_particion/home/israel_gpu_data/audio_data_raw/audio_participantes_validos --arch wav2vec --wav2vec-bundle WAV2VEC2_BASE --results-prefix AudioW2V_v1 --class-weighted --epochs 10 --batch-size 8`

- Entrenamiento unimodal video (CNN/LSTM)
  - `python mains/run_video_training.py --pickle data/processed/X_proc_final_linked.pkl --path-col gpu_tensor_path --label-col action --prefer-df-label --timestamp-col timestamp --window-id-col window --batch-size 16 --epochs 20 --lr 1e-4 --weight-decay 1e-4 --cnn-emb 128 --lstm-hidden 128 --lstm-layers 1 --bidirectional --num-classes 5 --arkoudi --dropout 0.0 --val-split 0.2 --class-weighted --scheduler step --step-size 5 --gamma 0.5 --t-max 20 --plateau-patience 3 --plateau-factor 0.5`
  - ViT/CLIP + LSTM (fine-tuning opcional):
    - `python mains/run_video_training.py --pickle data/processed/X_proc_final_linked.pkl --path-col gpu_tensor_path --label-col action --prefer-df-label --timestamp-col timestamp --window-id-col window --batch-size 16 --epochs 20 --lr 1e-4 --weight-decay 1e-4 --lstm-hidden 256 --lstm-layers 1 --bidirectional --num-classes 5 --arkoudi --dropout 0.0 --val-split 0.2 --scheduler cosine --arch frame_lstm --backbone vit --backbone-name vit_b_16 --backbone-trainable --backbone-unfreeze-last 1 --target-size 224`
    - Para CLIP: reemplaza `--backbone vit --backbone-name vit_b_16` por `--backbone clip --backbone-name RN50` (u otro nombre soportado). 

- Entrenamiento multimodal (VAE determinista/variacional)
  - Determinista (ejemplo):
    - `python mains/run_multimodal_vae_train.py --pkl data/processed/multimodal_join.pkl --features-file utils/feature_sets/exp1.json --label-col action_proc --batch-size 16 --epochs 20 --lr 1e-4 --deterministic --fuse-dropout 0.2 --label-smoothing 0.05 --scheduler cosine --weight-decay 2e-4 --w-align 0.1 --w-contrastive 0.1 --contrastive-temp 0.07 --proj-dim 128 --modality-dropout 0.1 --grad-clip 1.0 --early-stop-patience 3 --early-stop-delta 0.02 --video-backbone vit --video-name vit_b_16 --video-target-size 224 --video-lstm-hidden 256 --video-lstm-layers 1 --video-bidirectional --video-dropout 0.0 --video-trainable --video-unfreeze-last 0 --val-split 0.2`
  - Variacional (añadir KL):
    - `python mains/run_multimodal_vae_train.py --pkl data/processed/multimodal_join.pkl --features-file utils/feature_sets/exp1.json --label-col action_proc --batch-size 16 --epochs 20 --lr 1e-4 --w-kl 1.0 --kl-anneal-steps 10000 --fuse-dropout 0.2 --label-smoothing 0.05 --scheduler cosine --weight-decay 2e-4 --w-align 0.1 --w-contrastive 0.1 --contrastive-temp 0.07 --proj-dim 128 --modality-dropout 0.1 --grad-clip 1.0 --early-stop-patience 3 --early-stop-delta 0.02 --video-backbone vit --video-name vit_b_16 --video-target-size 224 --video-lstm-hidden 256 --video-lstm-layers 1 --video-bidirectional --video-dropout 0.0 --video-trainable --video-unfreeze-last 0 --val-split 0.2`

- Entrenamiento multimodal con audio (nuevo)
  - Determinista (tab + video + audio):
    - `python mains/run_multimodal_vae_audio_train.py --pkl data/processed/multimodal_join.pkl --features-file utils/feature_sets/exp1.json --label-col action_proc --batch-size 32 --epochs 30 --lr 1e-4 --deterministic --fusion early --tabular-scaler robust --video-norm imagenet --audio-sr 16000 --audio-duration 5.0 --audio-norm per_channel --class-weighted --warmup-epochs 5 --warmup-modality both --warmup-disable-contrastive --w-align 0.0 --w-contrastive 0.0 --modality-dropout 0.0 --video-backbone vit --video-name vit_b_16 --video-trainable --video-unfreeze-last 1 --video-target-size 224 --video-lstm-hidden 256 --video-lstm-layers 1 --video-bidirectional`

- Artefactos (ambos VAE)
  - Modelo: `results/MMVAE_*‑model‑<hash>.pt`
  - Preprocesador tabular: `results/MMVAE_*‑preprocessor‑<hash>.pkl`
  - History: `results/*‑history‑<hash>.csv`
  - Reporte y probabilidades: `results/*‑eval_report‑<hash>.txt`, `results/*‑eval_proba‑<hash>.csv`
  - Embeddings: `results/*‑embeddings‑<hash>.csv` con `z_*` (+ `mu_*`/`std_*` en variacional) y metadata

## Requisitos

- Python 3.10+ (probado con 3.11)
- Instalar dependencias:
  - `pip install -r dataset_bicicletas/requirements.txt`
  - Nota: `torch` depende de tu GPU/CUDA; instala segun guia oficial.

## Flujo Rapido

1) Limpiar CSV crudo y guardar procesado.
2) Seleccionar features (o usar `utils/feature_sets/exp1.json`).
3) Entrenar baseline / MNLogit / Torch (tabular).
4) Preparar audios (renombrar + validar) y entrenar baseline audio si aplica.
5) Linkear tensores de video y verificar.
6) Alinear CSV al ancla de video y revisar.
7) Hacer join multimodal por timestamp.
8) Entrenar VAE multimodal (determinista o variacional).

## Carpetas y Rol

- `data/`
  - `raw/`: datos crudos (ej. `all_data.csv`).
  - `processed/`: intermedios limpios y pickles.
- `mains/`: ejecutables CLI del flujo.
- `src/`: codigo modular (cleaning, loading, features, models).
- `utils/`: utilidades para features y resultados.

## Notas y Consejos

- Etiquetas: usar `action_proc` cuando exista; si aparecen strings, los mapeos se aplican internamente.
- Preprocesamiento tabular: StandardScaler + OneHot; el preprocessor entrenado se guarda junto al modelo para inferencia reproducible.
- Multimodal: el VAE guarda embeddings (`z_*`) y, en variacional, `mu_*`/`std_*` para analisis econometrico.

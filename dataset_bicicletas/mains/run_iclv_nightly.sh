#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# MM-ICLV (10 epochs, 50% participants)
python mains/run_multimodal_icl_v.py \
  --pkl data/processed/multimodal_av_join_audio_cached.pkl \
  --label-col action_proc \
  --obs-lt-cols-file utils/feature_sets/obs_lt_mm.txt \
  --obs-u-cols-file utils/feature_sets/obs_u_mm.txt \
  --indicator-cols-file utils/feature_sets/obs_i_mm.txt \
  --epochs 10 \
  --batch-size 64 \
  --val-split 0.2 \
  --test-split 0.1 \
  --participant-frac 0.5 \
  --participant-col participant \
  --n-latent 64 \
  --alpha 1.0 \
  --lr 5e-4 \
  --grad-clip 1.0 \
  --cat-unique-threshold 5 \
  --audio-sr 16000 \
  --audio-duration 2 \
  --freeze-video \
  --freeze-audio \
  --seed 42

# ICLV Biogeme (sanity/standard, 50% participants, minimal draws)
python mains/run_icl_v_biogeme.py \
  --data data/raw/all_data.csv \
  --obs-lt-cols utils/feature_sets/obs_lt.txt \
  --obs-u-cols utils/feature_sets/obs_u.txt \
  --obs-i-cols utils/feature_sets/obs_i.txt \
  --label-col action_proc \
  --participant-col participant \
  --val-split 0.2 \
  --test-split 0.1 \
  --seed 42 \
  --half-data \
  --n-draws 100 \
  --n-latent 1

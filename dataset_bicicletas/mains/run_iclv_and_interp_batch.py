from __future__ import annotations

import subprocess
from pathlib import Path


def run_cmd(cmd: list[str], workdir: Path) -> int:
    print("\n=== Ejecutando:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=workdir)
    if res.returncode != 0:
        print(f"[ERROR] Comando falló con código {res.returncode}")
    return res.returncode


def main():
    """
    Lanza corridas representativas:
      - VAE interpretable (det y var) con hiperparámetros agresivos (3 frames, 2s audio).
      - ICLV tabular estándar.
      - ICLV multimodal estándar.
    """
    root = Path(__file__).resolve().parent.parent  # dataset_bicicletas/

    # Config agresiva para VAE interpretable (similar cfgB 16ep)
    interp_common = [
        "--epochs", "16",
        "--batch-size", "24",
        "--val-split", "0.2",
        "--test-split", "0.1",
        "--participant-frac", "0.9",
        "--participant-col", "participant",
        "--path-col", "frames_route",
        "--audio-col", "audio_cached_path",
        "--features-file", "utils/feature_sets/exp1.json",
        "--label-col", "action_proc",
        "--seed", "42",
        "--freeze-video",
        "--freeze-audio",
        "--lr", "5e-4",
        "--weight-decay", "5e-5",
        "--dropout", "0.1",
        "--label-smoothing", "0.02",
        "--class-weighted",
        "--early-stop-patience", "16",
        "--audio-sr", "16000",
        "--audio-duration", "2",
    ]

    # Config estándar para ICLV tabular
    iclv_common = [
        "--pkl", "data/processed/multimodal_av_join_audio_cached.pkl",
        "--label-col", "action_proc",
        "--features-file", "utils/feature_sets/exp1.json",
        "--obs-lt-cols-file", "utils/feature_sets/obs_lt.txt",
        "--obs-u-cols-file", "utils/feature_sets/obs_u.txt",
        "--indicator-cols-file", "utils/feature_sets/obs_i.txt",
        "--val-split", "0.2",
        "--test-split", "0.1",
        "--participant-col", "participant",
        "--participant-frac", "1.0",
        "--batch-size", "64",
        "--epochs", "20",
        "--lr", "1e-3",
        "--n-latent", "3",
        "--alpha", "1.0",
        "--seed", "42",
    ]

    # Config estándar para ICLV multimodal
    mm_iclv_common = [
        "--pkl", "data/processed/multimodal_av_join_audio_cached.pkl",
        "--label-col", "action_proc",
        "--features-file", "utils/feature_sets/exp1.json",
        "--obs-lt-cols-file", "utils/feature_sets/obs_lt_mm.txt",
        "--obs-u-cols-file", "utils/feature_sets/obs_u_mm.txt",
        "--indicator-cols-file", "utils/feature_sets/obs_i_mm.txt",
        "--val-split", "0.2",
        "--test-split", "0.1",
        "--participant-col", "participant",
        "--participant-frac", "1.0",
        "--batch-size", "8",
        "--epochs", "10",
        "--lr", "1e-4",
        "--n-latent", "64",
        "--alpha", "1.0",
        "--seed", "42",
        "--audio-duration", "2",
        "--audio-sr", "16000",
        "--freeze-video",
        "--freeze-audio",
    ]

    jobs = [
        {
            "name": "InterpVAE_det",
            "cmd": ["python", "mains/run_multimodal_vae_audio_interpretable.py", "--deterministic", *interp_common],
        },
        {
            "name": "InterpVAE_var",
            "cmd": ["python", "mains/run_multimodal_vae_audio_interpretable.py", *interp_common],
        },
        {
            "name": "ICLV_tabular",
            "cmd": ["python", "mains/run_icl_v.py", *iclv_common],
        },
        {
            "name": "ICLV_multimodal",
            "cmd": ["python", "mains/run_multimodal_icl_v.py", *mm_iclv_common],
        },
    ]

    for job in jobs:
        rc = run_cmd(job["cmd"], workdir=root)
        if rc != 0:
            print(f"[WARN] Deteniendo ejecución; fallo en {job['name']}")
            break


if __name__ == "__main__":
    main()

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
    Batch simple para ICLV tabular y multimodal.
    Edita aquí para corridas estándar o exigentes.
    """
    root = Path(__file__).resolve().parent.parent  # dataset_bicicletas/

    iclv_common = [
        "--pkl", "data/processed/multimodal_av_join_audio_with_iclv.pkl",
        "--label-col", "action_proc",
        "--obs-lt-cols-file", "utils/feature_sets/obs_lt.txt",
        "--obs-u-cols-file", "utils/feature_sets/obs_u.txt",
        "--indicator-cols-file", "utils/feature_sets/obs_i.txt",
        "--val-split", "0.2",
        "--test-split", "0.1",
        "--participant-col", "participant",
        "--participant-frac", "1.0",
        "--batch-size", "64",
        "--epochs", "10",
        "--lr", "1e-3",
        "--n-latent", "3",
        "--alpha", "1.0",
        "--seed", "42",
    ]

    mm_iclv_common = [
        "--pkl", "data/processed/multimodal_av_join_audio_cached.pkl",
        "--label-col", "action_proc",
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
        {"name": "ICLV_tabular", "cmd": ["python", "mains/run_icl_v.py", *iclv_common]},
        {"name": "ICLV_multimodal", "cmd": ["python", "mains/run_multimodal_icl_v.py", *mm_iclv_common]},
    ]

    for job in jobs:
        rc = run_cmd(job["cmd"], workdir=root)
        if rc != 0:
            print(f"[WARN] Deteniendo ejecución; fallo en {job['name']}")
            break


if __name__ == "__main__":
    main()

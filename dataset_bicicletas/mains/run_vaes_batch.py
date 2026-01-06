from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], workdir: Path) -> int:
    print("\n=== Ejecutando:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=workdir)
    if result.returncode != 0:
        print(f"[ERROR] Comando falló con código {result.returncode}")
    return result.returncode


def main():
    """
    Lanza una tanda representativa de corridas:
      - VAE 2 modal determinista y variacional
      - VAE 3 modal determinista y variacional
    Se usan configuraciones moderadas (más epochs que el smoke test, pero sin ser pesadas),
    con 3 frame de video y 2s de audio (ya manejado en los mains).
    """
    root = Path(__file__).resolve().parent.parent  # dataset_bicicletas/

    # Dos configuraciones: una intermedia y otra más exigente
    configs = [
        {
            "tag": "cfgA_12ep",
            "common": [
                "--epochs", "12",
                "--batch-size", "24",
                "--val-split", "0.2",
                "--test-split", "0.1",
                "--participant-frac", "0.75",
                "--participant-col", "participant",
                "--path-col", "frames_route",
                "--features-file", "utils/feature_sets/exp1.json",
                "--label-col", "action_proc",
                "--seed", "42",
                "--freeze-video",
                "--lr", "3e-4",
                "--weight-decay", "1e-4",
                "--dropout", "0.2",
                "--label-smoothing", "0.05",
                "--class-weighted",
                "--early-stop-patience", "12",
            ],
        },
        {
            "tag": "cfgB_16ep",
            "common": [
                "--epochs", "16",
                "--batch-size", "24",
                "--val-split", "0.2",
                "--test-split", "0.1",
                "--participant-frac", "0.9",
                "--participant-col", "participant",
                "--path-col", "frames_route",
                "--features-file", "utils/feature_sets/exp1.json",
                "--label-col", "action_proc",
                "--seed", "42",
                "--freeze-video",
                "--lr", "5e-4",
                "--weight-decay", "5e-5",
                "--dropout", "0.1",
                "--label-smoothing", "0.02",
                "--class-weighted",
                "--early-stop-patience", "16",
            ],
        },
    ]

    audio_common = [
        "--audio-col", "audio_cached_path",
        "--audio-sr", "16000",
        "--audio-duration", "2",
        "--freeze-audio",
    ]

    jobs = []
    for cfg in configs:
        base_common = cfg["common"]
        tag = cfg["tag"]
        jobs.extend([
            {
                "name": f"MMVAE_det_{tag}",
                "cmd": ["python", "mains/run_multimodal_vae_train.py", "--deterministic", *base_common],
            },
            {
                "name": f"MMVAE_var_{tag}",
                "cmd": ["python", "mains/run_multimodal_vae_train.py", *base_common],
            },
            {
                "name": f"MMVAEAudio_det_{tag}",
                "cmd": ["python", "mains/run_multimodal_vae_audio_train.py", "--deterministic", *base_common, *audio_common],
            },
            {
                "name": f"MMVAEAudio_var_{tag}",
                "cmd": ["python", "mains/run_multimodal_vae_audio_train.py", *base_common, *audio_common],
            },
        ])

    for job in jobs:
        rc = run_cmd(job["cmd"], workdir=root)
        if rc != 0:
            print(f"[WARN] Deteniendo ejecución; fallo en {job['name']}")
            break


if __name__ == "__main__":
    main()

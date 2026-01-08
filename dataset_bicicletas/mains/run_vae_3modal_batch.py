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
    Batch de 5 corridas robustas para MMVAEAudio (3 modal),
    con variaciones de LR, warmup y pesos auxiliares.
    """
    root = Path(__file__).resolve().parent.parent  # dataset_bicicletas/

    base = [
        "--epochs", "30",
        "--batch-size", "24",
        "--val-split", "0.2",
        "--test-split", "0.1",
        "--participant-frac", "0.9",
        "--participant-col", "participant",
        "--path-col", "frames_route",
        "--audio-col", "audio_cached_path",
        "--features-file", "utils/feature_sets/exp1.json",
        "--label-col", "action_proc",
        "--audio-sr", "16000",
        "--audio-duration", "2",
        "--seed", "42",
        "--shared-dim", "128",
        "--proj-dim", "256",
        "--dropout", "0.1",
        "--label-smoothing", "0.0",
        "--w-rec-tab", "0.5",
        "--w-rec-vid", "1.0",
        "--w-cls", "1.0",
        "--w-kl", "1.0",
        "--w-align", "0.2",
        "--w-contrastive", "0.2",
        "--contrastive-temp", "0.07",
        "--modality-dropout", "0.2",
        "--w-aux-tab", "0.2",
        "--w-aux-vid", "0.6",
        "--w-aux-aud", "0.6",
        "--class-weighted",
        "--grad-clip", "1.0",
        "--lr-tab-mult", "0.5",
        "--lr-video-mult", "2.0",
        "--lr-audio-mult", "2.0",
    ]

    # Variaciones
    jobs = [
        {
            "name": "run1_base_lr5e-4",
            "cmd": ["python", "mains/run_multimodal_vae_audio_train.py", *base, "--lr", "5e-4", "--weight-decay", "5e-5", "--warmup-epochs", "4", "--warmup-modality", "video", "--warmup-disable-contrastive"],
        },
        {
            "name": "run2_high_lr1e-3",
            "cmd": ["python", "mains/run_multimodal_vae_audio_train.py", *base, "--lr", "1e-3", "--weight-decay", "1e-4", "--warmup-epochs", "4", "--warmup-modality", "audio", "--warmup-disable-contrastive"],
        },
        {
            "name": "run3_more_aux",
            "cmd": ["python", "mains/run_multimodal_vae_audio_train.py", *base, "--lr", "5e-4", "--weight-decay", "5e-5", "--w-aux-vid", "1.0", "--w-aux-aud", "1.0", "--w-aux-tab", "0.1", "--warmup-epochs", "3", "--warmup-modality", "both"],
        },
        {
            "name": "run4_more_dropout",
            "cmd": ["python", "mains/run_multimodal_vae_audio_train.py", *base, "--lr", "5e-4", "--weight-decay", "5e-5", "--modality-dropout", "0.35", "--dropout", "0.2", "--warmup-epochs", "3", "--warmup-modality", "both"],
        },
        {
            "name": "run5_no_label_smoothing",
            "cmd": ["python", "mains/run_multimodal_vae_audio_train.py", *base, "--lr", "5e-4", "--weight-decay", "5e-5", "--label-smoothing", "0.0", "--warmup-epochs", "3", "--warmup-modality", "both"],
        },
    ]

    for job in jobs:
        rc = run_cmd(job["cmd"], workdir=root)
        if rc != 0:
            print(f"[WARN] Deteniendo ejecución; fallo en {job['name']}")
            break


if __name__ == "__main__":
    main()

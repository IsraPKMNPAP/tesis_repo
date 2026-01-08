from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run_cmd(cmd: List[str]) -> int:
    print("\n== Running ==")
    print(" ".join(cmd))
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity run de todos los modelos base NEUMA.")
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument("--tabular-data", type=str, default="/mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed/data_base_neuma.csv")
    parser.add_argument("--tabular-config", type=str, default="configs/tabular_cols.json")
    parser.add_argument("--eeg-index", type=str, default="./data/processed/eeg_segments_index.csv")
    parser.add_argument("--products", type=str, default="./data/processed/products_all_with_images.csv")
    parser.add_argument("--embeddings-dir", type=str, default="./data/processed/image_embeddings")
    parser.add_argument("--tabular-join", type=str, default="/mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed/data_latente_neuma.csv")
    parser.add_argument("--multimodal-data", type=str, default="./data/processed/multimodal_join_with_eeg_emb_aug.csv")
    parser.add_argument("--obs-lt-cols", type=str, default="./utils/columns/iclv/obs_lt.txt")
    parser.add_argument("--obs-u-cols", type=str, default="./utils/columns/iclv/obs_u.txt")
    parser.add_argument("--obs-i-cols", type=str, default="./utils/columns/iclv/obs_i.txt")
    parser.add_argument("--obs-lt-cols-mm", type=str, default="./utils/columns/iclv_multimodal/obs_lt.txt")
    parser.add_argument("--obs-u-cols-mm", type=str, default="./utils/columns/iclv_multimodal/obs_u.txt")
    parser.add_argument("--results-root", type=str, default="./results/sanity_runs")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    cmds = []
    cmds.append([
        args.python, "-m", "mains.run_tabular_baseline",
        "--data", args.tabular_data,
        "--config", args.tabular_config,
        "--epochs", str(args.epochs),
        "--results-dir", str(results_root / "tabular_baseline"),
    ])
    cmds.append([
        args.python, "-m", "mains.run_eeg_baseline",
        "--index-csv", args.eeg_index,
        "--epochs", str(args.epochs),
        "--results-dir", str(results_root / "eeg_baseline"),
    ])
    cmds.append([
        args.python, "-m", "mains.run_multimodal_tabular_clip_baseline",
        "--products", args.products,
        "--embeddings-dir", args.embeddings_dir,
        "--tabular", args.tabular_join,
        "--config", args.tabular_config,
        "--epochs", str(args.epochs),
        "--results-dir", str(results_root / "multimodal_tabular_clip"),
    ])
    # VAE multimodal fusion (stochastic)
    cmds.append([
        args.python, "-m", "mains.run_multimodal_fusion_baseline",
        "--data", args.multimodal_data,
        "--mode", "vae",
        "--use-tabular", "--use-clip", "--use-eeg",
        "--epochs", str(args.epochs),
        "--results-dir", str(results_root / "multimodal_fusion_vae"),
    ])
    # VAE multimodal logits (stochastic)
    cmds.append([
        args.python, "-m", "mains.run_multimodal_logits_baseline",
        "--data", args.multimodal_data,
        "--mode", "vae",
        "--use-tabular", "--use-clip", "--use-eeg",
        "--epochs", str(args.epochs),
        "--results-dir", str(results_root / "multimodal_logits_vae"),
    ])
    # ICLV clásico
    cmds.append([
        args.python, "-m", "mains.run_icl_v",
        "--data", args.multimodal_data,
        "--obs-lt-cols", args.obs_lt_cols,
        "--obs-u-cols", args.obs_u_cols,
        "--obs-i-cols", args.obs_i_cols,
        "--epochs", str(args.epochs),
        "--results-dir", str(results_root / "icl_v"),
    ])
    # ICLV multimodal
    cmds.append([
        args.python, "-m", "mains.run_multimodal_icl_v",
        "--data", args.multimodal_data,
        "--obs-lt-cols", args.obs_lt_cols_mm,
        "--obs-u-cols", args.obs_u_cols_mm,
        "--img-emb-col", "embedding_path",
        "--eeg-emb-col", "eeg_emb_path",
        "--epochs", str(args.epochs),
        "--results-dir", str(results_root / "multimodal_icl_v"),
    ])

    failed = False
    for cmd in cmds:
        code = run_cmd(cmd)
        if code != 0:
            print(f"[WARN] command failed with exit code {code}")
            failed = True

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

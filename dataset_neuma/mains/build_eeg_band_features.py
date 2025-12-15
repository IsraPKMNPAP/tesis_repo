"""
Calcula features de potencia por banda y canal, agrupando todas las instancias de EEG
para cada (subject, page, product).

Entrada:
  - data/processed/eeg_segments_index.csv (con start/end y npy_path por sujeto)
  - Archivos npy/meta en /mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed/eeg

Salida:
  - data/EDA/eda_results_tabular/eeg_band_features.csv con columnas:
      subject, page, product_id, bought, channel, band, power_mean, power_std, power_rel

Uso (desde dataset_neuma):
  python -m dataset_neuma.mains.build_eeg_band_features \
    --segments-index ./data/processed/eeg_segments_index.csv \
    --output ./data/EDA/eda_results_tabular/eeg_band_features.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch

# Permite ejecución desde carpeta dataset_neuma
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))


BANDS: List[Tuple[str, Tuple[float, float]]] = [
    ("theta", (4.0, 7.0)),
    ("alpha", (8.0, 12.0)),
    ("beta", (13.0, 30.0)),
    ("gamma_low", (30.0, 45.0)),
]


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def band_powers(segment: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula potencias por banda y potencia total por canal para un segmento.
    segment: shape [n_channels, n_samples]
    Retorna: band_powers (bands x channels), total_power (channels,)
    """
    # Welch: usa ventana por defecto; nperseg limitado por longitud
    freqs, psd = welch(segment, fs=fs, axis=1, nperseg=min(1024, segment.shape[1]))
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0
    total_power = np.sum(psd, axis=1) * df if df else np.sum(psd, axis=1)
    band_power_list = []
    for _, (low, high) in BANDS:
        mask = (freqs >= low) & (freqs <= high)
        if not mask.any():
            band_power = np.zeros(psd.shape[0])
        else:
            band_power = np.sum(psd[:, mask], axis=1) * df if df else np.sum(psd[:, mask], axis=1)
        band_power_list.append(band_power)
    band_powers_arr = np.stack(band_power_list, axis=0)  # [bands, channels]
    return band_powers_arr, total_power


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye features EEG por banda y canal, agrupando segmentos.")
    parser.add_argument(
        "--segments-index",
        type=Path,
        default=Path("./data/processed/eeg_segments_index.csv"),
        help="CSV con segmentos EEG y rutas npy.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./data/EDA/eda_results_tabular/eeg_band_features.csv"),
        help="Ruta de salida para el CSV de features.",
    )
    args = parser.parse_args()

    if not args.segments_index.exists():
        raise SystemExit(f"No se encontró {args.segments_index}")

    df_idx = pd.read_csv(args.segments_index)
    required_cols = {"subject", "page", "product_id", "bought", "npy_path", "start", "end"}
    if not required_cols.issubset(df_idx.columns):
        raise SystemExit(f"Faltan columnas requeridas en segments_index: {required_cols - set(df_idx.columns)}")

    rows = []
    # Procesar por sujeto para cargar npy una sola vez
    for subj, df_subj in df_idx.groupby("subject"):
        if df_subj.empty:
            continue
        npy_path = Path(df_subj.iloc[0]["npy_path"])
        if not npy_path.exists():
            print(f"[WARN] npy no encontrado para {subj}: {npy_path}")
            continue
        # intenta fs desde columna fs o meta
        fs = df_subj["fs"].dropna().iloc[0] if "fs" in df_subj.columns and df_subj["fs"].notna().any() else None
        if fs is None:
            meta_path = npy_path.with_name(npy_path.stem.replace("_data", "_meta") + ".json")
            meta = load_json(meta_path)
            fs = meta.get("Fs")
        if fs is None:
            print(f"[WARN] Fs no encontrado para {subj}, se omite")
            continue
        eeg = np.load(npy_path)  # shape [channels, samples]

        # Agrupar por (page, product_id)
        for (page, product), df_grp in df_subj.groupby(["page", "product_id"]):
            # lista de segmentos
            segs = []
            for _, r in df_grp.iterrows():
                start = int(r["start"])
                end = int(r["end"])
                end = min(end, eeg.shape[1] - 1)
                segs.append(eeg[:, start : end + 1])
            # concatenar en tiempo
            concat = np.concatenate(segs, axis=1) if len(segs) > 1 else segs[0]
            band_pows_list = []
            total_list = []
            # calculamos power por banda/segmento y acumulamos
            # (aquí ya concatenado, equivalente a tratar como un solo segmento)
            bp, tot = band_powers(concat, fs=fs)
            band_pows_list.append(bp)
            total_list.append(tot)

            band_pows_arr = np.stack(band_pows_list, axis=0)  # [n_segments=1, bands, channels]
            total_arr = np.stack(total_list, axis=0)  # [n_segments=1, channels]
            band_mean = band_pows_arr.mean(axis=0)  # [bands, channels]
            band_std = band_pows_arr.std(axis=0)   # [bands, channels]
            total_mean = total_arr.mean(axis=0)    # [channels]

            bought_val = df_grp["bought"].dropna().iloc[0] if df_grp["bought"].notna().any() else None

            n_ch = eeg.shape[0]
            for ch_idx in range(n_ch):
                for b_idx, (bname, _) in enumerate(BANDS):
                    p_mean = band_mean[b_idx, ch_idx]
                    p_std = band_std[b_idx, ch_idx]
                    rel = p_mean / total_mean[ch_idx] if total_mean[ch_idx] not in (0, None) else np.nan
                    rows.append(
                        {
                            "subject": subj,
                            "page": page,
                            "product_id": product,
                            "bought": bought_val,
                            "channel_idx": ch_idx,
                            "band": bname,
                            "power_mean": p_mean,
                            "power_std": p_std,
                            "power_rel": rel,
                            "fs": fs,
                        }
                    )

    df_out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"Guardado: {args.output} (filas: {len(df_out)})")


if __name__ == "__main__":
    main()

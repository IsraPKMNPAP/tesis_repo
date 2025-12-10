"""
Utilidades de exportación para el dataset NEUMA.

Incluye extracción de:
- Productos y segmentos (EEG/ET) -> CSV/JSON ligeros.
- Perfil y demografía -> CSV (una fila por sujeto).
- EEG_clean -> .npy + meta.
- ET_clean -> CSV (coords normalizadas y en píxeles) + meta.

Pensado para ser usado desde mains/run_export_all.py en GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.io.matlab import mat_struct


# ---------------------------------------------------------------------------
# utilidades generales
# ---------------------------------------------------------------------------

def _first_mat_struct(data: Dict[str, Any]) -> mat_struct:
    for v in data.values():
        if isinstance(v, mat_struct):
            return v
    raise ValueError("No se encontró mat_struct raíz en el .mat")


def _field(ms: mat_struct, name: str, default=None):
    return getattr(ms, name) if hasattr(ms, name) else default


def _clean_reasons(obj: Any) -> List[str]:
    if obj is None:
        return []
    arr = np.asarray(obj)
    if arr.size == 0:
        return []
    if arr.dtype == object:
        return [str(x) for x in arr.tolist() if str(x).strip()]
    return [str(x) for x in arr.reshape(-1).tolist() if str(x).strip()]


def _segments_to_list(arr: Any) -> List[Tuple[int, int]]:
    if arr is None:
        return []
    na = np.asarray(arr)
    if na.ndim != 2 or na.shape[1] < 2:
        return []
    return [(int(a), int(b)) for a, b in na[:, :2]]


def _mat_struct_to_dict(ms: mat_struct) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for field in getattr(ms, "_fieldnames", []) or []:
        val = getattr(ms, field)
        if isinstance(val, np.ndarray):
            if val.size == 0:
                out[field] = None
            elif val.size == 1:
                scalar = val.reshape(-1)[0]
                out[field] = scalar.item() if hasattr(scalar, "item") else scalar
            else:
                out[field] = val.tolist()
        else:
            out[field] = val
    return out


# ---------------------------------------------------------------------------
# Productos y segmentos
# ---------------------------------------------------------------------------

def export_products_and_segments(root: mat_struct, subject: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    products_rows: List[Dict[str, Any]] = []
    segments_rows: List[Dict[str, Any]] = []
    pages_json: Dict[str, Any] = {}

    for page_idx in range(1, 7):
        page_name = f"Page{page_idx}"
        if not hasattr(root, page_name):
            continue
        page_obj = getattr(root, page_name)
        page_dict: Dict[str, Any] = {}

        if isinstance(page_obj, mat_struct):
            for field in getattr(page_obj, "_fieldnames", []) or []:
                if not field.lower().startswith("product"):
                    continue
                product_obj = getattr(page_obj, field)
                if not isinstance(product_obj, mat_struct):
                    continue

                pinfo = _field(product_obj, "ProductInfo")
                pinfo_dict = _mat_struct_to_dict(pinfo) if isinstance(pinfo, mat_struct) else {}
                desc = pinfo_dict.get("Description")
                bought = pinfo_dict.get("Bought")
                familiarity = pinfo_dict.get("Familiarity")
                frequent_buy = pinfo_dict.get("FrequentBuy")
                reasons = _clean_reasons(pinfo_dict.get("Reasons"))

                products_rows.append(
                    {
                        "subject": subject,
                        "page": page_name,
                        "product_id": field,
                        "description": desc,
                        "bought": bought,
                        "familiarity": familiarity,
                        "frequent_buy": frequent_buy,
                        "reasons": reasons,
                    }
                )

                eeg_segments = _segments_to_list(_field(product_obj, "EEG_segments"))
                et_segments = _segments_to_list(_field(product_obj, "ET_segments"))

                page_dict[field] = {
                    "ProductInfo": {
                        "Description": desc,
                        "Bought": bought,
                        "Familiarity": familiarity,
                        "FrequentBuy": frequent_buy,
                        "Reasons": reasons,
                    },
                    "EEG_segments": eeg_segments,
                    "ET_segments": et_segments,
                }

                for idx, (start, end) in enumerate(eeg_segments):
                    segments_rows.append(
                        {
                            "subject": subject,
                            "page": page_name,
                            "product_id": field,
                            "modality": "EEG",
                            "seg_idx": idx,
                            "start": start,
                            "end": end,
                        }
                    )
                for idx, (start, end) in enumerate(et_segments):
                    segments_rows.append(
                        {
                            "subject": subject,
                            "page": page_name,
                            "product_id": field,
                            "modality": "ET",
                            "seg_idx": idx,
                            "start": start,
                            "end": end,
                        }
                    )

        pages_json[page_name] = page_dict

    products_df = pd.DataFrame(products_rows)
    segments_df = pd.DataFrame(segments_rows)
    return products_df, segments_df, pages_json


# ---------------------------------------------------------------------------
# Perfil y demografía
# ---------------------------------------------------------------------------

def export_profile_demographics(root: mat_struct, subject: str) -> pd.DataFrame:
    row: Dict[str, Any] = {"subject": subject}
    if hasattr(root, "Profile") and isinstance(getattr(root, "Profile"), mat_struct):
        row.update(_mat_struct_to_dict(getattr(root, "Profile")))
    if hasattr(root, "Demographics") and isinstance(getattr(root, "Demographics"), mat_struct):
        row.update(_mat_struct_to_dict(getattr(root, "Demographics")))
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# EEG
# ---------------------------------------------------------------------------

def export_eeg(root: mat_struct, subject: str, out_dir: Path) -> Dict[str, Any]:
    if not hasattr(root, "EEG_clean") or not isinstance(getattr(root, "EEG_clean"), mat_struct):
        raise ValueError("EEG_clean no encontrado en el .mat")
    eeg = getattr(root, "EEG_clean")
    if not hasattr(eeg, "Data"):
        raise ValueError("EEG_clean.Data no encontrado")

    eeg_data = np.asarray(eeg.Data)
    eeg_fs = getattr(eeg, "Fs") if hasattr(eeg, "Fs") else None

    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{subject}_eeg_data.npy"
    np.save(data_path, eeg_data)
    meta = {"subject": subject, "Fs": float(eeg_fs) if eeg_fs is not None else None, "shape": list(eeg_data.shape)}
    meta_path = out_dir / f"{subject}_eeg_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"data_path": data_path, "meta_path": meta_path, **meta}


# ---------------------------------------------------------------------------
# ET
# ---------------------------------------------------------------------------

def _normalize_channels(chans_obj: Any) -> List[str]:
    arr = np.asarray(chans_obj)
    if arr.size == 0:
        return []
    chans: List[str] = []
    for el in arr.reshape(-1):
        try:
            chans.append(str(el))
        except Exception:
            chans.append("")
    return chans


def _find_channel_indices(chans: List[str], aliases: Dict[str, List[str]]) -> Dict[str, Optional[int]]:
    lower = [c.lower() for c in chans]
    idxs: Dict[str, Optional[int]] = {}
    for key, names in aliases.items():
        found = None
        for name in names:
            if name.lower() in lower:
                found = lower.index(name.lower())
                break
        idxs[key] = found
    return idxs


def _extract_channel(data_cf: np.ndarray, idx: Optional[int]) -> np.ndarray:
    if idx is None:
        return np.full(data_cf.shape[1], np.nan)
    return data_cf[idx, :]


def _ensure_channel_first(data: np.ndarray, n_chans: int) -> Tuple[np.ndarray, str]:
    if data.shape[0] == n_chans:
        return data, "channel_first"
    if data.shape[1] == n_chans:
        return data.T, "time_first_transposed"
    return data, "unknown"


def export_et(
    root: mat_struct,
    subject: str,
    out_dir: Path,
    screen_w: int = 1920,
    screen_h: int = 1080,
) -> Dict[str, Any]:
    if not hasattr(root, "ET_clean") or not isinstance(getattr(root, "ET_clean"), mat_struct):
        raise ValueError("ET_clean no encontrado en el .mat")
    et = getattr(root, "ET_clean")
    if not hasattr(et, "Data"):
        raise ValueError("ET_clean.Data no encontrado")

    raw_data = np.asarray(et.Data)
    chans = _normalize_channels(getattr(et, "chans", []))
    n_chans_declared = len(chans)
    data_cf, orientation = _ensure_channel_first(raw_data, n_chans_declared or raw_data.shape[0])
    n_ch, n_samples = data_cf.shape

    aliases = {
        "left_x": ["left_x", "lx"],
        "left_y": ["left_y", "ly"],
        "left_pupil": ["left_pupil", "lpupil", "left_pupildiam"],
        "right_x": ["right_x", "rx"],
        "right_y": ["right_y", "ry"],
        "right_pupil": ["right_pupil", "rpupil", "right_pupildiam"],
    }
    idxs = _find_channel_indices(chans, aliases) if chans else {k: (i if i < n_ch else None) for i, k in enumerate(aliases.keys())}

    lx = _extract_channel(data_cf, idxs.get("left_x"))
    ly = _extract_channel(data_cf, idxs.get("left_y"))
    rx = _extract_channel(data_cf, idxs.get("right_x"))
    ry = _extract_channel(data_cf, idxs.get("right_y"))
    lp = _extract_channel(data_cf, idxs.get("left_pupil"))
    rp = _extract_channel(data_cf, idxs.get("right_pupil"))

    lx_pix = lx * screen_w
    rx_pix = rx * screen_w
    ly_pix = ly * screen_h
    ry_pix = ry * screen_h

    fs = getattr(et, "Fs", None)
    time_s = np.arange(n_samples) / float(fs) if fs else np.arange(n_samples)

    df_et = pd.DataFrame(
        {
            "sample_idx": np.arange(n_samples, dtype=int),
            "time_s": time_s,
            "left_x_norm": lx,
            "left_y_norm": ly,
            "right_x_norm": rx,
            "right_y_norm": ry,
            "left_x_px": lx_pix,
            "left_y_px": ly_pix,
            "right_x_px": rx_pix,
            "right_y_px": ry_pix,
            "left_pupil": lp,
            "right_pupil": rp,
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{subject}_et_data.csv"
    df_et.to_csv(csv_path, index=False)

    meta = {
        "subject": subject,
        "Fs": float(fs) if fs is not None else None,
        "shape_original": list(raw_data.shape),
        "shape_used": [int(n_ch), int(n_samples)],
        "channels": chans,
        "channel_indices": idxs,
        "orientation": orientation,
        "screen_width": screen_w,
        "screen_height": screen_h,
    }
    meta_path = out_dir / f"{subject}_et_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"data_path": csv_path, "meta_path": meta_path, **meta}


# ---------------------------------------------------------------------------
# Segmentos con tiempos (usa Fs de EEG/ET)
# ---------------------------------------------------------------------------

def enrich_segments_with_time(segments_df: pd.DataFrame, eeg_fs: Optional[float], et_fs: Optional[float], eeg_shape: Optional[List[int]], et_shape: Optional[List[int]]) -> pd.DataFrame:
    fs_map = {"EEG": eeg_fs, "ET": et_fs}
    shape_map = {"EEG": eeg_shape, "ET": et_shape}

    def compute_times(row):
        modality = row["modality"]
        fs = fs_map.get(modality)
        start = row["start"]
        end = row["end"]
        if fs is None or fs == 0:
            return pd.Series({"start_time_s": np.nan, "end_time_s": np.nan, "duration_s": np.nan, "in_bounds": np.nan})
        start_t = start / fs
        end_t = end / fs
        duration = (end - start) / fs
        shape = shape_map.get(modality)
        if shape and len(shape) >= 2:
            max_idx = shape[1] - 1
            in_bounds = int((start >= 0) and (end <= max_idx))
        else:
            in_bounds = np.nan
        return pd.Series({"start_time_s": start_t, "end_time_s": end_t, "duration_s": duration, "in_bounds": in_bounds})

    times_df = segments_df.apply(compute_times, axis=1)
    return pd.concat([segments_df, times_df], axis=1)


# ---------------------------------------------------------------------------
# Pipeline por participante
# ---------------------------------------------------------------------------

def process_participant(
    mat_path: Path,
    subject: Optional[str],
    out_processed_root: Path,
    out_repo_processed_root: Path,
    screen_w: int = 1920,
    screen_h: int = 1080,
) -> Dict[str, Any]:
    """
    Procesa un participante:
      - Extrae productos, segmentos, pages.json (guardado en repo processed)
      - Extrae perfil/demografía (guardado en repo processed)
      - Exporta EEG (npy+meta) -> processed externo (disco GPU)
      - Exporta ET (csv+meta)  -> processed externo (disco GPU)
      - Enriquecer segments.csv con tiempos -> repo processed
    """
    subject_id = subject or mat_path.stem
    data = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    root = _first_mat_struct(data)

    # Productos y segmentos
    products_df, segments_df, pages_json = export_products_and_segments(root, subject_id)
    # Perfil/demografía
    profile_df = export_profile_demographics(root, subject_id)

    # EEG / ET
    eeg_out = export_eeg(root, subject_id, out_processed_root / "eeg")
    et_out = export_et(root, subject_id, out_processed_root / "et", screen_w=screen_w, screen_h=screen_h)

    # Segments con tiempo
    seg_with_time = enrich_segments_with_time(
        segments_df,
        eeg_fs=eeg_out.get("Fs"),
        et_fs=et_out.get("Fs"),
        eeg_shape=eeg_out.get("shape"),
        et_shape=et_out.get("shape_original"),
    )

    # Guardar en repo processed
    repo_dir = out_repo_processed_root
    repo_dir.mkdir(parents=True, exist_ok=True)
    products_df.to_csv(repo_dir / f"{subject_id}_products.csv", index=False)
    segments_df.to_csv(repo_dir / f"{subject_id}_segments.csv", index=False)
    seg_with_time.to_csv(repo_dir / f"{subject_id}_segments_with_times.csv", index=False)
    profile_df.to_csv(repo_dir / f"{subject_id}_profile_demographics.csv", index=False)
    (repo_dir / f"{subject_id}_pages.json").write_text(json.dumps(pages_json, indent=2, ensure_ascii=False), encoding="utf-8")

    meta_global = {
        "subject": subject_id,
        "eeg": {"data_path": str(eeg_out["data_path"]), "meta_path": str(eeg_out["meta_path"])},
        "et": {"data_path": str(et_out["data_path"]), "meta_path": str(et_out["meta_path"])},
    }
    (repo_dir / f"{subject_id}_meta.json").write_text(json.dumps(meta_global, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "subject": subject_id,
        "repo_dir": repo_dir,
        "eeg_dir": out_processed_root / "eeg",
        "et_dir": out_processed_root / "et",
    }


# ---------------------------------------------------------------------------
# Helpers para listas de participantes
# ---------------------------------------------------------------------------

def find_participants(raw_dir: Path, pattern: str = "S*.mat") -> List[Path]:
    return sorted(raw_dir.glob(pattern))

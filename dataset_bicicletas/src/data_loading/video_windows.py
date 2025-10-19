from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import re
import json
import pandas as pd
import torch
from torch.utils.data import Dataset


def _coerce_list(val):
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, str):
        # Could be JSON list or a single path
        s = val.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return [val]
        return [val]
    return [val]


def adjust_paths_prefix(df: pd.DataFrame, column: str, src_prefix: str, dst_prefix: str) -> pd.DataFrame:
    """Replace a Windows/OneDrive prefix to a Linux/GPU root in a dataframe column.

    - Works for columns containing strings or lists of strings.
    - Returns a copy.
    """
    src_prefix = str(src_prefix)
    dst_prefix = str(dst_prefix)

    def _replace(x):
        if isinstance(x, str):
            return x.replace(src_prefix, dst_prefix)
        if isinstance(x, (list, tuple)):
            return [str(xx).replace(src_prefix, dst_prefix) for xx in x]
        return x

    out = df.copy()
    out[column] = out[column].map(_replace)
    return out


def extract_window_index(filename: str) -> Optional[int]:
    m = re.search(r"window_(\d+)\.pt$", filename)
    if m:
        return int(m.group(1))
    return None


def map_by_timestamp_or_order(
    df: pd.DataFrame,
    timestamp_col: str,
    gpu_root: str | Path,
    candidate_dirnames: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Assign correct GPU paths to rows using timestamp when available, else order.

    Assumptions:
      - GPU partition holds files named `window_i.pt` (i from 0..N-1 or 1..N).
      - Each file is either a tensor or a dict with optional key 'timestamp'.
      - Dataframe `df` contains chronological rows by `timestamp_col`.

    Strategy:
      1) List all window_*.pt files under `gpu_root` (and subfolders in candidate_dirnames if provided).
      2) Try reading a small sample to detect presence of 'timestamp' metadata.
      3) If timestamps are present across files, map by timestamp equality.
      4) Else map by index order (sorted df by timestamp vs sorted filenames by index).
    """
    gpu_root = Path(gpu_root)
    files: List[Path] = []
    if candidate_dirnames:
        for d in candidate_dirnames:
            dpath = gpu_root / d
            if dpath.exists():
                files.extend(sorted(dpath.rglob("window_*.pt")))
    else:
        if gpu_root.exists():
            files = sorted(gpu_root.rglob("window_*.pt"))

    if not files:
        raise FileNotFoundError(f"No se encontraron archivos window_*.pt en {gpu_root}")

    # Probe subset for timestamps
    file_ts: Dict[Path, Optional[str]] = {}
    probe = files[:: max(1, len(files) // 20) ]  # ~20 samples
    for p in probe:
        try:
            obj = torch.load(p, map_location="cpu")
            ts: Optional[str] = None
            if isinstance(obj, dict):
                ts = obj.get("timestamp") or obj.get("ts")
            # tensors may not carry timestamp; skip
            file_ts[p] = ts
        except Exception:
            file_ts[p] = None

    has_ts = any(v is not None for v in file_ts.values())

    out = df.copy()
    out = out.sort_values(by=timestamp_col).reset_index(drop=True)

    if has_ts:
        # Build full mapping lazily: only if needed
        ts_to_path: Dict[str, Path] = {}
        for p in files:
            try:
                obj = torch.load(p, map_location="cpu")
                ts = None
                if isinstance(obj, dict):
                    ts = obj.get("timestamp") or obj.get("ts")
                if ts is not None:
                    ts_to_path[str(ts)] = p
            except Exception:
                continue
        missing = []
        gpu_paths: List[Optional[str]] = []
        for ts in out[timestamp_col].astype(str).tolist():
            p = ts_to_path.get(ts)
            if p is None:
                missing.append(ts)
                gpu_paths.append(None)
            else:
                gpu_paths.append(str(p))
        if missing:
            raise ValueError(
                f"No se encontraron archivos para {len(missing)} timestamps. Ejemplo: {missing[:3]}"
            )
        out["gpu_tensor_path"] = gpu_paths
        return out

    # Fallback: order mapping using filename index
    files_with_idx = [(extract_window_index(f.name), f) for f in files]
    files_with_idx = [(i, p) for i, p in files_with_idx if i is not None]
    if not files_with_idx:
        raise ValueError("Archivos window_*.pt no siguen el patrón esperado con índice.")
    files_with_idx.sort(key=lambda x: x[0])
    if len(files_with_idx) < len(out):
        raise ValueError(
            f"Menos archivos ({len(files_with_idx)}) que filas ({len(out)}). No se puede mapear por orden."
        )
    out["gpu_tensor_path"] = [str(p) for _, p in files_with_idx[: len(out)]]
    return out


@dataclass
class WindowSample:
    x: torch.Tensor
    y: Optional[int]
    timestamp: Optional[str]
    window_id: Optional[int]
    participant: Optional[str]


class VideoWindowsDataset(Dataset):
    """Dataset that loads precomputed window tensors and labels.

    Expects a dataframe with at least one of the following columns:
      - 'tensor_path' or 'gpu_tensor_path': path to a .pt file per window
      - OR 'paths': list of per-frame .pt or image files (not recommended here)
    Optionally:
      - 'label' (int) or specified via `label_col`
      - 'timestamp' string
      - 'window_id' numeric

    The .pt file may be either a dict with keys {'x': Tensor, 'y': int?, 'timestamp': str?}
    or a raw Tensor. If raw Tensor, `label_col` is used from the dataframe.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        path_col: str = "gpu_tensor_path",
        label_col: Optional[str] = None,
        timestamp_col: Optional[str] = None,
        window_id_col: Optional[str] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.path_col = path_col
        self.label_col = label_col
        self.timestamp_col = timestamp_col
        self.window_id_col = window_id_col
        self.transform = transform

        if path_col not in self.df.columns:
            # try fallbacks commonly used
            for alt in ["tensor_path", "paths", "path", "file_path"]:
                if alt in self.df.columns:
                    self.path_col = alt
                    break
        if self.path_col not in self.df.columns:
            raise KeyError(
                f"No se encontró una columna de rutas ('{path_col}' ni alternativas) en el dataframe"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> WindowSample:
        row = self.df.iloc[idx]
        p = Path(str(row[self.path_col]))
        obj = torch.load(p, map_location="cpu")
        x: torch.Tensor
        y: Optional[int] = None
        ts: Optional[str] = None
        wid: Optional[int] = None
        part: Optional[str] = None

        if isinstance(obj, dict):
            # Buscar el tensor de frames sin evaluación booleana
            if "frames" in obj:
                x = obj["frames"]
            elif "x" in obj:
                x = obj["x"]
            else:
                raise KeyError(f"No se encontró un tensor 'frames' o 'x' en {p}")

            # Extraer etiqueta
            y = obj.get("label")
            if isinstance(y, torch.Tensor):
                y = int(y.item())
            elif isinstance(obj.get("y"), (int, torch.Tensor)):
                y = obj.get("y")
                if isinstance(y, torch.Tensor):
                    y = int(y.item())

            ts = obj.get("timestamp") or obj.get("ts")
            wid = obj.get("window_id") or obj.get("idx")
            part = obj.get("participant")
        elif isinstance(obj, torch.Tensor):
            x = obj
        else:
            raise TypeError(f"Contenido no soportado en {p}: {type(obj)}")

        if self.transform is not None:
            x = self.transform(x)

        # If no label inside .pt, try dataframe column
        if y is None and self.label_col and self.label_col in self.df.columns:
            val = row[self.label_col]
            if pd.notna(val):
                try:
                    y = int(val)
                except Exception:
                    y = None

        if ts is None and self.timestamp_col and self.timestamp_col in self.df.columns:
            ts = row[self.timestamp_col]
        if wid is None and self.window_id_col and self.window_id_col in self.df.columns:
            wid = row[self.window_id_col]
        if part is None and 'participant' in self.df.columns:
            part = row['participant']

        return WindowSample(x=x, y=y, timestamp=ts, window_id=wid, participant=part)

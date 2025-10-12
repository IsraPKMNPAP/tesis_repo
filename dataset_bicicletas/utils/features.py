from __future__ import annotations

from typing import Iterable, List, Sequence
import hashlib
from pathlib import Path


def save_features_file(path: str | Path, features: Sequence[str], meta: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        import json
        payload = {"features": list(features)}
        if meta:
            payload["meta"] = meta
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        path.write_text("\n".join(features), encoding="utf-8")


def load_features_file(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        return []
    if path.suffix.lower() == ".json":
        import json
        data = json.loads(path.read_text(encoding="utf-8"))
        feats = data.get("features", [])
        return [f for f in feats if isinstance(f, str) and f]
    else:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [l.strip() for l in lines if l.strip()]


def feature_hash(features: Sequence[str]) -> str:
    """Return a short stable hash for a feature list.

    Uses SHA1 over a comma-joined list and returns first 8 hex chars.
    """
    joined = ",".join(features)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:8]


def chunk_columns(cols: Sequence[str], page_size: int = 10) -> List[List[str]]:
    """Split column names into chunks of size `page_size`."""
    return [list(cols[i : i + page_size]) for i in range(0, len(cols), page_size)]


def format_columns_paged(cols: Sequence[str], page_size: int = 10) -> str:
    """Format columns with indices, grouped by `page_size` per block."""
    lines: List[str] = []
    for start in range(0, len(cols), page_size):
        block = cols[start : start + page_size]
        for idx, name in enumerate(block, start=start):
            lines.append(f"[{idx:4d}] {name}")
    return "\n".join(lines)


def apply_feature_diff(
    all_cols: Sequence[str],
    base: Sequence[str] | None = None,
    add: Sequence[str] | None = None,
    remove: Sequence[str] | None = None,
) -> List[str]:
    """Start from `base` (or `all_cols` if None), add and/or remove features.

    Preserves the original order from `all_cols` in the final list.
    Unknown features (not in `all_cols`) are ignored.
    """
    all_set = set(all_cols)
    selected = set(base) if base is not None else set(all_cols)
    if add:
        selected |= {f for f in add if f in all_set}
    if remove:
        selected -= {f for f in remove if f in all_set}
    return [f for f in all_cols if f in selected]

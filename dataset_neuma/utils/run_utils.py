from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def save_run_metadata(args: Any, results_dir: Path, filename: str = "run_metadata.json") -> None:
    """Guarda args y comando ejecutado para trazabilidad."""
    results_dir.mkdir(parents=True, exist_ok=True)
    cmd = " ".join(os.environ.get("CMDLINE", "").split())
    try:
        cmdline = " ".join(os.sys.argv)
    except Exception:
        cmdline = ""
    data: Dict[str, Any] = {
        "cmd_env": cmd,
        "cmd_argv": cmdline,
    }
    # intentar serializar args (argparse Namespace)
    if hasattr(args, "__dict__"):
        cleaned = {}
        for k, v in vars(args).items():
            if isinstance(v, Path):
                cleaned[k] = str(v)
            else:
                cleaned[k] = v
        data["args"] = cleaned
    out = results_dir / filename
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def next_run_dir(base_dir: Path, prefix: str = "run_") -> Path:
    """Crea un directorio incremental run_XXXX."""
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    nums = []
    for name in existing:
        tail = name[len(prefix) :]
        if tail.isdigit():
            nums.append(int(tail))
    next_id = max(nums) + 1 if nums else 1
    run_name = f"{prefix}{next_id:04d}"
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

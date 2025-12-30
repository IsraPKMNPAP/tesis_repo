from __future__ import annotations

from pathlib import Path
from typing import List


def load_features_file(path: str | Path) -> List[str]:
    """Lee un archivo de texto con una columna por línea y devuelve la lista de strings."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe el archivo de columnas: {p}")
    lines = []
    for line in p.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return lines

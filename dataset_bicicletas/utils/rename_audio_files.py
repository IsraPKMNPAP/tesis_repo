#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

ID_PATTERN = re.compile(r"P\d{2}", re.IGNORECASE)
SOURCE_PATTERN = re.compile(r"(?:copia\s+de\s+)?final_audio", re.IGNORECASE)


def infer_participant_token(name: str) -> str | None:
    normalized = name.replace(" ", "")
    match = ID_PATTERN.search(normalized)
    if match:
        return match.group(0).upper()
    digits = re.findall(r"\d+", normalized)
    if not digits:
        return None
    return f"P{int(digits[-1]):02d}"


def rename_files(audio_root: Path, dry_run: bool, overwrite: bool) -> None:
    renamed, skipped = 0, 0
    for wav in sorted(audio_root.rglob("*.wav")):
        if wav.name.lower().startswith("raw_audio_"):
            continue
        name_compact = wav.name.lower().replace(" ", "")
        if not SOURCE_PATTERN.search(name_compact):
            continue
        token = infer_participant_token(wav.name)
        if not token:
            print(f"[skip] No se pudo inferir participante para {wav.name}")
            skipped += 1
            continue
        target = wav.with_name(f"raw_audio_{token}.wav")
        if target.exists() and target != wav and not overwrite:
            print(f"[skip] {target.name} ya existe (usar --overwrite para reemplazarlo)")
            skipped += 1
            continue
        prefix = "[dry-run]" if dry_run else "[rename]"
        print(f"{prefix} {wav.name} -> {target.name}")
        if not dry_run and target != wav:
            wav.replace(target)
        renamed += 1
    print(f"Renombrados: {renamed} | Omitidos: {skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Renombra archivos tipo 'Copia de final_audio_P02.wav' a 'raw_audio_PXX.wav'."
    )
    parser.add_argument("--audio-root", required=True, help="Carpeta con los .wav a renombrar")
    parser.add_argument("--dry-run", action="store_true", help="Solo mostrar acciones sin modificar archivos")
    parser.add_argument("--overwrite", action="store_true", help="Permitir sobrescribir si el destino existe")
    args = parser.parse_args()

    audio_root = Path(args.audio_root)
    if not audio_root.exists():
        raise SystemExit(f"No existe la carpeta {audio_root}")
    rename_files(audio_root, dry_run=args.dry_run, overwrite=args.overwrite)


if __name__ == "__main__":
    main()

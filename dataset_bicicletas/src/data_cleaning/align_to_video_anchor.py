from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import os
import sys

import pandas as pd

# Ensure package root on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p))


def _to_dt(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        raise KeyError(f"No se encontró la columna '{col}' en el DataFrame")
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    if out[col].isna().any():
        n = int(out[col].isna().sum())
        raise ValueError(f"{n} valores inválidos en '{col}' (no convertibles a datetime)")
    return out


def _ensure_col(df: pd.DataFrame, col: str, default) -> pd.DataFrame:
    if col in df.columns:
        return df
    out = df.copy()
    out[col] = default
    return out


def _copy_row_with_new_ts(row: pd.Series, ts_col: str, new_ts) -> pd.Series:
    r = row.copy()
    r[ts_col] = new_ts
    return r


def _ensure_paths_column(
    df_csv: pd.DataFrame,
    raw_csv_path: Optional[Path],
    timestamp_col: str,
    participant_col: str,
    paths_col: str = "paths",
) -> pd.DataFrame:
    """Ensure 'paths' exists; if missing, fetch from raw CSV by (participant, timestamp), then drop nulls."""
    if paths_col in df_csv.columns:
        out = df_csv.copy()
    else:
        if raw_csv_path is None or not raw_csv_path.exists():
            raise FileNotFoundError(
                "No 'paths' in clean CSV and raw CSV not found to import it"
            )
        df_raw = pd.read_csv(raw_csv_path)
        if timestamp_col not in df_raw.columns:
            raise KeyError(
                f"Raw CSV lacks timestamp column '{timestamp_col}' required to merge 'paths'"
            )
        df_raw = _to_dt(df_raw, timestamp_col)
        cols = [participant_col, timestamp_col]
        if paths_col not in df_raw.columns:
            raise KeyError("Raw CSV lacks 'paths' column")
        df_paths = df_raw[cols + [paths_col]].drop_duplicates(cols)
        out = df_csv.merge(df_paths, on=cols, how="left")
    before = len(out)
    out = out[~out[paths_col].isna()].reset_index(drop=True)
    after = len(out)
    print(f"[paths] before={before} after_dropna={after}")
    return out


def fill_missing_windows(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    participant_col: str = "participant",
    imputed_col: str = "is_imputed",
    min_gap: int = 5,
    max_gap: int = 120,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fill temporal gaps per participant by repeating last valid row; returns (df_filled, large_gaps)."""
    df = df.sort_values(by=[participant_col, timestamp_col]).reset_index(drop=True)
    rows: List[pd.Series] = []
    large_gaps: List[Dict] = []

    for pid, sub in df.groupby(participant_col):
        sub = sub.reset_index(drop=True)
        prev_row: Optional[pd.Series] = None
        for _, row in sub.iterrows():
            if prev_row is not None:
                delta = (row[timestamp_col] - prev_row[timestamp_col]).total_seconds()
                if delta > min_gap and delta <= max_gap:
                    n_missing = int(delta // min_gap) - 1
                    for i in range(n_missing):
                        fake = prev_row.copy()
                        fake[timestamp_col] = prev_row[timestamp_col] + pd.Timedelta(seconds=min_gap * (i + 1))
                        fake[imputed_col] = True
                        rows.append(fake)
                elif delta > max_gap:
                    large_gaps.append(
                        {
                            "participant": pid,
                            "prev_timestamp": prev_row[timestamp_col],
                            "curr_timestamp": row[timestamp_col],
                            "delta": delta,
                        }
                    )
            row = row.copy()
            row[imputed_col] = False
            rows.append(row)
            prev_row = row
    df_filled = pd.DataFrame(rows).reset_index(drop=True)
    large_gaps_df = pd.DataFrame(large_gaps)
    return df_filled, large_gaps_df


@dataclass
class AlignReport:
    added_rows: int
    missing_originals: int
    unresolved_imputations: int
    groups_processed: int
    violations_5s: int
    violations_detail: List[Tuple[str, str, int]]  # (participant, session_id, count)
    pickle_impute_mismatch: int


def align_csv_to_anchor(
    df_csv: pd.DataFrame,
    df_anchor: pd.DataFrame,
    timestamp_col: str = "timestamp",
    participant_col: str = "participant",
    session_id_col: str = "session_id",
    imputed_col: str = "is_imputed",
    expected_step_seconds: int = 5,
) -> Tuple[pd.DataFrame, AlignReport]:
    # Ensure datetime
    df_csv = _to_dt(df_csv, timestamp_col)
    df_anchor = _to_dt(df_anchor, timestamp_col)

    # Ensure imputed column present in CSV, default 0
    df_csv = _ensure_col(df_csv, imputed_col, 0)

    # If session_id missing in CSV, try to map from anchor on matching keys
    if session_id_col not in df_csv.columns and session_id_col in df_anchor.columns:
        df_csv = df_csv.merge(
            df_anchor[[participant_col, timestamp_col, session_id_col]].drop_duplicates(
                [participant_col, timestamp_col]
            ),
            on=[participant_col, timestamp_col],
            how="left",
        )

    # Index for fast lookup
    key = [participant_col, session_id_col, timestamp_col]
    if session_id_col not in df_csv.columns:
        # fallback to participant+timestamp if no session_id available
        key = [participant_col, timestamp_col]

    csv_idx = df_csv.set_index(key)
    added_rows: List[pd.Series] = []
    missing_originals = 0
    unresolved_imputations = 0
    pickle_impute_mismatch = 0

    # Validate that imputed rows in anchor look like repetition (except timestamp)
    if imputed_col in df_anchor.columns:
        df_anchor_sorted = df_anchor.sort_values([participant_col, session_id_col, timestamp_col])
        cols_to_compare = [
            c
            for c in df_anchor.columns
            if c
            not in {timestamp_col, imputed_col, "window", "window_id", "gpu_tensor_path", "tensor_path", "idx"}
        ]
        grp = df_anchor_sorted.groupby([participant_col, session_id_col], dropna=False)
        for (_, _), g in grp:
            prev = None
            for _, row in g.iterrows():
                if row.get(imputed_col, 0) in (1, True) and prev is not None:
                    # Compare selected columns
                    if not all((row[c] == prev[c]) or (pd.isna(row[c]) and pd.isna(prev[c])) for c in cols_to_compare):
                        pickle_impute_mismatch += 1
                prev = row

    # Walk anchor timeline and add missing imputed timestamps
    gcols = [participant_col]
    if session_id_col in df_anchor.columns:
        gcols.append(session_id_col)
    df_anchor_sorted = df_anchor.sort_values(gcols + [timestamp_col])
    grp_anchor = df_anchor_sorted.groupby(gcols, dropna=False)

    for gkey, g in grp_anchor:
        # Existing CSV timestamps in this group
        if len(key) == 3:
            # group by participant+session_id
            if isinstance(gkey, tuple):
                pval, sval = gkey
            else:
                pval, sval = gkey, None
            mask = (df_csv[participant_col] == pval) & (df_csv[session_id_col] == sval)
        else:
            # group by participant only
            pval = gkey if not isinstance(gkey, tuple) else gkey[0]
            mask = df_csv[participant_col] == pval
        csv_g = df_csv.loc[mask].sort_values(timestamp_col)
        csv_times = set(csv_g[timestamp_col].tolist())

        # Iterate anchor rows in chronological order
        prev_anchor_ts = None
        for _, arow in g.iterrows():
            ats = arow[timestamp_col]
            is_imp = bool(arow.get(imputed_col, 0))

            if ats in csv_times:
                prev_anchor_ts = ats
                continue

            if not is_imp:
                # An original anchor row is missing in CSV
                missing_originals += 1
                prev_anchor_ts = ats
                continue

            # For imputed anchor timestamp, duplicate previous CSV row in the group
            # Use previous anchor ts to find the source row in CSV
            prev_ts = prev_anchor_ts
            if prev_ts is None:
                unresolved_imputations += 1
                prev_anchor_ts = ats
                continue

            # Find source row
            if len(key) == 3:
                src_key = (pval, sval, prev_ts)
            else:
                src_key = (pval, prev_ts) if isinstance(prev_ts, pd.Timestamp) else (pval, prev_ts)

            try:
                src_row = csv_idx.loc[src_key]
                if isinstance(src_row, pd.DataFrame):
                    # If duplicated index in CSV, take the last
                    src_row = src_row.iloc[-1]
            except KeyError:
                unresolved_imputations += 1
                prev_anchor_ts = ats
                continue

            new_row = _copy_row_with_new_ts(src_row, timestamp_col, ats)
            # Ensure imputed flag = 1
            new_row[imputed_col] = 1
            # Ensure session_id copied from anchor if present
            if session_id_col in df_anchor.columns:
                new_row[session_id_col] = arow.get(session_id_col, new_row.get(session_id_col))
            added_rows.append(new_row)
            csv_times.add(ats)
            prev_anchor_ts = ats

    # Build final DataFrame
    if added_rows:
        to_add_df = pd.DataFrame(added_rows)
        df_out = pd.concat([df_csv, to_add_df], ignore_index=True, axis=0)
    else:
        df_out = df_csv.copy()

    # Sort final
    sort_cols = [participant_col]
    if session_id_col in df_out.columns:
        sort_cols.append(session_id_col)
    sort_cols.append(timestamp_col)
    df_out = df_out.sort_values(sort_cols).reset_index(drop=True)

    # Validate 5-second step per group
    violations_5s = 0
    violations_detail: List[Tuple[str, str, int]] = []
    gcols_out = [participant_col]
    if session_id_col in df_out.columns:
        gcols_out.append(session_id_col)
    for gkey, g in df_out.groupby(gcols_out, dropna=False):
        gg = g.sort_values(timestamp_col)
        deltas = gg[timestamp_col].diff().dropna()
        bad = deltas.dt.total_seconds() != expected_step_seconds
        n_bad = int(bad.sum())
        if n_bad > 0:
            violations_5s += n_bad
            # format key
            if isinstance(gkey, tuple):
                p, s = gkey[0], str(gkey[1])
            else:
                p, s = gkey, "None"
            violations_detail.append((str(p), s, n_bad))

    report = AlignReport(
        added_rows=len(added_rows),
        missing_originals=missing_originals,
        unresolved_imputations=unresolved_imputations,
        groups_processed=len(list(grp_anchor.groups.keys())),
        violations_5s=violations_5s,
        violations_detail=violations_detail,
        pickle_impute_mismatch=pickle_impute_mismatch,
    )
    return df_out, report


def main():
    ap = argparse.ArgumentParser(description="Alinear CSV tabular a la línea de tiempo del pickle (ancla)")
    ap.add_argument("--csv-in", type=str, required=True, help="CSV tabular limpio a alinear")
    ap.add_argument("--pkl-ref", type=str, required=True, help="Pickle ancla con timestamps e is_imputed")
    ap.add_argument("--csv-out", type=str, required=True, help="Salida CSV alineado al pickle")
    ap.add_argument("--timestamp-col", type=str, default="timestamp")
    ap.add_argument("--participant-col", type=str, default="participant")
    ap.add_argument("--session-id-col", type=str, default="session_id")
    ap.add_argument("--imputed-col", type=str, default="is_imputed")
    ap.add_argument("--expected-step-seconds", type=int, default=5)
    ap.add_argument("--raw-csv", type=str, default="data/raw/all_data.csv", help="raw CSV to fetch 'paths' if missing")
    ap.add_argument("--paths-col", type=str, default="paths")
    ap.add_argument("--max-gap", type=int, default=120, help="max gap (s) to fill with imputed windows")
    ap.add_argument("--strict", action="store_true", help="Fallar si hay violaciones o faltantes importantes")
    args = ap.parse_args()

    csv_path = _expand(args.csv_in)
    ref_path = _expand(args.pkl_ref)
    out_path = _expand(args.csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el CSV: {csv_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"No existe el Pickle de referencia: {ref_path}")

    df_csv = pd.read_csv(csv_path)
    df_ref = pd.read_pickle(ref_path)

    print(f"[load] csv.shape={df_csv.shape}")
    print(f"[load] ref.shape={df_ref.shape}")

    # Timestamps to datetime before merges/fills
    df_csv = _to_dt(df_csv, args.timestamp_col)
    df_ref = _to_dt(df_ref, args.timestamp_col)
    print(f"[to_datetime] csv.shape={df_csv.shape}")

    # Ensure 'paths' column and drop nulls
    raw_csv_path = _expand(args.raw_csv) if args.raw_csv else None
    df_csv = _ensure_paths_column(
        df_csv,
        raw_csv_path=raw_csv_path,
        timestamp_col=args.timestamp_col,
        participant_col=args.participant_col,
        paths_col=args.paths_col,
    )
    print(f"[paths_ready] csv.shape={df_csv.shape}")

    # Fill temporal gaps per participant using 5s step up to max_gap
    df_csv_filled, large_gaps_df = fill_missing_windows(
        df_csv,
        timestamp_col=args.timestamp_col,
        participant_col=args.participant_col,
        imputed_col=args.imputed_col,
        min_gap=args.expected_step_seconds,
        max_gap=args.max_gap,
    )
    print(f"[fill_missing_windows] csv.shape={df_csv_filled.shape} | large_gaps={len(large_gaps_df)}")

    df_aligned, rep = align_csv_to_anchor(
        df_csv_filled,
        df_ref,
        timestamp_col=args.timestamp_col,
        participant_col=args.participant_col,
        session_id_col=args.session_id_col,
        imputed_col=args.imputed_col,
        expected_step_seconds=args.expected_step_seconds,
    )
    print(f"[anchor_align] csv.shape={df_aligned.shape}")

    # Report
    print("=== Align Report ===")
    print(f"added_rows: {rep.added_rows}")
    print(f"missing_originals_in_csv: {rep.missing_originals}")
    print(f"unresolved_imputations: {rep.unresolved_imputations}")
    print(f"pickle_impute_mismatch: {rep.pickle_impute_mismatch}")
    print(f"groups_processed: {rep.groups_processed}")
    print(f"violations_5s_total: {rep.violations_5s}")
    if rep.violations_detail:
        print("violations_5s_detail (participant, session_id, count):")
        for p, s, n in rep.violations_detail[:20]:
            print(f"  - {p} | {s} -> {n}")
        if len(rep.violations_detail) > 20:
            print(f"  (y {len(rep.violations_detail)-20} grupos más)")

    if args.strict:
        if rep.missing_originals > 0 or rep.unresolved_imputations > 0 or rep.violations_5s > 0:
            raise SystemExit(2)

    # Save
    df_aligned.to_csv(out_path, index=False)
    print(f"Guardado CSV alineado en: {out_path} | filas: {len(df_aligned)}")


if __name__ == "__main__":
    main()

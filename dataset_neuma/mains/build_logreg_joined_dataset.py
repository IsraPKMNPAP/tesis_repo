from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def normalize_subject(val: str) -> str:
    s = str(val).strip().lower()
    digits = "".join([c for c in s if c.isdigit()])
    if digits:
        return f"s{int(digits):02d}"
    return s


def parse_page(val: str) -> int | None:
    s = str(val).strip().lower()
    digits = "".join([c for c in s if c.isdigit()])
    return int(digits) if digits else None


def parse_product(val: str) -> int | None:
    s = str(val).strip().lower()
    digits = "".join([c for c in s if c.isdigit()])
    return int(digits) if digits else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Join EEG features + multimodal + latente and run logreg.")
    parser.add_argument("--eeg-features", type=Path, default=Path("./data/EDA/eda_results_tabular/eeg_band_features_wide.csv"))
    parser.add_argument("--multimodal", type=Path, default=Path("./data/processed/multimodal_join_with_eeg_emb_aug.csv"))
    parser.add_argument("--latente", type=Path, default=Path("/mnt/otra_particion/home/israel_gpu_data/dataset_neuma/processed/data_latente_estimulo_neuma.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("./data/processed/joined_eeg_multimodal_latente.csv"))
    parser.add_argument("--label-col", type=str, default="bought")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eeg_df = pd.read_csv(args.eeg_features)
    mm_df = pd.read_csv(args.multimodal)
    lat_df = pd.read_csv(args.latente)

    eeg_df.columns = eeg_df.columns.str.lower()
    mm_df.columns = mm_df.columns.str.lower()
    lat_df.columns = lat_df.columns.str.lower()

    # Normalize keys for eeg/mm
    for df in (eeg_df, mm_df):
        df["subject_key"] = df["subject"].apply(normalize_subject)
        df["page_key"] = df["page"].astype(str).str.lower()
        df["product_key"] = df["product_id"].astype(str).str.lower()

    # Build keys for latente (left join base)
    if not {"id_sub", "id_prod"}.issubset(lat_df.columns):
        raise SystemExit("latente CSV must include id_sub and id_prod columns.")
    lat_df["subject_key"] = lat_df["id_sub"].apply(normalize_subject)
    lat_df["page_num"] = ((lat_df["id_prod"] - 1) // 24) + 1
    lat_df["prod_num"] = ((lat_df["id_prod"] - 1) % 24) + 1
    lat_df["page_key"] = lat_df["page_num"].apply(lambda v: f"page{int(v)}")
    lat_df["product_key"] = lat_df["prod_num"].apply(lambda v: f"product{int(v)}")

    # Join multimodal (left join to latente)
    if args.label_col.lower() not in mm_df.columns:
        candidates = [c for c in mm_df.columns if args.label_col.lower() in c]
        if candidates:
            mm_df = mm_df.rename(columns={candidates[0]: args.label_col.lower()})
        else:
            raise SystemExit(f"Label '{args.label_col}' not found in multimodal dataset.")
    mm_label = args.label_col.lower()
    if mm_label not in mm_df.columns:
        candidates = [c for c in mm_df.columns if mm_label in c]
        if candidates:
            mm_df = mm_df.rename(columns={candidates[0]: mm_label})
        else:
            raise SystemExit(f"Label '{args.label_col}' not found in multimodal dataset.")

    mm_cols = [
        "familiarity",
        "frequent_buy",
        "reasons",
        "weeklysupermarketvisits",
        "supermarketvisitduration",
        "priceimpact",
        "brandimpact",
        "discountimpact",
        "advertisementimpact",
        "suggestionimpact",
        "shoppinglist",
        "verbalvisual",
        "spontaneous",
        "varietyseeker",
        "utilitarianmotivation",
        "hedonicmotivation",
        "extraversion",
        "neuroticism",
        "agreeableness",
        "openness",
        "conscientiousness",
        "bargainhunter",
        "age",
        "gender",
        "dominanthand",
        "maritalstatus",
        "children",
        "price",
        "offer",
        "len_med",
        mm_label,
    ]
    mm_cols = [c for c in mm_cols if c in mm_df.columns]
    # Filtrar solo columnas disponibles
    mm_cols = [c for c in mm_cols if c in mm_df.columns]
    mm_small = mm_df[["subject_key", "page_key", "product_key"] + mm_cols].copy()
    merged = lat_df.merge(mm_small, on=["subject_key", "page_key", "product_key"], how="left")
    if mm_label not in merged.columns:
        if f"{mm_label}_y" in merged.columns:
            merged = merged.rename(columns={f"{mm_label}_y": mm_label})
        elif f"{mm_label}_x" in merged.columns:
            merged = merged.rename(columns={f"{mm_label}_x": mm_label})
    if mm_label in merged.columns:
        print("After multimodal join, label present:", True, "non-null:", int(merged[mm_label].notna().sum()))
    else:
        print("After multimodal join, label present:", False, "cols:", merged.columns.tolist())

    # Join EEG features (left join to latente base)
    eeg_drop = {"subject", "page", "product_id", args.label_col.lower(), "subject_key", "page_key", "product_key"}
    eeg_feat_cols = [c for c in eeg_df.columns if c not in eeg_drop]
    eeg_small = eeg_df[["subject_key", "page_key", "product_key"] + eeg_feat_cols].copy()
    merged = merged.merge(eeg_small, on=["subject_key", "page_key", "product_key"], how="left")
    if mm_label in merged.columns:
        print("After EEG join, label present:", True, "non-null:", int(merged[mm_label].notna().sum()))
    else:
        print("After EEG join, label present:", False, "cols:", merged.columns.tolist())

    # Image attributes from latente
    img_cols = [
        "intensidad_media",
        "intensidad_std",
        "promedio_color",
        "saturacion_media",
        "cantidad_bordes",
        "contraste",
        "homogeneidad",
        "relacion_aspecto",
    ]
    img_cols = [c for c in img_cols if c in merged.columns]

    # Build final feature set
    label_col = mm_label

    keep_cols = []
    keep_cols += eeg_feat_cols
    keep_cols += [c for c in mm_cols if c != args.label_col.lower()]
    keep_cols += img_cols
    keep_cols = [c for c in keep_cols if c in merged.columns]
    # asegurar que la etiqueta no quede como feature (cualquier columna con 'bought')
    keep_cols = [c for c in keep_cols if "bought" not in c]

    if label_col not in merged.columns:
        raise SystemExit(f"Label '{label_col}' not found after joins.")

    # Drop rows without label
    merged = merged.dropna(subset=[label_col])
    merged[label_col] = merged[label_col].astype(int)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print(f"Saved joined dataset: {args.out_csv} (rows: {len(merged)})")
    print("Columns used (batch size 10):")
    cols_print = keep_cols + [label_col]
    for i in range(0, len(cols_print), 10):
        batch = cols_print[i : i + 10]
        print(f"  {i:03d}-{i+len(batch)-1:03d}: {batch}")

    # Logistic regression on selected features
    X = merged[keep_cols]
    y = merged[label_col].to_numpy()
    X.drop("label",inplace=True)
    # Identify numeric/categorical
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    clf = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("logreg", LogisticRegression(max_iter=500, class_weight="balanced")),
        ]
    )
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    print(f"AUC: {auc:.4f}")
    print(classification_report(y_test, (prob >= 0.5).astype(int), digits=3))


if __name__ == "__main__":
    main()

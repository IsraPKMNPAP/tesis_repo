from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Ensure package root on path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loading.multimodal import MultimodalDataset, collate_multimodal
from src.models.mm_vae import DeterministicMMVAE, VariationalMMVAE
from src.features.prepare import build_preprocessor
from src.data_cleaning.cleaning import categorias_a_str, convertir_a_categorico
from utils.results_io import (
    ensure_dir,
    save_text,
    save_model_pickle,
    compute_run_hash,
    artifact_name,
    register_run,
)
from utils.features import load_features_file


def _to_float_tensor(mat):
    import numpy as np
    try:
        arr = mat.toarray()
    except Exception:
        try:
            arr = np.asarray(mat)
        except Exception:
            arr = mat
    return torch.tensor(arr.astype(np.float32), dtype=torch.float32)


def split_train_val(df: pd.DataFrame, label_col: str, val_split: float = 0.2, seed: int = 42):
    rng = np.random.RandomState(seed)
    if val_split <= 0 or val_split >= 1:
        return df.reset_index(drop=True), df.iloc[0:0].copy()
    y = pd.to_numeric(df[label_col], errors='coerce') if df[label_col].dtype != object else df[label_col]
    # Stratified split by label when possible
    if y.notna().all():
        labels = y
        uniq = pd.Series(labels).unique()
        val_idx = []
        for c in uniq:
            idx = np.where(labels == c)[0]
            k = int(max(1, round(len(idx) * val_split)))
            val_idx.extend(rng.choice(idx, size=min(k, len(idx)), replace=False))
        val_idx = sorted(set(val_idx))
    else:
        n = len(df)
        k = int(round(n * val_split))
        val_idx = sorted(rng.choice(np.arange(n), size=k, replace=False).tolist())
    mask = np.zeros(len(df), dtype=bool)
    mask[val_idx] = True
    df_val = df.iloc[mask].reset_index(drop=True)
    df_tr = df.iloc[~mask].reset_index(drop=True)
    return df_tr, df_val


def main():
    ap = argparse.ArgumentParser(description="Train Multimodal VAE (tabular + video) end-to-end")
    ap.add_argument("--pkl", type=str, default="data/processed/multimodal_join.pkl", help="Ruta al pickle multimodal")
    ap.add_argument("--label-col", type=str, default="action")
    ap.add_argument("--features", nargs="*", default=None, help="Columnas tabulares a usar")
    ap.add_argument("--features-file", type=str, default=None, help="Archivo con lista de features (json o txt)")
    ap.add_argument("--path-col", type=str, default="gpu_tensor_path")
    ap.add_argument("--timestamp-col", type=str, default="timestamp")
    ap.add_argument("--window-id-col", type=str, default="window")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--deterministic", action="store_true", help="Usar VAE determinista (por defecto, variacional)")
    ap.add_argument("--tab-emb", type=int, default=128)
    ap.add_argument("--shared-dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--w-rec-tab", type=float, default=1.0)
    ap.add_argument("--w-rec-vid", type=float, default=1.0)
    ap.add_argument("--w-cls", type=float, default=1.0)
    ap.add_argument("--w-kl", type=float, default=1.0)
    ap.add_argument("--kl-anneal-steps", type=int, default=1000)
    ap.add_argument("--save-embeddings", action="store_true")
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        raise FileNotFoundError(f"No existe el pickle multimodal: {pkl_path}")
    df = pd.read_pickle(pkl_path).reset_index(drop=True)

    # Resolve features
    tab_cols = args.features
    if args.features_file:
        loaded = load_features_file(args.features_file)
        if loaded:
            tab_cols = loaded
    if not tab_cols:
        # Heuristic: drop known non-tabular columns
        drop_cols = {args.path_col, args.label_col, args.timestamp_col, args.window_id_col, 'participant', 'session_id'}
        tab_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

    # Optional label coercion
    if df[args.label_col].dtype == object:
        classes = sorted(df[args.label_col].dropna().unique().tolist())
        class_to_idx = {c: i for i, c in enumerate(classes)}
        df[args.label_col] = df[args.label_col].map(class_to_idx)
    num_classes = int(pd.Series(df[args.label_col]).nunique())

    # Split
    df_tr, df_val = split_train_val(df, label_col=args.label_col, val_split=args.val_split)

    # Preprocesamiento tabular (StandardScaler + OneHot) similar al pipeline baseline
    X_tr_raw = df_tr[tab_cols].copy()
    X_val_raw = df_val[tab_cols].copy()
    # Convertir objetos a categorías para que OneHotEncoder las procese
    X_tr_prep = convertir_a_categorico(categorias_a_str(X_tr_raw))
    X_val_prep = convertir_a_categorico(categorias_a_str(X_val_raw))
    preprocessor, numeric, categorical = build_preprocessor(X_tr_prep)
    X_tr_mat = preprocessor.fit_transform(X_tr_prep)
    X_val_mat = preprocessor.transform(X_val_prep)

    # Persistir preprocessor para reproducibilidad
    results_dir = Path("results")
    ensure_dir(results_dir)
    tmp_cfg = {
        "pkl": str(pkl_path),
        "features": tab_cols,
        "label_col": args.label_col,
    }
    pre_hash = compute_run_hash(tmp_cfg, sys.argv, model="MMVAE_Preproc")
    save_model_pickle(preprocessor, results_dir / artifact_name("MMVAE", "preprocessor", pre_hash, "pkl"))

    # Datasets / loaders
    ds_tr = MultimodalDataset(
        df_tr,
        tab_columns=tab_cols,
        X_tab_array=_to_float_tensor(X_tr_mat),
        path_col=args.path_col,
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        window_id_col=args.window_id_col,
        prefer_df_label=True,
    )
    ds_val = MultimodalDataset(
        df_val,
        tab_columns=tab_cols,
        X_tab_array=_to_float_tensor(X_val_mat),
        path_col=args.path_col,
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        window_id_col=args.window_id_col,
        prefer_df_label=True,
    )

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_multimodal)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_multimodal)

    # Model
    tab_in_dim = ds_tr.X_tab_array.shape[1]
    video_kwargs = dict(backbone="vit", backbone_name="vit_b_16", backbone_trainable=False, lstm_hidden=256, num_classes=num_classes)
    if args.deterministic:
        model = DeterministicMMVAE(
            tab_in_dim=tab_in_dim,
            tab_emb_dim=args.tab_emb,
            shared_dim=args.shared_dim,
            num_classes=num_classes,
            dropout=args.dropout,
            video_kwargs=video_kwargs,
            classifier_arkoudi=True,
        )
    else:
        model = VariationalMMVAE(
            tab_in_dim=tab_in_dim,
            tab_emb_dim=args.tab_emb,
            shared_dim=args.shared_dim,
            num_classes=num_classes,
            dropout=args.dropout,
            video_kwargs=video_kwargs,
            classifier_arkoudi=True,
            kl_anneal_steps=args.kl_anneal_steps,
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {"loss": [], "acc": []}
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        tr_loss, tr_total, tr_correct = 0.0, 0, 0
        for b in dl_tr:
            x_tab = b.x_tab.to(device)
            x_vid = b.x_vid.to(device)
            y = b.y.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x_tab, x_vid)
            if isinstance(model, VariationalMMVAE):
                loss, logs = model.loss(out, y=y, w_rec_tab=args.w_rec_tab, w_rec_vid=args.w_rec_vid, w_cls=args.w_cls, w_kl=args.w_kl, step=global_step)
            else:
                loss, logs = model.loss(out, y=y, w_rec_tab=args.w_rec_tab, w_rec_vid=args.w_rec_vid, w_cls=args.w_cls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += float(loss.item())
            pred = out["logits"].argmax(dim=1)
            tr_correct += int((pred == y).sum().item())
            tr_total += int(y.numel())
            global_step += 1
        tr_acc = tr_correct / max(1, tr_total)
        history["loss"].append(tr_loss / max(1, len(dl_tr)))
        history["acc"].append(tr_acc)

        # Validation
        model.eval()
        v_total, v_correct = 0, 0
        v_probs = []
        with torch.no_grad():
            for b in dl_val:
                x_tab = b.x_tab.to(device)
                x_vid = b.x_vid.to(device)
                y = b.y.to(device)
                out = model(x_tab, x_vid)
                logits = out["logits"]
                v_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                pred = logits.argmax(dim=1)
                v_correct += int((pred == y).sum().item())
                v_total += int(y.numel())
        val_acc = v_correct / max(1, v_total)
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={history['loss'][-1]:.4f} | train_acc={tr_acc:.3f} | val_acc={val_acc:.3f}")

    # Results
    results_dir = Path("results")
    ensure_dir(results_dir)
    model_name = "MMVAE_Det" if args.deterministic else "MMVAE_Var"
    cfg = {
        "pkl": str(pkl_path),
        "label_col": args.label_col,
        "path_col": args.path_col,
        "features": tab_cols,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "tab_emb": args.tab_emb,
        "shared_dim": args.shared_dim,
        "dropout": args.dropout,
        "w_rec_tab": args.w_rec_tab,
        "w_rec_vid": args.w_rec_vid,
        "w_cls": args.w_cls,
        "w_kl": args.w_kl,
        "kl_anneal_steps": args.kl_anneal_steps,
    }
    run_hash = compute_run_hash(cfg, sys.argv, model=model_name)
    torch.save(model.state_dict(), results_dir / artifact_name(model_name, "model", run_hash, "pt"))
    pd.DataFrame(history).to_csv(results_dir / artifact_name(model_name, "history", run_hash, "csv"), index=False)

    # Validation report + Guardado de embeddings
    if len(df_val) > 0:
        # Re-run validation to gather predictions
        all_true, all_pred, all_probs = [], [], []
        model.eval()
        with torch.no_grad():
            for b in dl_val:
                x_tab = b.x_tab.to(device)
                x_vid = b.x_vid.to(device)
                y = b.y.to(device)
                out = model(x_tab, x_vid)
                logits = out["logits"]
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                pred = logits.argmax(dim=1).cpu().numpy().tolist()
                all_true.extend(y.cpu().numpy().tolist())
                all_pred.extend(pred)
                all_probs.append(probs)
        report = classification_report(all_true, all_pred, zero_division=0)
        print("\n=== Validation (Multimodal VAE) ===")
        print(report)
        save_text(report, results_dir / artifact_name(model_name, "eval_report", run_hash, "txt"))
        if all_probs:
            probs = np.concatenate(all_probs, axis=0)
            pd.DataFrame(probs, columns=[f"class_{i}" for i in range(probs.shape[1])]).to_csv(
                results_dir / artifact_name(model_name, "eval_proba", run_hash, "csv"), index=False
            )

    # Extraer y guardar embeddings finales (para análisis econométrico)
    X_all_raw = df[tab_cols].copy()
    X_all_prep = convertir_a_categorico(categorias_a_str(X_all_raw))
    X_all_mat = preprocessor.transform(X_all_prep)
    ds_all = MultimodalDataset(
        df,
        tab_columns=tab_cols,
        X_tab_array=_to_float_tensor(X_all_mat),
        path_col=args.path_col,
        label_col=args.label_col,
        timestamp_col=args.timestamp_col,
        window_id_col=args.window_id_col,
        prefer_df_label=True,
    )
    dl_all = DataLoader(ds_all, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_multimodal)
    model.eval()
    zs, mus, logvars, ys_all, ts_all, wids_all, parts_all = [], [], [], [], [], [], []
    with torch.no_grad():
        for b in dl_all:
            x_tab = b.x_tab.to(device)
            x_vid = b.x_vid.to(device)
            out = model(x_tab, x_vid)
            z = out["z"].detach().cpu().numpy()
            zs.append(z)
            if "mu" in out and "logvar" in out:
                mus.append(out["mu"].detach().cpu().numpy())
                logvars.append(out["logvar"].detach().cpu().numpy())
            ys_all.extend(b.y.numpy().tolist())
            ts_all.extend(b.timestamp)
            wids_all.extend(b.window_id)
            parts_all.extend(b.participant)
    emb_df = pd.DataFrame(np.concatenate(zs, axis=0), columns=[f"z_{i}" for i in range(np.concatenate(zs, axis=0).shape[1])])
    emb_df["label"] = ys_all
    emb_df["timestamp"] = ts_all
    emb_df["window_id"] = wids_all
    emb_df["participant"] = parts_all
    if mus:
        mu_mat = np.concatenate(mus, axis=0)
        lv_mat = np.concatenate(logvars, axis=0) if logvars else np.zeros_like(mu_mat)
        std_mat = np.exp(0.5 * lv_mat)
        for i in range(mu_mat.shape[1]):
            emb_df[f"mu_{i}"] = mu_mat[:, i]
            emb_df[f"std_{i}"] = std_mat[:, i]
    emb_df.to_csv(results_dir / artifact_name(model_name, "embeddings", run_hash, "csv"), index=False)

    # Save run config
    (results_dir / artifact_name(model_name, "config", run_hash, "json")).write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    register_run(results_dir, run_hash, model_name, cmd=" ".join(sys.argv), config=cfg)


if __name__ == "__main__":
    main()

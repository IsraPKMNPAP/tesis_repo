from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading.multimodal_icl_v import MultimodalICLVDataset, collate_fn
from src.models.multimodal_icl_v import MultimodalICLVDeterministic
from src.models.icl_v import compute_hessian_stats, param_names
from torch.nn.utils import parameters_to_vector
from utils.features import load_features_file
from utils.metrics import classification_metrics, pseudo_r2_mcfadden, save_metrics, summarize_coefs
from utils.run_utils import next_run_dir, save_run_metadata
from utils.splits import split_by_subject_train_val_test, save_split_info


def resolve_cols(df: pd.DataFrame, file_path: str | None, fallback_numeric: bool, drop_cols: set) -> List[str]:
    if file_path:
        cols = [c.strip().lower() for c in load_features_file(file_path)]
    else:
        cols = []
    if not cols and fallback_numeric:
        cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if c in df.columns]
    return cols


def preprocess_block(train_df: pd.DataFrame, val_df: pd.DataFrame, cols: List[str], prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    import pandas.api.types as ptypes

    num_cols = [c for c in cols if ptypes.is_numeric_dtype(train_df[c])]
    cat_cols = [c for c in cols if c not in num_cols]

    out_tr_parts = []
    out_val_parts = []
    new_names = []

    if num_cols:
        means = train_df[num_cols].mean()
        stds = train_df[num_cols].std().replace(0, 1)
        tr_num = (train_df[num_cols] - means) / stds
        val_num = (val_df[num_cols] - means) / stds
        new_names.extend([f"{prefix}{c}" for c in num_cols])
        tr_num.columns = new_names[: len(num_cols)]
        val_num.columns = new_names[: len(num_cols)]
        out_tr_parts.append(tr_num)
        out_val_parts.append(val_num)

    if cat_cols:
        tr_cat = pd.get_dummies(train_df[cat_cols].astype(str), prefix=[f"{prefix}{c}" for c in cat_cols])
        val_cat = pd.get_dummies(val_df[cat_cols].astype(str), prefix=[f"{prefix}{c}" for c in cat_cols])
        tr_cols = tr_cat.columns
        val_cat = val_cat.reindex(columns=tr_cols, fill_value=0)
        new_names.extend(tr_cols.tolist())
        out_tr_parts.append(tr_cat)
        out_val_parts.append(val_cat)

    if out_tr_parts:
        tr_block = pd.concat(out_tr_parts, axis=1)
        val_block = pd.concat(out_val_parts, axis=1)
    else:
        tr_block = pd.DataFrame(index=train_df.index)
        val_block = pd.DataFrame(index=val_df.index)
    return tr_block, val_block, new_names


def run_epoch(model, loader, device, train=True, optimizer=None):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    total_choice = 0.0
    total_meas = 0.0
    total_ll = 0.0
    y_true_all = []
    y_prob_all = []
    total = 0
    with torch.set_grad_enabled(train):
        for obs_lt, obs_u, eeg_emb, img_emb, choice in loader:
            obs_lt = obs_lt.to(device)
            obs_u = obs_u.to(device)
            eeg_emb = eeg_emb.to(device)
            img_emb = img_emb.to(device)
            choice_t = choice.to(device)

            out = model(obs_lt, obs_u, eeg_emb, img_emb, choice_t)
            loss = out["loss"]
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * obs_lt.size(0)
            total_choice += float(out["loss_choice"].item()) * obs_lt.size(0)
            total_meas += float(out["loss_meas"].item()) * obs_lt.size(0)
            total_ll += float(out["log_likelihood"].item())
            prob1 = torch.exp(out["logp"][:, 1]) if out["logp"].shape[1] > 1 else torch.exp(out["logp"][:, 0])
            y_true_all.append(choice_t.detach().cpu().numpy())
            y_prob_all.append(prob1.detach().cpu().numpy())
            total += obs_lt.size(0)

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_prob = np.concatenate(y_prob_all) if y_prob_all else np.array([])
    return {
        "loss": total_loss / max(1, total),
        "loss_choice": total_choice / max(1, total),
        "loss_meas": total_meas / max(1, total),
        "log_likelihood": total_ll,
        "n": total,
        "y_true": y_true,
        "y_prob": y_prob,
    }


def main():
    parser = argparse.ArgumentParser(description="ICLV multimodal (tab + img_emb + EEG_emb como indicador).")
    parser.add_argument("--data", type=Path, default=Path("./data/processed/multimodal_join_with_eeg_emb.csv"))
    parser.add_argument("--label-col", type=str, default="bought")
    parser.add_argument("--obs-lt-cols", type=str, default="./utils/columns/iclv_multimodal/obs_lt.txt")
    parser.add_argument("--obs-u-cols", type=str, default="./utils/columns/iclv_multimodal/obs_u.txt")
    parser.add_argument("--img-emb-col", type=str, default="embedding_path")
    parser.add_argument("--eeg-emb-col", type=str, default="eeg_emb_path")
    parser.add_argument("--num-choices", type=int, default=2)
    parser.add_argument("--n-latent", type=int, default=3)
    parser.add_argument("--img-proj-dim", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta-per-alt", action="store_true", help="Usar betas distintos por alternativa.")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=Path("./results/multimodal_icl_v"))
    parser.add_argument("--skip-hessian", action="store_true", help="No calcular Hessiano (ahorra memoria).")
    parser.add_argument("--hessian-max-params", type=int, default=2000, help="Maximo de parametros para Hessiano completo.")
    parser.add_argument("--hessian-device", type=str, default="cpu", help="Dispositivo para Hessiano: cpu o cuda.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.lower()
    label_col = args.label_col.lower()
    img_emb_col = args.img_emb_col.lower()
    eeg_emb_col = args.eeg_emb_col.lower()
    if "subject" not in df.columns and "id_sub" in df.columns:
        df["subject"] = df["id_sub"].astype(str)
    if "subject" not in df.columns:
        raise ValueError("Se requiere columna 'subject' para split por sujeto.")

    # imputaciones clave
    cat_impute = ["gender", "maritalstatus", "supermarketvisitduration", "shoppinglist", "offer"]
    num_impute = ["price", "len_med"]
    for c in cat_impute:
        if c in df.columns:
            mode = df[c].mode(dropna=True)
            fill_val = mode.iloc[0] if len(mode) else "missing"
            df[c] = df[c].fillna(fill_val)
    for c in num_impute:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    df = df.dropna(subset=[label_col, img_emb_col, eeg_emb_col])

    drop_cols = {label_col}
    orig_obs_lt_cols = resolve_cols(df, args.obs_lt_cols, fallback_numeric=False, drop_cols=drop_cols)
    orig_obs_u_cols = resolve_cols(df, args.obs_u_cols, fallback_numeric=True, drop_cols=drop_cols)

    train_df, val_df, test_df, split_info = split_by_subject_train_val_test(
        df, subject_col="subject", val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
    )
    print(
        f"[split] subjects={split_info['n_subjects']} train={split_info['n_train_subjects']} "
        f"val={split_info['n_val_subjects']} test={split_info['n_test_subjects']} | "
        f"rows train={split_info['train_rows']} val={split_info['val_rows']} test={split_info['test_rows']}"
    )

    # one-hot + estandarizar
    lt_tr, lt_val, lt_names = preprocess_block(train_df, val_df, orig_obs_lt_cols, prefix="lt_")
    u_tr, u_val, u_names = preprocess_block(train_df, val_df, orig_obs_u_cols, prefix="u_")
    train_df = train_df.join(lt_tr).join(u_tr)
    val_df = val_df.join(lt_val).join(u_val)
    obs_lt_cols = lt_names
    obs_u_cols = u_names

    # aplicar mismas columnas a test
    lt_tr2, lt_te, _ = preprocess_block(train_df, test_df, orig_obs_lt_cols, prefix="lt_")
    u_tr2, u_te, _ = preprocess_block(train_df, test_df, orig_obs_u_cols, prefix="u_")
    test_df = test_df.join(lt_te).join(u_te)

    train_ds = MultimodalICLVDataset(train_df, obs_lt_cols, obs_u_cols, label_col, img_emb_col, eeg_emb_col, num_choices=args.num_choices)
    val_ds = MultimodalICLVDataset(val_df, obs_lt_cols, obs_u_cols, label_col, img_emb_col, eeg_emb_col, num_choices=args.num_choices)
    test_ds = MultimodalICLVDataset(test_df, obs_lt_cols, obs_u_cols, label_col, img_emb_col, eeg_emb_col, num_choices=args.num_choices)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    sample = train_ds[0]
    dim_obs_lt = sample[0].shape[-1]
    dim_obs_u = sample[1].shape[-1]
    dim_eeg_emb = sample[2].shape[-1]
    dim_img_emb = sample[3].shape[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalICLVDeterministic(
        dim_obs_lt=dim_obs_lt,
        dim_obs_u=dim_obs_u,
        dim_img_emb=dim_img_emb,
        dim_eeg_emb=dim_eeg_emb,
        n_latent=args.n_latent,
        n_choices=args.num_choices,
        alpha=args.alpha,
        img_proj_dim=args.img_proj_dim,
        beta_per_alt=args.beta_per_alt,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, train=True, optimizer=optim)
        val = run_epoch(model, val_loader, device, train=False)
        tr_cls = classification_metrics(tr["y_true"], tr["y_prob"])
        val_cls = classification_metrics(val["y_true"], val["y_prob"])
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"tr_loss={tr['loss']:.4f} tr_acc={tr_cls['acc']:.3f} tr_f1={tr_cls['f1_macro']:.3f} "
            f"val_loss={val['loss']:.4f} val_acc={val_cls['acc']:.3f} val_f1={val_cls['f1_macro']:.3f}"
        )

    tr = run_epoch(model, train_loader, device, train=False)
    val = run_epoch(model, val_loader, device, train=False)
    te = run_epoch(model, test_loader, device, train=False)

    # Hessiano (opcional y reducido)
    hess = None
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not args.skip_hessian and param_count <= args.hessian_max_params:
        h_dev = torch.device(args.hessian_device)
        model_h = model.to(h_dev)

        def loss_closure():
            out = []
            for obs_lt, obs_u, eeg_emb, img_emb, choice in train_loader:
                obs_lt = obs_lt.to(h_dev)
                obs_u = obs_u.to(h_dev)
                eeg_emb = eeg_emb.to(h_dev)
                img_emb = img_emb.to(h_dev)
                choice_t = choice.to(h_dev)
                o = model_h(obs_lt, obs_u, eeg_emb, img_emb, choice_t)
                out.append(o["loss"])
            return torch.stack(out).mean()

        hess = compute_hessian_stats(model_h, loss_closure)

    run_dir = next_run_dir(args.results_dir)
    torch.save(model.state_dict(), run_dir / "model_last.pt")
    save_split_info(split_info, run_dir)
    save_run_metadata(args, run_dir)

    def pack_iclv_metrics(m):
        cls = classification_metrics(m["y_true"], m["y_prob"])
        nll = -m["log_likelihood"] / max(1, m["n"])
        return {
            **cls,
            "mean_nll": float(nll),
            "log_likelihood": float(m["log_likelihood"]),
            "pseudo_r2": float(pseudo_r2_mcfadden(m["log_likelihood"], m["y_true"], args.num_choices)),
        }

    names = param_names(model)
    theta = parameters_to_vector([p for p in model.parameters() if p.requires_grad]).detach().cpu()
    coef_summary = summarize_coefs(hess.names, hess.theta, hess.std, top_k=5) if hess is not None else summarize_coefs(names, theta, None, top_k=5)

    metrics = {
        "train": pack_iclv_metrics(tr),
        "val": pack_iclv_metrics(val),
        "test": pack_iclv_metrics(te),
        "obs_lt_cols": obs_lt_cols,
        "obs_u_cols": obs_u_cols,
        "img_emb_col": img_emb_col,
        "eeg_emb_col": eeg_emb_col,
        "coef_summary": coef_summary,
        "hessian_skipped": bool(hess is None),
        "hessian_param_count": int(param_count),
    }
    save_metrics(metrics, run_dir)

    if hess is not None:
        with open(run_dir / "hessian.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "theta": hess.theta.tolist(),
                    "std": hess.std.tolist(),
                    "tstat": hess.tstat.tolist(),
                    "names": hess.names,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    print(f"Guardado en {run_dir}")


if __name__ == "__main__":
    main()

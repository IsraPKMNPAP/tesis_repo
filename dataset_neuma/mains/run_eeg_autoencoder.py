from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loading.eeg_autoencoder import EEGAutoencoderDataset
from src.models.eeg_autoencoder import EEGAutoencoder


def collate(batch):
    xs, paths = zip(*batch)
    return torch.stack(xs, dim=0), paths


def main():
    parser = argparse.ArgumentParser(description="Autoencoder para EEG concatenado y extracción de embeddings.")
    parser.add_argument("--data", type=Path, default=Path("./data/processed/multimodal_join.csv"), help="CSV con columna eeg_concat_path.")
    parser.add_argument("--results-dir", type=Path, default=Path("./results/eeg_autoencoder"))
    parser.add_argument("--eeg-len", type=int, default=2048)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", type=Path, default=Path("./data/processed/multimodal_join_with_eeg_emb.csv"))
    parser.add_argument("--emb-dir", type=Path, default=Path("./data/processed/eeg_embeddings"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import pandas as pd

    df = pd.read_csv(args.data)
    if "eeg_concat_path" not in df.columns:
        raise ValueError("Se requiere columna eeg_concat_path en el CSV.")
    paths = df["eeg_concat_path"].dropna().unique().tolist()
    dataset = EEGAutoencoderDataset(paths, eeg_len=args.eeg_len)

    n_total = len(dataset)
    n_val = int(n_total * args.val_split)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # infer channels
    sample_x, _ = dataset[0]
    in_ch = sample_x.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGAutoencoder(in_channels=in_ch, eeg_len=args.eeg_len, emb_dim=args.emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()

    def run_epoch(loader, train=True):
        if train:
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        total = 0
        with torch.set_grad_enabled(train):
            for x, _ in loader:
                x = x.to(device)
                x_hat, _ = model(x)
                loss = criterion(x_hat, x)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                total_loss += loss.item() * x.size(0)
                total += x.size(0)
        return total_loss / max(1, total)

    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(train_loader, train=True)
        val_loss = run_epoch(val_loader, train=False)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={tr_loss:.4f} val_loss={val_loss:.4f}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.results_dir / "eeg_autoencoder.pt")

    # Extraer embeddings para todas las filas y guardar
    args.emb_dir.mkdir(parents=True, exist_ok=True)
    emb_paths = []
    model.eval()
    with torch.no_grad():
        for path in df["eeg_concat_path"]:
            if pd.isna(path):
                emb_paths.append("")
                continue
            arr = np.load(path)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            x = torch.tensor(arr, dtype=torch.float32)
            if x.shape[1] > args.eeg_len:
                x = x[:, : args.eeg_len]
            elif x.shape[1] < args.eeg_len:
                pad = args.eeg_len - x.shape[1]
                x = torch.cat([x, torch.zeros((x.shape[0], pad), dtype=x.dtype)], dim=1)
            x = x.unsqueeze(0).to(device)
            _, z = model(x)
            z_np = z.squeeze(0).cpu().numpy()
            out_path = args.emb_dir / (Path(path).stem + "_emb.npy")
            np.save(out_path, z_np)
            emb_paths.append(str(out_path))

    df_out = df.copy()
    df_out["eeg_emb_path"] = emb_paths
    df_out.to_csv(args.output_csv, index=False)

    with open(args.results_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data": str(args.data),
                "output_csv": str(args.output_csv),
                "emb_dir": str(args.emb_dir),
                "eeg_len": args.eeg_len,
                "emb_dim": args.emb_dim,
                "train_loss": tr_loss,
                "val_loss": val_loss,
            },
            f,
            indent=2,
        )
    print(f"Embeddings guardados en {args.output_csv}")


if __name__ == "__main__":
    main()

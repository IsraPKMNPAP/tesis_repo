from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 3, emb_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        e = self.proj(h)
        e = F.normalize(e, p=2, dim=-1)
        return e


class ArkoudiHead(nn.Module):
    """Arkoudi-style classifier: class embeddings as parameters; logits = z @ E^T.

    - No bias.
    - Optionally normalize embeddings to compute cosine-similarity-like logits.
    """

    def __init__(self, emb_dim: int, num_classes: int, normalize: bool = True):
        super().__init__()
        self.class_emb = nn.Parameter(torch.randn(num_classes, emb_dim) * 0.02)
        self.normalize = normalize

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            z = F.normalize(z, p=2, dim=-1)
            w = F.normalize(self.class_emb, p=2, dim=-1)
        else:
            w = self.class_emb
        logits = z @ w.t()
        return logits


class VideoCNNLSTM(nn.Module):
    """CNN per frame + LSTM over time windows.

    Accepts input as:
      - [B, T, C, H, W]  or  [T, C, H, W] (single sample mode)
      - If [B, C, H, W] (no time), treats T=1.
    """

    def __init__(
        self,
        in_channels: int = 3,
        cnn_emb_dim: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = False,
        num_classes: int = 3,
        arkoudi: bool = False,
        arkoudi_normalize: bool = True,
    ):
        super().__init__()
        self.cnn = SmallCNN(in_channels=in_channels, emb_dim=cnn_emb_dim)
        self.lstm = nn.LSTM(
            input_size=cnn_emb_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        head_in = lstm_hidden * (2 if bidirectional else 1)
        if arkoudi:
            self.head = ArkoudiHead(head_in, num_classes, normalize=arkoudi_normalize)
        else:
            self.head = nn.Linear(head_in, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Normalize input shapes
        if x.dim() == 4:  # [T, C, H, W] (single sample) or [B, C, H, W]
            if x.size(1) in (1, 3) and x.size(0) > 8:
                # assume [T, C, H, W]
                x = x.unsqueeze(0)  # [1, T, C, H, W]
            else:
                # [B, C, H, W] -> add T=1
                x = x.unsqueeze(1)
        elif x.dim() == 3:  # [C, H, W]
            x = x.unsqueeze(0).unsqueeze(1)  # [1, 1, C, H, W]
        # else expect [B, T, C, H, W]

        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        f = self.cnn(x)  # [B*T, cnn_emb_dim]
        f = f.view(B, T, -1)
        out, (hn, cn) = self.lstm(f)  # out: [B, T, H]
        # Take last time-step hidden state
        z = out[:, -1, :]
        logits = self.head(z)
        return logits, z  # also return embedding for extraction


@dataclass
class TrainHistory:
    epochs: int
    losses: List[float]
    accs: List[float]


def train_gpu(
    model: nn.Module,
    train_loader,
    val_loader=None,
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    amp: bool = True,
    device: Optional[torch.device] = None,
):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_val_acc = -1.0
    best_state = None
    tr_losses: List[float] = []
    tr_accs: List[float] = []

    for epoch in range(epochs):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for batch in train_loader:
            x = batch.x.to(device)
            y = torch.tensor(batch.y, device=device) if not isinstance(batch.y, torch.Tensor) else batch.y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits, _ = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item())
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
        tr_losses.append(running_loss / max(1, len(train_loader)))
        tr_accs.append(correct / max(1, total))

        # Optional validation
        if val_loader is not None:
            model.eval()
            v_total, v_correct = 0, 0
            with torch.no_grad():
                for vb in val_loader:
                    vx = vb.x.to(device)
                    vy = torch.tensor(vb.y, device=device) if not isinstance(vb.y, torch.Tensor) else vb.y.to(device)
                    v_logits, _ = model(vx)
                    v_pred = v_logits.argmax(dim=1)
                    v_correct += int((v_pred == vy).sum().item())
                    v_total += int(vy.numel())
            v_acc = v_correct / max(1, v_total)
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, TrainHistory(epochs=epochs, losses=tr_losses, accs=tr_accs)


@torch.no_grad()
def extract_embeddings(model: nn.Module, loader, device: Optional[torch.device] = None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)
    model.eval()
    all_embs: List[torch.Tensor] = []
    all_labels: List[int] = []
    all_meta: List[Tuple[Optional[str], Optional[int], Optional[str]]] = []
    for b in loader:
        x = b.x.to(device)
        logits, z = model(x)
        all_embs.append(z.detach().cpu())
        # Labels may be None for some splits
        if b.y is not None:
            if isinstance(b.y, torch.Tensor):
                all_labels.extend(b.y.detach().cpu().tolist())
            else:
                all_labels.extend([int(b.y)])
        else:
            all_labels.extend([None] * z.size(0))
        # metadata
        ts = b.timestamp if isinstance(b.timestamp, list) else [b.timestamp] * z.size(0)
        wid = b.window_id if isinstance(b.window_id, list) else [b.window_id] * z.size(0)
        part = b.participant if hasattr(b, 'participant') and isinstance(b.participant, list) else [getattr(b, 'participant', None)] * z.size(0)
        for i in range(z.size(0)):
            all_meta.append((
                ts[i] if isinstance(ts, list) else b.timestamp,
                wid[i] if isinstance(wid, list) else b.window_id,
                part[i] if isinstance(part, list) else getattr(b, 'participant', None)
            ))

    embs = torch.cat(all_embs, dim=0).numpy()
    return embs, all_labels, all_meta

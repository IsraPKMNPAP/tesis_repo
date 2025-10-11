from __future__ import annotations

from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim


class ArkoudiStyleLogit(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.emb = nn.Linear(input_dim, num_classes, bias=False)

    def forward(self, x):
        return self.emb(x)


@dataclass
class TrainHistory:
    epochs: int
    losses: List[float]
    accs: List[float]


def train_simple(
    X_train_tensor: torch.Tensor,
    y_train_tensor: torch.Tensor,
    input_dim: int,
    num_classes: int,
    lr: float = 0.01,
    epochs: int = 150,
):
    model = ArkoudiStyleLogit(input_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses, accs = [], []
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train_tensor)
        loss = criterion(logits, y_train_tensor)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            acc = (logits.argmax(1) == y_train_tensor).float().mean().item()
        losses.append(loss.item())
        accs.append(acc)
    return model, TrainHistory(epochs=epochs, losses=losses, accs=accs)


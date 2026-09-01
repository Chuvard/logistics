"""PyTorch model definitions and training loop.

This module is imported lazily and only used when ``torch`` is installed. It
holds the five architectures the brief asks for:

* **MLP** - the tabular workhorse baseline.
* **Autoencoder** - unsupervised reconstruction; reconstruction error doubles
  as an anomaly score, which is directly comparable to Isolation Forest.
* **LSTM** - treats each delivery's feature vector as a short sequence of
  feature groups, so recurrence has something to chew on. For true sequence
  modelling, feed it the GPS trace instead (see ``make_sequences``).
* **Transformer** - self-attention over feature tokens, in the spirit of
  FT-Transformer: every feature becomes a token with a learned embedding.
* **TabNet** - a faithful compact implementation of sequential attention with
  sparse feature selection over decision steps.

Nothing here executes unless torch is present; ``deep.py`` handles the fallback.
"""

from __future__ import annotations

import numpy as np

from ..utils import get_logger, optional_import

logger = get_logger()

torch = optional_import("torch")

__all__ = [
    "available", "MLP", "AutoEncoder", "LSTMNet", "TabTransformer", "TabNet",
    "train_supervised", "train_autoencoder", "make_sequences",
]


def available() -> bool:
    return torch is not None


if torch is not None:
    import torch.nn as nn

    # ----------------------------------------------------------------- MLP --
    class MLP(nn.Module):
        """Plain feed-forward net with batch norm and dropout."""

        def __init__(self, n_features: int, n_outputs: int,
                     hidden: list[int] | None = None, dropout: float = 0.2):
            super().__init__()
            hidden = hidden or [256, 128, 64]
            layers: list[nn.Module] = []
            prev = n_features
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
                prev = h
            layers.append(nn.Linear(prev, n_outputs))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    # --------------------------------------------------------- Autoencoder --
    class AutoEncoder(nn.Module):
        """Symmetric encoder/decoder. The latent code is usable as a compressed
        feature set; the reconstruction error is an anomaly score."""

        def __init__(self, n_features: int, hidden: list[int] | None = None, latent: int = 16):
            super().__init__()
            hidden = hidden or [128, 64]
            enc: list[nn.Module] = []
            prev = n_features
            for h in hidden:
                enc += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
                prev = h
            enc.append(nn.Linear(prev, latent))
            self.encoder = nn.Sequential(*enc)

            dec: list[nn.Module] = []
            prev = latent
            for h in reversed(hidden):
                dec += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
                prev = h
            dec.append(nn.Linear(prev, n_features))
            self.decoder = nn.Sequential(*dec)

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z), z

    # ---------------------------------------------------------------- LSTM --
    class LSTMNet(nn.Module):
        """Recurrent model over a sequence of feature-group tokens."""

        def __init__(self, n_features_per_step: int, n_outputs: int,
                     hidden: int = 64, layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.lstm = nn.LSTM(n_features_per_step, hidden, num_layers=layers,
                                batch_first=True, dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, n_outputs))

        def forward(self, x):                      # x: (batch, seq, features)
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])        # last hidden state

    # --------------------------------------------------------- Transformer --
    class TabTransformer(nn.Module):
        """Self-attention over feature tokens (FT-Transformer style).

        Each scalar feature is projected to a d_model embedding and treated as a
        token, so attention learns which features interact with which.
        """

        def __init__(self, n_features: int, n_outputs: int, d_model: int = 64,
                     heads: int = 4, layers: int = 2, dropout: float = 0.1):
            super().__init__()
            self.feature_embed = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
            self.feature_bias = nn.Parameter(torch.zeros(n_features, d_model))
            self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=heads, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_outputs))

        def forward(self, x):                      # x: (batch, n_features)
            tokens = x.unsqueeze(-1) * self.feature_embed + self.feature_bias
            cls = self.cls.expand(x.size(0), -1, -1)
            h = self.encoder(torch.cat([cls, tokens], dim=1))
            return self.head(h[:, 0])              # read out the CLS token

    # -------------------------------------------------------------- TabNet --
    class _GLUBlock(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.fc = nn.Linear(in_dim, out_dim * 2, bias=False)
            self.bn = nn.BatchNorm1d(out_dim * 2)

        def forward(self, x):
            h = self.bn(self.fc(x))
            a, b = h.chunk(2, dim=-1)
            return a * torch.sigmoid(b)

    def _sparsemax(logits, dim: int = -1):
        """Sparsemax - like softmax but produces exact zeros, which is what
        gives TabNet its interpretable hard feature selection."""
        srt, _ = torch.sort(logits, dim=dim, descending=True)
        cum = srt.cumsum(dim) - 1
        k = torch.arange(1, logits.size(dim) + 1, device=logits.device,
                         dtype=logits.dtype).view(*([1] * (logits.dim() - 1)), -1)
        support = (srt - cum / k) > 0
        k_sup = support.sum(dim=dim, keepdim=True).clamp(min=1)
        tau = cum.gather(dim, k_sup - 1) / k_sup.to(logits.dtype)
        return torch.clamp(logits - tau, min=0)

    class TabNet(nn.Module):
        """Sequential attention over decision steps with sparse masks.

        Each step picks a sparse subset of features (via sparsemax), processes
        them, and contributes to the output. The accumulated masks are directly
        interpretable as feature importance.
        """

        def __init__(self, n_features: int, n_outputs: int, n_d: int = 16,
                     n_a: int = 16, n_steps: int = 3, gamma: float = 1.3):
            super().__init__()
            self.n_steps, self.n_d, self.n_a, self.gamma = n_steps, n_d, n_a, gamma
            self.bn = nn.BatchNorm1d(n_features)
            self.shared = _GLUBlock(n_features, n_d + n_a)
            self.steps = nn.ModuleList([_GLUBlock(n_features, n_d + n_a) for _ in range(n_steps)])
            self.attention = nn.ModuleList([nn.Linear(n_a, n_features, bias=False)
                                            for _ in range(n_steps)])
            self.head = nn.Linear(n_d, n_outputs)

        def forward(self, x, return_masks: bool = False):
            x = self.bn(x)
            prior = torch.ones_like(x)
            out = torch.zeros(x.size(0), self.n_d, device=x.device)
            a = self.shared(x)[:, self.n_d:]
            masks = []
            for step in range(self.n_steps):
                mask = _sparsemax(self.attention[step](a) * prior)
                prior = prior * (self.gamma - mask)          # discourage reuse
                h = self.steps[step](x * mask)
                d, a = h[:, :self.n_d], h[:, self.n_d:]
                out = out + torch.relu(d)
                masks.append(mask)
            logits = self.head(out)
            return (logits, torch.stack(masks, 1)) if return_masks else logits

    # ------------------------------------------------------------ training --
    def _loader(X, y, batch_size, shuffle):
        from torch.utils.data import DataLoader, TensorDataset
        tensors = [torch.tensor(X, dtype=torch.float32)]
        if y is not None:
            tensors.append(torch.tensor(y))
        return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)

    def train_supervised(model, X_train, y_train, X_val, y_val, task_type: str,
                         epochs: int = 30, batch_size: int = 512, lr: float = 1e-3,
                         patience: int = 5, seq: bool = False) -> dict:
        """Train with AdamW, cosine schedule and early stopping on val loss."""
        device = torch.device("cpu")
        model = model.to(device)

        if task_type == "regression":
            loss_fn = nn.MSELoss()
            y_tr = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
            y_va = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
        else:
            loss_fn = nn.CrossEntropyLoss()
            y_tr = np.asarray(y_train, dtype=np.int64)
            y_va = np.asarray(y_val, dtype=np.int64)

        train_dl = _loader(X_train, y_tr, batch_size, True)
        val_dl = _loader(X_val, y_va, batch_size, False)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

        best, best_state, bad, history = np.inf, None, 0, []
        for epoch in range(epochs):
            model.train()
            total = 0.0
            for xb, yb in train_dl:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                total += float(loss) * len(xb)
            sched.step()

            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    val_total += float(loss_fn(model(xb), yb)) * len(xb)
            train_loss = total / len(train_dl.dataset)
            val_loss = val_total / len(val_dl.dataset)
            history.append({"epoch": epoch + 1, "train_loss": round(train_loss, 6),
                            "val_loss": round(val_loss, 6)})

            if val_loss < best - 1e-6:
                best, bad = val_loss, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return {"history": history, "best_val_loss": round(float(best), 6),
                "epochs_run": len(history)}

    def train_autoencoder(model, X_train, X_val, epochs: int = 30,
                          batch_size: int = 512, lr: float = 1e-3, patience: int = 5) -> dict:
        device = torch.device("cpu")
        model = model.to(device)
        loss_fn = nn.MSELoss()
        train_dl = _loader(X_train, None, batch_size, True)
        val_dl = _loader(X_val, None, batch_size, False)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

        best, best_state, bad, history = np.inf, None, 0, []
        for epoch in range(epochs):
            model.train()
            total = 0.0
            for (xb,) in train_dl:
                opt.zero_grad()
                recon, _ = model(xb)
                loss = loss_fn(recon, xb)
                loss.backward()
                opt.step()
                total += float(loss) * len(xb)

            model.eval()
            val_total = 0.0
            with torch.no_grad():
                for (xb,) in val_dl:
                    recon, _ = model(xb)
                    val_total += float(loss_fn(recon, xb)) * len(xb)
            val_loss = val_total / len(val_dl.dataset)
            history.append({"epoch": epoch + 1,
                            "train_loss": round(total / len(train_dl.dataset), 6),
                            "val_loss": round(val_loss, 6)})
            if val_loss < best - 1e-7:
                best, bad = val_loss, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return {"history": history, "best_val_loss": round(float(best), 6),
                "epochs_run": len(history)}

else:  # torch missing - export placeholders so imports never explode
    MLP = AutoEncoder = LSTMNet = TabTransformer = TabNet = None  # type: ignore
    train_supervised = train_autoencoder = None  # type: ignore


def make_sequences(X: np.ndarray, sequence_length: int) -> np.ndarray:
    """Reshape a flat feature vector into ``(batch, sequence_length, features)``.

    Tabular rows are not sequences, so we chunk each row's features into
    `sequence_length` groups. This gives the recurrent and attention models a
    genuine sequence axis to operate over while keeping the comparison against
    the tabular models fair - same information, different inductive bias.
    """
    n, f = X.shape
    per_step = int(np.ceil(f / sequence_length))
    padded = np.zeros((n, per_step * sequence_length), dtype=np.float32)
    padded[:, :f] = X
    return padded.reshape(n, sequence_length, per_step)

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Trainer:
    model: object
    loss_fn: object
    optimizer: object
    history: list[float] = field(default_factory=list)

    def fit(self, X, y=None, epochs: int = 1, batch_size: int | None = None, shuffle: bool = True, on_epoch_end=None):
        for _ in range(epochs):
            if y is None and hasattr(X, "__iter__"):
                batches = list(X)
            elif batch_size is None:
                batches = [(X, y)]
            else:
                indices = torch.randperm(X.shape[0]) if shuffle else torch.arange(X.shape[0])
                batches = [
                    (
                        X[indices[start:start + batch_size]],
                        y[indices[start:start + batch_size]],
                    )
                    for start in range(0, X.shape[0], batch_size)
                ]

            epoch_losses = []
            for X_batch, y_batch in batches:
                predictions = self.model.forward(X_batch)
                loss = self.loss_fn.forward(predictions, y_batch)
                grad = self.loss_fn.backward(predictions, y_batch)
                self.model.backward(grad)
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_value = float(loss.detach().cpu()) if hasattr(loss, "detach") else float(loss)
                epoch_losses.append(loss_value)

            mean_loss = sum(epoch_losses) / len(epoch_losses)
            self.history.append(mean_loss)
            if on_epoch_end is not None:
                on_epoch_end(mean_loss)
        return self.history

"""Lightning callbacks used by train.py."""
from __future__ import annotations

import time

import pytorch_lightning as pl
import torch


class StepTimer(pl.Callback):
    """Log per-step wall time so a run's throughput is visible in the logs.

    Two quantities are logged every optimisation step (TensorBoard tags ``time/step_ms`` and
    ``time/data_wait_ms``) and printed every ``print_every`` steps on rank 0:

    * ``step_ms``: from ``on_train_batch_start`` to ``on_train_batch_end`` with a CUDA
      synchronisation, i.e. transfer + forward + backward + optimiser for this batch.
    * ``data_wait_ms``: from the previous ``on_train_batch_end`` to this ``on_train_batch_start``,
      i.e. the time the trainer waited for the DataLoader (0 when the loader keeps up).
    """

    def __init__(self, print_every: int = 20):
        super().__init__()
        self.print_every = print_every
        self._t_start: float | None = None
        self._t_prev_end: float | None = None

    @staticmethod
    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        self._data_wait_ms = 0.0 if self._t_prev_end is None else 1e3 * (now - self._t_prev_end)
        self._t_start = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._sync()
        now = time.perf_counter()
        step_ms = 1e3 * (now - self._t_start) if self._t_start is not None else float("nan")
        self._t_prev_end = now
        pl_module.log("time/step_ms", step_ms, on_step=True, on_epoch=False, prog_bar=True, rank_zero_only=True)
        pl_module.log("time/data_wait_ms", self._data_wait_ms, on_step=True, on_epoch=False, rank_zero_only=True)
        if trainer.is_global_zero and trainer.global_step % self.print_every == 0:
            mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            loss = outputs["loss"] if isinstance(outputs, dict) and "loss" in outputs else None
            loss_s = f" loss={float(loss):.4f}" if loss is not None else ""
            print(
                f"[step {trainer.global_step:6d}] step {step_ms:7.1f} ms | data wait {self._data_wait_ms:6.1f} ms"
                f" | {1e3 / step_ms if step_ms > 0 else 0:5.2f} steps/s | peak mem {mem:5.1f} GB{loss_s}",
                flush=True,
            )

#!/usr/bin/env python3
"""Invertible U-Net for DIBCO document binarization.

BinarySegmentation trainer and Bunet model are defined here.
When the trainer proves useful beyond this example it can be
promoted to mentor/trainers.py.

Usage::

    python bunet.py
    python bunet.py -resume_path ./tmp/bunet.pt -epochs 20 -verbose true
    python bunet.py -train_split 2009-2016 -val_split ^2009-2016 -patch_size 256
    python bunet.py -device cuda
"""

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from fargv import fargv
from iunets import iUNet

from mentor import make_mentee
from mentor.trainers import MentorTrainer
from dibco_ds import DibcoDataset


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BinarySegmentation(MentorTrainer):
    """Training strategy for binary pixel-wise segmentation.

    * Loss   : NLL on 2-channel output (registered by :meth:`create_train_objects`).
    * Metrics: F1 (primary), IoU, pixel accuracy, loss.

    The model must output 2-channel logits ``(B, 2, H, W)``;
    targets must be long integer masks ``(B, H, W)`` with values in ``{0, 1}``.
    """

    @classmethod
    def default_training_step(
        cls,
        model: Any,
        batch: Any,
        loss_fn: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        x, y = batch
        x = x.to(model.device)
        target = y.squeeze(1).long().to(model.device)  # (B, H, W)

        logits = model(x)  # (B, 2, H, W)
        log_probs = F.log_softmax(logits, dim=1)

        if loss_fn is not None:
            loss = loss_fn(log_probs, target)
        else:
            loss = F.nll_loss(log_probs, target)

        with torch.no_grad():
            pred = logits.argmax(dim=1)  # (B, H, W), values 0 or 1
            tp = ((pred == 1) & (target == 1)).float().sum().item()
            fp = ((pred == 1) & (target == 0)).float().sum().item()
            fn = ((pred == 0) & (target == 1)).float().sum().item()
            tn = ((pred == 0) & (target == 0)).float().sum().item()
            pr = tp / (tp + fp + 1e-6)
            rec = tp / (tp + fn + 1e-6)
            f1        = (2.0 * pr * rec ) / (pr + rec + 1e-6)
            iou       = (tp / (tp + fp + fn + 1e-6))
            pixel_acc = ((tp + tn) / (tp + tn + fp + fn + 1e-6))

        return loss, {"f1": f1, "iou": iou, "pixel_acc": pixel_acc, "loss": loss.item()}

    def create_train_objects(
        self,
        model: Any,
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.1,
        loss_fn: Optional[Any] = None,
        overwrite_default_loss: bool = False,
    ) -> Dict[str, Any]:
        if loss_fn is not None and (overwrite_default_loss or self._loss_fn is None):
            self._loss_fn = loss_fn
        # NLL loss is stateless (F.nll_loss), leave self._loss_fn as None to use it
        self._optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self._lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, step_size=step_size, gamma=gamma
        )
        return {
            "optimizer": self._optimizer,
            "lr_scheduler": self._lr_scheduler,
            "loss_fn": self._loss_fn,
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@make_mentee(trainer=BinarySegmentation)
class Bunet(nn.Module):
    """Invertible U-Net for binary document segmentation.

    Architecture::

        Conv2d(in_channels → base_channels, k=1)   # channel lift
        iUNet(base_channels, depth levels, blocks_per_level blocks)
        Conv2d(base_channels → 2, k=1)              # 2-class segmentation head

    The iUNet is memory-efficient: activations are recomputed during
    backprop via the invertible structure, so deeper/wider networks fit
    on the same GPU budget.

    Parameters
    ----------
    in_channels : int
        Input image channels (1 for grayscale, 3 for RGB).
    base_channels : int
        Channels fed into the iUNet.  Must be ≥ 2.  Default ``8``.
    depth : int
        Number of resolution levels in the iUNet.  Default ``3``.
    blocks : int
        Invertible residual blocks per level.  Default ``2``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 8,
        depth: int = 3,
        blocks: int = 2,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth,
            blocks=blocks,
        )
        self.encoder = nn.Conv2d(in_channels, base_channels, kernel_size=1)
        self.iunet = iUNet(
            in_channels=base_channels,
            architecture=tuple([blocks] * depth),
            dim=2,
            verbose=0,
        )
        self.head = nn.Conv2d(base_channels, 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.iunet(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Training script
# ---------------------------------------------------------------------------

params = {
    "epochs":       [10,            "Epochs to train"],
    "batch_size":   [4,             "Samples per batch"],
    "lr":           [1e-3,          "Learning rate"],
    "base_channels":[8,             "iUNet base channels (>= 2)"],
    "depth":        [3,             "iUNet depth (resolution levels)"],
    "blocks":       [2,             "Invertible blocks per level"],
    "patch_size":   [256,           "Training crop size (px)"],
    "in_channels":  [1,             "Input channels: 1=grayscale, 3=RGB"],
    "train_split":  ["2009-2016",   "Year range for training"],
    "val_split":    ["^2009-2016",  "Year range for validation"],
    "data":         ["./tmp",       "DIBCO data root directory"],
    "resume_path":  ["./tmp/bunet.pt", "Checkpoint path"],
    "device":       ["cuda",         "Device: cpu, cuda, cuda:0, mps, …"],
    "verbose":      [False,         "tqdm progress bars"],
}


def main() -> None:
    p, _ = fargv(params)
    device = torch.device(p.device)

    train_ds = DibcoDataset(
        p.data, split=p.train_split, channels=p.in_channels,
        min_size=p.patch_size, max_size=p.patch_size, download=True,
    )
    val_ds = DibcoDataset(
        p.data, split=p.val_split, channels=p.in_channels, download=True,
    )
    train_loader = DataLoader(train_ds, batch_size=p.batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False)

    if Path(p.resume_path).exists():
        model, opt, sched = Bunet.resume_training(
            p.resume_path, model_class=Bunet, lr=p.lr, device=device,
        )
    else:
        model = Bunet(
            in_channels=p.in_channels,
            base_channels=p.base_channels,
            depth=p.depth,
            blocks=p.blocks,
        )
        model.to(device)
        _to   = model.create_train_objects(lr=p.lr)
        opt   = _to["optimizer"]
        sched = _to["lr_scheduler"]

    print(f"Starting at epoch {model.current_epoch}, device={device}, "
          f"training on {len(train_ds)} patches, "
          f"validating on {len(val_ds)} images.")

    for _ in range(p.epochs):
        tr = model.train_epoch(train_loader, opt, lr_scheduler=sched,
                               verbose=p.verbose)
        vl = model.validate_epoch(val_loader, verbose=p.verbose)
        print(f"epoch {model.current_epoch:3d} | "
              f"train f1={tr['f1']:.4f} iou={tr['iou']:.4f} loss={tr['loss']:.4f} | "
              f"val   f1={vl['f1']:.4f} iou={vl['iou']:.4f} | "
              f"best={model._best_epoch_so_far}")
        Path(p.resume_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(p.resume_path, optimizer=opt, lr_scheduler=sched)


if __name__ == "__main__":
    main()

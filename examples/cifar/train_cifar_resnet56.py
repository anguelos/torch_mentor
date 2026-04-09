#!/usr/bin/env python3
"""CIFAR-10 with the original ResNet-56 architecture from He et al. (2016).

The paper reports 6.97% test error for ResNet-56 on CIFAR-10.  This script
reproduces the training recipe: SGD + momentum, weight-decay 1e-4, and a
multi-step LR schedule at 32K/48K/64K iterations (~82/123/164 epochs with
batch_size=128 on 50K training samples).

Architecture: ResNet-56 is a CIFAR-specific design (3x3 convs, no max-pool,
global avg-pool before the classifier) using torchvision's BasicBlock.
It is NOT the same as the ImageNet ResNet-50 / ResNet-101 variants.

Usage:
    python train_cifar_resnet56.py
    python train_cifar_resnet56.py -resume_path ./tmp/resnet56.pt -epochs 200 -verbose
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.resnet import BasicBlock
import fargv

from mentor import Mentee
from mentor.trainers import MentorTrainer


# ---------------------------------------------------------------------------
# Custom trainer: SGD + iteration-based LR (matches He et al. recipe)
# ---------------------------------------------------------------------------

class CifarSGDResnetClassifier(MentorTrainer):
    """SGD + momentum + weight-decay + iteration-based LR decay for CIFAR.

    Matches the training recipe in He et al. (2016):
    lr=0.1, momentum=0.9, weight_decay=1e-4, divide LR by 10 at
    32K, 48K, and 64K iterations (~82, 123, 164 epochs with batch 128).

    The LR scheduler (IterationMultiStepLR) reads mentee.total_train_iterations
    directly, so its state_dict() is empty -- no extra state to checkpoint.
    Resume is automatic because total_train_iterations is persisted by Mentee.
    """

    class IterationMultiStepLR:
        """LR scheduler that reads its state from mentee.total_train_iterations.

        state_dict() returns {} -- all persistent state lives in the Mentee.
        load_state_dict({}) re-applies the correct LR from the restored counter.
        """

        def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            mentee: Any,
            base_lr: float = 0.1,
            milestones: tuple = (32000, 48000, 64000),
            gamma: float = 0.1,
        ) -> None:
            self.optimizer  = optimizer
            self.mentee     = mentee
            self.base_lr    = base_lr
            self.milestones = list(milestones)
            self.gamma      = gamma
            self._apply_lr()

        def _apply_lr(self) -> None:
            done = self.mentee.total_train_iterations
            factor = self.gamma ** sum(1 for m in self.milestones if done >= m)
            lr = self.base_lr * factor
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

        def step(self) -> None:
            self._apply_lr()

        def state_dict(self) -> dict:
            return {}

        def load_state_dict(self, state: dict) -> None:  # noqa: ARG002
            self._apply_lr()

    def __init__(
        self,
        milestones: tuple = (32000, 48000, 64000),
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.milestones   = list(milestones)
        self.momentum     = momentum
        self.weight_decay = weight_decay

    @classmethod
    def default_training_step(
        cls,
        model: Any,
        batch: Any,
        loss_fn: Optional[Any] = None,
    ) -> tuple:
        x, y = batch
        x, y = x.to(model.device), y.to(model.device)
        logits = model(x)
        eff_fn = loss_fn if loss_fn is not None else F.cross_entropy
        loss = eff_fn(logits, y)
        acc  = (logits.argmax(1) == y).float().mean().item()
        return loss, {"acc": acc, "loss": loss.item()}

    def create_train_objects(
        self,
        model: Any,
        lr: float = 0.1,
        step_size: int = 10,      # unused -- kept for interface compatibility
        gamma: float = 0.1,
        loss_fn: Optional[Any] = None,
        overwrite_default_loss: bool = False,
    ) -> Dict[str, Any]:
        if loss_fn is not None and (overwrite_default_loss or self._loss_fn is None):
            self._loss_fn = loss_fn
        elif self._loss_fn is None:
            self._loss_fn = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        self._lr_scheduler = CifarSGDResnetClassifier.IterationMultiStepLR(
            self._optimizer,
            model,
            base_lr=lr,
            milestones=self.milestones,
            gamma=gamma,
        )
        return {
            "optimizer":    self._optimizer,
            "lr_scheduler": self._lr_scheduler,
            "loss_fn":      self._loss_fn,
        }


# ---------------------------------------------------------------------------
# ResNet-56 model (uses torchvision BasicBlock)
# ---------------------------------------------------------------------------

class CifarResNet56(Mentee):
    """CIFAR-10 ResNet-56 as described in He et al. (2016).

    Architecture: 6n+2 layers with n=9 -> 56 layers.
    Three groups of n BasicBlocks with 16, 32, 64 filters.
    Global average pooling before the 10-class linear head.
    """

    def __init__(self, num_classes: int = 10, n: int = 9) -> None:
        super().__init__(num_classes=num_classes, n=n)
        self.conv   = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn     = nn.BatchNorm2d(16)
        self.layer1 = self._make_group(16, 16, n, stride=1)
        self.layer2 = self._make_group(16, 32, n, stride=2)
        self.layer3 = self._make_group(32, 64, n, stride=2)
        self.fc     = nn.Linear(64, num_classes)
        self.trainer = CifarSGDResnetClassifier()

    def _make_group(self, in_ch: int, out_ch: int, n: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        blocks = [BasicBlock(in_ch, out_ch, stride=stride, downsample=downsample)]
        for _ in range(n - 1):
            blocks.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_loaders(data_dir: str, batch_size: int, num_workers: int):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train = DataLoader(
        torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=train_tf),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers,
    )
    val = DataLoader(
        torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=val_tf),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train, val


# def main(epochs: int=200, batch_size: int=128, lr: float=0.1, resume_path: str="./tmp/resnet56.pt", 
#          data: str="./tmp/data", device: str="cuda", verbose: bool=False, num_workers: int=2):
#     train_loader, val_loader = make_loaders(data, batch_size, num_workers)

#     model, opt, sched = CifarResNet56.resume_training(
#         resume_path,
#         model_class=CifarResNet56,
#         device=device,
#         lr=lr,
#         tolerate_irresumable_trainstate=True,
#     )

#     model.fit(
#         train_loader,
#         val_data=val_loader,
#         epochs=epochs,
#         lr=lr,
#         checkpoint_path=resume_path,
#         verbose=verbose,
#     )

#     best = model._validate_history.get(model._best_epoch_so_far, {})
#     print(f"\nBest epoch {model._best_epoch_so_far}: "
#           f"acc={best.get('acc', 0):.4f}  "
#           f"error={100*(1-best.get('acc', 0)):.2f}%")


# if __name__ == "__main__":
#     p, _ = fargv.parse_and_launch(main)


def main():
    args, _ = fargv.parse({
        "epochs": 200,
        "batch_size": 128,
        "lr": 0.1,
        "resume_path": "./tmp/resnet56.pt",
        "data": "./tmp/data",
        "device": "cuda",
        "num_workers": 2,
        "wandb": False,
        "gradio": False,
    })
    print(f"Training CIFAR-10 ResNet-56 with args: {args}", flush=True)
    train_loader, val_loader = make_loaders(args.data, args.batch_size, args.num_workers)
    model, opt, sched = CifarResNet56.resume_training(args.resume_path, device=args.device, lr=args.lr)
    model.fit(
        train_loader,
        val_data=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        checkpoint_path=args.resume_path,
        verbose=args.verbosity > 0,
        report_wandb=args.wandb,
        report_gradio=args.gradio,
    )


if __name__ == "__main__":
    main()
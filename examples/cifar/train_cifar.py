#!/usr/bin/env python3
"""CIFAR-10 with Mentee + torchvision ResNet.

Usage:
    python train_cifar.py -epochs 10 -resnet resnet34 -pretrained
    python train_cifar.py -resume_path ./runs/cifar.pt -verbose
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as tvm
from torch.utils.data import DataLoader
from torchvision import transforms
from fargv import fargv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mentor import Mentee


params = {
    "epochs":          [5,        "Number of epochs to train"],
    "batch_size":      [5,        "Samples per batch"],
    "pseudo_batch":    [2,        "Gradient accumulation steps"],
    "lr":              [1e-3,     "Learning rate"],
    "resnet":          ["resnet18", "torchvision ResNet variant"],
    "pretrained":      [False,    "Use pretrained ImageNet weights"],
    "data":            ["./tmp/data", "Directory for CIFAR-10 data"],
    "resume_path":     ["./tmp/cifar_mentee.pt", "Checkpoint path"],
    "tensorboard_dir": ["./tmp/tb/", "TensorBoard log dir (empty = disabled)"],
    "device":          ["cuda" if __import__("torch").cuda.is_available() else "cpu", "Device"],
    "verbose":         [False,    "Show tqdm progress bars"],
}


class CifarResNet(Mentee):
    """ResNet wrapper for CIFAR-10 classification."""

    def __init__(self, num_classes=10, resnet_variant="resnet18", pretrained=False):
        super().__init__(num_classes=num_classes, resnet_variant=resnet_variant, pretrained=pretrained)
        self.backbone = tvm.get_model(resnet_variant, weights="DEFAULT" if pretrained else None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.eval()
        return self

    def forward(self, x):
        return self.backbone(x)

    def _infer(self, sample):
        x, y = sample
        x, y = x.to(self.device), y.to(self.device)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = float(logits.argmax(1).eq(y).float().mean())
        return loss, {"accuracy": acc, "loss": loss.item()}

    def training_step(self, sample):
        loss, metrics = self._infer(sample)
        return loss, metrics

    def validation_step(self, sample):
        _, metrics = self._infer(sample)
        return metrics


def make_datasets(data_dir):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True,  download=True,
        transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize(mean, std)]))
    val_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
    return train_set, val_set


def main():
    p, _ = fargv(params)
    train_set, val_set = make_datasets(p.data)
    train_loader = DataLoader(train_set, batch_size=p.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=p.batch_size, shuffle=False, drop_last=False)

    if Path(p.resume_path).exists():
        model, optimizer, lr_scheduler = CifarResNet.resume_training(
            p.resume_path, model_class=CifarResNet, device=p.device, lr=p.lr)
    else:
        model = CifarResNet(resnet_variant=p.resnet, pretrained=p.pretrained)
        model.to(p.device)
        _to = model.create_train_objects(lr=p.lr)

        optimizer, lr_scheduler = _to["optimizer"], _to["lr_scheduler"]

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(p.tensorboard_dir) if p.tensorboard_dir else None

    print(f"Starting at epoch {model.current_epoch}, training for {p.epochs} epochs on {p.device}.")
    for _ in range(p.epochs):
        train_metrics = model.train_epoch(train_loader, optimizer, lr_scheduler=lr_scheduler,
            pseudo_batch_size=p.pseudo_batch, memfail="ignore",
            tensorboard_writer=writer, verbose=p.verbose)
        val_metrics = model.validate_epoch(val_loader, tensorboard_writer=writer, verbose=p.verbose)
        print(f"Epoch {model.current_epoch:3d} | "
              f"train acc {train_metrics.get('accuracy', 0):.4f} loss {train_metrics.get('loss', 0):.4f} | "
              f"val acc {val_metrics.get('accuracy', 0):.4f} loss {val_metrics.get('loss', 0):.4f} | "
              f"best {model._best_epoch_so_far}")
        model.save(p.resume_path, optimizer=optimizer, lr_scheduler=lr_scheduler)

    if writer:
        writer.close()


if __name__ == "__main__":
    main()

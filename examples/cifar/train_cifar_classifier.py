#!/usr/bin/env python3
"""CIFAR-10 with Classifier trainer — minimal version.

Usage:
    python train_cifar_classifier.py
    python train_cifar_classifier.py -resume_path ./tmp/cifar2.pt -epochs 20 -verbose
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch.nn as nn
import torchvision, torchvision.models as tvm
from torch.utils.data import DataLoader
from torchvision import transforms
from fargv import fargv

from mentor import Mentee, Classifier


params = {
    "epochs":       [5,           "Epochs to train"],
    "batch_size":   [32,          "Samples per batch"],
    "lr":           [1e-3,        "Learning rate"],
    "resnet":       ["resnet18",  "torchvision ResNet variant"],
    "resume_path":  ["./tmp/cifar_classifier.pt", "Checkpoint path"],
    "data":         ["./tmp/data", "CIFAR-10 data directory"],
    "verbose":      [False,       "tqdm progress bars"],
}


class CifarNet(Mentee):
    def __init__(self, num_classes=10, resnet="resnet18"):
        super().__init__(num_classes=num_classes, resnet=resnet)
        self.net = tvm.get_model(resnet, weights=None)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        self.trainer = Classifier()

    def forward(self, x):
        return self.net(x)


def make_loaders(data, batch_size):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    val_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train = DataLoader(torchvision.datasets.CIFAR10(data, train=True,  download=True, transform=train_tf),
                       batch_size=batch_size, shuffle=True,  drop_last=True)
    val   = DataLoader(torchvision.datasets.CIFAR10(data, train=False, download=True, transform=val_tf),
                       batch_size=batch_size, shuffle=False)
    return train, val


def main():
    p, _ = fargv(params)
    train_loader, val_loader = make_loaders(p.data, p.batch_size)

    if Path(p.resume_path).exists():
        model, opt, sched = CifarNet.resume_training(p.resume_path, model_class=CifarNet, lr=p.lr)
    else:
        model = CifarNet(resnet=p.resnet)
        _to = model.create_train_objects(lr=p.lr)
        opt, sched = _to["optimizer"], _to["lr_scheduler"]

    if model.current_epoch == 0:
        vl = model.validate_epoch(val_loader, verbose=p.verbose)
        print(f"epoch   0  val loss={vl['loss']:.4f} acc={vl['acc']:.4f}  (baseline)")
        model.save(p.resume_path, optimizer=opt, lr_scheduler=sched)

    for _ in range(p.epochs):
        tr = model.train_epoch(train_loader, opt, lr_scheduler=sched, verbose=p.verbose)
        vl = model.validate_epoch(val_loader, verbose=p.verbose)
        print(f"epoch {model.current_epoch:3d}  "
              f"train loss={tr['loss']:.4f} acc={tr['acc']:.4f}  "
              f"val loss={vl['loss']:.4f} acc={vl['acc']:.4f}  "
              f"best={model._best_epoch_so_far}")
        model.save(p.resume_path, optimizer=opt, lr_scheduler=sched)


if __name__ == "__main__":
    main()

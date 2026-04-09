#!/usr/bin/env python3
"""Fine-tune MobileNetV3-Small (ImageNet) on CIFAR-10 using Mentee.

MobileNetV3-Small is loaded from torchvision with IMAGENET1K_V1 weights --
the same model and weights published on HuggingFace Hub as
timm/mobilenetv3_small_100.lamb_in1k.

Transfer learning is done in two stages:

    Stage 1  (epochs 1 .. epochs_head)
        Backbone frozen, only the classification head is trained.
        Runs at lr_head with Adam + StepLR.

    Stage 2  (epochs epochs_head+1 .. epochs_head+epochs_finetune)
        Full model unfrozen.  The backbone uses a lower effective LR
        (lr_finetune * backbone_lr_coeff) while the head uses lr_finetune,
        preventing catastrophic forgetting of ImageNet features.

Usage::

    python examples/hf/train_mobilenet_cifar10.py

    # resume, increase fine-tune budget
    python examples/hf/train_mobilenet_cifar10.py
        -resume_path ./tmp/mobilenet_cifar10.pt -epochs_finetune 40

Layer freeze reference
----------------------
Two named groups relevant for Mentee.freeze():

    backbone.features.*    152 convolutional sub-layers (backbone)
    backbone.classifier.*  two-linear classification head
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

import fargv

from mentor import Mentee
from mentor.trainers import Classifier

# ImageNet normalisation -- must match the pretrained backbone's training stats.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


class MobileNetV3SmallCIFAR10(Mentee):
    """MobileNetV3-Small (ImageNet pretrained) adapted for CIFAR-10.

    The torchvision backbone is stored under self.backbone.
    Only the final linear layer (1024 -> 1000) is replaced with a fresh
    (1024 -> num_classes) layer; all other weights start from ImageNet
    pretrained values.

    Parameters
    ----------
    num_classes : int
        Number of output classes.  Default is 10 (CIFAR-10).
    pretrained : bool
        Load IMAGENET1K_V1 weights when True (default).
    """

    def __init__(self, num_classes: int = 10, pretrained: bool = True) -> None:
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        weights = (
            torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        backbone = torchvision.models.mobilenet_v3_small(weights=weights)
        in_features = backbone.classifier[3].in_features   # 1024
        backbone.classifier[3] = nn.Linear(in_features, num_classes)
        self.backbone = backbone
        self.trainer = Classifier()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def make_loaders(data_dir: str, batch_size: int, num_workers: int) -> tuple:
    """Build CIFAR-10 DataLoaders resized to 224x224 with ImageNet stats."""
    train_tf = T.Compose([
        T.Resize(256),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
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


def main(
    epochs_head: int = 10,
    epochs_finetune: int = 20,
    batch_size: int = 64,
    lr_head: float = 1e-3,
    lr_finetune: float = 1e-4,
    backbone_lr_coeff: float = 0.1,
    resume_path: str = "./tmp/mobilenet_cifar10.pt",
    data: str = "./tmp/data",
    device: str = "cuda",
    num_workers: int = 2,
    pretrained: bool = True,
    verbose: bool = False,
) -> None:
    """Run two-stage fine-tuning and print the best result."""
    train_loader, val_loader = make_loaders(data, batch_size, num_workers)

    model, _, _ = MobileNetV3SmallCIFAR10.resume_training(
        resume_path,
        model_class=MobileNetV3SmallCIFAR10,
        device=device,
        lr=lr_head,
        tolerate_irresumable_trainstate=True,
        pretrained=pretrained,
    )

    # Stage 1: head-only
    if model.current_epoch < epochs_head:
        remaining = epochs_head - model.current_epoch
        print(f"Stage 1: head-only "
              f"(epochs {model.current_epoch + 1}-{epochs_head}, lr={lr_head})")
        model.freeze("backbone\\.features.*")
        if model.optimizer is None:
            model.create_train_objects(lr=lr_head)
        model.fit(
            train_loader,
            val_data=val_loader,
            epochs=remaining,
            checkpoint_path=resume_path,
            verbose=verbose,
        )

    # Stage 2: full fine-tune
    total = epochs_head + epochs_finetune
    if model.current_epoch < total:
        remaining = total - model.current_epoch
        backbone_lr = lr_finetune * backbone_lr_coeff
        print(f"Stage 2: full fine-tune "
              f"(epochs {model.current_epoch + 1}-{total}, "
              f"head lr={lr_finetune}, backbone lr={backbone_lr:.2e})")
        model.unfreeze("backbone\\.features.*")
        model.create_train_objects(lr=lr_finetune)
        model.set_lr_coefficient(backbone_lr_coeff, "backbone\\.features.*")
        model.fit(
            train_loader,
            val_data=val_loader,
            epochs=remaining,
            checkpoint_path=resume_path,
            verbose=verbose,
        )

    best = model._validate_history.get(model._best_epoch_so_far, {})
    print(f"\nBest epoch {model._best_epoch_so_far}: "
          f"acc={best.get('acc', 0):.4f}  "
          f"top-1 error={100 * (1 - best.get('acc', 0)):.2f}%")


if __name__ == "__main__":
    p, _ = fargv.parse_and_launch(main)

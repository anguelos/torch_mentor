import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification, AutoImageProcessor
import fargv

from mentor import Mentee
from mentor.trainers import Classifier

MODEL_ID = "google/mobilenet_v2_1.0_224"


class MobileNetV2CIFAR10(Mentee):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        # Load HF model; replace 1001-class head with num_classes
        self.hf = AutoModelForImageClassification.from_pretrained(MODEL_ID) if pretrained \
            else AutoModelForImageClassification.from_config(
                AutoModelForImageClassification.config_class.from_pretrained(MODEL_ID))
        self.hf.classifier = nn.Linear(self.hf.classifier.in_features, num_classes)
        self.trainer = Classifier()

    def forward(self, x):
        return self.hf(pixel_values=x).logits


def make_loaders(data_dir, batch_size, num_workers):
    # Normalization comes from the HF image processor for this checkpoint
    proc = AutoImageProcessor.from_pretrained(MODEL_ID)
    mean, std = proc.image_mean, proc.image_std
    train_tf = T.Compose([
        T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize(mean, std),
    ])
    val_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(), T.Normalize(mean, std),
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
    epochs_head=10,
    epochs_finetune=20,
    batch_size=64,
    lr_head=1e-3,
    lr_finetune=1e-4,
    backbone_lr_coeff=0.1,
    resume_path="./tmp/mobilenetv2_cifar10.pt",
    data="./tmp/data",
    device="cuda",
    num_workers=2,
    pretrained=True,
    verbose=False,
):
    train_loader, val_loader = make_loaders(data, batch_size, num_workers)

    model, _, _ = MobileNetV2CIFAR10.resume_training(
        resume_path, model_class=MobileNetV2CIFAR10,
        device=device, lr=lr_head,
        tolerate_irresumable_trainstate=True, pretrained=pretrained,
    )

    # Stage 1: freeze backbone, train head only
    if model.current_epoch < epochs_head:
        model.freeze("hf\\.mobilenet_v2.*")
        if model.optimizer is None:
            model.create_train_objects(lr=lr_head)
        model.fit(train_loader, val_data=val_loader,
                  epochs=epochs_head - model.current_epoch,
                  checkpoint_path=resume_path, verbose=verbose)

    # Stage 2: unfreeze backbone with lower lr to avoid forgetting
    total = epochs_head + epochs_finetune
    if model.current_epoch < total:
        model.unfreeze("hf\\.mobilenet_v2.*")
        model.create_train_objects(lr=lr_finetune)
        model.set_lr_coefficient(backbone_lr_coeff, "hf\\.mobilenet_v2.*")
        model.fit(train_loader, val_data=val_loader,
                  epochs=total - model.current_epoch,
                  checkpoint_path=resume_path, verbose=verbose)

    best = model._validate_history.get(model._best_epoch_so_far, {})
    print(f"best epoch {model._best_epoch_so_far}: "
          f"acc={best.get('acc', 0):.4f}  error={100*(1-best.get('acc',0)):.2f}%")


if __name__ == "__main__":
    fargv.parse_and_launch(main)

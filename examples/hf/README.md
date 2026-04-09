# MobileNetV3-Small fine-tuned on CIFAR-10

This example shows how to wrap a pretrained torchvision model as a Mentee
and adapt it to a new task via two-stage transfer learning.

The model is MobileNetV3-Small with IMAGENET1K_V1 weights (2.5 M parameters),
identical to `timm/mobilenetv3_small_100.lamb_in1k` on HuggingFace Hub.
CIFAR-10 images (32x32) are upsampled to 224x224 to match the backbone input.

## Training recipe

| | Stage 1 (head) | Stage 2 (fine-tune) |
|---|---|---|
| Epochs | 10 | 20 |
| Optimizer | Adam | Adam |
| LR (head) | 1e-3 | 1e-4 |
| LR (backbone) | frozen | 1e-5 (coeff 0.1) |
| Backbone | frozen | unfrozen |

## Usage

    # fresh run
    python examples/hf/train_mobilenet_cifar10.py

    # resume or extend fine-tuning
    python examples/hf/train_mobilenet_cifar10.py \
        -resume_path ./tmp/mobilenet_cifar10.pt

    # all parameters
    python examples/hf/train_mobilenet_cifar10.py \
        -epochs_head 10 -epochs_finetune 40 \
        -lr_head 1e-3 -lr_finetune 1e-4 -backbone_lr_coeff 0.1 \
        -batch_size 64 -device cuda -verbose true

## How it works

### Wrapping a pretrained model as Mentee

`MobileNetV3SmallCIFAR10` subclasses `Mentee` and stores the torchvision
backbone as `self.backbone`. Only the final linear layer is replaced:

    backbone.classifier[3] = nn.Linear(1024, num_classes)

All other weights keep their ImageNet values. `self.trainer = Classifier()`
wires up cross-entropy loss and top-1 accuracy automatically.

### Two-stage training

**Stage 1** -- `model.freeze("backbone\.features.*")` sets `requires_grad=False`
on all 152 feature sub-layers. Only the classification head is updated.
This quickly trains the head to recognise CIFAR-10 classes without
disturbing the ImageNet features.

**Stage 2** -- `model.unfreeze("backbone\.features.*")` re-enables gradients.
`model.set_lr_coefficient(0.1, "backbone\.features.*")` gives the backbone
a 10x lower LR than the head, preventing catastrophic forgetting.

### Transparent resume

Calling the script twice with the same `-resume_path` always continues from
the last checkpoint. The current stage is inferred from `model.current_epoch`:

    epoch < epochs_head          -> still in stage 1
    epochs_head <= epoch < total -> still in stage 2

### Layer names

Use `mtr_checkpoint view -paths ./tmp/mobilenet_cifar10.pt` to inspect the
full layer tree. The two freeze-relevant groups are:

    backbone.features.*    backbone (152 sub-layers, frozen in stage 1)
    backbone.classifier.*  head (kept trainable throughout)

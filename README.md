# mentor

[![License](https://img.shields.io/github/license/anguelos/mentor)](https://github.com/anguelos/mentor/blob/main/LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/anguelos/mentor/releases)
[![Python](https://img.shields.io/badge/python-3.7%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.9-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Last commit](https://img.shields.io/github/last-commit/anguelos/mentor)](https://github.com/anguelos/mentor/commits/main)
[![Open issues](https://img.shields.io/github/issues/anguelos/mentor)](https://github.com/anguelos/mentor/issues)
[![Open PRs](https://img.shields.io/github/issues-pr/anguelos/mentor)](https://github.com/anguelos/mentor/pulls)
[![Repo size](https://img.shields.io/github/repo-size/anguelos/mentor)](https://github.com/anguelos/mentor)
[![Stars](https://img.shields.io/github/stars/anguelos/mentor?style=social)](https://github.com/anguelos/mentor/stargazers)
[![Forks](https://img.shields.io/github/forks/anguelos/mentor?style=social)](https://github.com/anguelos/mentor/forks)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checking: mypy](https://img.shields.io/badge/type%20checking-mypy-2A6DB2)](https://mypy-lang.org/)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Docs: Sphinx RTD](https://img.shields.io/badge/docs-Sphinx%20RTD-blue?logo=sphinx)](https://mentor.readthedocs.io)
[![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS%20%7C%20windows-lightgrey)](https://github.com/anguelos/mentor)

A lightweight PyTorch training framework built around a single idea: **a model should carry its own training history**.

`Mentee` is a `torch.nn.Module` subclass that transparently records every epoch of training, validation metrics, software environment, and command-line invocation — all saved into a single `.pt` checkpoint.  Resuming on a different machine, reporting on a run, or rolling back to the best epoch requires no extra bookkeeping.

---

## Installation

```bash
pip install mentor
```

Or from source:

```bash
git clone https://github.com/anguelos/mentor
pip install -e mentor/
```

---

## Quick start

### 1. Subclass `Mentee`

```python
import torch, torch.nn as nn, torch.nn.functional as F
from mentor import Mentee

class MyNet(Mentee):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__(num_classes=num_classes)   # kwargs become constructor_params
        self.fc = nn.Linear(784, num_classes)

    def forward(self, x):
        return self.fc(x.flatten(1))

    def training_step(self, sample):
        x, y = sample
        logits = self(x.to(self.device))
        loss = F.cross_entropy(logits, y.to(self.device))
        acc = float(logits.argmax(1).eq(y.to(self.device)).float().mean())
        return loss, {"accuracy": acc, "loss": loss.item()}

    def validation_step(self, sample):
        x, y = sample
        logits = self(x.to(self.device))
        acc = float(logits.argmax(1).eq(y.to(self.device)).float().mean())
        return {"accuracy": acc}
```

### 2. Train, validate, save

```python
from torch.utils.data import DataLoader

model = MyNet(num_classes=10)
model.to("cuda")
train_objs = model.create_train_objects(lr=1e-3)
optimizer, scheduler = train_objs["optimizer"], train_objs["lr_scheduler"]

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32)

for epoch in range(20):
    train_metrics = model.train_epoch(train_loader, optimizer, lr_scheduler=scheduler,
                                      pseudo_batch_size=4, verbose=True)
    val_metrics   = model.validate_epoch(val_loader, verbose=True)
    print(model)                              # shows live training summary
    model.save("checkpoint.pt", optimizer=optimizer, lr_scheduler=scheduler)
```

### 3. Resume anywhere

```python
model, optimizer, scheduler = MyNet.resume_training(
    "checkpoint.pt", model_class=MyNet, device="cuda", lr=1e-3
)
print(f"Resuming from epoch {model.current_epoch}")
```

---

## Key features

| Feature | Description |
|---|---|
| **Self-contained checkpoints** | Weights, optimizer state, full training & validation history in one `.pt` file |
| **Automatic provenance** | Git hash, Python & PyTorch versions, hostname, user, and `sys.argv` recorded every epoch |
| **Best-weights tracking** | `_best_weights_so_far` updated whenever the principal validation metric improves |
| **Gradient accumulation** | `pseudo_batch_size` in `train_epoch` accumulates gradients over N samples before each optimizer step |
| **OOM tolerance** | `memfail="ignore"` skips samples that raise `MemoryError` and counts them separately |
| **tqdm progress bars** | Pass `verbose=True` to `train_epoch` / `validate_epoch`; postfix refreshes every `refresh_freq` iterations |

---

## The `Mentee` API

### Properties

```python
model.current_epoch   # int — len(train_history)
model.device          # torch.device — inferred from parameters
```

### Core methods to implement in your subclass

```python
def training_step(self, sample) -> tuple[Tensor, dict[str, float]]: ...
def validation_step(self, sample)    -> dict[str, float]: ...
```

The **first key** of the returned dict is the *principal metric* used for best-checkpoint selection.

### Training

```python
model.create_train_objects(lr=1e-3, step_size=10, gamma=0.1)
# -> {"optimizer": Adam, "lr_scheduler": StepLR, "loss_fn": <fn or None>}

model.train_epoch(dataset, optimizer,
                  lr_scheduler=None, pseudo_batch_size=1,
                  memfail="raise", tensorboard_writer=None,
                  verbose=False, refresh_freq=20)

model.validate_epoch(dataset,
                     recalculate=False, memfail="raise",
                     tensorboard_writer=None, verbose=False, refresh_freq=20)
```

### Checkpointing

```python
model.save("run.pt", optimizer=optimizer, lr_scheduler=scheduler)

# load weights only (no optimizer)
model = MyNet.resume("run.pt", model_class=MyNet)

# full resume (weights + optimizer + scheduler, moved to device)
model, optimizer, scheduler = MyNet.resume_training(
    "run.pt", model_class=MyNet, device="cuda", lr=1e-3
)
```

All tensors are saved on **CPU** regardless of the training device.

---

## Included example — CIFAR-10 with ResNet

```bash
cd examples/cifar

# first run — downloads data, starts fresh
python train_cifar.py -resume_path ./runs/cifar.pt -epochs 20 -verbose

# subsequent runs — auto-resumes from the same path
python train_cifar.py -resume_path ./runs/cifar.pt -epochs 20 -verbose

# different architecture
python train_cifar.py -resume_path ./runs/cifar34.pt -resnet resnet34 -pretrained
```

---

## Reporting CLI

After installation a command-line tool is registered:

```bash
mtr_report_file -path ./runs/cifar.pt
```

Example output:

```
Checkpoint: /runs/cifar.pt
File size:  89.3 KB

Model class:   examples.cifar.train_cifar.CifarResNet
Importable:    OK (found in 'examples.cifar.train_cifar')

Architecture (inferred from state_dict):
  Parameters:   11,181,642 in 122 tensors
  Modules:      61 parameter-bearing
  Input:        3 channels  (inferred from first conv)
  Output:       10 features  (inferred from last linear)

Epochs trained: 5
  First epoch:  accuracy=0.1823  loss=2.2987  memfails=0.0000
  Last epoch:   accuracy=0.6341  loss=1.0201  memfails=0.0000
...
```

---

## Design philosophy

- **No magic**: `Mentee` is a plain `nn.Module`. Models work identically whether used through the framework or as bare PyTorch modules.
- **Single file**: one `.pt` file holds everything. No sidecar JSON, no separate history database.
- **You own the loop**: `train_epoch` and `validate_epoch` are helpers, not a `Trainer` class. Call them however you like.
- **Reproducibility first**: every change in git hash, environment, or invocation is recorded automatically.

---

## License

MIT

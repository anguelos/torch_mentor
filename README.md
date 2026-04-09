# mentor : A pytorch training framework for lazy and impatient people

<p align="center"><img src="https://raw.githubusercontent.com/anguelos/torch_mentor/main/docs/_static/logo.svg" alt="mentor logo" width="200"/></p>

[![License](https://img.shields.io/github/license/anguelos/torch_mentor)](https://github.com/anguelos/torch_mentor/blob/main/LICENSE)
[![Tests](https://github.com/anguelos/torch_mentor/actions/workflows/tests.yml/badge.svg)](https://github.com/anguelos/torch_mentor/actions/workflows/tests.yml)
[![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/anguelos/torch_mentor/badges/badges/coverage.json)](https://github.com/anguelos/torch_mentor/actions/workflows/tests.yml)
[![Docs](https://readthedocs.org/projects/torch-mentor/badge/?version=latest)](https://torch-mentor.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/torch-mentor)](https://pypi.org/project/torch-mentor/)
[![PyPI downloads](https://img.shields.io/pypi/dm/torch-mentor)](https://pypi.org/project/torch-mentor/)
[![GitHub repo size](https://img.shields.io/github/repo-size/anguelos/torch_mentor)](https://github.com/anguelos/torch_mentor)
[![GitHub last commit](https://img.shields.io/github/last-commit/anguelos/torch_mentor)](https://github.com/anguelos/torch_mentor/commits/main)

A lightweight PyTorch training framework built around a single idea: **a model should carry its own training history**.

Designed for **progressive adoption**: use only what you need — a single feature such as self-contained checkpointing, a built-in trainer that removes all boilerplate, or anything in between. Every abstraction level has an escape hatch to the level below; you are never locked in.

`Mentee` is a `torch.nn.Module` subclass that transparently records every epoch of training, validation metrics, software environment, and command-line invocation — all saved into a single `.pt` checkpoint.  Resuming on a different machine, reporting on a run, or rolling back to the best epoch requires no extra bookkeeping.

---

## Installation

```bash
pip install torch-mentor
```

Or from source:

```bash
git clone https://github.com/anguelos/torch_mentor
pip install -e mentor/
```

---

## Quick start

### Option A — Wrap a pretrained model (zero code changes)

Use `wrap_as_mentee` to add checkpointing and training history to any existing
`nn.Module` instance — including pretrained models from `torchvision` (already
a torch_mentor dependency):

```python
import torchvision.models as tvm
from mentor import wrap_as_mentee, Classifier

net = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
model = wrap_as_mentee(
    net,
    # Store weights=None: at resume time the architecture is reconstructed
    # from scratch and the fine-tuned weights are loaded from the checkpoint.
    # Storing the pretrained enum would be fragile across torchvision versions.
    constructor_params={"weights": None},
    trainer=Classifier,
)
model.fit(train_data, val_data=val_data, epochs=5, lr=1e-4,
          checkpoint_path="resnet50_cifar.pt")

# Resume later — weights, history, and optimizer state all restored
model, opt, sched = type(model).resume_training(
    "resnet50_cifar.pt", model_class=type(model), lr=1e-4
)
```

`constructor_params` records how to reconstruct the model architecture at
resume time — `weights=None` is correct here because the fine-tuned weights
come from the checkpoint, not from ImageNet pretraining.  The class of *net*
must be importable by its qualified name; `torchvision.models` and similar
libraries satisfy this automatically.

### Option B — Built-in trainer (least code)

Assign a `Classifier` or `Regressor` trainer to `self.trainer` and only implement `forward`:

```python
import torch.nn as nn
from mentor import Mentee, Classifier

class MyNet(Mentee):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__(num_classes=num_classes)
        self.fc = nn.Linear(784, num_classes)
        self.trainer = Classifier()   # supplies training_step + validation_step

    def forward(self, x):
        return self.fc(x.flatten(1))
```

### Option C — Custom training step

Override `training_step` and optionally `validation_step` for full control:

```python
import torch.nn as nn, torch.nn.functional as F
from mentor import Mentee

class MyNet(Mentee):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__(num_classes=num_classes)
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


### Train, validate, save

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

### Resume anywhere

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
| **Automatically resumable** | `save()` + `resume_training()` restore weights, optimizer state, and full history — one line to pick up where you left off, on any machine |
| **Self-contained checkpoints** | Everything in one `.pt` file: weights, optimizer, LR scheduler, training & validation history, best weights, and inference state |
| **Automatic TensorBoard** | Pass `tensorboard_writer` to `train_epoch` / `validate_epoch` and all metrics are logged with no extra code |
| **Automatic provenance** | Git hash, Python & PyTorch versions, hostname, user, and `sys.argv` recorded automatically every epoch |
| **Best-weights tracking** | Best checkpoint is updated whenever the principal validation metric improves; roll back with one call |
| **Built-in trainers** | `Classifier` and `Regressor` supply loss, `training_step`, and `validation_step` via composition — only `forward` required |
| **Gradient accumulation** | `pseudo_batch_size` accumulates gradients over N batches before each optimizer step |
| **OOM tolerance** | `memfail="ignore"` skips samples that raise `MemoryError` and counts them in the epoch metrics |

---

## How mentor compares

| Feature | mentor | [Lightning](https://lightning.ai) | [HF Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) | [fastai](https://www.fast.ai) | [Ignite](https://pytorch.org/ignite) |
|---|:---:|:---:|:---:|:---:|:---:|
| Model is plain `nn.Module` | ✅ | ✅ | ✅ | ⚠️ | ✅ |
| You own the training loop | 👍 | ❌ | ❌ | ❌ | ✅ |
| Full metric history in checkpoint | ✅ | ❌ | ❌ | ❌ | ❌ |
| One-call resume (weights + optimizer + history) | ✅ | ✅ | ✅ | ⚠️ | ⚠️ |
| Model carries its own history | ✅ | ❌ | ❌ | ❌ | ❌ |
| Automatic provenance (git, env, argv) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Best-epoch weights auto-saved | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| Inference state bundled in checkpoint | ✅ | ❌ | ❌ | ❌ | ❌ |
| TensorBoard logging | ✅ | ✅ | ✅ | ✅ | ✅ |
| Gradient accumulation | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| OOM-tolerant training | ✅ | ❌ | ❌ | ❌ | ❌ |
| High-level `fit()` | 👍 | ✅ | ✅ | ✅ | ❌ |
| Early stopping | 👍 | ✅ | ✅ | ✅ | ✅ |
| Framework integration | opt-in | opt-out | opt-out | opt-out | opt-in |
| Multi-GPU / distributed training | ❌ | ✅ | ✅ | ✅ | ⚠️ |
| Mixed precision (AMP) | 👍 | ✅ | ✅ | ✅ | ⚠️ |
| Callback / hook system | ❌ | ✅ | ✅ | ✅ | ✅ |
| LR finder | 👍 | ✅ | ❌ | ✅ | ❌ |

✅ built-in · 👍 optional · ⚠️ partial or via plugin · ❌ not supported

---

## The `Mentee` API

### Properties

```python
model.current_epoch   # int — len(train_history)
model.device          # torch.device — inferred from parameters
model.optimizer       # cached optimizer (from trainer or create_train_objects)
model.lr_scheduler    # cached LR scheduler
model.loss_fn         # cached default loss function
```

### Core methods to implement in your subclass

```python
def training_step(self, sample) -> tuple[Tensor, dict[str, float]]: ...
def validation_step(self, sample)    -> dict[str, float]: ...
```

Both are **optional** when `self.trainer` is set — the trainer's classmethods are used instead.

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

## Built-in trainers

`MentorTrainer` is a pure-Python strategy class (not an `nn.Module`) that is composed into a `Mentee` via `self.trainer`.  It separates stateless logic (classmethods) from stateful training objects (cached per-instance):

| Trainer | Default loss | Metrics |
|---|---|---|
| `Classifier` | `nn.CrossEntropyLoss` | `loss`, `acc` |
| `Regressor` | `nn.MSELoss` | `loss`, `rmse` |

```python
from mentor import Mentee, Classifier, Regressor

class MyClassifier(Mentee):
    def __init__(self, num_classes=10):
        super().__init__(num_classes=num_classes)
        self.fc = nn.Linear(128, num_classes)
        self.trainer = Classifier()
    def forward(self, x): return self.fc(x.flatten(1))

class MyRegressor(Mentee):
    def __init__(self, in_features=8):
        super().__init__(in_features=in_features)
        self.fc = nn.Linear(in_features, 1)
        self.trainer = Regressor()
    def forward(self, x): return self.fc(x).squeeze(-1)
```

Custom trainers can be added by subclassing `MentorTrainer` and implementing
`default_training_step` (classmethod) and `create_train_objects`.

---

## Included examples — CIFAR-10 with ResNet

```bash
cd examples/cifar

# full control — custom training_step
python train_cifar.py -resume_path ./runs/cifar.pt -epochs 20 -verbose

# minimal — uses Classifier trainer
python train_cifar_classifier.py -resume_path ./runs/cifar2.pt -epochs 20 -verbose
```

---

## Reporting CLI

After installation a command-line tool is registered:

```bash
mtr_checkpoint -path ./runs/cifar.pt
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
- **Composition over inheritance**: trainers are strategy objects assigned to `self.trainer`, not base classes. A model is always a `Mentee`; a trainer is always a `MentorTrainer`.
- **Reproducibility first**: every change in git hash, environment, or invocation is recorded automatically.
- **Progressive adoption with escape hatches**: use just one feature or the full stack. Every abstraction level drops cleanly to the one below — plain , trainer strategy, or  — without fighting the framework.

---

## License

MIT

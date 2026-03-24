# Quickstart

## Installation

```bash
pip install pytorch-mentor
```

Or in editable mode from source:

```bash
git clone <repo>
cd mentor
pip install -e .
```

## Subclassing Mentee

Every model is a subclass of {class}`~mentor.Mentee`.  There are two ways
to define training behaviour.

### Option A — Built-in trainer (least code)

Assign a {class}`~mentor.Classifier` or {class}`~mentor.Regressor` trainer to
`self.trainer` and only implement `forward`.  The trainer supplies
`training_step` and `validation_step` automatically.

```python
import torch.nn as nn
from mentor import Mentee, Classifier

class MyClassifier(Mentee):
    def __init__(self, num_classes: int = 10):
        super().__init__(num_classes=num_classes)
        self.fc = nn.Linear(128, num_classes)
        self.trainer = Classifier()   # cross-entropy loss + accuracy out of the box

    def forward(self, x):
        return self.fc(x)
```

```python
import torch.nn as nn
from mentor import Mentee, Regressor

class MyRegressor(Mentee):
    def __init__(self, in_features: int = 8):
        super().__init__(in_features=in_features)
        self.fc = nn.Linear(in_features, 1)
        self.trainer = Regressor()    # MSE loss + RMSE metric out of the box

    def forward(self, x):
        return self.fc(x).squeeze(-1)
```

### Option B — Custom training step

Override {meth}`~mentor.Mentee.training_step` and optionally
{meth}`~mentor.Mentee.validation_step` for full control.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from mentor import Mentee

class MyClassifier(Mentee):
    def __init__(self, num_classes: int = 10):
        super().__init__(num_classes=num_classes)   # kwargs stored in checkpoint
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        loss = F.cross_entropy(self(x), y)
        return loss, {"loss": loss.item()}

    def validation_step(self, batch):
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        acc = (self(x).argmax(1) == y).float().mean().item()
        return {"acc": acc}
```

## Training loop

The loop is the same regardless of whether you use a built-in trainer or
custom step methods.

```python
model = MyClassifier(num_classes=10)
_to = model.create_train_objects(lr=1e-3)

opt, sched = _to["optimizer"], _to["lr_scheduler"]

for epoch in range(20):
    train_metrics = model.train_epoch(train_loader, opt, sched, verbose=True)
    val_metrics   = model.validate_epoch(val_loader)
    model.save("checkpoint.pt", optimizer=opt, lr_scheduler=sched)
    print(f"epoch {model.current_epoch}  "
          f"loss={train_metrics['loss']:.4f}  "
          f"acc={val_metrics['acc']:.4f}")
```

Alternatively, pass the cached `model.optimizer` / `model.lr_scheduler`
properties directly instead of unpacking `_to`:

```python
model.create_train_objects(lr=1e-3)

for epoch in range(20):
    model.train_epoch(train_loader, model.optimizer, model.lr_scheduler)
    model.validate_epoch(val_loader)
    model.save("checkpoint.pt")   # optimizer and scheduler auto-saved from cache
```

## Resuming training

```python
from mentor import Mentee

model, opt, sched = Mentee.resume_training(
    "checkpoint.pt",
    model_class=MyClassifier,
    device="cuda",
    lr=1e-4,
)
model.train_epoch(train_loader, opt, sched)
```

## Inference-only loading

```python
model = Mentee.resume("checkpoint.pt", model_class=MyClassifier)
model.eval()
with torch.no_grad():
    logits = model(x)
```

## Batteries-included inference state

Use {meth}`~mentor.Mentee.register_inference_state` to attach any data
computed from the training set (label names, vocabulary, normalisation
statistics) so the checkpoint is fully self-contained.

```python
# during training
model.register_inference_state("classes", ["cat", "dog", "bird"])
model.save("checkpoint.pt")

# at inference time — no external config needed
model = Mentee.resume("checkpoint.pt", model_class=MyClassifier)
classes = model.get_inference_state("classes")
```

## Gradient accumulation

Pass `pseudo_batch_size` to {meth}`~mentor.Mentee.train_epoch` to
accumulate gradients over multiple samples before each optimiser step:

```python
# effective batch size = 64, memory footprint of 8
model.train_epoch(train_loader, opt, pseudo_batch_size=8)
```

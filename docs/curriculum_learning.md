# Curriculum Learning Support

Curriculum learning — training in phases where different parts of the model
are active or learn at different speeds — requires two orthogonal controls:

- **which layers are trainable** (frozen / unfrozen)
- **how fast each layer learns** (per-layer LR coefficients)

torch_mentor keeps both as first-class Mentee state, persisted in every
checkpoint.  The optimizer is treated as a *derivative* of that state: it
can be updated in-place when possible, or rebuilt from scratch when the
structure changes.

---

## Design principles

**Progressive adoption with escape hatches.**
Every control can be used independently.  Freeze layers without touching LR
coefficients.  Set LR coefficients before the optimizer even exists.  Rebuild
the optimizer explicitly or let torch_mentor do it automatically.

**Source of truth lives in the model, not the optimizer.**
`_frozen_modules` and `_lr_coefficients` are the authoritative state.
The optimizer reflects them but is always reproducible from them via
`create_train_objects`.

**Rebuild on phase change, update in-place for fine adjustments.**
Changing which layers are trainable is a training-phase transition — fresh
Adam state for newly active layers is often desirable.  Adjusting a
coefficient by a small ratio is a fine adjustment — the ratio update
preserves accumulated scheduler decay.

---

## Layer inspection

Before freezing or assigning coefficients, inspect the available layer paths:

```python
print(model.layer_names)
# ['backbone', 'backbone.conv1', 'backbone.layer1', ..., 'head']
```

`layer_names` lists every parameter-bearing module in traversal order.
These are the strings accepted by `freeze`, `unfreeze`, `set_lr_coefficient`,
and `select_layers`.

Patterns are matched with `re.fullmatch`, so plain strings act as exact
selectors and regex patterns select groups:

```python
model.select_layers(["backbone"])               # exact match
model.select_layers([r"backbone\.layer[34]"])   # regex
model.select_layers(["head", "backbone"])       # order follows layer_names
```

---

## Freezing and unfreezing

### Basic usage

`freeze` and `unfreeze` accept a single string or a list of strings / regex
patterns:

```python
model.freeze("backbone")                     # freeze entire backbone
model.freeze([r"backbone\.layer[12]"])       # freeze first two layer groups
model.unfreeze("backbone.layer4")            # unfreeze one sub-layer
model.unfreeze(["backbone"])                 # unfreeze whole subtree
```

Both methods return `self` for chaining:

```python
model.freeze("backbone").freeze("neck")
```

### State persistence

Frozen state is saved in every checkpoint and restored automatically by
`resume` and `resume_training` — no extra bookkeeping required.

### Optimizer interaction

`freeze` sets `requires_grad=False` on the affected parameters.  If an
optimizer already exists, its param groups are left untouched: Adam skips
parameters whose gradient is `None`, so the frozen groups become inert
without any restructuring.

`unfreeze` sets `requires_grad=True`.  The optimizer interaction depends on
when the layer was frozen relative to when the optimizer was built:

| Situation | Behaviour |
|---|---|
| Layer was frozen *after* the optimizer was built | param group already exists; parameters become live again; Adam initialises their state on the first gradient step |
| Layer was frozen *before* the optimizer was built | no param group exists; a rebuild is required |

```python
# Layer frozen after optimizer was built — fast path, no state loss
model.create_train_objects(lr=1e-3)
model.freeze("backbone")
model.unfreeze("backbone")              # fast path — group already exists

# Layer frozen before optimizer was built — rebuild required
model.freeze("backbone")
model.create_train_objects(lr=1e-3)    # optimizer built without backbone
model.unfreeze("backbone", reset_optimizer_if_needed=True)   # triggers rebuild
# or raise if reset_optimizer_if_needed=False (default)
```

### Typical fine-tuning workflow

```python
import torch.nn as nn
from torchvision.models import resnet50
from mentor import Mentee, Classifier

class MyResNet(Mentee):
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet50(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Linear(2048, num_classes)
        self.trainer = Classifier()

    def forward(self, x):
        return self.head(self.backbone(x).flatten(1))

model = MyResNet(num_classes=10).to("cuda")

# Phase 1 — train head only
model.freeze("backbone")
model.create_train_objects(lr=1e-3)
model.fit(train_data, val_data, epochs=5, checkpoint_path="phase1.pt")

# Phase 2 — fine-tune everything with a lower global LR
model.unfreeze("backbone", reset_optimizer_if_needed=True)
model.create_train_objects(lr=1e-4)
model.fit(train_data, val_data, epochs=10, checkpoint_path="phase2.pt")
```

---

## Per-layer learning rate coefficients

`set_lr_coefficient` assigns a multiplier relative to the global learning
rate.  The effective LR for a layer is `global_lr × coefficient`.

```python
model.set_lr_coefficient(0.1, "backbone")          # 10× lower than global LR
model.set_lr_coefficient(0.0, "backbone.layer1")   # zero out one sub-layer
model.set_lr_coefficient(1.0, "backbone")          # restore default (removes entry)
model.set_lr_coefficient(0.1, [r"backbone\..*"])   # regex — all backbone sub-layers
```

Coefficients are stored in `_lr_coefficients` (a sparse dict; absent key
means 1.0) and are persisted in every checkpoint.

### Ancestor inheritance

Setting a coefficient on a parent module propagates to its children's param
groups.  This means you can set one coefficient for `"backbone"` and every
sub-layer (`backbone.layer1`, `backbone.layer1.conv1`, …) inherits it,
without needing to enumerate each one:

```python
model.set_lr_coefficient(0.01, "backbone")   # all of backbone at 1% LR
model.set_lr_coefficient(0.1,  "backbone.layer4")  # layer4 overrides to 10%
```

When `create_train_objects` is called, each param group resolves its
coefficient from the most specific matching entry in `_lr_coefficients`.

### Setting before the optimizer exists

Coefficients can be set at any point — before or after `create_train_objects`:

```python
model = MyResNet()
model.set_lr_coefficient(0.01, "backbone")   # stored only; no optimizer yet
model.create_train_objects(lr=1e-3)          # optimizer built with 0.01 × 1e-3 for backbone
```

### Live in-place update

When the optimizer already has a dedicated param group for the target layer
(built by `create_train_objects`), `set_lr_coefficient` updates the group's
`lr` in-place:

```python
group["lr"] *= new_coefficient / old_coefficient
```

This preserves any LR decay the scheduler has already applied.  The update
also propagates to descendant groups — setting a coefficient for `"backbone"`
updates `"backbone.layer1"`, `"backbone.layer1.conv1"`, and so on.

### Rebuild path

A rebuild via `create_train_objects` is triggered automatically when:

- the target layer has no dedicated param group (optimizer was built as a
  single flat group, or the layer was frozen when the optimizer was built), or
- the old coefficient was `0.0` and the new one is non-zero (ratio undefined).

```python
# Explicit rebuild (always safe)
model.set_lr_coefficient(0.1, "backbone")
model.create_train_objects(lr=1e-3)

# Automatic rebuild on demand
model.set_lr_coefficient(0.1, "backbone", reset_optimizer_if_needed=True)

# Raise instead of rebuild (default)
model.set_lr_coefficient(0.1, "backbone")   # RuntimeError if rebuild needed
```

### Layer-wise learning rate decay (LLRD)

A common transfer-learning technique applies exponentially decaying LRs from
the output layer back to the input:

```python
layers = model.layer_names          # ordered from input to output
n = len(layers)
decay = 0.9
for i, layer in enumerate(layers):
    coeff = decay ** (n - 1 - i)   # highest LR at output, lowest at input
    model.set_lr_coefficient(coeff, layer)

model.create_train_objects(lr=1e-3)
```

### Coefficient = 0.0

Setting a coefficient to `0.0` zeroes the LR for that layer without freezing
it (`requires_grad` is unchanged).  Gradients still flow — the layer just
receives no parameter update.  This is occasionally useful to "pause" a
layer temporarily while keeping it in the computational graph.

---

## Combining freeze and LR coefficients

The two systems are fully independent.  `_frozen_modules` controls gradient
flow; `_lr_coefficients` controls update magnitude.  A frozen layer with a
coefficient set will have the coefficient applied when it is unfrozen and the
optimizer is rebuilt:

```python
model.set_lr_coefficient(0.01, "backbone")
model.freeze("backbone")
# ... train head only ...
model.unfreeze("backbone", reset_optimizer_if_needed=True)
# backbone is now trainable at 0.01 × global_lr, as stored in _lr_coefficients
```

---

## API reference

### `freeze(patterns, optimizer=None, reset_optimizer_if_needed=False)`

Freeze layers matched by *patterns* (`str` or `list[str]`).  Updates
`_frozen_modules` and sets `requires_grad=False`.  Returns `self`.

### `unfreeze(patterns, optimizer=None, reset_optimizer_if_needed=False)`

Unfreeze layers matched by *patterns* (`str` or `list[str]`).  Updates
`_frozen_modules` and sets `requires_grad=True`.  If the optimizer lacks a
group for an unfrozen layer, raises `RuntimeError` unless
`reset_optimizer_if_needed=True`.  Returns `self`.

### `set_lr_coefficient(coefficient, patterns, optimizer=None, reset_optimizer_if_needed=False)`

Set LR coefficient for layers matched by *patterns* (`str` or `list[str]`).
Updates `_lr_coefficients`.  Updates optimizer param groups in-place where
possible; triggers rebuild or raises when not possible (controlled by
`reset_optimizer_if_needed`).  Returns `self`.

### `select_layers(patterns)`

Return the layer paths from `layer_names` that match any pattern in
*patterns*, deduplicated and sorted in module traversal order.  Raises
`ValueError` if a pattern matches nothing.  Used internally by `freeze`,
`unfreeze`, and `set_lr_coefficient`.

### `create_train_objects(lr, step_size, gamma, ...)`

Always reads `_frozen_modules` and `_lr_coefficients` to build one param
group per non-frozen layer with `lr = global_lr × coefficient`.  Calling
this is the safe "full rebuild" path after any structural change.

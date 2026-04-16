# HuggingFace Transfer Learning with mentor

This example fine-tunes a HuggingFace pretrained image classifier on
**Oxford Flowers-102** (102 flower categories, ~1020 training images) using
`mentor`'s two-stage curriculum: frozen backbone first, then full fine-tune.

It is the primary guide for wrapping HuggingFace models as `Mentee` objects
and covers the non-obvious complications that arise from doing so.

## Quick start

```bash
# Stage 1 + 2 — fresh run (downloads model and dataset automatically)
python examples/hf/classify.py train

# Resume from where you left off
python examples/hf/classify.py train -resume_fname ./tmp/mobilenetv2_flowers102.mentor.pt

# Inference
python examples/hf/classify.py inference -img flower.jpg
```

## Why Flowers-102?

| Property | Value |
|---|---|
| Training images | 1020 (~10 per class) |
| Validation images | 1020 |
| Classes | 102 fine-grained flower species |
| Native resolution | variable, 300–800 px |
| ImageNet overlap | large (many flower classes present) |

With only 10 labelled examples per class, training a classifier from scratch
is hopeless. Pretrained ImageNet features (edges, textures, shapes) transfer
directly, so a frozen MobileNetV2 backbone + a fresh linear head converges to
useful accuracy within a single epoch.  This gap is exactly what makes
Flowers-102 a good knowledge-transfer benchmark.

## Training recipe

| | Stage 1 (head only) | Stage 2 (full fine-tune) |
|---|---|---|
| Epochs | 0–10 | 10–40 |
| Backbone | frozen | unfrozen |
| LR | 1e-3 | 1e-4 |
| Optimizer | Adam | Adam |

The stage boundary is stored in the checkpoint as `current_epoch`, so the
script always resumes at the right stage with no manual bookkeeping.

## Wrapping HuggingFace models as Mentee

### The core pattern

```python
from transformers import AutoModelForImageClassification
import mentor

model = AutoModelForImageClassification.from_pretrained(
    "google/mobilenet_v2_1.0_224",
    num_labels=102,
    ignore_mismatched_sizes=True,   # replaces the 1000-class head
)
model = mentor.wrap_as_mentee(
    model,
    trainer=HFClassifier,
    constructor_params={"config": model.config},  # required — see below
)
```

`wrap_as_mentee` injects the full `Mentee` API (`fit`, `freeze`, `save`, …)
into the live HF model object without touching its weights or submodules.

### Why `constructor_params={"config": model.config}` is required

When `mentor` saves a checkpoint it records `class_name` and `class_module`
so it can reconstruct the model at resume time via:

```python
model_class(**constructor_params)
```

For a plain `nn.Module` the constructor typically takes simple scalar
arguments.  HuggingFace models require a `PretrainedConfig` object as their
first argument — passing `{}` results in a `TypeError` on resume.

The `config` object is picklable and self-contained: it captures
`num_labels`, hidden sizes, and all architectural hyperparameters, so
`MobileNetV2ForImageClassification(config=checkpoint["constructor_params"]["config"])`
faithfully reconstructs the architecture before the saved weights are loaded
on top.

### The classifier head must stay architecture-compatible

`mentor` reconstructs the model by calling `model_class(config=...)`, then
loading the saved `state_dict`.  If the classifier head was replaced with a
different `nn.Module` *after* construction (e.g. wrapping `nn.Linear` in
`nn.Sequential`), the `state_dict` keys change and `load_state_dict` fails
silently under the default `tolerate_irresumable_model=True`, returning a
**fresh random model** instead.

```python
# WRONG — Sequential changes state_dict keys (classifier.1.weight vs classifier.weight)
model.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(in_features, 102))

# RIGHT — keep the architecture that from_pretrained produces
# num_labels=102 already gives nn.Linear(hidden_size, 102) as the head
```

If you need dropout in the head, use weight decay in the optimizer:

```python
model.fit(..., weight_decay=1e-4)
```

Or subclass the model so the custom head is part of `__init__` and therefore
matches what `model_class(config=...)` produces.

### HF models return structured outputs, not bare tensors

`AutoModelForImageClassification` returns an `ImageClassifierOutput` object,
not a raw tensor.  The default `mentor.Classifier` trainer calls
`loss_fn(model(x), y)` which fails because `loss_fn` expects a tensor.

The fix is a one-method subclass that unwraps `.logits`:

```python
class HFClassifier(mentor.Classifier):
    @classmethod
    def default_training_step(cls, model, batch, loss_fn=None):
        x, y = batch
        x, y = x.to(model.device), y.to(model.device)
        logits = model(pixel_values=x).logits   # unwrap structured output
        eff_fn = loss_fn if loss_fn is not None else F.cross_entropy
        loss = eff_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean().item()
        return loss, {"acc": acc, "loss": loss.item()}
```

Note `pixel_values=x` — HF image models use keyword arguments, not
positional, for their primary input tensor.

### Leeching preprocessing from the HF processor

HF ships an `AutoImageProcessor` alongside each model that encodes the
correct resize size, centre-crop, mean, and std.  Rather than hardcoding
these values, extract them at startup and build a native torchvision
transform pipeline:

```python
def make_transform(processor):
    size = processor.size.get("shortest_edge", processor.size.get("height", 224))
    return tv.transforms.Compose([
        tv.transforms.Resize(size),
        tv.transforms.CenterCrop(size),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
    ])
```

Calling the processor object directly inside the DataLoader workers (one
call per sample) is 3–4× slower because of Python-level overhead and
PIL → numpy → tensor round-trips.  The torchvision-native pipeline above
runs in C++ inside the worker processes and is the correct approach for
training throughput.

## Curriculum training: two-stage freeze / fine-tune

```python
# Stage 1 — freeze backbone, train head only
model.freeze("mobilenet_v2")
model.fit(train_data, val_data=val_data, epochs=10, lr=1e-3, ...)

# Stage 2 — unfreeze backbone, fine-tune everything
model.unfreeze("mobilenet_v2")
model.fit(train_data, val_data=val_data, epochs=40, lr=1e-4, ...)
```

`mentor.Mentee.freeze` and `unfreeze` accept regex patterns matched against
the full dotted layer path.  Use `mtr_checkpoint view` to explore the layer
tree:

```bash
mtr_checkpoint view -path ./tmp/mobilenetv2_flowers102.mentor.pt
```

The `epochs` argument to `fit` is a **ceiling**, not a count.  After resuming
at epoch 7, `fit(epochs=10)` runs 3 more epochs to reach epoch 10 — not 10
additional ones.  This means the stage boundary logic works unchanged across
resume:

```python
if model.current_epoch < args.freeze_epochs:
    model.freeze("mobilenet_v2")
    model.fit(..., epochs=args.freeze_epochs, ...)

model.unfreeze("mobilenet_v2")
model.fit(..., epochs=args.total_epochs, ...)
```

## Resuming a HuggingFace Mentee checkpoint

```python
model = mentor.Mentee.resume(
    "./tmp/mobilenetv2_flowers102.mentor.pt",
    trainer=HFClassifier,   # trainer is not serialised — must be re-supplied
)
```

Two things are re-applied automatically at resume time:

1. **Mentee mixin** — `wrap_as_mentee` stores the original HF class name in
   the checkpoint.  `resume` detects this and re-inserts `Mentee` into the
   MRO so the returned object is always a proper `Mentee` instance.

2. **Instance state** — attributes like `_train_history`, `_frozen_modules`,
   and `_lr_coefficients` are injected before the `state_dict` is loaded,
   because HF model `__init__` does not call `Mentee.__init__` via `super()`.

The `trainer` argument must be re-supplied because trainer instances are not
serialised inside the checkpoint (they are stateless strategy objects and may
live in user scripts that are not importable at resume time).

## What cannot easily be done

| Limitation | Workaround |
|---|---|
| Custom head architecture diverges from `from_pretrained` output | Subclass the HF model so the custom head is built in `__init__` |
| Optimizer state not restored by `Mentee.resume` | Use `Mentee.resume_training` instead |
| Multi-modal HF models (e.g. CLIP, LLaVA) | Override `default_training_step` and `default_validate_step` for each input modality |
| HF models with multiple outputs (detection, segmentation) | Same — the `.logits` unwrap pattern extends naturally |

## Source

```{literalinclude} ../../examples/hf/classify.py
:language: python
:linenos:
```

# CIFAR-10 with ResNet

Three example scripts train on CIFAR-10 using mentor, in order of
increasing complexity.  They cover the full workflow — argument parsing,
training, validation, checkpointing, and resuming — from a hand-written
training step all the way to reproducing a published benchmark.

---

## train_cifar.py — custom training step

`examples/cifar/train_cifar.py` subclasses {class}`~mentor.Mentee` directly
and overrides `training_step` and `validation_step` for full control over
the loss and metrics.

```bash
python examples/cifar/train_cifar.py \
    -resume_path cifar.pt \
    -epochs 30 -batch_size 64 -pseudo_batch 2 \
    -lr 0.001 -resnet resnet18 -pretrained true \
    -device cuda -verbose true
```

Resuming uses the same path:

```bash
python examples/cifar/train_cifar.py -resume_path cifar.pt -epochs 60
```

### Key design decisions

**Single path for save and load**
: `-resume_path` is used both for loading an existing checkpoint and for
  writing each new checkpoint.  If the file does not exist, training starts
  from scratch.

**BatchNorm in eval mode during training**
: CIFAR batches can be as small as 1 sample (e.g. the last batch with
  `drop_last=False`), which would cause BatchNorm to fail.
  `CifarResNet` overrides {meth}`~torch.nn.Module.train` to call `m.eval()`
  on every BatchNorm layer after `super().train()`.

**Gradient accumulation**
: `pseudo_batch` accumulates gradients over several mini-batches before
  calling `optimizer.step()`, allowing a larger effective batch size without
  increasing GPU memory.

**Inference state**
: The CIFAR class names are registered with
  {meth}`~mentor.Mentee.register_inference_state` so any checkpoint is
  self-contained for inference.

### Source

```{literalinclude} ../../examples/cifar/train_cifar.py
:language: python
:linenos:
```

---

## train_cifar_classifier.py — built-in Classifier trainer

`examples/cifar/train_cifar_classifier.py` demonstrates the composition
pattern: the model subclasses {class}`~mentor.Mentee`, assigns
`self.trainer = Classifier()` in `__init__`, and only implements `forward`.
The trainer supplies cross-entropy loss, top-1 accuracy, and a default Adam
optimiser automatically.

```bash
python examples/cifar/train_cifar_classifier.py \
    -resume_path cifar2.pt -epochs 20 -verbose true
```

### Source

```{literalinclude} ../../examples/cifar/train_cifar_classifier.py
:language: python
:linenos:
```

---

## train_cifar_resnet56.py — reproducing He et al. (2016)

`examples/cifar/train_cifar_resnet56.py` replicates the CIFAR-10 result
from *Deep Residual Learning for Image Recognition* (He et al., 2016):
**6.97 % test error (~93 % top-1 accuracy)** with ResNet-56, matching the
paper's SGD recipe exactly.

It is the primary example of writing a **custom** {class}`~mentor.trainers.MentorTrainer`
that deviates from the built-in Adam + StepLR defaults.

```bash
# fresh run
python examples/cifar/train_cifar_resnet56.py

# resume, show progress bars
python examples/cifar/train_cifar_resnet56.py \
    -resume_path ./tmp/resnet56.pt -epochs 200 -verbose true
```

### Performance

Measured on an RTX 3090 (batch size 128, single GPU, < 1 GB GPU memory):

| Metric | Value |
|---|---|
| Throughput | ~43 iterations / sec |
| Total runtime | ~30 min (78 K iterations) |
| Peak GPU memory | < 1 GB |
| Best validation accuracy | ~93.02 % |

The validation-loss curve below shows the characteristic three-step staircase
produced by the iteration-based LR schedule:

```{figure} ../_static/cifar_56_loss.png
:alt: Validation loss over 200 epochs — three sharp drops at the LR milestones
:align: center

Validation loss over 200 epochs.  The dotted vertical line marks epoch 0
(baseline before training); the three drops correspond to LR reductions
at 32 K, 48 K, and 64 K iterations (~epochs 82, 123, 164).
```

Reproduce the plot from a finished checkpoint:

```bash
mtr_plot_file_hist -paths ./tmp/resnet56.pt -verbose \
    -values validate/loss -output /tmp/cifar_56_loss.png
```

### Key design decisions

**SGD instead of Adam**
: The built-in {class}`~mentor.trainers.Classifier` and
  {class}`~mentor.trainers.Regressor` trainers use Adam.
  `CifarSGDResnetClassifier` overrides `create_train_objects` to create
  an SGD optimiser with momentum 0.9 and weight decay 1e-4 — the settings
  from the paper.  Assigning `self.trainer = CifarSGDResnetClassifier()` in
  the model's `__init__` is sufficient; {class}`~mentor.Mentee` delegates
  `create_train_objects`, `training_step`, and `validation_step` to the
  trainer automatically.

**Iteration-based LR schedule**
: The paper's milestones (32 K / 48 K / 64 K iterations) do not align with
  epoch boundaries for all batch sizes.  `IterationMultiStepLR` reads
  {attr}`~mentor.Mentee.total_train_iterations` — a cumulative batch counter
  maintained and checkpointed by {class}`~mentor.Mentee` — instead of
  carrying its own state.  `state_dict()` therefore returns `{}`, and
  `load_state_dict()` simply re-derives the correct LR from the restored
  counter.  The schedule survives resume unchanged, even across machines or
  batch-size changes.

**First metric key is the principal metric**
: `default_training_step` returns `{"acc": acc, "loss": loss.item()}` with
  `acc` first.  {meth}`~mentor.Mentee.validate_epoch` always *maximises* the
  first key when selecting the best checkpoint, so a higher-is-better metric
  must come first.  Putting `loss` first would cause the untrained model
  (highest loss) to be permanently recorded as "best".

### Source

```{literalinclude} ../../examples/cifar/train_cifar_resnet56.py
:language: python
:linenos:
```

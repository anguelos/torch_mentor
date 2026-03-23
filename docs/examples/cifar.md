# CIFAR-10 with ResNet

Two example scripts train a torchvision ResNet on CIFAR-10 using mentor.
They demonstrate the same full workflow — argument parsing, training,
validation, checkpointing, and resuming — using the two main patterns
the framework supports.

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

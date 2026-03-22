# mentor.trainers

Built-in training strategies for {class}`~mentor.Mentee`.

A {class}`~mentor.trainers.MentorTrainer` is a **pure-Python strategy object**
(not an `nn.Module`) that is composed into a `Mentee` via `self.trainer`.
It separates:

- **State** — the optimizer, LR scheduler, and default loss function cached
  by {meth}`~mentor.trainers.MentorTrainer.create_train_objects`
  (exposed as read-only properties).
- **Logic** — the forward/loss/metrics computation in
  {meth}`~mentor.trainers.MentorTrainer.default_training_step` and
  {meth}`~mentor.trainers.MentorTrainer.default_validate_step` (classmethods,
  callable without a trainer instance).

When `self.trainer` is set on a `Mentee`, its `training_step` and
`validation_step` automatically delegate to the trainer's classmethods,
injecting the cached `loss_fn`.

## MentorTrainer

```{autoclass} mentor.trainers.MentorTrainer
:members:
:special-members: __init__
:show-inheritance:
```

## Classifier

```{autoclass} mentor.trainers.Classifier
:members:
:show-inheritance:
```

## Regressor

```{autoclass} mentor.trainers.Regressor
:members:
:show-inheritance:
```

"""
Training-task strategy classes (trainers) for Mentee.

A trainer bundles the task-specific parts of a training loop — the default
loss function, the training step, and ``create_train_objects`` — so that a
concrete model only needs to implement ``forward``.

Trainers are **not** ``nn.Module`` subclasses.  Their state is limited to
the training objects cached by :meth:`MentorTrainer.create_train_objects`
(optimizer, LR scheduler, loss function), exposed as read-only properties.

Training logic lives in two **classmethods** so it can be called without a
trainer instance:

* :meth:`~MentorTrainer.default_training_step` — abstract, must be implemented
* :meth:`~MentorTrainer.default_validate_step`  — defaults to unpacking
  ``default_training_step``

When a :class:`~mentor.Mentee` has ``self.trainer`` set, its
``training_step`` and ``validation_step`` automatically delegate to the
trainer's classmethods, injecting the cached ``loss_fn``.

Usage::

    class MyNet(Mentee):
        def __init__(self, num_classes: int = 10) -> None:
            super().__init__(num_classes=num_classes)
            self.fc = torch.nn.Linear(128, num_classes)
            self.trainer = Classifier()

        def forward(self, x):
            return self.fc(x.flatten(1))

    model = MyNet(num_classes=10)
    train_objs = model.create_train_objects(lr=1e-3)
    model.train_epoch(loader, model.optimizer)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MentorTrainer(ABC):
    """Abstract base class for Mentee training strategies.

    A trainer separates *state* (the optimizer, LR scheduler, and loss
    function produced by :meth:`create_train_objects`) from *logic* (the
    forward/loss/metrics computation in :meth:`default_training_step` and
    :meth:`default_validate_step`).

    **State** — per-instance, ``None`` until :meth:`create_train_objects`
    is called:

    * :attr:`optimizer`
    * :attr:`lr_scheduler`
    * :attr:`loss_fn`

    **Logic** — class-level, callable without an instance:

    * :meth:`default_training_step` ``(cls, model, batch, loss_fn=None)``
      — **abstract**, must be overridden.
    * :meth:`default_validate_step` ``(cls, model, batch, loss_fn=None)``
      — default unpacks :meth:`default_training_step`; override when the
      validation pass differs from training.

    When a trainer is assigned to a :class:`~mentor.Mentee`, the model's
    ``training_step`` and ``validation_step`` automatically route to these
    classmethods with the cached :attr:`loss_fn` pre-injected.
    """

    def __init__(self) -> None:
        self._optimizer: Optional[Any] = None
        self._lr_scheduler: Optional[Any] = None
        self._loss_fn: Optional[Any] = None

    # ------------------------------------------------------------------
    # Read-only state properties
    # ------------------------------------------------------------------

    @property
    def optimizer(self) -> Optional[Any]:
        """Optimizer cached by the last :meth:`create_train_objects` call."""
        return self._optimizer

    @property
    def lr_scheduler(self) -> Optional[Any]:
        """LR scheduler cached by the last :meth:`create_train_objects` call."""
        return self._lr_scheduler

    @property
    def loss_fn(self) -> Optional[Any]:
        """Default loss callable registered by :meth:`create_train_objects`."""
        return self._loss_fn

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def default_training_step(
        cls,
        model: Any,
        batch: Any,
        loss_fn: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the loss and metrics for one training batch.

        This is a **classmethod** so it can be inspected and called without
        a trainer instance.  The :class:`~mentor.Mentee` injects the
        trainer's cached :attr:`loss_fn` before calling this method.

        Parameters
        ----------
        model : Mentee
            The model being trained.
        batch : Any
            One element from the training DataLoader.
        loss_fn : callable, optional
            Effective loss function — either an explicit override or the
            cached default forwarded by the Mentee.

        Returns
        -------
        loss : torch.Tensor
            Scalar differentiable loss.
        metrics : dict[str, float]
            Scalar metrics; the first key is the principal metric.
        """

    @classmethod
    def default_validate_step(
        cls,
        model: Any,
        batch: Any,
        loss_fn: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Evaluate one validation batch.

        Default implementation calls :meth:`default_training_step` and
        strips the loss tensor, returning only the metrics dict.  Override
        when the validation forward pass differs from training.

        Parameters
        ----------
        model : Mentee
            The model being evaluated.
        batch : Any
            One element from the validation DataLoader.
        loss_fn : callable, optional
            Effective loss function forwarded by the Mentee.

        Returns
        -------
        dict[str, float]
            Scalar evaluation metrics.
        """
        result = cls.default_training_step(model, batch, loss_fn)
        if isinstance(result, tuple) and len(result) == 2:
            _, metrics = result
            return metrics
        return result

    @abstractmethod
    def create_train_objects(
        self,
        model: Any,
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.1,
        loss_fn: Optional[Any] = None,
        overwrite_default_loss: bool = False,
    ) -> Dict[str, Any]:
        """Create and cache the optimizer, LR scheduler, and default loss.

        Parameters
        ----------
        model : Mentee
            Model whose ``parameters()`` are passed to the optimizer.
        lr, step_size, gamma
            Standard Adam + StepLR hyperparameters.
        loss_fn : callable, optional
            Loss to register as default.
        overwrite_default_loss : bool, optional
            Replace an existing cached loss when ``True``.

        Returns
        -------
        dict
            ``{"optimizer": ..., "lr_scheduler": ..., "loss_fn": ...}``
        """


class Classifier(MentorTrainer):
    """Training strategy for multi-class classification with cross-entropy loss.

    :meth:`default_training_step` computes cross-entropy loss and top-1
    accuracy.  :meth:`create_train_objects` registers
    ``nn.CrossEntropyLoss()`` automatically.

    The effective loss passed to :meth:`default_training_step` is resolved
    by the owning :class:`~mentor.Mentee` in this order:

    1. Explicit ``loss_fn`` argument to ``model.training_step``.
    2. :attr:`loss_fn` cached by :meth:`create_train_objects`.
    3. :func:`torch.nn.functional.cross_entropy` (stateless fallback).

    Examples
    --------
    >>> class MyNet(Mentee):
    ...     def __init__(self, num_classes=10):
    ...         super().__init__(num_classes=num_classes)
    ...         self.fc = nn.Linear(128, num_classes)
    ...         self.trainer = Classifier()
    ...     def forward(self, x):
    ...         return self.fc(x.flatten(1))
    >>> model = MyNet(num_classes=5)
    >>> model.create_train_objects(lr=1e-3)
    >>> isinstance(model.loss_fn, nn.CrossEntropyLoss)
    True
    """

    @classmethod
    def default_training_step(
        cls,
        model: Any,
        batch: Any,
        loss_fn: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Cross-entropy loss and top-1 accuracy.

        Parameters
        ----------
        model : Mentee
            Classification model; ``model(x)`` returns logits.
        batch : tuple[Tensor, Tensor]
            ``(inputs, targets)`` — targets are class indices (long).
        loss_fn : callable, optional
            Effective loss; falls back to ``F.cross_entropy`` when ``None``.
        """
        x, y = batch
        x, y = x.to(model.device), y.to(model.device)
        logits = model(x)
        eff_fn = loss_fn if loss_fn is not None else F.cross_entropy
        loss = eff_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean().item()
        return loss, {"loss": loss.item(), "acc": acc}

    def create_train_objects(
        self,
        model: Any,
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.1,
        loss_fn: Optional[Any] = None,
        overwrite_default_loss: bool = False,
    ) -> Dict[str, Any]:
        """Adam + StepLR with ``nn.CrossEntropyLoss`` as default loss.

        Registers ``nn.CrossEntropyLoss()`` automatically when no loss is
        cached yet and *loss_fn* is ``None``.
        """
        if loss_fn is not None and (overwrite_default_loss or self._loss_fn is None):
            self._loss_fn = loss_fn
        elif self._loss_fn is None:
            self._loss_fn = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self._lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, step_size=step_size, gamma=gamma
        )
        return {
            "optimizer": self._optimizer,
            "lr_scheduler": self._lr_scheduler,
            "loss_fn": self._loss_fn,
        }


class Regressor(MentorTrainer):
    """Training strategy for regression with mean-squared-error loss.

    :meth:`default_training_step` computes MSE loss and RMSE metric.
    :meth:`create_train_objects` registers ``nn.MSELoss()`` automatically.

    Targets are cast to ``float`` automatically.

    Examples
    --------
    >>> class MyNet(Mentee):
    ...     def __init__(self, in_features=10):
    ...         super().__init__(in_features=in_features)
    ...         self.fc = nn.Linear(in_features, 1)
    ...         self.trainer = Regressor()
    ...     def forward(self, x):
    ...         return self.fc(x).squeeze(-1)
    >>> model = MyNet(in_features=8)
    >>> model.create_train_objects(lr=1e-3)
    >>> isinstance(model.loss_fn, nn.MSELoss)
    True
    """

    @classmethod
    def default_training_step(
        cls,
        model: Any,
        batch: Any,
        loss_fn: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """MSE loss and RMSE metric.

        Parameters
        ----------
        model : Mentee
            Regression model; output and target must have compatible shapes.
        batch : tuple[Tensor, Tensor]
            ``(inputs, targets)`` — targets cast to float.
        loss_fn : callable, optional
            Effective loss; falls back to ``F.mse_loss`` when ``None``.
        """
        x, y = batch
        x, y = x.to(model.device), y.float().to(model.device)
        pred = model(x)
        eff_fn = loss_fn if loss_fn is not None else F.mse_loss
        loss = eff_fn(pred, y)
        return loss, {"loss": loss.item(), "rmse": loss.item() ** 0.5}

    def create_train_objects(
        self,
        model: Any,
        lr: float = 1e-3,
        step_size: int = 10,
        gamma: float = 0.1,
        loss_fn: Optional[Any] = None,
        overwrite_default_loss: bool = False,
    ) -> Dict[str, Any]:
        """Adam + StepLR with ``nn.MSELoss`` as default loss.

        Registers ``nn.MSELoss()`` automatically when no loss is cached yet
        and *loss_fn* is ``None``.
        """
        if loss_fn is not None and (overwrite_default_loss or self._loss_fn is None):
            self._loss_fn = loss_fn
        elif self._loss_fn is None:
            self._loss_fn = nn.MSELoss()
        self._optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self._lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self._optimizer, step_size=step_size, gamma=gamma
        )
        return {
            "optimizer": self._optimizer,
            "lr_scheduler": self._lr_scheduler,
            "loss_fn": self._loss_fn,
        }
